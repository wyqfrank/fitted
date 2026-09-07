#!/usr/bin/env python
"""Train the frozen-encoder pairwise outfit ranker (PRD § ML System, Plan A).

    python services/inference/scripts/train_ranker.py
    python services/inference/scripts/train_ranker.py --dims 16 --artifact-dir artifacts/new-run

Embeds the labelling pool with a frozen DINOv2-S, reduces to a few dozen
dimensions, and fits a linear pairwise ranker on the collected A/B decisions.

The frozen encoder is a data-driven baseline for the small labelled image pool.
Its head capacity is an experimental choice, not an established upper bound.
See docs/ranker-audit.md for limitations of the historical capacity experiments.

    score(image)      = w · P(embed(image))
    P(A preferred)    = sigmoid((score(A) - score(B)) / temperature)

There is no intercept, so the model is antisymmetric by construction: swapping
A and B provably flips the prediction. The PRD asks for swap consistency as an
evaluation metric; here it is a property of the architecture rather than
something to measure and hope for.

Training-only code. It lives outside `src/fitted_inference` because the wheel
ships only the deployable package.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ranker_artifact import (
    ACTIVATIONS,
    CALIBRATION_QUANTILES,
    load_scorer,
    numpy_activation,
)
from ranker_evaluation import (
    embedding_metadata,
    file_hash,
    group_split,
    load_embeddings,
    load_manifest,
    paired_summary,
    partition_pairs,
    read_jsonl,
    validate_teacher_cache,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
POOL_DIR = REPO_ROOT / "apps" / "web" / "public" / "label-pool"
DECISIONS_DIR = REPO_ROOT / "data" / "labelling"
ARTIFACT_DIR = REPO_ROOT / "models" / "ranker"
CACHE_PATH = ARTIFACT_DIR / "embeddings.npz"
TEACHER_POOL_DIR = REPO_ROOT / "data" / "teacher-pool"
TEACHER_CACHE_PATH = ARTIFACT_DIR / "teacher-embeddings.npz"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
ENCODER = "facebook/dinov2-small"
RANKER_VERSIONS = {"linear": "dinov2s-pca-linear-v1", "mlp": "dinov2s-pca-mlp-v1"}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


@dataclass
class Comparison:
    """One rater decision, in the order the rater actually saw it.

    We deliberately use `shownLeftId`/`shownRightId` rather than parsing the
    pair id. `pairId()` in apps/web/lib/labelling/pairing.ts *sorts* the two
    ids, so the canonical order behind `verdict`/`target` cannot be recovered
    from the id string. The shown fields carry their own orientation and need
    no reconstruction.
    """

    left: str
    right: str
    # 1.0 = left preferred, 0.0 = right preferred, 0.5 = too close to call.
    label: float
    split: str
    rater: str


def load_decisions(
    directory: Path = DECISIONS_DIR, raters: set[str] | None = None
) -> list[Comparison]:
    """Read every rater's JSONL file. One file per rater, merged here.

    Point `directory` at a frozen snapshot when comparing configurations.
    Raters label continuously, so the evaluation splits grow underneath a live
    directory and two runs minutes apart are scored on different data.

    `raters` restricts the set by id. FITTED targets a defined audience, and
    pooling judges who do not share a preference produces a target that no
    single viewer holds — measurably so: see the PRD's inter-rater table.
    """
    files = sorted(directory.glob("decisions.*.jsonl"))
    if not files:
        sys.exit(f"No decision files in {directory}. Collect labels at /label first.")

    verdict_to_label = {"a": 1.0, "b": 0.0, "close": 0.5}
    out: list[Comparison] = []
    unjudgeable = 0

    for file in files:
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if raters is not None and row["raterId"].upper() not in raters:
                continue
            label = verdict_to_label.get(row["shownVerdict"])
            if label is None:
                # "Cannot judge" is a frame-quality signal, not a preference.
                # The PRD excludes these rows from the training target.
                unjudgeable += 1
                continue
            out.append(
                Comparison(
                    left=row["shownLeftId"],
                    right=row["shownRightId"],
                    label=label,
                    split=row["split"],
                    rater=row["raterId"],
                )
            )

    per_rater = Counter(c.rater for c in out)
    present = sorted(per_rater)
    print(
        f"decisions: {len(out)} usable, {unjudgeable} unjudgeable, "
        f"raters={ {r: per_rater[r] for r in present} }"
    )
    if raters is not None:
        missing = sorted(raters - {r.upper() for r in present})
        if missing:
            sys.exit(f"Requested rater(s) with no decisions on file: {', '.join(missing)}")
    if len(present) == 1:
        print(
            "  NOTE: one rater only. Inter-rater agreement is unmeasurable, so\n"
            "  there is no ceiling to compare the test number against."
        )
    return out


def resolve_images(comparisons: list[Comparison]) -> dict[str, Path]:
    """Map every referenced image id to a file on disk.

    Ids are filename stems (see readPool in apps/web/lib/labelling/store.ts),
    so this is a lookup rather than a search.
    """
    needed = {c.left for c in comparisons} | {c.right for c in comparisons}

    on_disk: dict[str, Path] = {}
    if POOL_DIR.is_dir():
        for path in POOL_DIR.iterdir():
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                on_disk[path.stem] = path

    missing = sorted(needed - on_disk.keys())
    if missing:
        print(f"\n{len(missing)} of {len(needed)} labelled images are missing from {POOL_DIR}")
        for stem in missing[:5]:
            print(f"  {stem}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")
        sys.exit(
            "\nThe pool is gitignored, so it does not survive a fresh clone.\n"
            "Restore the exact same photo set and re-ingest:\n"
            "  node scripts/ingest-label-pool.mjs <photo-dir> --clear\n"
            "  node scripts/ingest-label-pool.mjs --fingerprint\n"
            "Ids are hashed from filenames, so a different photo set produces\n"
            "different ids and orphans every decision already collected."
        )

    return {stem: on_disk[stem] for stem in needed}


def image_splits(comparisons: list[Comparison]) -> dict[str, str]:
    """Derive each image's split from the decisions themselves.

    Splits are assigned per subject in TypeScript. Rather than reimplement that
    RNG in Python and risk a silent mismatch, take the split each decision
    recorded. Every pair lies within one split, so both of its images inherit
    it — and a contradiction means the pool changed under the labels.
    """
    splits: dict[str, str] = {}
    for c in comparisons:
        for image in (c.left, c.right):
            previous = splits.setdefault(image, c.split)
            if previous != c.split:
                sys.exit(
                    f"Image {image} appears in both '{previous}' and '{c.split}' pairs.\n"
                    "The pool changed after these labels were collected; re-ingest the\n"
                    "original photo set before training."
                )
    return splits


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------


def embed_pool(
    images: dict[str, Path],
    batch_size: int,
    refresh: bool,
    cache_path: Path = CACHE_PATH,
    *,
    allow_legacy: bool = False,
) -> dict[str, np.ndarray]:
    """Frozen DINOv2-S embeddings, cached by image id.

    Cached because embedding is the slow half (~70 ms/image on CPU) and the
    regularisation sweep below wants to re-run in seconds.
    """
    cached: dict[str, np.ndarray] = {}
    if cache_path.exists() and not refresh:
        cached, provenance = load_embeddings(cache_path, images, ENCODER, allow_legacy=allow_legacy)
        print(f"embedding cache provenance: {provenance}")

    todo = sorted(set(images) - set(cached))
    if not todo:
        print(f"embeddings: {len(cached)} cached, 0 to compute")
        return {stem: cached[stem] for stem in images}

    # Imported late so --help and the data checks above work without torch.
    import torch
    from PIL import Image
    from transformers import AutoModel

    print(f"embeddings: {len(cached)} cached, {len(todo)} to compute with {ENCODER}")
    model = AutoModel.from_pretrained(ENCODER).eval()

    # ImageNet statistics, matching the encoder's own preprocessing.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def to_tensor(path: Path) -> np.ndarray:
        """Letterbox to square, then resize to 224.

        The stock processor resizes the short edge and centre-crops, which
        slices the shoes or the head off a full-body outfit photo — exactly
        the evidence being judged. Padding preserves the whole garment at the
        cost of some border.
        """
        image = Image.open(path).convert("RGB")
        side = max(image.size)
        square = Image.new("RGB", (side, side), (255, 255, 255))
        square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        array = np.asarray(square.resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255.0
        return ((array - mean) / std).transpose(2, 0, 1)

    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(todo), batch_size):
            chunk = todo[start : start + batch_size]
            batch = torch.from_numpy(np.stack([to_tensor(images[stem]) for stem in chunk]))
            # CLS token: DINOv2's global image descriptor.
            output = model(pixel_values=batch).last_hidden_state[:, 0].numpy()
            for stem, vector in zip(chunk, output, strict=True):
                cached[stem] = vector.astype(np.float32)
            done = min(start + batch_size, len(todo))
            print(f"  {done}/{len(todo)}  ({(time.time() - started) / done:.2f}s per image)")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **cached)
    metadata = embedding_metadata(images, ENCODER)
    metadata["cacheSha256"] = file_hash(cache_path)
    metadata["encoderRevision"] = getattr(model.config, "_commit_hash", None)
    write_json(cache_path.with_suffix(".metadata.json"), metadata)
    print(f"  cached to {cache_path}")
    return {stem: cached[stem] for stem in images}


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def fit_projection(
    train_vectors: np.ndarray, dims: int, *, mode: str = "pca"
) -> tuple[np.ndarray, np.ndarray]:
    """PCA basis fitted on training images only.

    Fitting on the whole pool would leak held-out images into the
    representation and inflate the test number.
    """
    if train_vectors.ndim != 2 or not len(train_vectors) or dims < 1:
        raise ValueError("Projection needs nonempty training vectors and positive dimensions")
    if not np.isfinite(train_vectors).all():
        raise ValueError("Projection vectors must be finite")
    centre = train_vectors.mean(axis=0)
    if mode == "identity":
        return centre, np.eye(train_vectors.shape[1], dtype=train_vectors.dtype)
    if mode != "pca":
        raise ValueError(f"Unknown projection mode: {mode}")
    _, _, vt = np.linalg.svd(train_vectors - centre, full_matrices=False)
    return centre, vt[:dims]


def project(
    vectors: np.ndarray, centre: np.ndarray, basis: np.ndarray, *, normalize: bool = True
) -> np.ndarray:
    """Centre, project, and L2-normalise so the ranker sees a bounded scale."""
    reduced = (vectors - centre) @ basis.T
    if not normalize:
        return reduced
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    return reduced / np.maximum(norms, 1e-8)


def _torch_activation(name: str):
    import torch

    if name == "tanh":
        return torch.tanh
    if name == "relu":
        return torch.relu
    if name == "gelu":
        # tanh-approximate form so the numpy scorer (ranker_artifact.numpy_activation)
        # can match it exactly without an erf dependency; torch's exact-erf gelu
        # differs from this by <1e-3 everywhere, immaterial next to the effect
        # being measured.
        return lambda x: torch.nn.functional.gelu(x, approximate="tanh")
    raise ValueError(f"unknown activation {name!r}, expected one of {ACTIVATIONS}")


def fit_head(
    z_left: np.ndarray,
    z_right: np.ndarray,
    y: np.ndarray,
    l2: float,
    *,
    head: str = "linear",
    hidden: int = 16,
    activation: str = "relu",
    prior: dict[str, np.ndarray] | None = None,
    seed: int = 0,
    steps: int = 400,
) -> dict[str, np.ndarray]:
    """Fit a per-image scorer f(z), trained on the margin f(z_left) - f(z_right).

    A per-image scorer rather than a function of the difference vector, so
    `margin(a, b) == -margin(b, a)` by construction (antisymmetry survives the
    head change for free) and the fitted parameters can score one image alone
    (what benchmark_judges.py and the live path both need).

    Soft labels: a 0.5 "too close to call" contributes genuine information
    (these two outfits are near-equal) and BCE handles it without a special
    case, so ties stay in the training set instead of being thrown away.

    `prior` turns the penalty into a proximal term pulling towards an existing
    parameter dict rather than towards zero, summed tensor-by-tensor. That is
    how the distilled student is fine-tuned: the teacher's parameters are the
    starting point, and the few hundred human pairs are only allowed to move
    them so far.

    linear reuses the original LBFGS body verbatim (same optimiser, same
    steps, same single `step(closure)`) so it is bit-identical to the old
    `fit_ranker`. mlp is non-convex, so LBFGS's strong-Wolfe line search would
    give seed-dependent, occasionally divergent results; full-batch Adam is
    steadier and the dataset is tiny enough that minibatching buys nothing.

    The MLP branch preserves the historical output-only penalty for reproduction.
    It does not adequately constrain the whole function: ReLU permits rescaling
    w1 and w2 without changing predictions. Both tanh and GELU are approximately
    linear near zero. These recipes cannot rule out nonlinear capacity; use a
    separately controlled experiment before interpreting their performance.
    """
    import torch

    dims = z_left.shape[1]
    left = torch.from_numpy(z_left.astype(np.float32))
    right = torch.from_numpy(z_right.astype(np.float32))
    targets = torch.from_numpy(y.astype(np.float32))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    act = _torch_activation(activation) if head == "mlp" else None

    def zeros() -> dict[str, torch.Tensor]:
        if head == "linear":
            return {"weights": torch.zeros(dims, dtype=torch.float32)}
        return {
            "w1": torch.zeros(hidden, dims, dtype=torch.float32),
            "b1": torch.zeros(hidden, dtype=torch.float32),
            "w2": torch.zeros(hidden, dtype=torch.float32),
        }

    anchor = zeros()
    if prior is not None:
        anchor = {key: torch.from_numpy(prior[key].astype(np.float32)) for key in anchor}

    torch.manual_seed(seed)
    params = {key: value.clone() for key, value in anchor.items()}
    if head == "mlp" and prior is None:
        # Zero-init hidden weights never break symmetry (every unit computes
        # the same gradient), so the first fit needs a small random kick.
        # Once a prior exists it is itself broken-symmetric, and fine-tuning
        # should start exactly there rather than perturb it further.
        params["w1"] = torch.empty(hidden, dims).normal_(std=0.2)
    for value in params.values():
        value.requires_grad_(True)

    def score(z: torch.Tensor) -> torch.Tensor:
        if head == "linear":
            return z @ params["weights"]
        pre = z @ params["w1"].T + params["b1"]
        return act(pre) @ params["w2"]

    def penalty() -> torch.Tensor:
        total = torch.zeros(())
        for key, value in params.items():
            if head == "mlp" and key in ("w1", "b1"):
                # Excluded, not just weakened: any positive coefficient here
                # still pulls pre-activations towards zero, which is exactly
                # the mechanism that collapses tanh onto the identity.
                continue
            offset = value - anchor[key]
            total = total + offset.flatten().dot(offset.flatten())
        return l2 * total

    def loss() -> torch.Tensor:
        margin = score(left) - score(right)
        return loss_fn(margin, targets) + penalty()

    if head == "linear":
        optimiser = torch.optim.LBFGS(
            list(params.values()), max_iter=steps, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            optimiser.zero_grad()
            value = loss()
            value.backward()
            return value

        optimiser.step(closure)
    else:
        optimiser = torch.optim.Adam(list(params.values()), lr=1e-2)
        for _ in range(1500):
            optimiser.zero_grad()
            value = loss()
            value.backward()
            optimiser.step()

    return {key: value.detach().numpy() for key, value in params.items()}


def make_scorer(params: dict[str, np.ndarray], head: str, activation: str = "relu"):
    """Vectorised numpy f(z) -> score from a fitted parameter dict."""
    if head == "linear":
        weights = params["weights"]
        return lambda z: z @ weights

    w1, b1, w2 = params["w1"], params["b1"], params["w2"]
    act = numpy_activation(activation)

    def scorer(z: np.ndarray) -> np.ndarray:
        return act(z @ w1.T + b1) @ w2

    return scorer


def fit_best_of_seeds(
    z_left: np.ndarray,
    z_right: np.ndarray,
    y: np.ndarray,
    l2: float,
    *,
    head: str,
    hidden: int,
    activation: str = "relu",
    prior: dict[str, np.ndarray] | None,
    seeds: int,
    select_left: np.ndarray,
    select_right: np.ndarray,
    select_y: np.ndarray,
) -> tuple[dict[str, np.ndarray], int]:
    """Fit `seeds` random restarts, keep the one scoring best on a held set.

    Only the mlp branch actually varies with seed: LBFGS on the linear head
    is a deterministic convex fit from a fixed starting point, so every
    restart returns identical weights and this is a single fit in disguise.
    Selection uses the supplied validation arrays: teacher validation for the
    teacher prior, human validation for human fine-tuning. The reported selected
    validation score is exploratory, not an independent generalisation estimate.
    With a fixed prior, MLP fine-tuning restarts share the same initial weights.
    """
    candidate_seeds = range(seeds) if head == "mlp" else (0,)
    best_params: dict[str, np.ndarray] | None = None
    best_score = -1.0
    best_seed = 0
    for seed in candidate_seeds:
        params = fit_head(
            z_left,
            z_right,
            y,
            l2,
            head=head,
            hidden=hidden,
            activation=activation,
            prior=prior,
            seed=seed,
        )
        scorer = make_scorer(params, head, activation)
        score, _ = accuracy(select_left, select_right, select_y, scorer)
        if score > best_score:
            best_params, best_score, best_seed = params, score, seed
    assert best_params is not None
    return best_params, best_seed


def collapse_diagnostic(
    params: dict[str, np.ndarray],
    activation: str,
    pool_z: np.ndarray,
    linear_score: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict:
    """Whether a fitted MLP is actually using its non-linearity.

    An MLP that fits the same accuracy as the linear head is not evidence
    against a non-linear function class unless it actually computed a
    non-linear function. tanh flattens to the identity near the origin, so
    a small enough w1 (exactly what an L2 penalty anchored at zero produces)
    collapses the network onto the linear solution — same accuracy, for the
    boring reason that it computed the same thing.

    Three checks, over every pool image (`pool_z`, already projected):
    - mean/max |pre-activation|: how far the network actually sits from the
      origin, where every one of these activations is approximately linear.
    - the fraction of the score's own variance a straight line through z
      cannot already explain (an OLS fit of the MLP's scores onto z; a
      collapsed network has near-zero residual because it IS linear in z).
    - correlation with a separately-fitted linear model's scores on the same
      images, when one is supplied. Values near 1 mean the two models are
      making the same ranking decisions — not proof of collapse by itself
      (two very different functions could still rank similarly) but taken
      together with the other two numbers, >0.99 here is the same signature
      reported against the earlier tanh runs.
    """
    w1, b1, w2 = params["w1"], params["b1"], params["w2"]
    pre = pool_z @ w1.T + b1
    act = numpy_activation(activation)
    scores = act(pre) @ w2

    design = np.hstack([pool_z, np.ones((pool_z.shape[0], 1))])
    coeffs, *_ = np.linalg.lstsq(design, scores, rcond=None)
    residual = scores - design @ coeffs
    total_std = float(np.std(scores))
    nonlinear_fraction = float(np.std(residual)) / total_std if total_std > 1e-12 else 0.0

    result = {
        "nImages": int(pool_z.shape[0]),
        "meanAbsPreActivation": float(np.abs(pre).mean()),
        "maxAbsPreActivation": float(np.abs(pre).max()),
        "nonlinearStdFraction": nonlinear_fraction,
    }
    if linear_score is not None:
        linear_scores = linear_score(pool_z)
        if total_std > 1e-12 and np.std(linear_scores) > 1e-12:
            corr = float(np.corrcoef(scores, linear_scores)[0, 1])
        else:
            corr = float("nan")
        result["corrWithLinear"] = corr
        result["collapsed"] = bool(corr > 0.99)
    return result


def accuracy(z_left: np.ndarray, z_right: np.ndarray, y: np.ndarray, scorer) -> tuple[float, int]:
    """Agreement on decided pairs. Ties are excluded — neither side is correct."""
    decided = y != 0.5
    if not decided.any():
        return float("nan"), 0
    margin = scorer(z_left[decided]) - scorer(z_right[decided])
    predicted = margin > 0
    return float((predicted == (y[decided] > 0.5)).mean()), int(decided.sum())


def json_number(value: float) -> float | None:
    """`None` for a non-finite measurement, because `NaN` is not valid JSON.

    An empty split has no accuracy and no interval, and `accuracy` /
    `wilson_interval` say so with NaN. Python's json module happily writes a
    bare `NaN` literal that every strict reader of ranker.json rejects, so the
    absence is recorded as null instead.
    """
    return value if math.isfinite(value) else None


def wilson_interval(correct: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. With 80-odd test pairs the CI is the honest number."""
    if n == 0:
        return (float("nan"), float("nan"))
    centre = (correct + z * z / (2 * n)) / (1 + z * z / n)
    spread = z * math.sqrt(correct * (1 - correct) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - spread), min(1.0, centre + spread))


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build_pairs(
    comparisons: list[Comparison], embeddings: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Left and right embeddings kept separate, plus labels.

    The old `build_matrix` collapsed each pair to `emb[left] - emb[right]`,
    which only works because a linear head is itself linear in that
    difference. A per-image scorer needs both sides individually so it can
    also be called pointwise, on one image, outside training.
    """
    if not comparisons:
        # np.stack rejects an empty sequence; --teacher-holdout 0 (or 1) hits
        # this legitimately by holding out none (or all) of the teacher pairs.
        dims = next(iter(embeddings.values())).shape[0] if embeddings else 0
        empty = np.zeros((0, dims), dtype=np.float32)
        return empty, empty, np.zeros(0, dtype=np.float32)
    z_left = np.stack([embeddings[c.left] for c in comparisons])
    z_right = np.stack([embeddings[c.right] for c in comparisons])
    y = np.array([c.label for c in comparisons], dtype=np.float32)
    return z_left, z_right, y


def teacher_split(
    comparisons: list[Comparison], holdout: float, *, mode: str = "image"
) -> tuple[list[Comparison], list[Comparison]]:
    """Image-disjoint development split; crossing pairs are discarded.

    ``mode='pair'`` exists only to reproduce historical exploratory runs.
    A supplied evaluation manifest is preferable for verified subject grouping.
    """
    import hashlib

    train: list[Comparison] = []
    held: list[Comparison] = []
    if not 0 <= holdout < 1 or mode not in {"image", "pair"}:
        raise ValueError("Expected holdout in [0, 1) and image/pair split mode")
    for c in comparisons:
        if mode == "pair":
            left, right = sorted((c.left, c.right))
            digest = hashlib.sha256(f"{left}|{right}".encode()).digest()
            (held if digest[0] / 256.0 < holdout else train).append(c)
        else:
            a, b = (group_split(i, val=holdout, test=0) for i in (c.left, c.right))
            if a == b:
                (train if a == "train" else held).append(c)
    return train, held


def load_teacher(path: Path, tie_margin: float, *, allow_legacy: bool = False) -> list[Comparison]:
    """VLM teacher preferences from distil_teacher.py.

    The teacher emits a continuous score margin. Anything inside `tie_margin`
    is recorded as a tie rather than forced to a side: a two-point gap on a
    0-100 rubric is the judge being indifferent, and training the student to
    call it confidently teaches it noise.
    """
    if not path.exists():
        sys.exit(f"No teacher labels at {path}. Run distil_teacher.py first.")

    if not np.isfinite(tie_margin) or tie_margin < 0:
        raise ValueError("Tie margin must be finite and nonnegative")
    rows = read_jsonl(path)
    expected = next((r["provenance"] for r in rows if "provenance" in r), {})
    provenance = validate_teacher_cache(rows, expected, allow_legacy=allow_legacy)
    print(f"teacher label provenance: {provenance}")
    out: list[Comparison] = []
    ties = 0
    seen = set()
    for row in rows:
        key = tuple(sorted((row["leftId"], row["rightId"])))
        if key[0] == key[1] or key in seen:
            raise ValueError(f"Duplicate, reversed or self teacher pair: {key}")
        seen.add(key)
        margin = row["margin"]
        if abs(margin) < tie_margin:
            label = 0.5
            ties += 1
        else:
            label = 1.0 if margin > 0 else 0.0
        out.append(
            Comparison(
                left=row["leftId"], right=row["rightId"], label=label, split="teacher", rater="vlm"
            )
        )
    print(f"teacher: {len(out)} pairs ({ties} within the {tie_margin}-point tie margin)")
    return out


def resolve_teacher_images(comparisons: list[Comparison], pool: Path) -> dict[str, Path]:
    needed = {c.left for c in comparisons} | {c.right for c in comparisons}
    missing = sorted(stem for stem in needed if not (pool / stem).exists())
    if missing:
        sys.exit(
            f"{len(missing)} teacher images are missing from {pool} "
            f"(e.g. {missing[0]}). Re-run prepare_teacher_pool.py with the same --seed."
        )
    return {stem: pool / stem for stem in needed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, default=32, help="PCA dimensions (default: 32)")
    parser.add_argument("--basis", choices=("pca", "identity"), default="pca")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-legacy-cache",
        action="store_true",
        help="explicitly reuse unversioned embeddings for exploratory runs",
    )
    parser.add_argument("--manifest", type=Path, help="frozen evaluation manifest")
    parser.add_argument("--teacher-split", choices=("image", "pair"), default="image")
    parser.add_argument("--teacher-l2", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--refresh", action="store_true", help="recompute cached embeddings")
    parser.add_argument(
        "--report-test",
        action="store_true",
        help="evaluate on the held-out test split. Run this ONCE, at the end.",
    )
    parser.add_argument(
        "--teacher",
        type=Path,
        help="teacher.jsonl from distil_teacher.py. Pretrains the head, then "
        "fine-tunes it on the human pairs (Plan B).",
    )
    parser.add_argument("--teacher-pool", type=Path, default=TEACHER_POOL_DIR)
    parser.add_argument(
        "--decisions-dir",
        type=Path,
        default=DECISIONS_DIR,
        help="frozen snapshot of the rater JSONL files. Use one when "
        "comparing configurations while raters are still labelling.",
    )
    parser.add_argument(
        "--raters",
        help="comma-separated rater ids to include, e.g. AC,DP. Omit for all.",
    )
    parser.add_argument(
        "--tie-margin",
        type=float,
        default=2.0,
        help="teacher score gaps smaller than this count as ties (default: 2.0)",
    )
    parser.add_argument(
        "--projection",
        choices=("teacher", "train"),
        default="teacher",
        help="which images fit the PCA basis when --teacher is used. 'teacher' "
        "uses the far larger teacher pool; 'train' keeps the label pool's own "
        "basis, isolating the prior's contribution from the basis change.",
    )
    parser.add_argument(
        "--head",
        choices=("linear", "mlp"),
        default="linear",
        help="scorer architecture: a weighted sum, or a one-hidden-layer MLP "
        "(default: linear, so the documented repro command keeps working).",
    )
    parser.add_argument(
        "--hidden", type=int, default=16, help="hidden units for --head mlp (default: 16)"
    )
    parser.add_argument(
        "--activation",
        choices=ACTIVATIONS,
        default="relu",
        help="hidden non-linearity for --head mlp (default: relu). tanh is linear "
        "near the origin, so an L2 penalty anchored at zero can shrink it into "
        "that regime and collapse the network onto a linear function; relu's "
        "kink sits at a fixed input, not a weight scale, so it cannot.",
    )
    parser.add_argument(
        "--compare-linear-artifact",
        type=Path,
        help="a previously-saved linear ranker.npz directory to correlate "
        "against in the --head mlp collapse diagnostic. Optional.",
    )
    parser.add_argument(
        "--teacher-holdout",
        type=float,
        default=0.2,
        help="fraction of teacher images held for development (default: 0.2); "
        "cross-split pairs are dropped. A manifest overrides this split.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="random restarts for --head mlp, best kept by validation accuracy "
        "(default: 3; irrelevant to --head linear, whose fit is deterministic)",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "ranker-training",
        help="new experiment output directory (default: artifacts/ranker-training)",
    )
    args = parser.parse_args()
    if args.artifact_dir.resolve() == ARTIFACT_DIR.resolve():
        parser.error(
            "Train into an experiment directory; promote artifacts separately after validation"
        )
    if (args.artifact_dir / "ranker.npz").exists():
        parser.error("Artifact already exists; use a new experiment directory")
    manifest = load_manifest(args.manifest, verify_files=True) if args.manifest else None
    if args.teacher and manifest is None and args.teacher_split != "pair":
        parser.error("Teacher training requires --manifest; --teacher-split pair is legacy-only")

    selected = (
        {r.strip().upper() for r in args.raters.split(",") if r.strip()} if args.raters else None
    )
    comparisons = load_decisions(args.decisions_dir, selected)
    files = resolve_images(comparisons)
    splits = image_splits(comparisons)
    if manifest:
        for stem, split in splits.items():
            if stem not in manifest["images"] or manifest["images"][stem]["split"] != split:
                parser.error("Human decision splits disagree with the frozen manifest")
        for path in args.decisions_dir.glob("decisions.*.jsonl"):
            if manifest.get("inputs", {}).get(str(path.resolve())) != file_hash(path):
                parser.error("Human decisions are not the snapshot recorded in the manifest")
        if args.teacher and manifest.get("inputs", {}).get(
            str(args.teacher.resolve())
        ) != file_hash(args.teacher):
            parser.error("Teacher labels are not the snapshot recorded in the manifest")
    embeddings = embed_pool(
        files, args.batch_size, args.refresh, allow_legacy=args.allow_legacy_cache
    )

    by_split = {
        name: [c for c in comparisons if c.split == name] for name in ("train", "val", "test")
    }
    counts = {name: len(rows) for name, rows in by_split.items()}
    images_per_split = {
        name: sum(1 for image, split in splits.items() if split == name)
        for name in ("train", "val", "test")
    }
    print(f"pairs: {counts}")
    print(f"images: {images_per_split}")

    teacher: list[Comparison] = []
    teacher_reduced: dict[str, np.ndarray] = {}

    if args.teacher:
        # Partition before fitting the projection. A teacher-pool basis must
        # never consume validation/test images, even without their labels.
        teacher = load_teacher(args.teacher, args.tie_margin, allow_legacy=args.allow_legacy_cache)
        if manifest:
            partitions, dropped = partition_pairs(teacher, manifest)
            teacher_train, teacher_held = partitions["train"], partitions["val"]
            print(f"teacher manifest: {dropped} crossing pairs dropped; test partition not scored")
        else:
            teacher_train, teacher_held = teacher_split(
                teacher, args.teacher_holdout, mode=args.teacher_split
            )
        if not teacher_train:
            parser.error("No teacher training pairs remain after the split")
        teacher_files = (
            {
                i: Path(manifest["images"][i]["path"])
                for c in teacher_train + teacher_held
                for i in (c.left, c.right)
            }
            if manifest
            else resolve_teacher_images(teacher, args.teacher_pool)
        )
        teacher_embeddings = embed_pool(
            teacher_files,
            args.batch_size,
            args.refresh,
            TEACHER_CACHE_PATH,
            allow_legacy=args.allow_legacy_cache,
        )
        if args.projection == "teacher":
            train_ids = sorted({i for c in teacher_train for i in (c.left, c.right)})
            source = np.stack([teacher_embeddings[i] for i in train_ids])
        else:
            train_images = sorted(image for image, split in splits.items() if split == "train")
            source = np.stack([embeddings[i] for i in train_images])
        print(f"projection fitted on {len(source)} {args.projection}-pool images")
        centre, basis = fit_projection(source, args.dims, mode=args.basis)
        teacher_reduced = dict(
            zip(
                teacher_embeddings,
                project(
                    np.stack(list(teacher_embeddings.values())),
                    centre,
                    basis,
                    normalize=args.normalize,
                ),
                strict=True,
            )
        )
    else:
        train_images = sorted(image for image, split in splits.items() if split == "train")
        centre, basis = fit_projection(
            np.stack([embeddings[i] for i in train_images]), args.dims, mode=args.basis
        )

    reduced = {
        stem: v
        for stem, v in zip(
            embeddings,
            project(np.stack(list(embeddings.values())), centre, basis, normalize=args.normalize),
            strict=True,
        )
    }

    x_train_l, x_train_r, y_train = build_pairs(by_split["train"], reduced)
    x_val_l, x_val_r, y_val = build_pairs(by_split["val"], reduced)

    teacher_report = None
    if args.teacher:
        tt_l, tt_r, tt_y = build_pairs(teacher_train, teacher_reduced)
        th_l, th_r, th_y = build_pairs(teacher_held, teacher_reduced)
        print(
            f"\npretraining on {len(tt_y)} teacher pairs ({args.dims} dims), "
            f"{len(th_y)} used for teacher validation (exploratory)"
        )
        # Teacher validation selects the prior. This is explicitly a development
        # score, not an independent test. sweep_ranker.py varies the teacher L2.
        prior, prior_seed = fit_best_of_seeds(
            tt_l,
            tt_r,
            tt_y,
            args.teacher_l2,
            head=args.head,
            hidden=args.hidden,
            activation=args.activation,
            prior=None,
            seeds=args.seeds,
            select_left=th_l if len(th_y) else tt_l,
            select_right=th_r if len(th_y) else tt_r,
            select_y=th_y if len(th_y) else tt_y,
        )
        prior_scorer = make_scorer(prior, args.head, args.activation)
        teacher_accuracy, teacher_n = accuracy(tt_l, tt_r, tt_y, prior_scorer)
        teacher_held_accuracy, teacher_held_n = accuracy(th_l, th_r, th_y, prior_scorer)
        held_pairs = [c for c in teacher_held if c.label != 0.5]
        held_values = [
            float(
                (prior_scorer(teacher_reduced[c.left]) > prior_scorer(teacher_reduced[c.right]))
                == (c.label > 0.5)
            )
            for c in held_pairs
        ]
        held_report = paired_summary(
            held_values,
            [(c.left, c.right) for c in held_pairs],
            groups={k: v["group"] for k, v in manifest["images"].items()} if manifest else None,
        )
        zero_shot, zero_n = accuracy(x_val_l, x_val_r, y_val, prior_scorer)
        if args.head == "mlp":
            print(f"  best seed: {prior_seed}")
        print(f"  teacher fit (in-sample): {teacher_accuracy:.3f} on n={teacher_n}")
        print(
            f"  teacher validation: {teacher_held_accuracy:.3f} "
            f"group bootstrap CI {held_report['ci95']} (n={teacher_held_n}; selected)"
        )
        print(f"  teacher-only stage, human val: {zero_shot:.3f} (n={zero_n}; exploratory)")
        teacher_report = {
            "holdout": args.teacher_holdout,
            "inSample": {"accuracy": teacher_accuracy, "n": teacher_n},
            "heldOut": {
                "accuracy": json_number(teacher_held_accuracy),
                "ci": held_report["ci95"],
                "n": teacher_held_n,
                "usedForSelection": True,
                "intervalMethod": "dyadic-group-product-bootstrap",
            },
            "zeroShotValAccuracy": zero_shot,
        }
    prior_params: dict[str, np.ndarray] | None = prior if args.teacher else None

    print(f"\nsweeping L2 on the validation split ({args.dims} dims, head={args.head})")
    if prior_params is not None:
        print("  (L2 now pulls towards the teacher's parameters, not towards zero)")
    best = (None, -1.0, 0.0, 0)
    for l2 in (0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        params, seed = fit_best_of_seeds(
            x_train_l,
            x_train_r,
            y_train,
            l2,
            head=args.head,
            hidden=args.hidden,
            activation=args.activation,
            prior=prior_params,
            seeds=args.seeds,
            select_left=x_val_l,
            select_right=x_val_r,
            select_y=y_val,
        )
        scorer = make_scorer(params, args.head, args.activation)
        train_accuracy, _ = accuracy(x_train_l, x_train_r, y_train, scorer)
        val_accuracy, val_n = accuracy(x_val_l, x_val_r, y_val, scorer)
        print(f"  l2={l2:<7} train={train_accuracy:.3f}  val={val_accuracy:.3f}  (n={val_n})")
        if val_accuracy > best[1]:
            best = (params, val_accuracy, l2, seed)

    params, val_accuracy, l2, seed = best
    print(f"\nbest: l2={l2}, val={val_accuracy:.3f}")
    if args.head == "mlp":
        print(f"  best seed: {seed}")
    print("A coin flip scores 0.500. Treat anything inside the interval as unproven.")

    test_report = None
    if args.report_test:
        x_test_l, x_test_r, y_test = build_pairs(by_split["test"], reduced)
        scorer = make_scorer(params, args.head, args.activation)
        test_accuracy, test_n = accuracy(x_test_l, x_test_r, y_test, scorer)
        print(f"\nHISTORICAL TEST (exploratory): {test_accuracy:.3f} (n={test_n})")
        test_report = {
            "accuracy": json_number(test_accuracy),
            "ci": None,
            "status": "historical-reused-test; not confirmatory",
            "n": test_n,
        }
    else:
        print(
            "\nTest split not scored in this run; historical reuse remains. "
            "Use fresh acceptance data."
        )

    diagnostic = None
    if args.head == "mlp":
        pool_z = np.stack([reduced[i] for i in reduced if splits[i] != "test"])
        compare_score = None
        if args.compare_linear_artifact:
            compare_scorer_by_id = load_scorer(args.compare_linear_artifact, embeddings)
            pool_stems = [i for i in reduced if splits[i] != "test"]

            def compare_score(_z: np.ndarray, _stems: list[str] = pool_stems) -> np.ndarray:
                # collapse_diagnostic scores by array position, but the linear
                # artifact's loader scores by image id — reassemble in the same
                # order pool_z (and therefore `_z`) was built in.
                return np.array([compare_scorer_by_id(s) for s in _stems])

        diagnostic = collapse_diagnostic(params, args.activation, pool_z, compare_score)
        print(
            f"\ncollapse diagnostic ({args.activation}, hidden={args.hidden}, "
            f"n={diagnostic['nImages']} pool images):"
        )
        print(
            f"  |pre-activation|: mean={diagnostic['meanAbsPreActivation']:.3f}  "
            f"max={diagnostic['maxAbsPreActivation']:.3f}"
        )
        print(
            f"  fraction of score std NOT explained by a straight line through z: "
            f"{diagnostic['nonlinearStdFraction']:.3f}"
        )
        if "corrWithLinear" in diagnostic:
            print(f"  corr(mlp, linear) over the pool: {diagnostic['corrWithLinear']:.4f}")
            if diagnostic["collapsed"]:
                print(
                    "  COLLAPSED: correlation with the linear model exceeds 0.99. "
                    "This config computed an approximately linear function — its "
                    "accuracy numbers above are not evidence about a non-linear "
                    "function class, only about this particular (collapsed) fit."
                )
        else:
            print("  (no --compare-linear-artifact given; corr(mlp, linear) not computed)")

    # Display calibration. The head is trained on the sign of a difference, so a
    # raw margin has no scale of its own, but the live path has to show 0-100.
    # Storing the score distribution over the training images lets the service
    # map a new score to its rank without re-deriving anything, and keeps that
    # mapping attached to the weights it belongs to — swap the artifact and the
    # calibration swaps with it, which is what makes a head change drop-in.
    # Train split only: val and test images do not inform anything the product shows.
    print()
    calibration_scorer = make_scorer(params, args.head, args.activation)
    calibration_images = sorted(i for i, split in splits.items() if split == "train")
    train_scores = np.sort(
        np.asarray(calibration_scorer(np.stack([reduced[i] for i in calibration_images])))
    )
    calibration = np.interp(
        np.linspace(0.0, 1.0, CALIBRATION_QUANTILES),
        np.linspace(0.0, 1.0, train_scores.size),
        train_scores,
    ).astype(np.float32)
    print(
        ""
        f"calibration: {train_scores.size} train images -> "
        f"{CALIBRATION_QUANTILES} quantiles, "
        f"raw range [{train_scores[0]:.3f}, {train_scores[-1]:.3f}]"
    )

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = args.artifact_dir / "ranker.npz"
    if args.head == "linear":
        np.savez(
            artifact,
            centre=centre,
            basis=basis,
            weights=params["weights"],
            normalize=np.array(args.normalize),
            calibration=calibration,
        )
    else:
        np.savez(
            artifact,
            centre=centre,
            basis=basis,
            normalize=np.array(args.normalize),
            w1=params["w1"],
            b1=params["b1"],
            w2=params["w2"],
            activation=np.array(args.activation),
            calibration=calibration,
        )
    (args.artifact_dir / "ranker.json").write_text(
        json.dumps(
            {
                "version": RANKER_VERSIONS[args.head],
                "encoder": ENCODER,
                "head": args.head,
                "hidden": args.hidden if args.head == "mlp" else None,
                "activation": args.activation if args.head == "mlp" else None,
                "collapseDiagnostic": diagnostic,
                "dims": int(basis.shape[0]),
                "requestedDims": args.dims,
                "basisMode": args.basis,
                "normalize": args.normalize,
                "evaluationStatus": "exploratory",
                "manifestFingerprint": manifest["fingerprint"] if manifest else None,
                "legacyCacheAllowed": args.allow_legacy_cache,
                "codeSha256": file_hash(Path(__file__)),
                "decisionHashes": {
                    p.name: file_hash(p)
                    for p in sorted(args.decisions_dir.glob("decisions.*.jsonl"))
                },
                "teacherSha256": file_hash(args.teacher) if args.teacher else None,
                "teacherSplit": "manifest" if manifest else args.teacher_split,
                "teacherL2": args.teacher_l2 if args.teacher else None,
                "l2": l2,
                "trainPairs": counts["train"],
                "valAccuracy": val_accuracy,
                "test": test_report,
                "raters": sorted({c.rater for c in comparisons}),
                "ratersRequested": sorted(selected) if selected else "all",
                "teacherPairs": len(teacher) if teacher else 0,
                "teacher": teacher_report,
                "calibration": {
                    "quantiles": CALIBRATION_QUANTILES,
                    "fittedOnImages": len(calibration_images),
                    "rawRange": [float(train_scores[0]), float(train_scores[-1])],
                },
                "projectionFittedOn": (
                    f"{args.projection}-pool" if args.teacher else "train-images"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        location = artifact.relative_to(REPO_ROOT)
    except ValueError:
        location = artifact
    print(f"saved {location}")


if __name__ == "__main__":
    main()
