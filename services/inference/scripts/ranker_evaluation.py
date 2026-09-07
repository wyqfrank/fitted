"""Offline manifests, cache provenance and paired evaluation for ranker experiments.

Build an exploratory manifest of the existing local pools:
    python services/inference/scripts/ranker_evaluation.py --out artifacts/ranker/manifest.json

Image IDs are only fallback grouping keys. A manifest without independently supplied
subject/outfit groups must never be described as subject-disjoint or confirmatory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SPLITS = {"train", "val", "test"}
PREPROCESSING = {
    "version": 1,
    "descriptor": "cls",
    "size": 224,
    "padding": "white-square",
    "resize": "PIL-bicubic",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def fingerprint(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def canonical_pair(left: str, right: str) -> tuple[str, str]:
    if left == right:
        raise ValueError(f"Self-comparison: {left}")
    return tuple(sorted((left, right)))


def group_split(group: str, *, seed: int = 1, val: float = 0.15, test: float = 0.15) -> str:
    if not 0 <= val < 1 or not 0 <= test < 1 or val + test >= 1:
        raise ValueError("Split fractions must be nonnegative and leave training images")
    bucket = int(hashlib.sha256(f"{seed}|{group}".encode()).hexdigest()[:16], 16) / 2**64
    return "test" if bucket < test else "val" if bucket < test + val else "train"


def image_record(path: Path, group: str, split: str, source: str) -> dict:
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
    return {
        "path": str(path.resolve()),
        "sha256": file_hash(path),
        "pixelSha256": pixels,
        "group": group,
        "split": split,
        "source": source,
    }


def validate_manifest(manifest: dict, *, verify_files: bool = False) -> None:
    if manifest.get("version") != 1 or not manifest.get("images"):
        raise ValueError("Expected a nonempty version-1 evaluation manifest")
    expected = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if manifest.get("fingerprint") != fingerprint(expected):
        raise ValueError("Manifest fingerprint mismatch")
    groups: dict[str, str] = {}
    contents: dict[str, str] = {}
    for stem, item in manifest["images"].items():
        split = item["split"]
        if split not in SPLITS or not item["group"]:
            raise ValueError(f"Invalid group/split for {stem}")
        for index, key in ((groups, item["group"]), (contents, item["pixelSha256"])):
            previous = index.setdefault(key, split)
            if previous != split:
                raise ValueError(f"Cross-split group or image content: {stem}")
        if verify_files and file_hash(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Image content changed: {stem}")
    if verify_files:
        for path, expected_hash in manifest.get("inputs", {}).items():
            if file_hash(Path(path)) != expected_hash:
                raise ValueError(f"Label snapshot changed: {path}")


def load_manifest(path: Path, *, verify_files: bool = False) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest, verify_files=verify_files)
    return manifest


def partition_pairs(pairs: list, manifest: dict) -> tuple[dict[str, list], int]:
    """Partition before learning anything; drop all edges crossing group splits."""
    result: dict[str, list] = {split: [] for split in sorted(SPLITS)}
    dropped = 0
    seen = set()
    for pair in pairs:
        key = canonical_pair(pair.left, pair.right)
        if key in seen:
            raise ValueError(f"Duplicate/reversed teacher pair: {key}")
        seen.add(key)
        a, b = (manifest["images"][stem] for stem in key)
        if a["group"] == b["group"] or a["split"] != b["split"]:
            dropped += 1
        else:
            result[a["split"]].append(pair)
    return result, dropped


def embedding_metadata(images: dict[str, Path], encoder: str) -> dict:
    return {
        "version": 1,
        "encoder": encoder,
        "preprocessing": PREPROCESSING,
        "images": {stem: file_hash(path) for stem, path in sorted(images.items())},
    }


def load_embeddings(
    path: Path, images: dict[str, Path], encoder: str, *, allow_legacy: bool = False
) -> tuple[dict[str, np.ndarray], str]:
    meta_path = path.with_suffix(".metadata.json")
    if not meta_path.exists():
        if not allow_legacy:
            raise ValueError(
                f"Unversioned embedding cache {path}; refresh it, or explicitly "
                "allow legacy caches for exploratory runs only"
            )
        provenance = "legacy-unverified"
    else:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = embedding_metadata(images, encoder)
        if (
            meta.get("version") != 1
            or meta.get("encoder") != encoder
            or meta.get("preprocessing") != PREPROCESSING
            or meta.get("cacheSha256") != file_hash(path)
            or any(meta.get("images", {}).get(k) != v for k, v in expected["images"].items())
        ):
            raise ValueError(f"Embedding provenance mismatch: {path}; refresh the cache")
        provenance = "verified"
    with np.load(path, allow_pickle=False) as data:
        missing = set(images) - set(data.files)
        if missing:
            raise ValueError(f"Cache lacks {len(missing)} images; refresh {path}")
        result = {stem: data[stem] for stem in sorted(images)}
    if any(v.ndim != 1 or not np.isfinite(v).all() for v in result.values()):
        raise ValueError(f"Invalid embeddings in {path}")
    return result, provenance


def teacher_provenance(model: str, prompt_version: str, prompt: str, resolution: str) -> dict:
    return {
        "version": 1,
        "model": model,
        "promptVersion": prompt_version,
        "promptSha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "mediaResolution": resolution,
        "protocol": "single-image-pair",
        "scoringSha256": file_hash(ROOT / "services/inference/src/fitted_inference/scoring.py"),
        "providerSha256": file_hash(ROOT / "services/inference/src/fitted_inference/vlm.py"),
    }


def validate_teacher_cache(rows: list[dict], expected: dict, *, allow_legacy: bool = False) -> str:
    legacy = False
    for row in rows:
        if "provenance" not in row:
            if not allow_legacy:
                raise ValueError(
                    "Teacher cache has no prompt provenance; use a new output "
                    "or explicitly allow legacy cached-only evaluation"
                )
            legacy = True
        elif row["provenance"] != expected:
            raise ValueError("Teacher cache configuration differs from requested judge")
        if not np.isfinite(row["margin"]):
            raise ValueError("Non-finite teacher margin")
    return "legacy-unverified" if legacy else "verified"


def outcome(margin: float, draw_threshold: float = 2.0) -> int:
    if not np.isfinite(margin) or draw_threshold < 0:
        raise ValueError("Expected finite margin and nonnegative draw threshold")
    return 0 if margin == 0 or abs(margin) < draw_threshold else (1 if margin > 0 else -1)


def paired_summary(
    values: list[float] | np.ndarray,
    pairs: list[tuple[str, str]],
    *,
    groups: dict[str, str] | None = None,
    draws: int = 5000,
    seed: int = 1,
) -> dict:
    """Dyadic product bootstrap: resample groups, retain all votes/predictions together.

    The same endpoint multiplicities weight every comparison in a replicate.
    Passing per-row candidate-minus-baseline outcomes gives a paired difference CI.
    This estimates uncertainty conditional on the chosen cohort/configuration; it
    cannot repair adaptive test reuse or turn image IDs into verified subjects.
    """
    values = np.asarray(values, dtype=float)
    if len(values) != len(pairs) or not np.isfinite(values).all() or draws < 1:
        raise ValueError("Expected finite values aligned with pairs and positive bootstrap draws")
    if not pairs:
        return {"estimate": None, "ci95": None, "rows": 0, "pairs": 0, "groups": 0}
    mapping = groups if groups is not None else {i: i for p in pairs for i in p}
    labels = sorted({mapping[i] for pair in pairs for i in pair})
    index = {name: i for i, name in enumerate(labels)}
    a = np.array([index[mapping[p[0]]] for p in pairs])
    b = np.array([index[mapping[p[1]]] for p in pairs])
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(draws):
        counts = rng.multinomial(len(labels), np.full(len(labels), 1 / len(labels)))
        weights = counts[a] * counts[b]
        if weights.sum():
            means.append(float(weights @ values / weights.sum()))
    ci = np.quantile(means, [0.025, 0.975]).tolist() if len(labels) > 2 and means else None
    return {
        "estimate": float(values.mean()),
        "ci95": ci,
        "rows": len(pairs),
        "pairs": len({canonical_pair(*p) for p in pairs}),
        "groups": len(labels),
        "intervalMethod": "dyadic-group-product-bootstrap",
        "bootstrapDraws": draws,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--decisions-dir", type=Path, default=ROOT / "data/labelling")
    parser.add_argument("--human-pool", type=Path, default=ROOT / "apps/web/public/label-pool")
    parser.add_argument("--teacher", type=Path, default=ROOT / "data/labelling/teacher.jsonl")
    parser.add_argument("--teacher-pool", type=Path, default=ROOT / "data/teacher-pool")
    parser.add_argument(
        "--groups", type=Path, help="JSON mapping image IDs to verified subject/outfit IDs"
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Manifest already exists; use a new output to preserve the snapshot")
    grouping = json.loads(args.groups.read_text()) if args.groups else None
    images = {}
    inputs = {}
    human_paths = {p.stem: p for p in args.human_pool.iterdir() if p.is_file()}
    for file in sorted(args.decisions_dir.glob("decisions.*.jsonl")):
        inputs[str(file.resolve())] = file_hash(file)
        for row in read_jsonl(file):
            for stem in (row["shownLeftId"], row["shownRightId"]):
                if stem in images:
                    if images[stem]["split"] != row["split"]:
                        raise ValueError(f"Human image crosses splits: {stem}")
                    continue
                group = grouping[stem] if grouping is not None else f"image:{stem}"
                images[stem] = image_record(human_paths[stem], group, row["split"], "human")
    inputs[str(args.teacher.resolve())] = file_hash(args.teacher)
    teacher_ids = {r[k] for r in read_jsonl(args.teacher) for k in ("leftId", "rightId")}
    for stem in sorted(teacher_ids):
        if stem in images:
            raise ValueError(f"Image ID occurs in both pools: {stem}")
        group = grouping[stem] if grouping is not None else f"image:{stem}"
        images[stem] = image_record(
            args.teacher_pool / stem, group, group_split(group, seed=args.seed), "teacher"
        )
    manifest = {
        "version": 1,
        "purpose": "exploratory-existing-labels",
        "grouping": "verified-subject-outfit" if grouping is not None else "image-id-only",
        "seed": args.seed,
        "teacherFractions": {"train": 0.7, "val": 0.15, "test": 0.15},
        "images": images,
        "inputs": inputs,
    }
    manifest["fingerprint"] = fingerprint(manifest)
    validate_manifest(manifest)
    write_json(args.out, manifest)
    print(
        json.dumps(
            {
                "manifest": str(args.out),
                "fingerprint": manifest["fingerprint"],
                "grouping": manifest["grouping"],
                "counts": dict(Counter(f"{x['source']}/{x['split']}" for x in images.values())),
            }
        )
    )


if __name__ == "__main__":
    main()
