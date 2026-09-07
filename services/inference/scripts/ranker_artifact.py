#!/usr/bin/env python
"""Load a ranker artifact and expose a pointwise `score(image) -> float`.

Owns the centre -> project -> L2-normalise math in one place. Before this
module existed, `benchmark_judges.py` reimplemented that projection inline
and `train_ranker.py` implemented it again for training — two copies that
would silently drift the moment the projection convention changed. Both now
delegate here.

Dispatches on which parameter keys the artifact's `.npz` contains: `weights`
means a linear head, `w1`/`b1`/`w2` means an MLP head (see train_ranker.py's
`fit_head` for how each is fitted).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

ACTIVATIONS = ("tanh", "relu", "gelu")


def numpy_activation(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """Numpy implementation of a hidden activation, by name.

    Single source of truth for both training-time scoring (train_ranker.py's
    `make_scorer`, `collapse_diagnostic`) and artifact inference (`load_scorer`
    below) — both call this rather than each defining their own, so the two
    paths can never silently drift apart the way the old inlined projection
    math did.
    """
    if name == "tanh":
        return np.tanh
    if name == "relu":
        return lambda x: np.maximum(x, 0.0)
    if name == "gelu":

        def gelu(x: np.ndarray) -> np.ndarray:
            return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

        return gelu
    raise ValueError(f"unknown activation {name!r}, expected one of {ACTIVATIONS}")


def project(
    vector: np.ndarray, centre: np.ndarray, basis: np.ndarray, *, normalize: bool = True
) -> np.ndarray:
    """Centre, project, and L2-normalise a single embedding.

    Mirrors `train_ranker.project`, which operates on a batch; kept as a
    single-vector version here because the artifact's consumers score one
    image at a time.
    """
    reduced = (vector - centre) @ basis.T
    if not normalize:
        return reduced
    norm = np.linalg.norm(reduced)
    return reduced / max(norm, 1e-8)


CALIBRATION_QUANTILES = 256


def percentile_of(margin: float, calibration: np.ndarray) -> float:
    """Where `margin` falls in a fitted score distribution, as 0..1.

    The head is only ever trained on the *sign* of a score difference, so the
    magnitude of `w.z` has no meaning on its own — nothing in the loss
    constrains its scale. Rank against a reference distribution is the one
    reading that survives a change of head, which is why the display mapping is
    a percentile rather than a rescaled margin.
    """
    if calibration.size == 0:
        return 0.5
    below = float(np.searchsorted(calibration, margin, side="left"))
    ties = float(np.searchsorted(calibration, margin, side="right")) - below
    # Midpoint of any tied run, so identical scores map to one percentile
    # rather than to the bottom of their run.
    return (below + ties / 2.0) / calibration.size


def load_calibration(artifact_dir: Path) -> np.ndarray:
    """Sorted reference scores from the artifact, or an empty array.

    Empty means the artifact predates calibration; callers should treat a
    missing distribution as "cannot map to a display score" rather than
    inventing one.
    """
    art = np.load(artifact_dir / "ranker.npz")
    if "calibration" not in art.files:
        return np.zeros(0, dtype=np.float64)
    return np.sort(np.asarray(art["calibration"], dtype=np.float64))


def load_scorer(artifact_dir: Path, embeddings: dict[str, np.ndarray]) -> Callable[[str], float]:
    """Build `score(image_id) -> float` from a ranker artifact.

    `embeddings` maps image id to its raw (pre-projection) DINOv2 vector, e.g.
    the dict loaded from embeddings.npz — the same cache train_ranker.py
    writes and reads.
    """
    art = np.load(artifact_dir / "ranker.npz")
    centre, basis = art["centre"], art["basis"]
    normalize = bool(art["normalize"]) if "normalize" in art.files else True

    if "weights" in art.files:
        weights = art["weights"]

        def score(stem: str) -> float:
            z = project(embeddings[stem], centre, basis, normalize=normalize)
            return float(z @ weights)

    elif "w1" in art.files:
        w1, b1, w2 = art["w1"], art["b1"], art["w2"]
        # Older mlp artifacts (before --activation existed) were always tanh.
        activation = str(art["activation"]) if "activation" in art.files else "tanh"
        act = numpy_activation(activation)

        def score(stem: str) -> float:
            z = project(embeddings[stem], centre, basis, normalize=normalize)
            return float(act(z @ w1.T + b1) @ w2)

    else:
        raise ValueError(
            f"{artifact_dir / 'ranker.npz'} has neither a 'weights' key (linear) "
            "nor 'w1'/'b1'/'w2' keys (mlp). Was it written by train_ranker.py?"
        )

    return score
