"""Live fit scoring: a frozen DINOv2 encoder behind a small trained head.

This is the fast path. The VLM judges once at finalisation and takes seconds;
this runs about once a second during the battle, so it has to be cheap. A
frozen encoder plus a tiny head is what makes that possible: the 71 ms is
almost entirely DINOv2, and the head itself is a dot product.

The artifact is produced offline by `scripts/train_ranker.py`, which lives
outside this package because the deployable wheel must not carry training
dependencies. What arrives here is a handful of small arrays.

**Preprocessing must match training exactly.** The encoder is frozen, so any
difference in how pixels reach it — resize, crop, normalisation — moves the
embedding somewhere the head was never fitted, and the scores become confident
nonsense rather than obviously broken. `_to_tensor` below is a deliberate copy
of `train_ranker.embed_pool.to_tensor`; change one and you must change both.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .config import Settings

ENCODER = "facebook/dinov2-small"

# services/inference/src/fitted_inference/ranker.py -> repository root.
_REPO_ROOT = Path(__file__).resolve().parents[4]

# ImageNet statistics, matching the encoder's own preprocessing.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class RankerModelNotReadyError(RuntimeError):
    """Raised when a fit score is requested but no ranker is loaded."""


class InvalidRankerImageError(ValueError):
    """Raised when the supplied bytes are not a decodable image."""


@dataclass(frozen=True, slots=True)
class FitScore:
    """One image's live score.

    `raw` is the head's own output and has no meaningful scale — the head is
    trained on the sign of a difference, never on magnitude. `percentile` is
    the honest quantity; `score` is `percentile` mapped into the display band.
    """

    score: float
    percentile: float
    raw: float
    latency_ms: float


class FitRanker(Protocol):
    @property
    def ready(self) -> bool: ...

    @property
    def model_version(self) -> str: ...

    def score(self, image: bytes) -> FitScore: ...


class UnavailableFitRanker:
    """Stands in when no artifact is configured, so the API can still start.

    Live scoring is an enhancement to the battle, not a precondition for it.
    A missing artifact should degrade the live estimate, not stop the service
    from serving garment perception and finalisation.
    """

    def __init__(self, reason: str) -> None:
        self._reason = reason

    @property
    def ready(self) -> bool:
        return False

    @property
    def model_version(self) -> str:
        return "unavailable"

    @property
    def reason(self) -> str:
        return self._reason

    def score(self, image: bytes) -> FitScore:
        raise RankerModelNotReadyError(self._reason)


class Dinov2FitRanker:
    """Frozen DINOv2-S -> PCA projection -> trained head -> calibrated score."""

    def __init__(
        self,
        artifact_dir: Path,
        *,
        display_min: float,
        display_max: float,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModel

        artifact = np.load(artifact_dir / "ranker.npz")
        self._centre = np.asarray(artifact["centre"], dtype=np.float32)
        self._basis = np.asarray(artifact["basis"], dtype=np.float32)
        self._normalize = bool(artifact["normalize"]) if "normalize" in artifact.files else True
        self._head = _load_head(artifact)

        if "calibration" not in artifact.files:
            raise RankerModelNotReadyError(
                f"{artifact_dir / 'ranker.npz'} has no calibration distribution. "
                "Regenerate it with a current train_ranker.py — without one there "
                "is no defensible way to turn a raw margin into a 0-100 score."
            )
        self._calibration = np.sort(np.asarray(artifact["calibration"], dtype=np.float64))

        metadata_path = artifact_dir / "ranker.json"
        self._metadata: dict[str, Any] = (
            json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        )
        self._display_min = display_min
        self._display_max = display_max

        self._torch = torch
        self._device = torch.device(device) if device else torch.device("cpu")
        self._model = AutoModel.from_pretrained(ENCODER).eval().to(self._device)

    @property
    def ready(self) -> bool:
        return True

    @property
    def model_version(self) -> str:
        return str(self._metadata.get("version", "dinov2s-pca-unknown"))

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def score(self, image: bytes) -> FitScore:
        started = time.perf_counter()
        pixels = _to_tensor(image)
        with self._torch.inference_mode():
            batch = self._torch.from_numpy(pixels[None, ...]).to(self._device)
            # CLS token: DINOv2's global image descriptor.
            embedding = self._model(pixel_values=batch).last_hidden_state[:, 0]
            embedding = embedding.to("cpu").numpy()[0].astype(np.float32)

        reduced = (embedding - self._centre) @ self._basis.T
        z = reduced / max(float(np.linalg.norm(reduced)), 1e-8) if self._normalize else reduced
        raw = float(self._head(z))
        percentile = _percentile_of(raw, self._calibration)
        score = self._display_min + percentile * (self._display_max - self._display_min)
        return FitScore(
            score=round(score, 1),
            percentile=percentile,
            raw=raw,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )


def _load_head(artifact: Any):
    """Build f(z) -> score from whichever parameters the artifact carries.

    Mirrors `scripts/ranker_artifact.load_scorer`. The two cannot import each
    other — that module is training-only and this package ships without it —
    so the dispatch is duplicated here deliberately and both sides name the
    other in a comment.
    """
    if "weights" in artifact.files:
        weights = np.asarray(artifact["weights"], dtype=np.float32)
        return lambda z: z @ weights

    if "w1" in artifact.files:
        w1 = np.asarray(artifact["w1"], dtype=np.float32)
        b1 = np.asarray(artifact["b1"], dtype=np.float32)
        w2 = np.asarray(artifact["w2"], dtype=np.float32)
        name = str(artifact["activation"]) if "activation" in artifact.files else "tanh"
        activation = _activation(name)
        return lambda z: activation(z @ w1.T + b1) @ w2

    raise RankerModelNotReadyError(
        "ranker.npz has neither a 'weights' key (linear) nor 'w1'/'b1'/'w2' (mlp)."
    )


def _activation(name: str):
    if name == "tanh":
        return np.tanh
    if name == "relu":
        return lambda x: np.maximum(x, 0.0)
    if name == "gelu":
        return lambda x: 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    raise RankerModelNotReadyError(f"unknown activation {name!r} in ranker.npz")


def _percentile_of(margin: float, calibration: np.ndarray) -> float:
    """Where a raw margin falls in the training-pool score distribution."""
    if calibration.size == 0:
        return 0.5
    below = float(np.searchsorted(calibration, margin, side="left"))
    ties = float(np.searchsorted(calibration, margin, side="right")) - below
    return (below + ties / 2.0) / calibration.size


def _to_tensor(image: bytes) -> np.ndarray:
    """Letterbox to square, then resize to 224.

    A copy of `train_ranker.embed_pool.to_tensor`. The stock processor resizes
    the short edge and centre-crops, which slices the shoes or the head off a
    full-body outfit photo — exactly the evidence being judged. Padding
    preserves the whole garment at the cost of some border.
    """
    from PIL import Image, UnidentifiedImageError

    try:
        decoded = Image.open(io.BytesIO(image))
        decoded.load()
    except (UnidentifiedImageError, OSError) as error:
        raise InvalidRankerImageError("Frame could not be decoded as an image.") from error

    decoded = decoded.convert("RGB")
    side = max(decoded.size)
    square = Image.new("RGB", (side, side), (255, 255, 255))
    square.paste(decoded, ((side - decoded.width) // 2, (side - decoded.height) // 2))
    array = np.asarray(square.resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255.0
    return ((array - _MEAN) / _STD).transpose(2, 0, 1)


def _resolve_artifact_dir(configured: str) -> Path:
    """Resolve the artifact directory against the repository, not the cwd.

    The service is launched from several places — uvicorn from
    `services/inference`, `scripts/dev.mjs` from the repository root — and a
    relative default that silently means something different in each is a
    confusing way to get "model unavailable". An absolute setting always wins.
    """
    path = Path(configured)
    if path.is_absolute():
        return path
    # A relative setting still wins when it resolves from the launch directory,
    # but it is returned absolute: the same string must not name two different
    # directories depending on who reads it later.
    if (path / "ranker.npz").exists():
        return path.resolve()
    return _REPO_ROOT / path


def create_fit_ranker(settings: Settings) -> FitRanker:
    """Load the live ranker, or a stand-in explaining why it is unavailable."""
    if not settings.ranker_enabled:
        return UnavailableFitRanker(
            "Live fit scoring is off. Set FITTED_RANKER_ENABLED=true to turn it on."
        )

    artifact_dir = _resolve_artifact_dir(settings.ranker_artifact_dir)
    if not (artifact_dir / "ranker.npz").exists():
        return UnavailableFitRanker(
            f"No ranker artifact at {artifact_dir / 'ranker.npz'}. The shipped one is "
            "committed at models/ranker, so this usually means "
            "FITTED_RANKER_ARTIFACT_DIR points somewhere else — check it, or restore "
            "the file with git checkout models/ranker."
        )

    try:
        return Dinov2FitRanker(
            artifact_dir,
            display_min=settings.ranker_display_min,
            display_max=settings.ranker_display_max,
            device=settings.ranker_device,
        )
    except RankerModelNotReadyError as error:
        return UnavailableFitRanker(str(error))
    except ImportError as error:
        return UnavailableFitRanker(
            f"Live fit scoring needs the 'ml' extra (torch, transformers, pillow): {error}"
        )
