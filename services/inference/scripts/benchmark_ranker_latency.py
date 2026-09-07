"""Measure the actual CPU score(bytes) path on development images, without test access.

Includes decoding, preprocessing, encoder, head and calibration. Excludes HTTP,
garment detection and concurrent service load: this is not an end-to-end live gate.
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np
from ranker_evaluation import file_hash, load_manifest, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, help="omit to retain the service's Torch default")
    parser.add_argument("--budget-ms", type=float, default=95)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Report exists; choose a new output")
    if min(args.images, args.repeats, args.warmup) < 1 or args.budget_ms <= 0:
        parser.error("Counts and latency budget must be positive")
    if args.threads is not None and args.threads < 1:
        parser.error("Thread count must be positive")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    import torch

    from fitted_inference.ranker import Dinov2FitRanker

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    manifest = load_manifest(args.manifest, verify_files=True)
    records = [
        (i, r)
        for i, r in sorted(manifest["images"].items())
        if r["source"] == "human" and r["split"] == "train"
    ][: args.images]
    if len(records) != args.images:
        parser.error("Not enough development images")
    photos = [Path(r["path"]).read_bytes() for _, r in records]
    models = [
        Dinov2FitRanker(path, display_min=55, display_max=85, device="cpu")
        for path in args.artifacts
    ]
    for model in models:
        for i in range(args.warmup):
            model.score(photos[i % len(photos)])
    timings: list[list[float]] = [[] for _ in models]
    for repetition in range(args.repeats):
        for image_index, photo in enumerate(photos):
            # Alternate model order so one candidate is not always first.
            indices = list(range(len(models)))
            shift = (repetition + image_index) % len(models)
            for index in indices[shift:] + indices[:shift]:
                started = time.perf_counter()
                models[index].score(photo)
                timings[index].append((time.perf_counter() - started) * 1000)
    report = {
        "scope": "warm CPU score(bytes); excludes HTTP, detection and concurrent workload",
        "promotionEligible": False,
        "manifestFingerprint": manifest["fingerprint"],
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "interopThreads": torch.get_num_interop_threads(),
            "threadSetting": "default" if args.threads is None else "explicit",
        },
        "imageIds": [i for i, _ in records],
        "warmupPerModel": args.warmup,
        "budgetMs": args.budget_ms,
        "codeSha256": file_hash(Path(__file__)),
        "runtimeCodeSha256": file_hash(
            Path(__file__).parents[1] / "src/fitted_inference/ranker.py"
        ),
        "models": [],
    }
    for path, samples in zip(args.artifacts, timings, strict=True):
        median, p95 = np.quantile(samples, [0.5, 0.95])
        result = {
            "artifact": str(path),
            "sha256": file_hash(path / "ranker.npz"),
            "samplesMs": samples,
            "meanMs": float(np.mean(samples)),
            "medianMs": float(median),
            "p95Ms": float(p95),
            "p95WithinBudget": bool(p95 <= args.budget_ms),
        }
        report["models"].append(result)
        print(f"{path}: median {median:.1f} ms, p95 {p95:.1f} ms", flush=True)
    write_json(args.out, report)


if __name__ == "__main__":
    main()
