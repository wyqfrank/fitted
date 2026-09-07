"""Experiment 1: 32 linear controls, selected only on image-disjoint teacher validation.

Requires a frozen manifest. Never scores its test split or overwrites the shipped
ranker. Every candidate, validation prediction and selection decision is retained.
Existing labels/caches are exploratory; this is not a new independent test result.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from ranker_artifact import load_scorer
from ranker_evaluation import (
    ROOT,
    file_hash,
    load_embeddings,
    load_manifest,
    paired_summary,
    partition_pairs,
    read_jsonl,
    write_json,
)
from train_ranker import ENCODER, build_pairs, fit_head, fit_projection, load_teacher, project


def evaluate(comparisons, reduced: dict, weights: np.ndarray) -> tuple[dict, list[dict]]:
    left, right, labels = build_pairs(comparisons, reduced)
    margins = (left - right) @ weights
    # Stable logistic cross-entropy, including the teacher's soft ties.
    loss = float(np.mean(np.logaddexp(0, margins) - labels * margins))
    decided = labels != 0.5
    correct = (margins > 0) == (labels > 0.5)
    records = [
        {
            "leftId": c.left,
            "rightId": c.right,
            "target": c.label,
            "margin": float(m),
            "correct": bool(hit) if c.label != 0.5 else None,
        }
        for c, m, hit in zip(comparisons, margins, correct, strict=True)
    ]
    return {
        "accuracy": float(correct[decided].mean()) if decided.any() else None,
        "decisiveDecisions": int(decided.sum()),
        "pairs": len(labels),
        "bce": loss,
    }, records


def human_validation(manifest: dict, args, selected: str, baseline: str) -> dict:
    """Secondary metric after selection; never reads or scores human test rows."""
    rows = []
    for filename in manifest["inputs"]:
        path = Path(filename)
        if not path.name.startswith("decisions."):
            continue
        rows.extend(
            r
            for r in read_jsonl(path)
            if r["split"] == "val"
            and r["raterId"].upper() in {"AC", "DP"}
            and r["shownVerdict"] in {"a", "b"}
        )
    if not rows:
        return {"status": "no human validation labels", "usedForSelection": False}
    pairs = [(r["shownLeftId"], r["shownRightId"]) for r in rows]
    images = {i: Path(manifest["images"][i]["path"]) for p in pairs for i in p}
    for i in images:
        if manifest["images"][i]["split"] != "val":
            raise ValueError("Human validation label disagrees with manifest split")
    embeddings, provenance = load_embeddings(
        args.human_embeddings, images, ENCODER, allow_legacy=args.allow_legacy_cache
    )
    result = {
        "usedForSelection": False,
        "raters": ["AC", "DP"],
        "embeddingProvenance": provenance,
        "models": {},
    }
    groups = {i: r["group"] for i, r in manifest["images"].items()}
    models = {
        "selectedTeacherOnly": args.out / selected,
        "baselineTeacherOnly": args.out / baseline,
    }
    for name, location in models.items():
        score = load_scorer(location, embeddings)
        predictions = [
            {
                "leftId": a,
                "rightId": b,
                "rater": r["raterId"],
                "correct": bool((score(a) > score(b)) == (r["shownVerdict"] == "a")),
            }
            for (a, b), r in zip(pairs, rows, strict=True)
        ]
        result["models"][name] = paired_summary(
            [float(p["correct"]) for p in predictions], pairs, groups=groups
        )
        write_json(args.out / f"human-validation-{name}.json", predictions)
    return result


def run(args) -> dict:
    import torch

    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Output directory is not empty; use a new run directory")
    if args.out.resolve() == (ROOT / "models/ranker").resolve():
        raise ValueError("Experiments must not overwrite the shipped ranker")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    manifest = load_manifest(args.manifest, verify_files=True)
    if manifest.get("inputs", {}).get(str(args.teacher.resolve())) != file_hash(args.teacher):
        raise ValueError("Teacher labels are not the snapshot recorded in this manifest")
    teacher = load_teacher(args.teacher, args.tie_margin, allow_legacy=args.allow_legacy_cache)
    splits, dropped = partition_pairs(teacher, manifest)
    train, val = splits["train"], splits["val"]
    if not train or not any(c.label != 0.5 for c in val):
        raise ValueError("Need training pairs and decisive teacher validation after splitting")
    train_ids = sorted({i for c in train for i in (c.left, c.right)})
    val_ids = sorted({i for c in val for i in (c.left, c.right)})
    if set(train_ids) & set(val_ids):
        raise ValueError("Training and validation images overlap")
    needed = sorted(set(train_ids) | set(val_ids))
    images = {i: Path(manifest["images"][i]["path"]) for i in needed}
    embeddings, provenance = load_embeddings(
        args.embeddings, images, ENCODER, allow_legacy=args.allow_legacy_cache
    )

    # Match the original train-pool PCA source while keeping every test feature out.
    # An alternative teacher-train source is recorded, never silently substituted.
    if args.projection_source == "human":
        projection_ids = sorted(
            i
            for i, r in manifest["images"].items()
            if r["source"] == "human" and r["split"] == "train"
        )
        projection_images = {i: Path(manifest["images"][i]["path"]) for i in projection_ids}
        human_embeddings, human_provenance = load_embeddings(
            args.human_embeddings,
            projection_images,
            ENCODER,
            allow_legacy=args.allow_legacy_cache,
        )
        source = np.stack([human_embeddings[i] for i in projection_ids])
    else:
        projection_ids = train_ids
        human_provenance = None
        source = np.stack([embeddings[i] for i in train_ids])

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "source").mkdir()
    for filename in (
        "sweep_ranker.py",
        "train_ranker.py",
        "ranker_evaluation.py",
        "ranker_artifact.py",
    ):
        path = Path(__file__).with_name(filename)
        (args.out / "source" / filename).write_bytes(path.read_bytes())
    write_json(args.out / "manifest.json", manifest)
    metadata = {
        "version": 1,
        "experiment": "full-coordinate-linear-capacity",
        "evaluationStatus": "exploratory-validation-selection; no test scored",
        "grouping": manifest["grouping"],
        "manifestFingerprint": manifest["fingerprint"],
        "teacherSha256": file_hash(args.teacher),
        "embeddingSha256": file_hash(args.embeddings),
        "embeddingProvenance": provenance,
        "humanEmbeddingSha256": file_hash(args.human_embeddings),
        "humanEmbeddingProvenance": human_provenance,
        "projectionSource": args.projection_source,
        "projectionImages": len(projection_ids),
        "selectionMetric": "teacher-validation-accuracy; ties broken by BCE then grid order",
        "teacherPairs": {
            "train": len(train),
            "val": len(val),
            "testNotScored": len(splits["test"]),
            "crossingDropped": dropped,
        },
        "images": {"train": len(train_ids), "val": len(val_ids), "overlap": 0},
        "tieMargin": args.tie_margin,
        "threads": args.threads,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "code": {
            p.name: file_hash(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("train_ranker.py"),
                Path(__file__).with_name("ranker_evaluation.py"),
                Path(__file__).with_name("ranker_artifact.py"),
            )
        },
        "candidates": [],
    }
    print(json.dumps({k: metadata[k] for k in ("teacherPairs", "images", "embeddingProvenance")}))
    started = time.perf_counter()
    predictions = {}
    # Include the fixed-penalty 16-d baseline in the same split and source.
    for mode, dims in (("pca", 16), ("pca", 64), ("pca", 128), ("identity", source.shape[1])):
        centre, basis = fit_projection(source, dims, mode=mode)
        for normalize in (True, False):
            z = project(
                np.stack([embeddings[i] for i in needed]), centre, basis, normalize=normalize
            )
            reduced = dict(zip(needed, z, strict=True))
            training_arrays = build_pairs(train, reduced)
            for l2 in (0.0, 0.0001, 0.001, 0.01):
                name = f"{mode}-{dims}-{'l2norm' if normalize else 'raw'}-reg-{l2:g}"
                weights = fit_head(*training_arrays, l2)["weights"]
                train_metrics, _ = evaluate(train, reduced, weights)
                val_metrics, records = evaluate(val, reduced, weights)
                candidate = {
                    "id": name,
                    "basisMode": mode,
                    "requestedDims": dims,
                    "effectiveDims": int(basis.shape[0]),
                    "normalize": normalize,
                    "l2": l2,
                    "train": train_metrics,
                    "val": val_metrics,
                }
                location = args.out / name
                location.mkdir()
                scores = np.sort(np.array([reduced[i] @ weights for i in train_ids]))
                calibration = np.quantile(scores, np.linspace(0, 1, 256)).astype(np.float32)
                np.savez(
                    location / "ranker.npz",
                    centre=centre,
                    basis=basis,
                    weights=weights,
                    normalize=np.array(normalize),
                    calibration=calibration,
                )
                write_json(
                    location / "ranker.json",
                    {
                        **candidate,
                        "version": "dinov2s-linear-experiment-v2",
                        "encoder": ENCODER,
                        "head": "linear",
                        "dims": int(basis.shape[0]),
                        "stage": "teacher-only",
                        "manifestFingerprint": manifest["fingerprint"],
                        "evaluationStatus": metadata["evaluationStatus"],
                        "calibration": {
                            "fittedOn": "teacher-training-images",
                            "images": len(scores),
                        },
                    },
                )
                write_json(location / "validation-predictions.json", records)
                metadata["candidates"].append(candidate)
                predictions[name] = records
                print(
                    f"{name:35} actual={basis.shape[0]:3} "
                    f"train={train_metrics['accuracy']:.3f} val={val_metrics['accuracy']:.3f}",
                    flush=True,
                )
    candidates = metadata["candidates"]
    best = max(candidates, key=lambda c: (c["val"]["accuracy"], -c["val"]["bce"]))
    baseline = next(c for c in candidates if c["id"] == "pca-16-l2norm-reg-0.01")
    rows = predictions[best["id"]]
    base_rows = predictions[baseline["id"]]
    indices = [i for i, r in enumerate(rows) if r["correct"] is not None]
    pairs = [(rows[i]["leftId"], rows[i]["rightId"]) for i in indices]
    difference = [int(rows[i]["correct"]) - int(base_rows[i]["correct"]) for i in indices]
    groups = {i: r["group"] for i, r in manifest["images"].items()}
    metadata["selected"] = best["id"]
    metadata["baseline"] = baseline["id"]
    metadata["selectedMinusBaseline"] = paired_summary(difference, pairs, groups=groups)
    metadata["selectedMinusBaseline"]["caveat"] = "Validation-selected; not confirmatory"
    metadata["humanValidation"] = human_validation(manifest, args, best["id"], baseline["id"])
    metadata["elapsedSeconds"] = time.perf_counter() - started
    metadata["promotion"] = "not eligible: fresh application benchmark and CPU latency required"
    write_json(args.out / "report.json", metadata)
    print(
        json.dumps(
            {
                "selected": best["id"],
                "difference": metadata["selectedMinusBaseline"],
                "report": str(args.out / "report.json"),
            },
            indent=2,
        )
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, default=ROOT / "data/labelling/teacher.jsonl")
    parser.add_argument(
        "--embeddings", type=Path, default=ROOT / "models/ranker/teacher-embeddings.npz"
    )
    parser.add_argument(
        "--human-embeddings", type=Path, default=ROOT / "models/ranker/embeddings.npz"
    )
    parser.add_argument("--projection-source", choices=("human", "teacher"), default="human")
    parser.add_argument("--allow-legacy-cache", action="store_true")
    parser.add_argument("--tie-margin", type=float, default=2.0)
    parser.add_argument("--threads", type=int, default=1)
    try:
        run(parser.parse_args())
    except ValueError as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
