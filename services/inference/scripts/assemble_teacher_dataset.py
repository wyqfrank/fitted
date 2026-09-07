"""Freeze newly collected train/validation judgments into an experiment dataset.

Copies complete JSONL rows into a new local output directory, verifies the judge,
image hashes and splits, and derives a manifest referencing the frozen copy. This
also permits reproducible checkpoints while an append-only collection continues.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from ranker_evaluation import (
    canonical_pair,
    file_hash,
    fingerprint,
    load_manifest,
    validate_manifest,
    validate_teacher_cache,
    write_json,
)


def snapshot_rows(path: Path) -> tuple[list[dict], bytes]:
    raw = path.read_bytes()
    # A writer may be partway through a JSON object. Only newline-terminated,
    # flushed rows enter the frozen dataset; never repair or invent a label.
    if raw and not raw.endswith(b"\n"):
        raw = raw.rsplit(b"\n", 1)[0] + b"\n" if b"\n" in raw else b""
    return [json.loads(line) for line in raw.splitlines() if line.strip()], raw


def assemble(
    manifest_path: Path,
    train_path: Path,
    val_path: Path,
    out: Path,
    *,
    min_train: int = 1,
    max_train: int | None = None,
    expected_val: int = 65,
) -> dict:
    if min_train < 1 or expected_val < 1 or (max_train is not None and max_train < min_train):
        raise ValueError("Invalid required training/validation counts")
    if out.exists():
        raise ValueError("Dataset output exists; use a new snapshot directory")
    manifest = load_manifest(manifest_path, verify_files=True)
    train, train_raw = snapshot_rows(train_path)
    val, val_raw = snapshot_rows(val_path)
    if len(train) < min_train or len(val) != expected_val:
        raise ValueError(
            f"Collection incomplete: train={len(train)} (need {min_train}), "
            f"val={len(val)} (need {expected_val})"
        )
    train = train[:max_train] if max_train is not None else train
    if "provenance" not in train[0]:
        raise ValueError("New datasets require versioned teacher provenance")
    judge = train[0]["provenance"]
    validate_teacher_cache(train + val, judge)
    seen = set()
    for split, rows in (("train", train), ("val", val)):
        for row in rows:
            a, b = row["leftId"], row["rightId"]
            key = canonical_pair(a, b)
            if key in seen:
                raise ValueError(f"Duplicate teacher pair: {key}")
            seen.add(key)
            images = manifest["images"]
            if any(images[i]["split"] != split for i in key):
                raise ValueError("Teacher judgment crosses the requested dataset split")
            if images[a]["group"] == images[b]["group"]:
                raise ValueError("Teacher comparison is within one subject/outfit group")
            if row.get("manifestFingerprint") != manifest["fingerprint"]:
                raise ValueError("Teacher judgment refers to another image manifest")
            if row.get("imageHashes") != {i: images[i]["sha256"] for i in key}:
                raise ValueError("Teacher judgment image hash mismatch")
    out.mkdir(parents=True)
    dataset = out / "teacher.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in train + val), encoding="utf-8")
    (out / "train-collection-snapshot.jsonl").write_bytes(train_raw)
    (out / "val-collection-snapshot.jsonl").write_bytes(val_raw)
    derived = copy.deepcopy(manifest)
    derived.pop("fingerprint")
    derived["parentFingerprint"] = manifest["fingerprint"]
    derived["inputs"][str(dataset.resolve())] = file_hash(dataset)
    derived["collectionSnapshots"] = {
        "train": hashlib.sha256(train_raw).hexdigest(),
        "val": hashlib.sha256(val_raw).hexdigest(),
    }
    derived["fingerprint"] = fingerprint(derived)
    validate_manifest(derived)
    write_json(out / "manifest.json", derived)
    report = {
        "trainPairs": len(train),
        "valPairs": len(val),
        "testPairs": 0,
        "provenance": judge,
        "status": "exploratory-application-photo-development",
        "manifestFingerprint": derived["fingerprint"],
        "teacherSha256": file_hash(dataset),
    }
    write_json(out / "dataset.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-train", type=int, default=1)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--expected-validation", type=int, default=65)
    args = parser.parse_args()
    report = assemble(
        args.manifest,
        args.train,
        args.validation,
        args.out,
        min_train=args.min_train,
        max_train=args.max_train,
        expected_val=args.expected_validation,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
