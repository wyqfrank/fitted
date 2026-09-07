"""Write a new, versioned development cache without reading test images."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ranker_evaluation import load_manifest
from train_ranker import embed_pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", choices=("human", "teacher"), default="human")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--offline", action="store_true", help="use only a locally cached encoder")
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(".metadata.json").exists():
        parser.error("Cache exists; use a new output path")
    if args.out.suffix != ".npz" or args.batch_size < 1 or args.threads < 1:
        parser.error("Need an .npz output and positive batch size / thread count")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    import torch

    torch.set_num_threads(args.threads)
    manifest = load_manifest(args.manifest, verify_files=True)
    images = {
        i: Path(r["path"])
        for i, r in manifest["images"].items()
        if r["source"] == args.source and r["split"] in {"train", "val"}
    }
    if not images:
        parser.error("No development images for the selected source")
    embed_pool(images, args.batch_size, False, args.out)


if __name__ == "__main__":
    main()
