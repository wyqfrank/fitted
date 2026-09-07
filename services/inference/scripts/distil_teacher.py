#!/usr/bin/env python
"""Collect VLM teacher preferences within a frozen development image split.

    python services/inference/scripts/distil_teacher.py --manifest <path> --dry-run

Replaces Fashion144k's engagement votes with judgements from the *same* Gemini
assessor that decides a real battle, using the same production prompt and
rubric. The student trained on these labels therefore approximates the judge it
has to agree with at finalisation — which is the point. The live estimate should
predict the final verdict, not some other notion of outfit quality.

The run is resumable and quota-aware. Every completed pair is appended to the
output immediately, an interrupted run continues where it stopped, and an
exhausted quota stops the run cleanly instead of burning retries against a wall.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from ranker_evaluation import (
    file_hash,
    load_manifest,
    read_jsonl,
    teacher_provenance,
    validate_teacher_cache,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "services" / "inference" / "src"))

OUTPUT_PATH = REPO_ROOT / "data" / "labelling" / "teacher.jsonl"

from fitted_inference.scoring import visual_fit_score  # noqa: E402
from fitted_inference.vlm import (  # noqa: E402
    VLM_PROMPTS,
    GeminiVlmProvider,
    VlmProviderError,
    VlmProviderQuotaError,
)


def load_dotenv(path: Path) -> None:
    """Fill missing environment variables from the repo's .env.

    `scripts/dev.mjs` loads .env for the dev servers, but a script run directly
    never sees it — so a key that is plainly present in the file would look
    absent. Existing environment variables win, matching how dev.mjs behaves.
    """
    if not path.exists():
        return
    # Collect first so a repeated name resolves the way Node's loadEnvFile
    # resolves it — last assignment wins. Reading first-wins here would make
    # the same .env mean different things to the service and to this script.
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name and value:
            values[name] = value
    for name, value in values.items():
        # A variable already exported for this shell still outranks the file.
        if name not in os.environ:
            os.environ[name] = value


def sample_pairs(images: list[str], count: int, seed: int) -> list[tuple[str, str]]:
    """Round-robin pairs: cover the pool before repeatedly reusing early images."""
    if len(set(images)) != len(images) or count < 0:
        raise ValueError("Expected unique image IDs and a nonnegative pair count")
    if len(images) < 2 or count == 0:
        return []
    rng = random.Random(seed)
    order = sorted(images)
    rng.shuffle(order)
    if len(order) % 2:
        order.append(None)
    pairs = []
    for _ in range(len(order) - 1):
        for i in range(len(order) // 2):
            a, b = order[i], order[-i - 1]
            if a is not None and b is not None:
                pairs.append(tuple(sorted((a, b))))
                if len(pairs) == count:
                    return pairs
        order = [order[0], order[-1], *order[1:-1]]
    return pairs


def pair_id(left: str, right: str) -> str:
    return f"{left}|{right}"


def parse_args(argv=None, *, env_path: Path = REPO_ROOT / ".env"):
    # Resolve defaults only after loading the same environment file as production.
    load_dotenv(env_path)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--model", default=os.getenv("FITTED_VLM_MODEL", "gemini-3.6-flash"))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-failures", type=int, default=3)
    # Defaults mirror the service's own environment. A teacher judging under
    # different settings than the finalisation path is a different judge, and
    # the student would be distilling something the product never runs.
    parser.add_argument("--prompt-version", default=os.getenv("FITTED_VLM_PROMPT_VERSION", "v2"))
    parser.add_argument(
        "--media-resolution",
        default=os.getenv("FITTED_VLM_MEDIA_RESOLUTION", "high"),
        help="'high' matches production; 'low' costs less quota per pair",
    )
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--source", choices=("human", "teacher"), default="teacher")
    parser.add_argument("--dry-run", action="store_true", help="validate/count; no API or writes")
    args = parser.parse_args(argv)
    if args.concurrency < 1 or args.pairs < 1 or args.max_failures < 1:
        parser.error("--concurrency, --pairs and --max-failures must be positive")
    return args


async def main() -> None:
    args = parse_args()

    manifest = load_manifest(args.manifest, verify_files=True)
    records = {
        i: r
        for i, r in manifest["images"].items()
        if r["source"] == args.source and r["split"] == args.split
    }
    images = sorted(records)
    if len(images) < 2:
        sys.exit(f"Teacher pool has {len(images)} images; need at least 2.")

    prompt = VLM_PROMPTS.get(args.prompt_version)
    if prompt is None:
        sys.exit(
            f"Unknown prompt version {args.prompt_version!r}. "
            f"Available: {', '.join(sorted(VLM_PROMPTS))}"
        )
    provenance = teacher_provenance(args.model, args.prompt_version, prompt, args.media_resolution)
    grouped = len({r["group"] for r in records.values()}) != len(records)
    count = len(images) * (len(images) - 1) // 2 if grouped else args.pairs
    pairs = [
        p
        for p in sample_pairs(images, count, args.seed)
        if records[p[0]]["group"] != records[p[1]]["group"]
    ][: args.pairs]
    if len(pairs) != args.pairs:
        sys.exit(f"Only {len(pairs)} eligible distinct pairs; requested {args.pairs}")
    existing = read_jsonl(args.out) if args.out.exists() else []
    validate_teacher_cache(existing, provenance)
    requested = {pair_id(*pair) for pair in pairs}
    for row in existing:
        if (
            row["pairId"] != pair_id(row["leftId"], row["rightId"])
            or row["pairId"] not in requested
        ):
            raise ValueError("Existing judgments differ from this requested pair list/seed")
        expected_hashes = {i: records[i]["sha256"] for i in (row["leftId"], row["rightId"])}
        if (
            row.get("manifestFingerprint") != manifest["fingerprint"]
            or row.get("imageHashes") != expected_hashes
        ):
            raise ValueError("Existing teacher rows use different images or manifest")
    done = {r["pairId"] for r in existing}
    if len(done) != len(existing):
        raise ValueError("Duplicate teacher cache rows")
    todo = [p for p in pairs if pair_id(*p) not in done]
    print(
        f"pool={len(images)} images pairs={len(pairs)} cached={len(done)} todo={len(todo)} "
        f"source={args.source} split={args.split} grouping={manifest['grouping']}"
    )
    print(
        f"teacher: {args.model}, prompt {args.prompt_version}, "
        f"media resolution {args.media_resolution}"
    )

    if args.dry_run or not todo:
        return
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY is not set")
    provider = GeminiVlmProvider(
        api_key=api_key,
        model=args.model,
        prompt=prompt,
        media_resolution=args.media_resolution,
        timeout_seconds=args.timeout,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    handle = args.out.open("a", encoding="utf-8")
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    quota_hit = asyncio.Event()
    failed_stop = asyncio.Event()
    failures = []
    counters = {"ok": 0, "unusable": 0, "failed": 0}
    started = time.time()

    async def judge(left: str, right: str) -> None:
        if quota_hit.is_set() or failed_stop.is_set():
            return
        async with semaphore:
            if quota_hit.is_set() or failed_stop.is_set():
                return
            paths = {i: Path(records[i]["path"]) for i in (left, right)}
            for i, path in paths.items():
                if file_hash(path) != records[i]["sha256"]:
                    raise ValueError(f"Image changed during collection: {i}")
            left_bytes = paths[left].read_bytes()
            right_bytes = paths[right].read_bytes()
            mime = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
            }
            try:
                assessment = await provider.assess(
                    player_a_images=[(left_bytes, mime[paths[left].suffix.lower()])],
                    player_b_images=[(right_bytes, mime[paths[right].suffix.lower()])],
                )
            except VlmProviderQuotaError:
                # Retrying is guaranteed to fail. Stop the whole run; the
                # output file already holds everything completed so far.
                quota_hit.set()
                return
            except VlmProviderError as error:
                counters["failed"] += 1
                cause = error.__cause__ or error
                message = str(cause).replace(api_key, "<redacted>")[:500]
                detail = {
                    "pairId": pair_id(left, right),
                    "errorType": type(cause).__name__,
                    "status": str(getattr(cause, "status_code", None)),
                    "message": message,
                }
                failures.append(detail)
                print(f"teacher request failed: {detail['errorType']}: {message}", flush=True)
                if counters["failed"] >= args.max_failures:
                    failed_stop.set()
                return

            a, b = assessment.player_a, assessment.player_b
            if a.component_quality is None or b.component_quality is None:
                # One side was unusable, so there is no preference to learn.
                counters["unusable"] += 1
                return

            score_a = visual_fit_score(
                component_quality=a.component_quality,
                outfit_coordination=a.outfit_coordination,
                body_fit=a.body_fit,
            )
            score_b = visual_fit_score(
                component_quality=b.component_quality,
                outfit_coordination=b.outfit_coordination,
                body_fit=b.body_fit,
            )
            row = {
                "pairId": pair_id(left, right),
                "leftId": left,
                "rightId": right,
                "scoreA": score_a,
                "scoreB": score_b,
                # Raw margin is kept so training can weight confident pairs or
                # treat near-ties as ties, without re-querying the teacher.
                "margin": score_a - score_b,
                "frameQualityA": a.frame_quality,
                "frameQualityB": b.frame_quality,
                "model": provider.model_version,
                "provenance": provenance,
                "manifestFingerprint": manifest["fingerprint"],
                "imageHashes": {i: records[i]["sha256"] for i in (left, right)},
                "split": args.split,
                "assessment": assessment.model_dump(mode="json"),
            }
            async with write_lock:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                counters["ok"] += 1
                total = counters["ok"]
                if total % 25 == 0:
                    rate = total / (time.time() - started)
                    remaining = (len(todo) - total) / rate / 60 if rate else 0
                    print(
                        f"  {total}/{len(todo)} ok  {counters['unusable']} unusable  "
                        f"{counters['failed']} failed  ~{remaining:.0f} min left",
                        flush=True,
                    )

    try:
        await asyncio.gather(*(judge(left, right) for left, right in todo))
    finally:
        handle.close()
        attempt = {
            "at": datetime.now(UTC).isoformat(),
            "provenance": provenance,
            "manifestFingerprint": manifest["fingerprint"],
            "requestedPairs": args.pairs,
            "samplingSeed": args.seed,
            "concurrency": args.concurrency,
            "counts": counters,
            "failures": failures,
            "quotaStopped": quota_hit.is_set(),
        }
        with args.out.with_suffix(".attempts.jsonl").open("a", encoding="utf-8") as attempts:
            attempts.write(json.dumps(attempt) + "\n")

    try:
        destination = args.out.relative_to(REPO_ROOT)
    except ValueError:
        destination = args.out
    print(
        f"\nwrote {counters['ok']} teacher labels to {destination}\n"
        f"unusable={counters['unusable']}  failed={counters['failed']}"
    )
    if quota_hit.is_set():
        print(
            "\nSTOPPED: the Gemini quota is exhausted. Everything above is saved —\n"
            "re-run the same command after the quota resets and it resumes."
        )
    if counters["failed"] or quota_hit.is_set():
        sys.exit("Teacher collection incomplete; inspect the attempt log before resuming")


if __name__ == "__main__":
    asyncio.run(main())
