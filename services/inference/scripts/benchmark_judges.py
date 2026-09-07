#!/usr/bin/env python
"""Score every available judge on the same held-out human-labelled pairs.

    python services/inference/scripts/benchmark_judges.py --raters AC,DP --split val
        --manifest <manifest.json>

Answers the question the distillation work left open: the student was trained on
Gemini's judgements of *Fashion144k* images, and Gemini has never been measured
against the project's own raters. Without that number there is no way to tell
whether the ranker underperforms because the linear head is lossy or because the
teacher it copied is itself weak.

Three judges are scored on identical pairs:

  human   - inter-rater agreement, not a population accuracy ceiling
  gemini  - the production assessor, run directly on the labelled pool
  ranker  - the trained student in models/ranker

Only pairs both raters judged decisively enter the human comparison, so every
judge is measured on the same evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "services" / "inference" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

POOL_DIR = REPO_ROOT / "apps" / "web" / "public" / "label-pool"
DECISIONS_DIR = REPO_ROOT / "data" / "labelling"
ARTIFACT_DIR = REPO_ROOT / "models" / "ranker"
CACHE_PATH = ARTIFACT_DIR / "judge-benchmark.jsonl"

MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}

from distil_teacher import load_dotenv  # noqa: E402
from ranker_artifact import load_scorer  # noqa: E402
from ranker_evaluation import (  # noqa: E402
    file_hash,
    load_embeddings,
    load_manifest,
    outcome,
    paired_summary,
    read_jsonl,
    teacher_provenance,
    validate_teacher_cache,
    write_json,
)

from fitted_inference.scoring import visual_fit_score  # noqa: E402
from fitted_inference.vlm import (  # noqa: E402
    VLM_PROMPTS,
    GeminiVlmProvider,
    VlmProviderError,
    VlmProviderQuotaError,
)


def load_pairs(directory: Path, raters: set[str], split: str) -> dict[tuple[str, str], dict]:
    """Canonical pair -> {rater: bool 'the canonically-first image won'}."""
    by_pair: dict[tuple[str, str], dict] = collections.defaultdict(dict)
    for file in directory.glob("decisions.*.jsonl"):
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["raterId"].upper() not in raters or row["split"] != split:
                continue
            if row["shownVerdict"] not in ("a", "b"):
                continue
            left, right = row["shownLeftId"], row["shownRightId"]
            winner = left if row["shownVerdict"] == "a" else right
            key = (min(left, right), max(left, right))
            by_pair[key][row["raterId"].upper()] = winner == key[0]
    return dict(by_pair)


def resolve(stem: str) -> Path:
    for extension in MIME:
        candidate = POOL_DIR / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    sys.exit(f"Image {stem} is not in {POOL_DIR}.")


async def run_gemini(
    pairs: list[tuple[str, str]],
    concurrency: int,
    out: Path,
    *,
    cached_only: bool = True,
    allow_legacy: bool = False,
    manifest: dict | None = None,
) -> dict:
    """Judge each pair with the production assessor. Resumable via `out`."""
    done: dict[str, float] = {}
    load_dotenv(REPO_ROOT / ".env")
    model = os.getenv("FITTED_VLM_MODEL", "gemini-3.6-flash")
    prompt_version = os.getenv("FITTED_VLM_PROMPT_VERSION", "v2")
    resolution = os.getenv("FITTED_VLM_MEDIA_RESOLUTION", "high")
    provenance = teacher_provenance(model, prompt_version, VLM_PROMPTS[prompt_version], resolution)
    if allow_legacy and not cached_only:
        raise ValueError("Legacy caches are read-only; collect new judgments into a new cache")
    if out.exists():
        rows = read_jsonl(out)
        validate_teacher_cache(rows, provenance, allow_legacy=allow_legacy)
        for row in rows:
            if row["pairId"] in done:
                raise ValueError("Duplicate benchmark judgments")
            if "provenance" in row:
                a, b = row["pairId"].split("|")
                paths = {
                    i: Path(manifest["images"][i]["path"]) if manifest else resolve(i)
                    for i in (a, b)
                }
                if row.get("imageHashes") != {i: file_hash(p) for i, p in paths.items()}:
                    raise ValueError("Benchmark cache image content mismatch")
            done[row["pairId"]] = row["margin"]

    todo = [p for p in pairs if f"{p[0]}|{p[1]}" not in done]
    print(f"gemini: {len(done)} cached, {len(todo)} to judge")
    if cached_only:
        return done
    if todo:
        if manifest is None:
            raise ValueError("New benchmark collection requires --manifest")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            sys.exit("GEMINI_API_KEY is not set; add it to the repo's .env.")

        provider = GeminiVlmProvider(
            api_key=api_key,
            model=model,
            prompt=VLM_PROMPTS[prompt_version],
            media_resolution=resolution,
            timeout_seconds=30.0,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        handle = out.open("a", encoding="utf-8")
        semaphore = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()
        stop = asyncio.Event()
        counters = collections.Counter()

        async def judge(first: str, second: str) -> None:
            if stop.is_set():
                return
            async with semaphore:
                if stop.is_set():
                    return
                images = []
                for stem in (first, second):
                    path = Path(manifest["images"][stem]["path"])
                    if file_hash(path) != manifest["images"][stem]["sha256"]:
                        raise ValueError(f"Image changed during collection: {stem}")
                    images.append((path.read_bytes(), MIME[path.suffix.lower()]))
                try:
                    assessment = await provider.assess(
                        player_a_images=[images[0]], player_b_images=[images[1]]
                    )
                except VlmProviderQuotaError:
                    stop.set()
                    return
                except VlmProviderError:
                    counters["failed"] += 1
                    return
                a, b = assessment.player_a, assessment.player_b
                if a.component_quality is None or b.component_quality is None:
                    counters["unusable"] += 1
                    return
                margin = visual_fit_score(
                    component_quality=a.component_quality,
                    outfit_coordination=a.outfit_coordination,
                    body_fit=a.body_fit,
                ) - visual_fit_score(
                    component_quality=b.component_quality,
                    outfit_coordination=b.outfit_coordination,
                    body_fit=b.body_fit,
                )
                async with lock:
                    handle.write(
                        json.dumps(
                            {
                                "pairId": f"{first}|{second}",
                                "leftId": first,
                                "rightId": second,
                                "split": manifest["images"][first]["split"],
                                "margin": margin,
                                "provenance": provenance,
                                "manifestFingerprint": manifest["fingerprint"],
                                "imageHashes": {
                                    i: manifest["images"][i]["sha256"] for i in (first, second)
                                },
                                "assessment": assessment.model_dump(mode="json"),
                            }
                        )
                        + "\n"
                    )
                    handle.flush()
                    done[f"{first}|{second}"] = margin
                    counters["ok"] += 1
                    if counters["ok"] % 25 == 0:
                        print(f"  {counters['ok']}/{len(todo)}")

        try:
            await asyncio.gather(*(judge(a, b) for a, b in todo))
        finally:
            handle.close()
        if stop.is_set():
            print("  STOPPED on quota; re-run to resume.")
        print(f"  ok={counters['ok']} unusable={counters['unusable']} failed={counters['failed']}")
    return done


def require_complete_judgments(pairs, margins: dict) -> None:
    missing = sum("|".join(pair) not in margins for pair in pairs)
    if missing:
        raise ValueError(
            f"Benchmark is missing {missing} requested judgments; "
            "collect the complete fixed pair set before evaluating"
        )


def summarize_judges(by_pair: dict, margins: dict, score, *, groups=None) -> dict:
    rows = []
    for pair, humans in sorted(by_pair.items()):
        if len(humans) != 2 or "|".join(pair) not in margins:
            continue
        teacher = outcome(margins["|".join(pair)])
        student = outcome(float(score(pair[0]) - score(pair[1])), draw_threshold=0)
        rows.append((pair, list(humans.values()), teacher, student))
    decisive = [r for r in rows if r[2] != 0]
    report = {
        "evaluationStatus": "exploratory; shared images and selected cohort; no population ceiling",
        "protocol": "single-image-pair; not a live-round flip-rate measurement",
        "studentDecisionRule": "raw-score ordering; no calibrated draw threshold",
        "teacherDecisionRule": "production score margin; draw when absolute margin < 2",
        "eligiblePairs": sum(len(v) == 2 for v in by_pair.values()),
        "scoredPairs": len(rows),
        "teacherDraws": len(rows) - len(decisive),
        "teacherDecisiveCoverage": len(decisive) / len(rows) if rows else None,
        "studentTeacherDecisiveAgreement": paired_summary(
            [float(t == s) for _, _, t, s in decisive], [p for p, *_ in decisive], groups=groups
        ),
        "studentTeacherOutcomeAgreement": paired_summary(
            [float(t == s) for _, _, t, s in rows], [p for p, *_ in rows], groups=groups
        ),
        "humanInterRaterAgreement": paired_summary(
            [float(h[0] == h[1]) for _, h, _, _ in rows], [p for p, *_ in rows], groups=groups
        ),
    }
    # Both human votes remain together in every bootstrap replicate. A model draw
    # does not match a decisive human vote; this differs from the old forced sign.
    pairs = [p for p, h, _, _ in rows for _ in h]
    for name, index in (("teacherHumanAgreement", 2), ("studentHumanAgreement", 3)):
        hits = [float(row[index] == (1 if vote else -1)) for row in rows for vote in row[1]]
        report[name] = paired_summary(hits, pairs, groups=groups)
    report["teacherMinusStudentHumanAgreement"] = paired_summary(
        [
            float(t == (1 if vote else -1)) - float(s == (1 if vote else -1))
            for _, h, t, s in rows
            for vote in h
        ],
        pairs,
        groups=groups,
    )
    return report


def compare_students(by_pair: dict, margins: dict, candidate, baseline, *, groups=None) -> dict:
    """Paired candidate-minus-baseline changes; never used to select a candidate."""
    decisive_delta, decisive_pairs, outcome_delta, pairs, human_delta, human_pairs = (
        [] for _ in range(6)
    )
    for pair, humans in sorted(by_pair.items()):
        if len(humans) != 2 or "|".join(pair) not in margins:
            continue
        teacher = outcome(margins["|".join(pair)])
        a = outcome(float(candidate(pair[0]) - candidate(pair[1])), draw_threshold=0)
        b = outcome(float(baseline(pair[0]) - baseline(pair[1])), draw_threshold=0)
        delta = float(a == teacher) - float(b == teacher)
        outcome_delta.append(delta)
        pairs.append(pair)
        if teacher:
            decisive_delta.append(delta)
            decisive_pairs.append(pair)
        for vote in humans.values():
            target = 1 if vote else -1
            human_delta.append(float(a == target) - float(b == target))
            human_pairs.append(pair)
    return {
        "direction": "candidate minus baseline",
        "caveat": "Exploratory validation; intervals do not correct candidate selection",
        "teacherDecisiveAgreement": paired_summary(decisive_delta, decisive_pairs, groups=groups),
        "teacherOutcomeAgreement": paired_summary(outcome_delta, pairs, groups=groups),
        "humanAgreement": paired_summary(human_delta, human_pairs, groups=groups),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raters", default="AC,DP")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--decisions-dir", type=Path, default=DECISIONS_DIR)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--baseline-artifact", type=Path, help="report paired model differences")
    parser.add_argument("--embeddings", type=Path, default=ARTIFACT_DIR / "embeddings.npz")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--collect", action="store_true", help="call Gemini for missing pairs")
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="collect versioned judgments without loading a student embedding cache",
    )
    parser.add_argument(
        "--allow-legacy-cache",
        action="store_true",
        help="allow unversioned caches only for exploratory cached-only evaluation",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.collect_only:
        args.collect = True
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.collect and args.allow_legacy_cache:
        parser.error("Legacy caches are read-only; collect into a new versioned cache")
    if args.report and args.report.exists():
        parser.error("Report exists; use a new output to preserve the previous evaluation")
    raters = {r.strip().upper() for r in args.raters.split(",") if r.strip()}
    if len(raters) != 2:
        parser.error("Exactly two raters are required")
    manifest = load_manifest(args.manifest, verify_files=True) if args.manifest else None
    by_pair = load_pairs(args.decisions_dir, raters, args.split)
    both = {k: v for k, v in by_pair.items() if len(v) == 2}
    if not both:
        sys.exit("No overlapping decisive pairs")
    ids = sorted({i for p in both for i in p})
    if manifest:
        for i in ids:
            if manifest["images"][i]["split"] != args.split:
                raise ValueError("Decision split differs from manifest")
    margins = await run_gemini(
        sorted(both),
        args.concurrency,
        args.cache,
        cached_only=not args.collect,
        allow_legacy=args.allow_legacy_cache,
        manifest=manifest,
    )
    if args.collect_only:
        collected = sum("|".join(pair) in margins for pair in both)
        print(
            json.dumps(
                {"eligiblePairs": len(both), "collectedPairs": collected, "cache": str(args.cache)}
            )
        )
        if collected != len(both):
            sys.exit("Benchmark collection incomplete; resume before evaluating")
        return
    require_complete_judgments(both, margins)
    images = {i: Path(manifest["images"][i]["path"]) if manifest else resolve(i) for i in ids}
    emb, provenance = load_embeddings(
        args.embeddings, images, "facebook/dinov2-small", allow_legacy=args.allow_legacy_cache
    )
    score = load_scorer(args.artifact_dir, emb)
    groups = {i: manifest["images"][i]["group"] for i in ids} if manifest else None
    report = summarize_judges(both, margins, score, groups=groups)
    if args.baseline_artifact:
        baseline = load_scorer(args.baseline_artifact, emb)
        report["candidateMinusBaseline"] = compare_students(
            both, margins, score, baseline, groups=groups
        )
        report["baselineArtifactSha256"] = file_hash(args.baseline_artifact / "ranker.npz")
    report.update(
        {
            "split": args.split,
            "raters": sorted(raters),
            "grouping": manifest["grouping"] if manifest else "image-id-only",
            "manifestFingerprint": manifest["fingerprint"] if manifest else None,
            "embeddingProvenance": provenance,
            "legacyCacheAllowed": args.allow_legacy_cache,
            "artifactSha256": file_hash(args.artifact_dir / "ranker.npz"),
            "embeddingSha256": file_hash(args.embeddings),
            "teacherCacheSha256": file_hash(args.cache) if args.cache.exists() else None,
            "decisionHashes": {
                p.name: file_hash(p) for p in sorted(args.decisions_dir.glob("decisions.*.jsonl"))
            },
            "codeSha256": file_hash(Path(__file__)),
        }
    )
    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    asyncio.run(main())
