# Ranker implementation and experiment log

Updated 2026-09-16. Implements the priorities in [ranker-audit.md](ranker-audit.md).
Primary objective: **live–Gemini agreement**. Human agreement is a separate quality
measure; copying Gemini does not establish human-level quality.

## Status and boundaries

- [x] **Makes the metric trustworthy:** frozen manifests with file/content hashes,
  group/split conflict checks, immutable input fingerprints, and explicit
  image-ID-only fallback grouping. Regression checks pass.
- [x] **Makes the metric trustworthy:** new teacher rows record model, prompt hash,
  media resolution, scoring/provider code hashes, image hashes and manifest.
  Resume rejects changed settings; collection loads `.env` before CLI defaults.
- [x] **Makes the metric trustworthy:** image-disjoint teacher validation,
  training-only projection, strict cache checks, paired group bootstrap, and
  cached-only benchmark by default. The evaluator reports teacher draws/coverage.
- [x] **Raises the metric (experiment implemented, gain not demonstrated):** true
  identity coordinates and PCA 16/64/128, each with normalisation on/off and
  L2 0/0.0001/0.001/0.01. All 32 artifacts and predictions are retained.
- [x] **Makes the metric trustworthy:** matched Gemini judgments collected for
  all 65 existing AC/DP validation pairs after explicit upload approval.
- [x] **Makes the metric trustworthy:** snapshot assembler rejects incomplete
  validation, mixed teachers, wrong splits and hashes; ignores an unfinished
  trailing row during append-only collection. Regression checks pass.
- [x] **Makes the metric trustworthy:** regenerate all 157 training/validation
  application embeddings locally, recording encoder revision, preprocessing,
  input hashes and cache hash. Test embeddings are excluded.
- [x] **Raises the metric (pilot only):** train all 32 candidates on the 339
  saved application-teacher pairs and measure against the shipped artifact on
  the identical 65 validation pairs. Independent gain remains unverified.
- [x] **Makes the metric trustworthy:** execute the real CPU `score(bytes)`
  latency harness on both artifacts. The strict 95 ms p95 gate did not pass.
- [x] The frozen 2,000-label candidate now ships as `dinov2s-linear-v2`. The live
  loader reproduces its validation margins.
- [x] **Implemented and measured, not independently verified:** all 2,000 teacher
  comparisons over the existing application training photographs are complete;
  immutable 500/1,000/2,000-label snapshots and all 96 candidate fits are retained.
  The reused validation set shows an exploratory gain, described below.

See [next-steps.md](next-steps.md) for the remaining latency, benchmarking, and
model-diagnostic work.

Current photographs contain 128 training, 29 validation and 25 test images.
Grouping is **image-ID-only**: exact content overlap is checked, but different
photographs of the same person/outfit could remain across splits. New labels do
not make these old images an independent test. No new test predictions are made
by the sweep or the new collection commands.

The in-domain experiment adds comparisons over the same 128 training images. It
tests denser application-photo supervision; it does not test adding new people
or justify encoder unfreezing. The original Fashion144k teacher lacks historical
prompt provenance, so comparing it with the new application teacher also changes
teacher settings. A difference cannot be attributed exclusively to source domain.

Photos, judgements, embeddings, local-path manifests, and experimental artifacts
stay in ignored local directories. The shipped artifact was updated only when the
new model was selected on 2026-09-16. The photo set does not need to be shared.

## Executed linear control

Local run: `artifacts/ranker-audit-20260906/capacity-v2/report.json` (with source
copies, input hashes, environment and every candidate). Uses historical,
explicitly unverified embedding/teacher caches. One Torch CPU thread.

The original teacher graph leaves 993 training pairs (971 decisive), 37 validation
pairs (all decisive), and 895 discarded cross-split edges. Training and validation
endpoints do not overlap. The 47 test edges were partitioned but not scored.

| Candidate | Training teacher agreement | Validation teacher agreement |
|---|---:|---:|
| PCA 16, normalised, L2 .01 baseline | .666 | 24/37 = .649 |
| Selected PCA 16, raw, L2 .0001 | .663 | 24/37 = .649 |
| Identity 384, normalised, L2 0 | .919 | 17/37 = .459 |
| Identity 384, raw, L2 0 | .920 | 16/37 = .432 |

Selection uses teacher validation accuracy, then BCE, then grid order. Selected
and baseline make identical predictions on all 37 validation pairs. Their paired
empirical delta is zero; its degenerate bootstrap interval is **not an equivalence
proof**. Secondary AC/DP human validation regresses from 111/158 = .703 for the
baseline to 105/158 = .665 for the selected teacher-only head. No candidate was
promoted and no performance gain was established.

True identity training fit around .92 refutes the old .695 *training-fit ceiling*.
Weak validation still leaves representation, generalisation, and teacher noise
unseparated. Selecting among 32 candidates on 37 pairs creates substantial
selection uncertainty; these numbers are exploratory.

## Matched application-photo teacher

New collection uses the configured `gemini-3.6-flash`, prompt **v1**, high media
resolution, one photo per side. Both training and validation record the same
provenance. This matches configured scoring settings, but the real final assessor
can receive a five-frame burst, so this is not a measured live-to-final flip rate.

Validation collection: **65/65 usable**, zero failures, 29 images. The shipped
artifact on this new teacher cache, using the historical embedding cache:

| Measurement | Result |
|---|---:|
| Teacher decisive coverage | 63/65 = .969 |
| Shipped ranker–teacher agreement on decisive teacher pairs | 47/63 = .746 |
| Agreement including teacher draws as distinct outcomes | 47/65 = .723 |
| Gemini agreement with both human votes | 90/130 = .692 |
| Shipped ranker agreement with both human votes | 93/130 = .715 |
| AC–DP inter-rater agreement | 50/65 = .769 |

Local evidence: `app-validation-shipped.json`; regenerated embeddings reproduced
every metric in `app-validation-shipped-refreshed.json` without a legacy-cache
override. The student uses its raw ordering;
it has no fitted draw threshold. The evaluator applies the production two-point
draw rule to Gemini scores. The apparent 2.3-point student advantage against humans
has paired bootstrap CI approximately **[-22.0, +15.4] points** for the opposite
teacher-minus-student difference. These previously used validation images, selected
raters, and broad uncertainty do not establish that the live ranker surpasses Gemini.

Training collection resumed after explicit training-photo upload consent. The
existing **339** judgments were reused and the remaining **1,661** were collected
at concurrency 16. The completed cache contains **2,000/2,000 usable judgments**,
zero unusable responses and zero failures. Its dry-run resume check reports zero
remaining pairs and verifies the same manifest, model, prompt v1, high media
resolution, single-image protocol, scoring hash and provider hash. Collection
completion implements the planned supervision budget; it does not demonstrate a
model-quality gain by itself.

## Fixed 500/1,000/2,000 checkpoints

Frozen datasets `data-500/`, `data-1000/` and `data-2000/` each contain the exact
append-order prefix and the same complete 65-pair validation cache. Fits are in
the corresponding `fit-500/`, `fit-1000/` and `fit-2000/` directories. Every
snapshot retains all 32 artifacts, predictions, source copies, input hashes and
the report. All runs verify teacher and embedding provenance, cover all 128
training images, and have zero training/validation image overlap.

| Training labels | Snapshot-selected configuration | Decisive Gemini validation | Fixed PCA-16 norm .01 | Selected minus fixed baseline, grouped CI |
|---:|---|---:|---:|---:|
| 500 | PCA 64, normalised, L2 .01 | 59/63 = .937 | 53/63 = .841 | +9.52 points [0.0, +25.01] |
| 1,000 | identity 384, normalised, L2 .0001 | 56/63 = .889 | 52/63 = .825 | +6.35 points [−13.85, +25.45] |
| 2,000 | identity 384, normalised, L2 .0001 | 57/63 = .905 | 51/63 = .810 | +9.52 points [−8.93, +30.30] |

The 500-label validation peak is not chosen: selecting the checkpoint after seeing
these results would add another adaptive choice. The frozen full-data candidate is
the 2,000-label snapshot's selected head,
`fit-2000/identity-384-l2norm-reg-0.0001`, SHA-256
`378241c75bb0c790b94dea49be45b5a80ed6db4af9678a63c9d2461e33341d30`.
The immutable report records that selection and marks it promotion-ineligible.

Paired cached-only evaluation against the shipped artifact is retained in
`app-validation-2000-paired.json`:

| Identical reused validation comparisons | Shipped artifact | Frozen 2,000-label candidate |
|---|---:|---:|
| Agreement with decisive Gemini judgments | 47/63 = .746 | 57/63 = .905 |
| Agreement including Gemini draws | 47/65 = .723 | 57/65 = .877 |
| Agreement with both human votes | 93/130 = .715 | 97/130 = .746 |

Candidate-minus-shipped decisive teacher agreement is **+15.87 points**, grouped
bootstrap CI **[+1.30, +34.38]**. Human agreement is **+3.08 points**, CI
**[−15.79, +20.00]**. The apparent teacher-fidelity gain is measured only on the
historically reused 29-image validation set and does not correct the 32-way model
selection. It is promising development evidence, not independent generalisation.
The separate human result does not establish that the candidate surpasses Gemini's
human-rated quality.

## Application-teacher pilot: 339 training pairs

Pipeline verification during the upload pause, **not** the planned 500-label
checkpoint. Frozen dataset: `data-pilot339/`; all candidate fits:
`fit-pilot339/report.json`. Both are under `artifacts/ranker-audit-20260906/`.
339 training judgments include eight teacher ties, leaving 331 decisive labels.
All 128 training images are represented, with zero validation-image overlap.
New teacher and embedding provenance checks pass without legacy overrides.

Selected candidate: **PCA 128, L2 normalisation, regularisation .001**. Selection
uses only the 65 teacher-validation pairs, of which 63 are decisive. The fixed
PCA-16 normalised .01 baseline trained on these same labels scores 51/63 = .810;
the selected head scores 55/63 = .873. The +6.35-point paired difference has
exploratory bootstrap CI **[-12.5, +26.2] points**.

| Identical validation comparisons | Shipped artifact | Selected pilot |
|---|---:|---:|
| Agreement with decisive Gemini judgments | 47/63 = .746 | 55/63 = .873 |
| Agreement including Gemini draws | 47/65 = .723 | 55/65 = .846 |
| Agreement with both human votes | 93/130 = .715 | 93/130 = .715 |

Paired report: `app-validation-pilot339-paired.json`. The selected-minus-shipped
teacher-agreement difference is **+12.70 points**, bootstrap CI **[-3.77, +32.31]**.
Human-agreement difference is zero, CI **[-20.34, +17.81] points**. Neither interval
corrects validation selection. Equal human totals do not mean equal predictions.
On the broader human-validation set (including pairs only one rater judged),
selected teacher-only accuracy is 113/158 = .715 versus 105/158 = .665 for the
same-data fixed PCA-16 teacher-only baseline; this is a different denominator.

This is a promising development result, **not evidence that the deployed live
ranker improves by 12.7 points or surpasses Gemini**. It changes training labels,
head configuration and fine-tuning stage relative to the shipped artifact.
There are only 29 validation images, the cohort was selected historically, and
the candidate was selected on these very teacher judgments. No artifact promotion.

## CPU latency and verification

`latency-pilot339-default.json`: 20 training photographs × three repetitions =
60 measured frames per artifact, five warm-ups per artifact, alternating model
order. Windows 11, Intel Family 6 Model 141, Torch 2.12.1, default eight intra-op
and eight inter-op threads, CPU. Includes image decoding, preprocessing, encoder,
projection, head and display calibration; excludes HTTP, detector work and live
concurrent load. This is a local development timing, not the deployment gate.

| Artifact | Median | p95 | Strict 95 ms p95 gate |
|---|---:|---:|---|
| Shipped | 77.0 ms | 101.7 ms | Fail |
| Selected pilot | 76.5 ms | 103.8 ms | Fail |

Typical scoring time did not rise in this sample. The new candidate cannot be
claimed within a strict 95 ms tail-latency budget; the old artifact also misses
that interpretation. This does not reproduce or disprove a historical *mean*
95 ms claim measured on another workload. Benchmark representative webcam crops
under the actual service workload before promotion.

The frozen 2,000-label candidate was measured again in
`latency-2000-default.json` with the same 20 training photographs, three
repetitions, five warm-ups, alternating order and default eight Torch threads:

| Artifact | Median | p95 | Strict 95 ms p95 gate |
|---|---:|---:|---|
| Shipped | 73.5 ms | 117.4 ms | Fail |
| Frozen 2,000-label candidate | 70.6 ms | 111.8 ms | Fail |

Exploratory candidate thread-setting runs (60 frames each) produced median/p95
of 162.3/182.8 ms at one thread, 99.2/119.6 ms at two, 75.5/96.7 ms at four,
76.4/97.9 ms at five, and 64.1/84.4 ms at six. Because six threads was selected
after seeing those small runs, `latency-2000-threads6-confirm.json` increased the
sample to 200 frames per artifact: the candidate records **68.2 ms median / 101.9
ms p95**, while shipped records **67.9/111.8 ms**. The tuned candidate therefore
still fails the strict p95 gate. No runtime thread setting was changed.

Median latency is within 95 ms, but the required p95 is not. This harness still
excludes HTTP, garment detection and concurrent service load, so realistic latency
is not independently verified. Fresh subject/outfit-grouped webcam acceptance data
has not been supplied. The model was still selected despite both missing checks.

Final verification: **108 inference tests passed, one opt-in Gemini smoke test
skipped**; Ruff passes for every changed Python file; `git diff --check` passes.
The existing Starlette/httpx deprecation warning remains. The completed 65-pair
validation collection separately exercises the external provider with consent.
Before that change, the shipped artifact had SHA-256
`309f67a0152de6399f1fe8a10e72f82abf54ee93d90f3e2b64658df3c0e4707d`.

## Model change on 2026-09-16

The team chose the frozen 2,000-label model without completing the two checks
above. A new subject- and outfit-grouped webcam set was out of scope, and the team
also skipped the webcam sanity check. Its 90.5% result is agreement with Gemini on
a reused validation set, not accuracy on new data.

- `models/ranker/ranker.npz` is the candidate unchanged, SHA-256
  `378241c75bb0c790b94dea49be45b5a80ed6db4af9678a63c9d2461e33341d30`.
- `ranker.json` keeps the training metadata, names the model
  `dinov2s-linear-v2`, and records the previous artifact for rollback. The old
  files remain in Git history.
- Loading `models/ranker` through `Dinov2FitRanker` reproduces all 65 saved
  validation margins from the 29 validation photos. The largest difference is
  8.3e-7, all margin signs match, and display scores range from 55.9 to 85.0.
- 108 inference tests passed, one skipped; Ruff passes.

### Latency on webcam-sized input

`latency-2000-640px-webp-default.json` resizes the 128 training photos to match
the browser input: at most 640 px wide, with WebP quality set to 0.82. It times
`score(bytes)` after 10 warm-ups, with three runs per image in alternating order
and eight Torch threads. It does not include HTTP, garment detection, or
concurrent requests.

| Artifact | Median | p95 | p99 | Frames over 95 ms | Strict 95 ms p95 gate |
|---|---:|---:|---:|---:|---|
| Previous PCA-16 | 88.9 ms | 110.1 ms | 140.8 ms | 100/384 | Fail |
| Current model | 89.2 ms | 117.4 ms | 145.0 ms | 103/384 | Fail |

An earlier estimate suggested that smaller images would meet the target. This
test shows that they do not. `latency-2000-format-diagnostic.txt` compares three
input formats for 40 images in one process using eight threads:

| Input | Median | p95 | Decode + preprocess median |
|---|---:|---:|---:|
| Original JPEG | 91.6 ms | 167.1 ms | 29.5 ms |
| 640 px WebP | 90.4 ms | 135.2 ms | 30.6 ms |
| 640 px JPEG | 78.4 ms | 118.7 ms | 16.3 ms |

The encoder takes about 60 ms regardless of image size, and slow runs already
exceed the p95 target. At 640 px, WebP decoding takes about 14 ms longer than
JPEG. The same original images ran about 20 ms slower than they did on 2026-09-07,
so only compare results from this run. The scripts are stored with the reports.

## Reproduction commands

Run from the repository root using `.venv/Scripts/python.exe`. Paths below are
local example run names; output commands deliberately refuse overwrites.

```powershell
.venv/Scripts/python.exe services/inference/scripts/ranker_evaluation.py --out artifacts/new-run/manifest.json
.venv/Scripts/python.exe services/inference/scripts/refresh_ranker_embeddings.py --manifest artifacts/new-run/manifest.json --out artifacts/new-run/app-embeddings.npz --offline
.venv/Scripts/python.exe services/inference/scripts/distil_teacher.py --manifest artifacts/new-run/manifest.json --source human --pairs 2000 --out artifacts/new-run/teacher-train.jsonl --dry-run
```

The dry run uploads nothing. Actual collection requires authorization for its
photo set. Validation collection uses `benchmark_judges.py --split val
--collect-only --manifest ... --cache ...`; ordinary benchmark execution is
cached-only. New caches reject old unversioned rows. `--allow-legacy-cache` is
an explicit exploratory escape hatch, not a provenance repair.

Freeze a complete checkpoint and run all candidates:

```powershell
.venv/Scripts/python.exe services/inference/scripts/assemble_teacher_dataset.py --manifest artifacts/new-run/manifest.json --train artifacts/new-run/teacher-train.jsonl --validation artifacts/new-run/teacher-val.jsonl --out artifacts/new-run/data-500 --min-train 500 --max-train 500
.venv/Scripts/python.exe services/inference/scripts/sweep_ranker.py --manifest artifacts/new-run/data-500/manifest.json --teacher artifacts/new-run/data-500/teacher.jsonl --embeddings artifacts/new-run/app-embeddings.npz --human-embeddings artifacts/new-run/app-embeddings.npz --out artifacts/new-run/fit-500
```

Use analogous fresh directories for 1,000 and 2,000. Snapshots are nested prefixes
of successfully completed collection rows; concurrent request completion order is
retained and is not a randomised label-efficiency study. Compare the fixed baseline
as well as the selected head at every checkpoint. Do not quietly choose the best
checkpoint and call its validation score an independent result.

`train_ranker.py` retains the historical MLP recipe for reproduction, labelled as
such; its output-only regularisation has not been repaired or validated as the
controlled nonlinear experiment. Human fine-tuning still selects human validation,
so it should not silently replace teacher-fidelity selection.

## Acceptance data explained

"Fresh webcam data" means new photos or short sequences from the actual camera
setup, followed by new A/B judgments. A local grouping file maps each image ID to
a stable subject/outfit group. Frames of one person/outfit must remain in one
split; if a person wears several outfits, keep that person's related groups
together when the intended claim is generalisation to new people. The existing
photos/comparisons remain useful for development. They cannot independently
validate choices already influenced by them.
