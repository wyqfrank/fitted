# FITTED next steps

This file tracks unfinished work. Check an item only after it has been completed
and verified.

The two-player flow, five-second round, garment detection, live ranker, and
Gemini final score are implemented. The main work left is testing on the demo
hardware and choosing where to deploy the app.

See [architecture.md](architecture.md) for the current system and [specs/](specs/)
for detailed plans.

## Before the next demo

- [ ] Test RF-DETR-Seg on 15 to 20 webcam crops from the demo hardware. The API
  and one-frame-per-second transport are ready, but this final check has not run.
  See [cv-detection.md section 12.1](specs/cv-detection.md).
- [ ] If that test passes, enable garment detection in live battles at about one
  frame per second. Live scoring uses the same frame pairs, so it also depends on
  garment detection being enabled.
- [ ] Run a full battle between two laptops with the current live ranker. The
  two-laptop flow was tested on 2026-08-22, before this ranker was added.
- [ ] Choose a hosting service and deploy both the web and inference services.
- [ ] Run the signalling and TURN tests on the venue network.

## Checks still needed

- [ ] Reduce ranker latency or change the target. The goal is 95 ms at p95. On
  640 px WebP input, the current model takes 89.2 ms at the median and 117.4 ms
  at p95. The previous model takes 88.9 ms and 110.1 ms.
- [ ] Measure latency with HTTP, garment detection, and two players active. The
  existing tests measure only the ranker.
- [ ] Test the live ranker on webcam frames. It was trained on full-body photos
  and may give confident scores for unfamiliar images. The `/preview` page has a
  **LIVE MODEL** view for checking score spread and separation.
- [ ] Measure how often live and final scores agree, how far apart they are, and
  how often the apparent leader changes. See
  [scoring-spec.md](specs/scoring-spec.md).
- [ ] Run the web, coordinator, and production-build checks with the current
  artifact. These were last run on 2026-08-22. The Python tests and TypeScript
  check were run on 2026-09-16.

The model's 90.5% result is agreement with Gemini on 63 reused validation pairs.
It is not accuracy on new data. Human agreement on those pairs is 74.6%, and the
same pairs were used to choose the model.

## Product changes

- [ ] Replace the single live score with a range or a neutral "leaning" state.
  The live score disagrees with the final result often enough that it should not
  appear definitive.
- [ ] Decide what to show when only one player is present. The current score bar
  is empty because scoring requires a pair of frames.
- [ ] Consider sending JPEG instead of WebP. On the measured CPU, decoding a
  640 px WebP takes about 14 ms longer. This change alone will not meet the p95
  target.

## Model work

These ideas may improve the model, but the product does not depend on them.

- [ ] Compare SigLIP 2 Base and FashionCLIP with DINOv2-S on person-disjoint
  data. A better head cannot recover information that the image features do not
  contain.
- [ ] Measure whether Gemini gives consistent answers when pairs are repeated,
  swapped, or arranged in cycles. See experiment 2 in
  [ranker-audit.md](ranker-audit.md).
- [ ] Compare the current CLS feature with pooled and patch features. See
  experiment 4.
- [ ] Test a properly regularised nonlinear head and measure its learning curve.
  The older MLP only regularised its output. See experiment 5.
- [ ] Fine-tune the last encoder block only if the earlier tests show that frozen
  features are the limit. See experiment 6.
- [ ] Test human agreement with a new group of raters chosen before results are
  inspected. The current AC and DP group was chosen after comparing agreement.

## Later research

- [ ] Build the full CV fixture set, compare Pose Lite and Pose Full, tune
  thresholds, and test face blurring and background removal. See
  [cv-detection.md](specs/cv-detection.md).
- [ ] Evaluate garment detection on DeepFashion2 and Fashionpedia after mapping
  their labels to the product's smaller category set.
- [ ] Build the learned combiner, including its dataset format, pairwise
  training, calibration, and temperature selection. See
  [scoring-spec.md](specs/scoring-spec.md).
- [ ] Decide how draws work and how final scores map to the 0 to 100 display.

## Out of scope

These are deliberate exclusions, not unfinished work.

- Instagram and Depop models, including social engagement signals.
- A new subject- and outfit-grouped webcam set for accepting this ranker.
- Mobile and narrow-screen layouts.
- Profiles, social feeds, matchmaking, global leaderboards, ecommerce, virtual
  try-on, and production-scale infrastructure.
