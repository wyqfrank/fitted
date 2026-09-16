# CV Detection and Frame Preparation Specification

**Product:** FITTED
**Status:** Browser pipeline and garment baseline implemented. Fixture calibration
and device testing remain.
**Scope:** Live person detection, pose tracking, frame-quality validation, and canonical outfit cropping  
**Out of scope for this implementation:** Taste scoring, source-expert training, pairwise calibration, VLM serving, and production deployment. Their contracts with detection are documented below.

This document expands the CV requirements in [`docs/PRD.md`](../PRD.md). See
[`docs/next-steps.md`](../next-steps.md) for unfinished work.

## 1. Purpose

The detection pipeline must convert a moving webcam feed into a recent, stable, consistently framed outfit image that is suitable for the visual encoder and scoring model.

Detection is an input-quality system. It must not decide whether an outfit is fashionable and must not use detection confidence as a fashion score.

```text
30 FPS webcam video ─────────────────────────► displayed video
          │
          └── latest frame only
                    ↓
          pose detection at approximately 10 FPS
                    ↓
          framing, visibility, motion, and quality checks
                    ↓
          stable padded full-outfit crop
                    ↓
          scoring capture at approximately 1 FPS
```

## 2. Goals

- Keep the displayed camera feed smooth while detection runs.
- Confirm that exactly one person is available for scoring.
- Determine whether enough of the outfit is visible.
- Follow normal movement without creating a stale frame queue.
- Reject or temporarily pause scoring for unusable frames.
- Produce a canonical padded crop without stretching the image.
- Use the same frame-preparation rules for training, evaluation, and live inference.
- Keep continuous detection local to the browser where practical.

## 3. Non-goals for the first implementation

- Identifying individual garment categories.
- Segmenting tops, bottoms, dresses, shoes, or accessories.
- Measuring garment fit from body landmarks.
- Recognising brands or estimating garment value.
- Scoring a person's face, body type, attractiveness, or identity.
- Reliably classifying front, side, and back outfit views.
- Building a guided multi-view outfit scan.

Garment detection and multi-view capture are later experiments and must not block the first whole-outfit scoring baseline.

## 4. Initial model choice

Use **MediaPipe Pose Landmarker** as the initial person and pose model.

Initial configuration:

| Setting | Initial value | Notes |
| --- | --- | --- |
| Model | Pose Landmarker Lite | Benchmark against Full before finalising |
| Running mode | `VIDEO` | Enables tracking across decoded video frames |
| Maximum poses | `2` | Detect a second person so multi-person frames can be rejected |
| Segmentation masks | Disabled | Avoid continuous mask-generation cost initially |
| Detection target | Approximately 10 FPS | Independent of the 30 FPS video presentation |
| In-flight detections | Maximum 1 | New frames are dropped while detection is busy |

MediaPipe provides 33 body landmarks, landmark visibility, and optional person segmentation. The Web implementation should run outside the React rendering path, preferably in a Web Worker. See the [official Pose Landmarker documentation](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/web_js).

## 5. Client-side architecture

Each browser analyses its own uncompressed local camera feed. The browser does not wait for a server round trip before showing framing guidance. The first implementation targets Chrome and Edge and reports `detector_unavailable` when the required browser APIs are missing.

```text
HTMLVideoElement at camera frame rate
              │
              ├──► normal video rendering
              │
              └──► requestVideoFrameCallback scheduler
                               │
                      latest-frame slot
                               │
                       Web Worker detector
                               │
                  timestamped detection result
                               │
                    temporal state machine
                               │
                    candidate-frame buffer
```

Each browser submits only its newest valid local candidate with room, player,
sample, and capture-time metadata. A server-side coordinator pairs fresh A/B
submissions, invokes the separate Python inference service once for the room, and
broadcasts one authoritative result. The service revalidates frame quality and
applies the selected canonical preprocessing consistently before scoring. Only
one comparison per room may be in flight; superseded or busy work is discarded.

```text
Player A local feed -> local framing guidance -> selected local candidate --\
                                                                        server-side
Player B local feed -> local framing guidance -> selected local candidate --/ pairing
                                                                        coordinator
                                                                            |
                                                                  paired comparison
```

This avoids asymmetric WebRTC compression in the scoring path because neither
browser captures the other player's decoded remote stream. The existing Node
room service owns the pairing coordinator and the Python inference service stays
stateless.

## 6. Scheduling and backpressure

Use `requestVideoFrameCallback()` where supported so detection is scheduled only when a new video frame exists.

Rules:

1. Cap detection scheduling at approximately one frame every 100 ms.
2. Allow only one detection operation in flight.
3. If the detector is busy, discard the pending frame and use the newest frame next.
4. Never queue frames for later detection.
5. Ignore detection results that arrive outside the accepted staleness window.
6. Timestamp every submitted frame and returned result.

```text
frame 1 ──► detecting
frame 2 ──► dropped
frame 3 ──► dropped
frame 4 ──► newest available frame after detection completes
```

If average detection time approaches or exceeds the scheduling interval, reduce the detection frequency instead of allowing work to accumulate.

## 7. Detection result contract

```ts
type OutfitFrameStatus =
  | "valid"
  | "no_person"
  | "multiple_people"
  | "partial_outfit"
  | "too_close"
  | "too_far"
  | "low_light"
  | "blurred"
  | "moving_too_fast"
  | "detector_unavailable";

type NormalizedRect = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type FrameQualityMetrics = {
  poseConfidence: number;
  landmarkVisibility: number;
  personFrameCoverage: number;
  sharpness: number;
  brightness: number;
  motion: number;
};

type VisibleRegions = {
  head: boolean;
  torso: boolean;
  legs: boolean;
  feet: boolean;
};

type OutfitDetectionResult = {
  capturedAt: number;
  observedStatus: OutfitFrameStatus;
  stableStatus: OutfitFrameStatus;
  scoreable: boolean;
  personBox: NormalizedRect | null;
  cropBox: NormalizedRect | null;
  landmarks: PoseLandmark[];
  visibleRegions: VisibleRegions;
  quality: FrameQualityMetrics;
  processingMs: number;
};
```

The exact metric scales must be documented alongside the implementation. Raw detector confidence must not be exposed to users as fashion confidence.

## 8. Frame-quality classification

### 8.1 Person count

- Zero detected poses → `no_person`.
- More than one sufficiently confident pose → `multiple_people`.
- Exactly one sufficiently confident pose → continue evaluating the frame.

### 8.2 Required body visibility

The first implementation should evaluate grouped landmarks rather than requiring every landmark independently:

- upper body: left/right shoulders;
- torso: left/right hips;
- legs: left/right knees;
- lower extent: ankles, heels, or foot-index landmarks.

A scoreable frame must provide reliable head, torso, and knee evidence. Ankles, heels, and foot-index landmarks are optional: missing feet do not invalidate an otherwise usable outfit, but `visibleRegions.feet` must be `false` so the future scorer knows that footwear evidence is unavailable. Exact confidence thresholds remain tunable until evaluated on representative webcam fixtures.

Long dresses, wide trousers, layering, and partial self-occlusion must be included in the fixture set so landmark visibility rules do not systematically reject valid outfits.

### 8.3 Framing

Reject or pause scoring when:

- the detected person occupies too little of the frame;
- the person fills the frame so tightly that padding cannot be added;
- the calculated person bounds touch the frame edge;
- the head, torso, or knees appear cropped;
- there is insufficient surrounding space for bags, outerwear, or silhouette.

Bottom-edge clipping is permitted when the knees remain usable. It must be recorded as missing foot evidence rather than silently implying that shoes were assessed.

Initial tunable values:

| Parameter | Initial value |
| --- | --- |
| Minimum person height | 40% of frame height |
| Maximum person height | 92% of frame height |
| Minimum border clearance | 3% of frame dimension |
| Crop padding | 15% around detected bounds |

These values are starting points, not acceptance criteria. Fixture evaluation must determine the final thresholds.

### 8.4 Brightness and blur

Brightness and sharpness checks should use inexpensive image statistics on a small grayscale sample.

- Brightness should be measured primarily within the person crop.
- Blur detection should distinguish camera softness from rapid motion where practical.
- Device-specific thresholds must be calibrated on the intended demo laptops.
- A single low-quality observation must not immediately invalidate an otherwise stable stream.

### 8.5 Motion

Estimate motion using changes in normalised landmarks and the person box between detections.

- Normal movement should keep tracking active.
- Rapid turns or high landmark displacement should return `moving_too_fast` for capture selection.
- The displayed camera must remain uninterrupted.
- The previous score may remain visible while capture is paused.

Pose detection tracks motion; it does not by itself classify whether the user is facing front, side, or back.

## 9. Temporal stability

Use hysteresis so transient misses do not make the UI flicker.

Initial state rules:

- Enter `valid` after 2 consecutive valid detections.
- Leave `valid` after 3 consecutive invalid detections.
- Retain the last usable pose for up to 300 ms during a brief tracking miss.
- Do not select a result older than 300 ms for a new scoring request.

Landmarks and crop bounds should use temporal smoothing. An initial exponential smoothing factor of `0.4` for the newest observation may be evaluated:

```text
smoothed = 0.4 × newest + 0.6 × previous
```

Smoothing must not allow an old crop to lag significantly behind a moving person.

## 10. Canonical crop

The crop must preserve the complete visible outfit evidence and enough context for silhouette, layering, headwear, and carried accessories. Shoes are retained when visible but are not mandatory for a valid crop.

Crop algorithm:

1. Calculate bounds from reliable pose landmarks and the detector's person region.
2. Expand the bounds by the configured padding.
3. Preserve available space around extremities and likely accessories.
4. Clamp the expanded rectangle to the source frame.
5. Reject the frame if clamping removes required outfit regions.
6. Pad to the encoder's expected aspect ratio.
7. Resize without stretching.

Do not create tight per-garment crops for the whole-outfit scoring path.

Face blurring and neutral-background segmentation should be controlled experiments:

- baseline: padded person crop;
- ablation A: padded crop with face blurred;
- ablation B: person mask over a neutral background.

The same selected preprocessing variant must be applied to social-source images, human-labelled images, evaluation fixtures, and live webcam frames.

## 11. Candidate-frame selection

Maintain a small buffer of recent valid candidates rather than automatically scoring the frame nearest the timer tick.

```ts
type CandidateFrame = {
  capturedAt: number;
  crop: ImageBitmap;
  quality: FrameQualityMetrics;
};
```

Initial behaviour:

- retain up to 5 recent candidates;
- remove candidates older than 300 ms;
- prefer high landmark visibility and sharpness;
- penalise excessive motion;
- select no frame when every recent candidate is unusable;
- preserve the previous score when a scoring tick is skipped.

The selection formula must be deterministic for the same candidate set.

## 12. Garment detection experiment

Garment detection is not a prerequisite for the first whole-outfit ranker.

If component-level analysis is explored, begin with a server-side open-vocabulary detection experiment using a fixed reduced taxonomy:

```text
top
bottoms
one-piece / dress
outerwear
shoes
bag
headwear
other accessory
```

Grounding DINO was the initial zero-shot experiment. Fashionpedia and DeepFashion2 remain useful sources of categories, masks, landmarks, and later evaluation data, but broad checkpoint comparison and dataset training are deferred until after the hackathon.

The implemented baseline uses Grounding DINO Tiny in the Python inference service. It queries a reduced set of text prompts and maps them into the product taxonomy above. `/v1/garments` accepts one canonical outfit crop and returns normalised boxes, canonical categories, confidence, a diagnostic matched prompt, and an explicit state for every queried category. The matched prompt must not be presented as a reliable fine-grained subtype; the canonical category is the product contract.

The zero-shot adapter uses `not_detected` when no box passes the configured thresholds. It does not emit `not_present`, because absence cannot be established reliably from a missed zero-shot detection. A later presence model may emit `not_present` after calibration.

DeepFashion2 and Fashionpedia enter as evaluation and improvement data after this zero-shot baseline:

- DeepFashion2 supplies garment categories, boxes, masks, landmarks, viewpoint, and occlusion cases for localisation and visibility evaluation.
- Fashionpedia supplies apparel masks, parts, fine-grained attributes, and an expert-defined ontology for later segmentation and attribute experiments.
- Neither dataset supplies FITTED preference targets or directly determines fashion quality.

Current experiment status:

- Define the reduced product garment taxonomy and typed service response.
- Implement a lazy, optional Grounding DINO Tiny adapter in the Python service.
- Add category mapping, box normalisation, duplicate suppression, and one-piece conflict handling tests.
- Smoke-test the configured model on a single-person full-body outfit image.
- Reject Grounding DINO Tiny as the live MVP runtime after measuring approximately 10.6 seconds for one CPU inference.
- Select the Fashionpedia-trained RF-DETR-Seg Small checkpoint as the only replacement candidate for the hackathon gate.
- Pin and record the candidate checkpoint revision, checksum, declared Apache 2.0 weight licence, and required attribution.
- Implement the RF-DETR-Seg adapter behind the existing garment-perception response.
- Carry canonical-crop geometry through room pairing and project reduced-category
  detections back onto both mirrored video feeds as labelled bounding boxes.
- Run the live acceptance gate on 15–20 representative webcam crops from the intended demo hardware.
- Connect passing garment results to the live battle loop at approximately 1 FPS with one request in flight.
- Post-hackathon: map DeepFashion2 and Fashionpedia evaluation labels into the reduced product taxonomy and run full per-category evaluation.

The single-image Grounding DINO CPU smoke test detected the expected top, bottoms, and two shoes after conservative category-consistency post-processing, but inference took approximately 10.6 seconds. That proves the adapter but fails the live-product requirement. Grounding DINO remains a diagnostic baseline and must not be integrated as a frozen-only garment result.

### 12.1 Hackathon live gate

Live garment perception means clothing categories update while the battle is running. It does not need to classify every displayed video frame. Schedule the newest canonical crop at approximately 1 FPS, allow one inference operation in flight, discard busy ticks, and keep the latest valid garment result visible between updates.

Use [`resoa/garment-detector-seg`](https://huggingface.co/resoa/garment-detector-seg), an RF-DETR-Seg Small checkpoint trained on Fashionpedia, as the only replacement candidate. Its model card declares the weights Apache 2.0 and reports Fashionpedia validation metrics. The underlying open-source RF-DETR package and Apache-designated model weights are Apache 2.0; the exact candidate revision, checksum, licence files, and attribution must be recorded before integration.

The integrated candidate is pinned to `rfdetr==1.9.3`, repository revision
`f1b64c11fa42d2f7455708b7a05f81c015461427`, and checkpoint SHA-256
`aafefc440ea8f3f388e894a898e4270a2eeb6e38a3c3ffd3751d07d0f30b26bb`.
The checkpoint model card declares its weights Apache 2.0 and identifies
Fashionpedia as CC BY 4.0 training data. Retain attribution to RF-DETR and
Fashionpedia/Jia et al. when redistributing or demonstrating the model. Startup
rejects any hash, safe-load structure, CUDA, package version, class count, or
ordered class-name mismatch before the detector becomes ready.

The first RTX 3060 Laptop GPU smoke gate used CUDA PyTorch `2.12.1+cu132`, three
warm-up batches and twenty measured batches of two at threshold `0.50`. On two
synthetic 640-pixel frames it completed all 20 batches with P50 `83.10 ms`, P95
`100.18 ms`, and peak reserved VRAM `171,966,464` bytes. This verifies loading,
batch stability, latency headroom, and memory headroom only. It does not satisfy
the correctness gate: the required 15–20 consented webcam-like outfit crops and
category/box review remain outstanding.

The live transport is integrated behind readiness/configuration. The Node room
coordinator requests one local crop from each browser approximately once per
second, derives Player A/B from the socket role, invokes `/v1/garments/pair` once,
and broadcasts category results separately from scoring. Busy ticks are skipped,
stale request IDs are rejected, image buffers are discarded after request
construction, and live garment work is aborted and paused during finalisation.
The browser displays reduced category chips and labelled detector boxes on both
video feeds. Each sample carries its normalized source crop rectangle; the room
coordinator preserves authoritative Player A/B geometry, and the browser maps each
crop-relative RF-DETR box back into source-video space before applying the mirrored
video transform. The 19 ignored Fashionpedia garment-part classes remain excluded.

Time-box adapter implementation and the first hardware measurement to **90 minutes**. The candidate passes only if:

- it loads once in the existing Python inference service;
- its outputs map into the existing reduced product taxonomy;
- it sustains approximately one result per second on the intended demo hardware without stalling the displayed video;
- scheduling maintains zero queued frames and no more than one inference operation in flight; and
- a 15–20 crop gate set shows usable top, bottoms, dress, outerwear, and shoe behaviour without a demo-breaking false-positive pattern.

If any gate fails within the time-box, remove live garment categorisation from the hackathon MVP. Do not substitute a frozen-only garment result. Whole-outfit scoring and pose/frame-quality detection must continue without component detections.

Requirements for any garment detector:

- evaluate on actual webcam crops, not only source-dataset images;
- distinguish `not present` from `not detected`;
- handle dresses without requiring separate tops and bottoms;
- avoid making optional garments mandatory;
- never multiply fashion quality directly by raw detection confidence;
- demonstrate measurable improvement over the whole-outfit baseline before becoming an MVP dependency.

RF-DETR-Seg may produce masks, but the first live integration should consume only canonical categories and boxes. Add masks to the product path only if they are available without breaking the time-box or latency gate and materially improve a required overlay or visibility check.

## 13. Multi-view and spinning

Ordinary movement is supported. Rapid movement may temporarily pause capture while pose tracking continues.

For the first version:

- prefer stable, primarily frontal frames for scoring;
- do not score a random blurred phase of a spin;
- keep the last stable result visible during rapid turns;
- do not claim that pose landmarks alone distinguish front from back.

A future guided scan may collect stable front, side, and back keyframes and pool their embeddings. This requires a separate orientation and multi-view specification.

## 14. Privacy and data handling

- Continuous pose detection should run locally where practical.
- Do not persist raw webcam frames as part of detection.
- Keep only the small in-memory candidate buffer required for selection.
- Release or close discarded `ImageBitmap`, `VideoFrame`, canvas, and mask resources promptly.
- Transmit only frames selected for inference.
- Document any future storage of captured or labelled webcam images separately and obtain appropriate consent.

## 15. Performance targets

Initial engineering targets on the intended demo laptop:

| Measure | Target |
| --- | --- |
| Displayed webcam | Approximately 30 FPS without visible detection-induced stutter |
| Pose scheduling | Approximately 10 FPS |
| Detection queue depth | 0 pending frames; maximum 1 in flight |
| Framing-feedback latency | Under 150 ms where hardware permits |
| Maximum candidate age | 300 ms |
| Scoring capture rate | Approximately 1 FPS |

Measured performance must be recorded for both players' intended devices. If targets are missed, reduce detection frequency before reducing video quality.

## 16. Testing and validation

### Unit tests

- Landmark-group visibility classification.
- Person-box and padded-crop calculations.
- Bounds clamping and safe pixel conversion.
- Encoder aspect-ratio padding and resize-without-stretching.
- Temporal hysteresis state transitions.
- Motion calculation.
- Candidate expiry and deterministic selection.
- Expired and overflow candidate-resource cleanup.
- Worker error and lifecycle cleanup where testable.

### Fixture set

Create and classify at least 50–100 representative stills or short sequences. Track fixture coverage explicitly:

- Correctly framed outfits.
- Person too near and too far.
- Cropped head, torso, legs, and shoes.
- Multiple people.
- No person.
- Low light and backlighting.
- Motion blur.
- Dark clothing.
- Dresses, skirts, wide-leg trousers, and layered outerwear.
- Bags, hats, and accessories extending beyond the body.
- Front, side, back, and turning poses.
- Varied body shapes, skin tones, mobility, and gender presentation.

Training, tuning, and evaluation fixtures must be separated where thresholds are learned from the data.

### Integration tests

- Video remains interactive while the worker runs on both intended laptops.
- No frame queue develops during sustained motion.
- Detection results retain correct timestamps.
- A brief miss does not cause UI flicker.
- A stale result is never sent as a new scoring frame.
- Camera stop, restart, room leave, and component unmount release detection resources.
- Each client submits only its latest stable local candidate with player,
  sample, and capture-time identity.
- The server pairs fresh submissions once per room and discards stale,
  duplicate, and superseded work.

### Acceptance criteria before finalising this spec

- Pose Lite and Pose Full are benchmarked on the intended laptops.
- Initial thresholds are evaluated on the fixture set.
- Valid crops retain the complete outfit and accessories at an agreed rate.
- False acceptance of cropped or multi-person frames is measured.
- Motion behaviour is tested with walking, turning, and a controlled spin.
- The selected preprocessing variant is evaluated for score stability and person/background bias.
- Per-client local-frame submission is validated for fresh, synchronised pair
  construction on the intended two-laptop setup.
- Detection ownership, pairing, and frame transport are recorded as final decisions in the PRD.

## 17. Proposed file organisation

```text
apps/web/
  hooks/
    useOutfitDetection.ts
  lib/
    cv/
      candidates.ts
      config.ts
      crop.ts
      frame-quality.ts
      motion.ts
      temporal.ts
      types.ts
  workers/
    pose-detection.worker.ts

services/inference/
  src/fitted_inference/
    engine.py
    main.py
    schemas.py
```

Keep pure crop, quality, motion, candidate-selection, and state-transition logic outside React and the worker so it can be unit tested directly. Keep model loading, paired-image validation, and comparison response construction inside the installable inference-service package; do not move ML dependencies into `apps/web/`.

## 18. Implementation sequence

- Build a still-image MediaPipe spike and debug overlay.
- Create and classify the initial fixture set.
- Implement landmark visibility and framing checks.
- Implement deterministic padded canonical cropping.
- Compare Pose Lite and Pose Full on target hardware.
- Move video detection into a Web Worker.
- Implement latest-frame scheduling with zero queueing.
- Add temporal smoothing and hysteresis.
- Add brightness, blur, and motion metrics.
- Implement the recent candidate-frame buffer and selector.
- Integrate detection status into the battle UI.
- Pad and resize selected crops to the encoder input shape without stretching.
- Connect per-client local-candidate submission and server-side pairing to the
  `/v1/compare` service boundary.
- On finalisation, capture three timed stable slots, preserve any one-to-three
  complete synchronised pairs, and reject late live detection results.
- Connect selected crops to the visual-encoder baseline.
- Test camera restart, disconnect, and cleanup behaviour.
- Run face-blur and background-neutralisation ablations.
- Implement and smoke-test the server-side garment-perception baseline.
- Decide that frozen-only garment perception does not satisfy the live product.
- Select RF-DETR-Seg Small as the only time-boxed live replacement candidate.
- Implement and unit-test the RF-DETR-Seg adapter and paired inference boundary.
- Add an idempotent checkpoint bootstrap that verifies the pinned byte size and
  SHA-256 before enabling the local RF-DETR runtime.
- Render reduced-category RF-DETR boxes on both mirrored player feeds using the
  exact crop geometry associated with each paired inference sample.
- Run the representative-crop portion of the 90-minute acceptance gate.
- Integrate garment perception at approximately 1 FPS only if the acceptance gate passes.
- Post-hackathon: evaluate garment perception on mapped DeepFashion2, Fashionpedia, and larger webcam fixtures.

## 19. Open decisions

- Pose Lite versus Pose Full after benchmarking.
- Final landmark visibility and framing thresholds.
- Exact brightness, blur, and motion thresholds.
- Whether face blurring becomes the default preprocessing path.
- Whether background segmentation improves fairness without removing useful accessories or silhouette.
- Whether garment boxes or masks materially improve target-audience preference accuracy.

Decisions locked for the initial implementation:

- detection runs independently on each player's local, uncompressed camera feed;
- missing feet are allowed when head, torso, and knee evidence is usable;
- Chrome and Edge are the supported initial browsers;
- the separate Python service owns online model inference;
- live garment perception runs asynchronously at approximately 1 FPS and never queues frames;
- RF-DETR-Seg Small is the only hackathon replacement candidate for the rejected Grounding DINO live runtime;
- a candidate that misses the 90-minute correctness or latency gate is cut rather than used only on a frozen frame;
- each browser submits only its own selected local frame;
- live candidate pairs feed provisional scoring only; they never declare or lock
  the battle winner;
- freeze/finalisation selects fresh stable evidence for the authoritative final
  path and prevents late live work from replacing it;
- the existing Node room service owns authoritative pairing, backpressure, and
  locked-result broadcast; the Python inference service remains stateless.

## 20. Webcam-like human calibration frames

FITTED needs a small human-labelled calibration set whose visual domain resembles the canonical crops produced by this pipeline. This is pairwise calibration data, not enough data to pretrain or fully fine-tune a visual encoder.

Start with approximately **200–400 unique images** of people wearing complete outfits and collect **500–1,000 individual A/B decisions**. Prefer:

- full-body images with shoes visible;
- mostly front-facing, neutral poses;
- one person per image;
- similar crop and resolution;
- clear lighting without heavy filters;
- varied styles, colours, silhouettes, layering and formality;
- varied people, backgrounds, lighting and body proportions; and
- faces blurred or excluded.

Do not show likes, prices, brands, captions or popularity cues. Avoid product-only photographs, flat lays, close-ups, heavily obscured outfits and pairs where framing quality determines the answer.

If frames come from video, select the best one to three representatives from a stable interval. Adjacent frames share an `outfitId`, `personId` and `sessionId`; they are not independent outfits.

Construct three pair groups:

1. **Clear contrasts** validate the task and rater instructions.
2. **Close comparisons** provide high-value preference signal.
3. **Robustness comparisons** test similar outfits across different people, backgrounds, lighting or cameras.

Randomise left/right placement and include a small number of deliberately swapped duplicate pairs to measure position bias. Each image should appear in several pairings, and the comparison graph should remain connected.

Use this prompt:

> Which outfit is better styled as a complete look? Judge the clothing, coordination and fit—not the person, photo quality or brand.

Offer `A wins`, `B wins`, `Too close` and `Cannot judge`. Treat a draw as a soft `0.5` preference target. Exclude `Cannot judge` from preference training and retain it for frame-quality evaluation.

Optional reason tags may cover individual pieces, coordination, proportion, silhouette, layering and colour. Collect them on a subset of decisions to avoid rater fatigue.

```ts
type PairwiseLabel = {
  pairId: string;
  leftOutfitId: string;
  rightOutfitId: string;
  result: "left" | "right" | "draw" | "cannot_judge";
  reasons?: Array<
    | "components"
    | "coordination"
    | "proportion"
    | "silhouette"
    | "layering"
    | "colour"
  >;
  anonymousRaterId: string;
  responseTimeMs: number;
};
```

Split by person, outfit and capture session before constructing pairs. Use an initial 70/15/15 train/validation/test allocation and build pairs only within each split. Never place the same image, adjacent frames, outfit session or—where possible—person/source creator across train and evaluation boundaries. Important validation and test pairs should receive multiple independent ratings so human agreement and the realistic model ceiling can be measured.

## 21. Pre-labelled fashion curriculum

Existing datasets should provide fashion perception and compatibility pretraining so hackathon effort is spent on FITTED-specific calibration:

- [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) provides clothing categories, boxes, dense landmarks, masks, viewpoint and occlusion annotations. Use it for garment localisation and visibility.
- [Fashionpedia](https://fashionpedia.github.io/home/index.html) provides apparel masks, categories, parts and fine-grained attributes. Use it for fashion-specific segmentation and attribute representation.
- [Polyvore Outfits](https://github.com/xthan/polyvore-dataset) provides outfit compatibility examples. Use it as optional compatibility pretraining while measuring the product-image-to-webcam domain gap.
- [Fashionpedia-Taste](https://arxiv.org/abs/2305.02307) provides human preference explanations involving localised attributes, attention and captions. Use it as optional explanation/preference pretraining, not as a replacement for FITTED labels.

Recommended curriculum:

```text
generic frozen encoder: DINOv2 Small or SigLIP 2 Base
                         ↓
fashion perception: DeepFashion2 / Fashionpedia
                         ↓
generic compatibility: Polyvore, optional
                         ↓
source experts: Instagram and Depop residual targets   [out of scope: hackathon]
                         ↓
FITTED calibration: webcam-like human A/B labels
```

The hackathon should load existing checkpoints or train small heads over cached embeddings. It should not train a detector or foundation encoder from scratch. Verify dataset and platform licences before redistribution or commercial use.

## 22. Detection-to-scoring contract

Detection locates evidence and establishes judgeability; it never directly determines fashion quality.

- The detection boundary between provisional live estimates and
  authoritative final scoring is documented.
- Final-frame capture, identity, freshness, and late-result rejection are
  implemented and verified on both clients.

During the current hackathon round, detection gates readiness but does not
produce the demo-only live estimate; that value is seeded presentation state.
At finalisation, detection supplies the latest valid outfit crop geometry so each
browser can crop its current local video frame for three timed slots. The newest
stable buffered candidate is used only if current-frame capture fails. Late live
candidates cannot overwrite the locked result.

The live range, smoothing, final-result, retry, and continuity rules belong to
the dedicated [`scoring-spec.md`](scoring-spec.md). Detection reports frame
quality, visibility, freshness, and garment evidence; it does not clamp scores or
decide whether a live estimate is close enough to the final result.

```ts
type GarmentCategory =
  | "top"
  | "bottoms"
  | "dress"
  | "outerwear"
  | "shoes"
  | "accessory";

type GarmentDetection = {
  category: GarmentCategory;
  box: NormalizedRect;
  confidence: number;
  visibleFraction?: number;
};
```

Garment boxes support component crops, visibility and UI overlays. Missing optional garments are removed from the score denominator, not assigned a zero. Dresses and one-piece garments must not be penalised for lacking separate tops and bottoms.

Pose landmarks may support body-aware visual-fit features such as shoulder/waist alignment, visible sleeve and trouser length, proportion, layering and silhouette balance. The system must not judge body type, facial appearance, attractiveness or gender presentation. Oversized, fitted and unconventional silhouettes are valid style choices; the target is visible coherence and intentionality.

The scoring representation may combine:

```text
global complete-outfit embedding
+ pose-relative upper-body crop embedding
+ pose-relative lower-body crop embedding
+ footwear crop embedding when visible
+ pose/proportion features
```

The global embedding is mandatory because whole-outfit coordination is an interaction between pieces and cannot be recovered reliably by averaging isolated crop scores.

For each outfit, the later scoring service forms a compact feature vector:

```text
x = [
  instagram_score,                 [out of scope: hackathon]
  depop_score,                     [out of scope: hackathon]
  component_quality,
  outfit_coordination,
  body_fit,
  vlm_holistic_score
]
```

Human pair labels calibrate a regularised pairwise combiner:

```text
P(A wins) = sigmoid(weights · (x(A) - x(B)) / temperature)
```

This low-capacity combiner is appropriate for 500–1,000 decisions. Direct unrestricted training on high-dimensional visual embeddings is not.

## 23. VLM and temporal-video boundary

At battle completion, an image-capable VLM analyses any one-to-three complete synchronised pairs from three timed capture slots in one request. Its structured response contains component quality, whole-outfit coordination, body-aware fit, a holistic diagnostic score, frame quality and visible clothing observations. The holistic value is not added to the fallback's deterministic 45/30/25 score.

The VLM prompt must prohibit assessment of faces, attractiveness, body type, perceived gender, brand value and popularity. The VLM is one expert and the explanation layer; it does not overwrite the application-owned deterministic combiner.

Do not run a large VLM on the full webcam frame rate:

```text
30 FPS WebRTC display
        |
        +--> local pose/framing detection at approximately 10 FPS
        |
        +--> stable local candidates retained briefly in memory
                              |
                        freeze/final trigger
                              |
                server pairs newest fresh A/B samples
                              |
                  one VLM request per room in flight
```

The later learned scorer may consume paired samples at approximately `1 FPS`.
The VLM fallback is freeze/final by default; `2–3 second` VLM polling is optional
experimentation only, and busy or stale work is discarded rather than queued.

StreamingVLM is deferred. The published [StreamingVLM](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6445dd88ebb9a6a3afa0b126ad87fe41-Abstract-Conference.html) architecture is aimed at stable understanding of effectively infinite video and depends on streaming-specific supervised fine-tuning. Revisit it for continuous commentary, garment-movement analysis or long-session memory, not for the initial slowly changing outfit state.

## 24. Thirty-hour decision gates

- Establish a working paired-image VLM response and fallback first.
- Prepare webcam-like calibration images and safe person/outfit/session splits.
- Cache DINOv2 Small and/or SigLIP 2 Base embeddings rather than fine-tuning an encoder.
- **Out of scope (hackathon):** Train the Instagram residual head if its cleaned metadata is ready.
- **Out of scope (hackathon):** Add Depop only if its data is ready and it improves held-out FITTED agreement.
- Train the small human-calibrated combiner.
- Integrate frame-quality states, explanation, and result synchronisation.
- Rehearse the complete two-laptop flow and failure recovery.

Stop source-expert work if the data cannot be cleaned and split safely. Do not integrate a learned ranker that fails person/creator-disjoint validation. Preserve a paired VLM-only path as the reliable demo fallback.

## 25. Production runtime direction

Python is appropriate for the hackathon inference service. At the learned
scorer's intended approximately one paired scoring request per second, image
preparation and model/GPU latency matter more than Python orchestration overhead.
The VLM fallback ordinarily runs only at freeze/finalisation.

After the model is validated, production may export stable models to ONNX and benchmark TensorRT or another optimised runtime. C++ is useful for measured bottlenecks such as GPU-native decode/preprocessing, buffer reuse, tight latency, high concurrency or edge deployment. It is not necessary to rewrite the browser or orchestration layer in C++: browser WebRTC and codecs already execute in native browser code, while the application samples frames independently for inference.

## 26. Calibration and expert evaluation

Evaluation tasks:

- Measure pairwise agreement with held-out human labels.
- Measure inter-rater agreement.
- Measure person-, creator-, outfit-, and session-disjoint performance.
- Verify A/B swap invariance.
- Evaluate draw and `cannot_judge` behaviour.
- Measure stability across representative frames of one outfit.
- Measure sensitivity to background, lighting, pose, and camera quality.
- Measure the incremental value of every expert.
- Record median and 95th-percentile live latency.

Compare at least a VLM-only baseline and the human-calibrated visual ensemble. Remove an expert if it does not add held-out signal, regardless of its training-set performance. The Instagram-only baseline is out of scope for the hackathon along with its expert.

This document becomes **Final** only after the acceptance criteria are met and the open decisions required for the MVP are resolved.
