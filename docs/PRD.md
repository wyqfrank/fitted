# FITTED — Product Requirements Document

## 1. Overview

**Working name:** FITTED  
**Status:** Hackathon prototype  
**Document status:** Draft — unresolved decisions are marked **TBD**

FITTED is a live fashion battle where two people connect through their cameras and an ML system compares their outfits.

The prototype should answer one central question: can an ML model act as a fun, fast referee for a live outfit comparison?

---

## Status

The prototype now supports two-player rooms, WebRTC video, browser pose checks,
a server-controlled five-second round, live DINOv2 scoring, RF-DETR garment
detection, and final scoring with Gemini.

- See [architecture.md](architecture.md) for the current system.
- See [next-steps.md](next-steps.md) for unfinished work.
- See [ranker-experiments.md](ranker-experiments.md) and
  [ranker-audit.md](ranker-audit.md) for model results.

This document describes the product. It does not track progress.

---

## 2. Product Goal

Create a reliable two-player experience in which:

1. one player creates a battle;
2. a second player joins from another device;
3. both players can see each other's live camera feed;
4. the system analyses both outfits; and
5. the UI presents a comparison quickly enough to feel interactive.

The exact meaning and presentation of the comparison are still **TBD**.

---

## 3. Target Experience

```text
Player A creates a battle
          ↓
Player B joins with the room code
          ↓
Both players grant camera access
          ↓
Both live feeds connect
          ↓
The ML system analyses both outfits
          ↓
A comparison result is shown
```

### Experience principles

- Fast to start.
- Clearly communicate camera, connection, opponent, and analysis states.
- Keep the video experience smooth while analysis runs separately.
- Make the result easy to understand at a glance.
- Treat the comparison as entertainment, not an objective judgement of a person.

---

## 4. MVP Requirements

### Required

- Create a two-player battle room.
- Join a battle using a room code.
- Connect two separate laptops or devices.
- Request and display each player's webcam feed.
- Show both live feeds in the battle room.
- Capture frames for ML inference without interrupting video playback.
- Pass outfit imagery through an ML inference boundary.
- Receive an outfit comparison result.
- Display the result and relevant loading, error, and connection states.
- Support leaving a battle and handling a disconnected opponent.

### Nice to have

**TBD — prioritise after the core two-player flow and ML approach have been validated.**

Questions to resolve before adding items here:

- Which additions make the battle meaningfully more fun or repeatable?
- Which additions improve demo reliability or make the ML result clearer?
- What can be completed without risking the required flow?

---

## 5. User Interface

**The UI is finalised for the hackathon.** The street/arcade visual system
described below is the shipped design; no further redesign is planned. Changes
from here should be corrective (legibility, overlap) rather than directional. The
layout targets laptop screens; mobile layouts are out of scope (§ 9).

### Visual system

- Urban street meets arcade cabinet: near-black concrete grounds (`#0b0b0b`
  ink, `#131313` shell, `#1e1e1e` panels) under hazard stripes and a halftone
  dot grid.
- Two type voices: a hand-tagged graffiti display face (GraffitiXenoa, vendored
  at `apps/web/app/fonts/`) for headlines and player names, and VT323 for every
  piece of UI text. VT323 is a pixel face and is not set below 16px.
- Player accents are fixed and semantic: **P1 electric magenta** (`#ff2ec4`),
  **P2 acid green** (`#b6ff1f`), with **hot orange** (`#ff6a00`) reserved for
  system/primary actions. A player's accent drives their panel bevel, number
  block, corner brackets, score plate and meter from a single token.
- Chunky bevelled controls with hard (unblurred) drop shadows and a real
  pressed state; a consistent `-9deg` skew on panels and controls.
- Camera feeds occupy roughly 60–65% of viewport height and stretch to fill
  available space. They carry no filter or scanline overlay.

### Landing screen

- FITTED branding and short explanation of the battle.
- **Create Battle** action.
- **Join Battle** action.
- Room-code entry and validation.

### Battle room

- Symmetrical local and opponent video panels, separated by a hairline seam with
  the VS badge straddling it.
- Player labels (`P1`/`P2` number blocks), room code, and copy affordance.
- Camera and peer-connection status, including ICE transport diagnostics
  (`STUN OK`, `TURN OK`, `NO ROUTE TO OPPONENT`, `NO VIDEO ROUTE`).
- Waiting, connecting, analysing, disconnected, and error states — drawn from a
  single shared vocabulary, never ad-hoc strings.
- Per-feed score HUD: an arcade `1UP`/`2UP` label and value set directly on the
  video over a gradient scrim, with a full-bleed segmented meter flush to the
  bottom edge at a 26px segment pitch. It is an overlay, not a card.
- Garment category chips and labelled bounding boxes derived from live perception.
- Outfit-detection overlay showing the canonical crop and pose landmarks.
- Controls for camera, copying the room code, **finalising** the score, retrying
  a failed final score, and leaving.

Canonical control and state labels currently shipped: `START CAMERA` /
`STOP CAMERA`, `FINALISE EARLY`, `RETRY SCORE`, `COPY CODE` / `COPIED`, `LEAVE`,
`WAITING FOR READY`, `STANDBY`, `LIVE`, `CAPTURING FITS`, `CAPTURING BOTH FITS`,
`ANALYSING FINAL RESULT`, `SCORE FINAL`, `TOO CLOSE TO CALL`,
`WAITING FOR OPPONENT`, `OPPONENT CONNECTED`, `OPPONENT CAMERA OFF`,
`LOCAL ONLY`, `SIGNAL LOST`, `CONNECTION LOST`, `NO BATTLE FOUND`.

### Result experience

The hackathon product uses **both** a lightweight live estimate and a more
accurate final result. They must be visually and semantically distinct.

During the hackathon demo, each player sees a one-decimal point value persistently
labelled **Live estimate**. It is a seeded, deterministic presentation value in
the `55..85` range, shared across both clients and updated every 500 ms.
It is not model output, does not declare a winner, and is replaced rather than
blended when the authoritative final result arrives.

The later learned fast path should replace that demo value with a smoothed score
range such as `72–82`, update at approximately one comparison per second, hold
the last valid range when work is skipped, and never block the video experience.

The initial range is the smoothed live estimate plus or minus five points. Tune
that width against the final scorer so at least 80% of representative per-player
final scores land inside the last live range. The full range may be widened from
ten to at most sixteen points. If useful coverage still requires a wider range,
remove the live numbers and retain only a qualitative current-leader treatment.
Do not imply that the range is a calibrated confidence interval.

When both players are connected and locally score-ready, the server starts one
5-second round. At zero, or when either player finalises early, both clients
enter the same final capture path and then **Analysing final result**. The server
captures five current paired crops approximately 750 ms apart, scores the
available one-to-five complete pairs in one request, broadcasts exact
one-decimal scores and the final winner or draw to both clients, and locks the
result. Late live responses cannot change it.

The accurate final score is not clamped to the previous live range. An
out-of-range result is revealed truthfully, described as an adjusted estimate,
and recorded for calibration. If final analysis fails or the images cannot be
judged, the product keeps the prior value labelled as an estimate, offers a
retry, and does not turn it into a final winner.

The detailed smoothing defaults, response states, continuity targets, and
implementation checklist live in
[`docs/specs/scoring-spec.md`](specs/scoring-spec.md).

**Still TBD:**

- the measured final draw threshold;
- how much final reasoning or calibrated confidence to show; and
- the exact retry/recapture interaction for poor framing or an outfit that is not sufficiently visible.

---

## 6. ML System

### Goal

Given images of two outfits, predict which outfit is more likely to be preferred by the defined FITTED target audience.

The model evaluates the visible outfit, not the attractiveness or value of the person wearing it. The result is an entertainment-oriented preference signal rather than an objective measure of fashion quality.

### Current design direction

Use broad social and commercial behaviour as weak supervision, then use direct A/B judgements from the target audience to calibrate the final FITTED scoring function.

> **Out of scope for the hackathon.** The Instagram residual, Depop residual,
> and visual-style-momentum source experts are descoped. The design below is
> retained for future reference. The shipped scoring path is visual-only
> (component quality, outfit coordination, body-aware fit) plus the VLM.

```text
Instagram residual expert                       [out of scope: hackathon]
          +
Depop residual expert                           [out of scope: hackathon]
          +
component, coordination and body-fit signals
          +
VLM holistic assessment and explanation
          +
visual style momentum                           [out of scope: hackathon]
          ↓
expert ensemble
          ↓
500–1,000 target-audience pair comparisons
          ↓
pairwise FITTED scoring function
```

This is a teacher–student design:

- social and commercial data provide large-scale but noisy offline supervision;
- target-audience comparisons determine how the weak signals and VLM assessment should be combined;
- the resulting visual scoring model runs without contacting social platforms during a live battle.
- the VLM supplies a grounded breakdown and final explanation, but does not own the score by itself.

### Shared visual representation

Each training image is passed through the same frozen visual encoder. Source-specific expert heads learn different signals from that shared embedding:

> **Out of scope for the hackathon.** The Instagram residual, Depop residual,
> and visual-style-momentum source experts are descoped. The design below is
> retained for future reference. The shipped scoring path is visual-only
> (component quality, outfit coordination, body-aware fit) plus the VLM.

```text
outfit image
     ↓
frozen visual encoder
     ↓
outfit embedding
     ├── Instagram expert → relative social engagement    [out of scope: hackathon]
     ├── Depop expert → relative commercial desirability  [out of scope: hackathon]
     └── momentum expert → visual-style momentum         [out of scope: hackathon]
```

With all three source experts descoped, the frozen encoder currently serves the
whole-outfit coordination signal only.

The first frozen-encoder experiment should benchmark **DINOv2 Small** and **SigLIP 2 Base**. DINOv2 is attractive for a lightweight learned visual ranker; SigLIP 2 also supports a prompt-based zero-shot baseline. The hackathon must not depend on full encoder fine-tuning.

**FashionCLIP** should be benchmarked as a second candidate, not assumed to be better. It is fashion-specific, but its published training domain is primarily isolated product imagery rather than webcam images of people wearing complete outfits.

The encoder remains frozen for the first implementation. Full fine-tuning should only be considered if the source experts and scoring head cannot learn a useful signal from frozen embeddings.

#### Measured baseline — DINOv2 Small, 2026-08-22

`services/inference/scripts/train_ranker.py` embeds the labelling pool with a
frozen DINOv2 Small, reduces to 16 PCA dimensions fitted on training images
only, and fits a linear pairwise ranker with no intercept, so swap consistency
holds by construction rather than by measurement.

Trained on 593 usable decisions from **one rater** across 182 images
(418 train / 95 val / 80 test pairs):

| Split | Agreement | n decided |
|---|---|---|
| Validation (model selection) | 0.791 | 86 |
| **Held-out test** | **0.622**, 95% CI [0.508, 0.724] | 74 |

**The result is positive but weak.** The interval's lower bound sits barely
above chance, so the honest claim is "better than a coin flip", not a usable
margin. The validation-to-test drop reflects selection optimism: the sweep chose
a configuration against 86 validation pairs.

Two constraints, not two bugs. A single rater means no inter-rater ceiling
exists to compare against, and 74 decided test pairs cannot resolve differences
smaller than roughly ten points. Both are addressed by the additional raters
rather than by a larger model — at 182 images, 16 dimensions was already the
plateau, and 64 dimensions overfit.

Do not re-read the test split to choose between encoders. The DINOv2 versus
SigLIP 2 comparison should run on validation once the additional raters have
widened both evaluation splits.

#### VLM distillation — replacing the descoped weak supervision

The descoped Instagram and Depop experts leave the frozen encoder with no
large-scale supervision, and 593 human decisions cannot fill that gap. The
replacement teacher is the **VLM assessor already used at finalisation**:
`distil_teacher.py` runs the production `GeminiVlmProvider` and the live rubric
prompt over pairs from an unlabelled image pool, and `train_ranker.py --teacher`
pretrains the pairwise head on those judgements before fine-tuning it on the
human pairs.

This targets the correct objective. The student is trained to predict the
verdict that actually settles a battle, so the live estimate becomes a fast
approximation of the final result rather than an unrelated second opinion. The
VLM is far too slow for a 1 FPS live path; a linear head over one 71 ms DINOv2
forward is not.

Fine-tuning uses a proximal penalty toward the teacher's weights rather than
toward zero, so a few hundred human pairs calibrate the teacher instead of
overwriting it.

The reported diagnostic is the student's accuracy on human validation pairs
*before* any human label is applied. That isolates what distillation contributed
from what the human pairs contributed.

##### Image source and its licence

The teacher pool is sampled from **Fashion144k** (Simo-Serra et al., CVPR 2015),
streamed from the distributed archive and re-encoded to WebP. Its
`relvotes.mat` fashionability scores are **deliberately discarded**: they carry
the photo-quality, fan-count and posting-era confounds that the descoped
residual experts existed to remove. Only the pixels are used.

**Licence constraint — the dataset is non-commercial.** The archive's terms
restrict use to "non-commercial research and educational purposes". A hackathon
prototype qualifies. A commercial FITTED would not, and neither would a shipped
model distilled from these images. If the product is ever commercialised, the
teacher pool must be rebuilt from a differently licensed image source and the
student retrained. Cite the CVPR 2015 paper wherever the dataset is credited.

Known domain gap: Fashion144k photos are 378×256 street-style shots, well lit
and full-body. Live input is a 640 px webcam crop under venue lighting. Matching
the codec closes part of that gap; resolution, lighting and pose remain
unmatched. A near-chance zero-shot number should be read as a domain-gap
symptom before it is read as a teacher-quality problem.

#### Measured results — 1,972 teacher pairs, three raters, 2026-08-22

Human labels: 1,252 usable decisions from three raters (AC 486, DP 593, FW 173)
over 182 images. Teacher labels: 1,972 Gemini pairs (19 unusable, 9 failed).
All configurations scored against one frozen label snapshot; comparing across a
live directory is invalid, because raters keep labelling and the evaluation
splits grow between runs.

| Configuration | Val (pooled) | Val (AC+DP) |
|---|---|---|
| Human labels only | 0.638 | 0.715 |
| Teacher zero-shot, **no human labels** | 0.621 | 0.709 |
| Teacher + proximal fine-tune | 0.660 | 0.734 |
| **Human inter-rater ceiling** | **0.686** | **0.778** |

Held-out test, frozen configuration (16 dims, train-pool basis):

| Label set | Test | 95% CI | n |
|---|---|---|---|
| Pooled, three raters | 0.553 | [0.485, 0.619] | 208 |
| AC+DP subgroup | 0.643 | [0.561, 0.717] | 140 |

Four conclusions, in order of importance:

1. **Rater agreement, not model capacity, is the binding constraint.** The model
   lands within roughly 4 points of the human ceiling on both label sets. There
   is very little headroom left for a better model to capture.
2. **The raters do not share a preference.** AC and DP agree 0.778; FW agrees
   with them 0.491 and 0.563 — the first is exactly chance. This is not
   carelessness: FW's median decision took 4.9 s, the slowest of the three.
   Their taste is simply uncorrelated with the other two.
3. **On a mixed audience the ranker does not beat chance.** The pooled test
   interval [0.485, 0.619] contains 0.5. On the coherent subgroup it clearly
   does: [0.561, 0.717]. FITTED's premise requires an audience that shares a
   notion of a better outfit, and three people were not enough to form one.
4. **Distillation replaces human labels almost entirely.** Zero-shot, with no
   human label at all, the student scores 0.709 against AC+DP's 0.715 from 761
   human training pairs. Fine-tuning adds about 2 points — within noise on these
   sample sizes, so treat the gain as directional. The value delivered is the
   near-elimination of the labelling requirement, not the 2 points.

Fitting the PCA basis on Fashion144k images rather than the label pool was worse
in every configuration (0.613 versus 0.660 pooled). The teacher's *judgements*
transfer across the domain gap; its *image statistics* do not. Use
`--projection train`.

Earlier single-rater figures (val 0.791, test 0.622) are superseded. That
validation set held 86 decided pairs from one person and was both small and
self-consistent; the three-rater numbers are lower and more trustworthy.

#### Cohort selection, and the caveat it carries

FITTED is an entertainment-oriented preference signal, not an objective measure,
so predicting *one* audience well is the correct product goal. The shipped
configuration is therefore:

```bash
python services/inference/scripts/train_ranker.py \
  --dims 16 --raters AC,DP --projection train \
  --teacher data/labelling/teacher.jsonl --report-test
```

Held out on that cohort, against a 0.778 cohort ceiling:

| Model | Test | 95% CI |
|---|---|---|
| Human labels only (761 pairs) | 0.636 | [0.553, 0.711] |
| Distilled + fine-tuned | 0.643 | [0.561, 0.717] |

**Distillation does not buy accuracy.** Seven tenths of a point on 140 pairs is
nothing. What it buys is independence from the labelling effort: the zero-shot
student, having seen no human label at all, scores 0.709 on validation against
the human-trained model's 0.715. Retargeting FITTED at a different cohort
therefore does not require another 600-decision labelling session — the teacher
supplies almost the entire signal, and human pairs are needed only to confirm
the cohort's taste is being tracked.

That is the result worth reporting: not a better ranker, a ranker that no longer
depends on collecting human preference data first.

**The caveat, stated plainly:** the cohort was chosen *because* its members
agreed, after seeing the agreement matrix. With three raters, "the two who
agree" is a post-hoc selection, and some of that agreement could be chance. The
0.643 is a genuine held-out number — no test label influenced training — but the
decision about whose taste counts was informed by the data.

The honest version of this claim is "FITTED predicts the AC+DP cohort's
preference", not "FITTED predicts good taste". To make the stronger claim, the
cohort must be defined by a recruiting criterion stated in advance (demographic,
subculture, self-reported style) and new judges recruited against it, rather
than by filtering raters on observed agreement.

FW's 173 decisions are kept in the repository. They are not noise — FW's median
decision took 4.9 s, the slowest of the three raters — and they are the evidence
that a second, uncorrelated cohort exists.

### Outfit composition and body-aware fit

The FIT score must evaluate both individual garments and the outfit as a complete composition. A collection of individually strong pieces should not automatically score well if their colours, proportions, silhouettes or styling do not work together.

The visual analysis should produce three distinct score groups:

1. **Component quality** — the visible styling quality of detected tops, bottoms, outerwear, shoes and accessories.
2. **Whole-outfit coordination** — how the pieces play off each other through colour harmony, silhouette, layering, proportion, material and style coherence.
3. **Body-aware fit and proportion** — how the garments sit and align on the wearer, using pose and silhouette information without judging the wearer's body type.

```text
latest outfit frame
        |
        +--> person and frame-quality check
        +--> garment detection / segmentation --> component scores
        +--> pose and silhouette estimation ----> body-fit score
        +--> full-outfit visual embedding ------> coordination score
                                                    |
                                                    v
                                           weighted FIT score
```

The body-aware branch may assess visible garment alignment, sleeve and trouser length, layering, silhouette balance and overall proportions. It must not score body shape, facial appearance, attractiveness, gender presentation or other personal characteristics. Face information is excluded from the competitive score.

Bounding-box detection is sufficient for locating initial garment components. Segmentation should be evaluated later for measurements that depend on garment boundaries, layering and silhouette.

#### Live garment perception decision

Garment-category recognition is useful to the hackathon only when it updates during the live battle. A result produced only after freezing the battle is out of scope for this branch.

The implemented Grounding DINO Tiny baseline took approximately 10.6 seconds for one image on the CPU-only development environment and is therefore rejected as the live runtime. Keep it as a diagnostic baseline only.

Time-box one replacement attempt to **90 minutes** using the Fashionpedia-trained [`resoa/garment-detector-seg`](https://huggingface.co/resoa/garment-detector-seg) RF-DETR-Seg Small checkpoint. The checkpoint model card declares its weights Apache 2.0. Before integration, pin its revision and checksum and record the model, architecture, and Fashionpedia attribution and licence notices.

The replacement passes only if it maps into the reduced FITTED taxonomy and sustains approximately one result per second on the intended demo hardware over a 15–20 crop gate set without stalling video. The client or coordinator permits one garment request in flight, discards busy ticks, and retains the latest valid boxes between updates.

If the candidate misses the latency, correctness, licensing-provenance, or 90-minute delivery gate, remove live garment categorisation from the hackathon MVP. Do not replace it with a frozen-only garment result. Whole-outfit scoring and pose/frame-quality detection continue without component detections.

#### Initial deterministic weighting

For the hackathon prototype, use deterministic weights that are easy to explain and tune:

```text
component quality        45%
whole-outfit coordination 30%
body-aware fit            25%
```

The initial component-quality weights are:

```text
top         30%
bottoms     25%
outerwear   20%
shoes       15%
accessories 10%
```

Only confidently detected and sufficiently visible components participate in the denominator. Missing optional garments, such as outerwear or accessories, must not be treated as zero-quality garments. Category-aware rules must also avoid penalising outfits such as dresses that do not contain separate tops and bottoms.

```text
component_score =
  sum(style_score_i * weight_i * detection_confidence_i * visibility_i)
  / sum(weight_i * detection_confidence_i * visibility_i)

visual_fit_score =
    0.45 * component_score
  + 0.30 * coordination_score
  + 0.25 * body_fit_score
```

These are product defaults, not claims about objective fashion importance. Once sufficient target-audience comparisons exist, the weights should be fitted against held-out human preferences with non-negative constraints and regularisation. Training may use stochastic techniques, but live inference should use deterministic learned weights so the same input does not receive a randomly different result.

The whole-outfit coordination score must be learned or evaluated from the complete outfit image rather than calculated as an average of component scores. This preserves interaction effects: two pieces can score differently together than either would in isolation.

### Source-signal construction

> **Out of scope for the hackathon.** The Instagram residual, Depop residual,
> and visual-style-momentum source experts are descoped. The design below is
> retained for future reference. The shipped scoring path is visual-only
> (component quality, outfit coordination, body-aware fit) plus the VLM.

Raw popularity metrics must not be treated as taste labels. Each source first needs its own residual signal that removes as much non-outfit influence as the available data permits.

#### Instagram residual expert (out of scope for the hackathon)

The Instagram expert learns to predict whether an outfit post overperforms relative to its expected engagement.

```text
Instagram residual =
actual engagement
− expected engagement given
  creator, audience size, normal engagement,
  post age, format, posting time and other available context
```

Where reach is unavailable, use within-creator and within-time-period ranks rather than comparing raw likes between creators.

#### Depop residual expert (out of scope for the hackathon)

The Depop expert learns whether an item or styled listing overperforms relative to comparable listings.

```text
Depop residual =
actual demand
− expected demand given
  brand, category, price, condition,
  seller, listing age and available promotion context
```

Depop is a secondary signal because listing performance measures product demand more directly than complete-outfit preference. Its contribution should be learned from target-audience comparisons rather than fixed in advance.

#### Visual style momentum (out of scope for the hackathon)

Style momentum is a proposed later signal, not required for the first model.
It is also structurally dependent on the two descoped residual experts: it embeds
their source images, tracks *their* residual popularity over time, and needs at
least two platforms for a cross-platform trend. With those experts descoped it has
no inputs, so it is descoped with them.

The retained design was:

1. embed images from available sources;
2. group visually similar outfits into style neighbourhoods;
3. measure changes in residual popularity for each neighbourhood over time;
4. use cross-platform acceleration as a weak trend signal.

If the source experts are ever rescoped, the first experiment should start with the Instagram and Depop experts. Add momentum only after the source residuals and time alignment can be measured reliably.

### Target-audience pair comparisons

The human dataset should initially contain approximately **500–1,000 individual A/B decisions** from people representing the intended FITTED audience.

```text
Which outfit fits best?

[ Outfit A ]    [ Outfit B ]

A wins | B wins | Too close | Cannot judge
```

Label interpretation:

- **A wins** → `1.0`;
- **B wins** → `0.0`;
- **Too close** → draw or soft target of `0.5`;
- **Cannot judge** → exclude from preference training and retain as a frame-quality example.

#### Labelling workflow

Collect the overall preference before asking the judge to explain it. Showing detailed scoring categories first may anchor the judge to the prototype's weighting scheme instead of capturing their genuine overall preference.

Every comparison uses this primary question:

```text
Which outfit is better styled as a complete look?

A wins | B wins | Too close | Cannot judge
```

The overall A/B decision is the primary training target for the final pairwise weighting function. `Too close` and `Cannot judge` remain valid answers and judges must not be forced to select a winner.

After an **A wins** or **B wins** decision, comparisons selected for additional annotation ask the optional follow-up:

```text
What most influenced your choice? Select up to two.

Individual pieces | Outfit coordination | Fit and proportion
Colour | Layering | Other
```

Reason tags support explanations, dataset analysis and debugging. They must not be treated as sufficient evidence for learning numerical weights because they identify a reported reason, not its magnitude or the contribution of competing factors.

Collect direct dimension judgements on approximately **20–30% of comparisons**, sampled across clear, close and robustness pairs:

| Dimension                  | Allowed answers              |
| -------------------------- | ---------------------------- |
| Component quality          | A / B / Equal / Cannot judge |
| Whole-outfit coordination  | A / B / Equal / Cannot judge |
| Garment fit and proportion | A / B / Equal / Cannot judge |

These dimension judgements align with the three visual score groups and are used to train or validate the individual scoring branches. The overall A/B decision remains the supervision used to fit the final combination of branch outputs.

Do not require judges to rate every top, bottom, shoe and accessory in the main workflow. Detailed garment-level annotation is expensive and likely to be inconsistent, so it should be limited to a small diagnostic subset if component-level errors need investigation.

Important evaluation pairs should receive multiple independent judgements. Rater cohort and fashion-engagement information should be retained so the team can measure whether different parts of the target audience disagree.

Pair selection should progress from random coverage to active learning: prioritise comparisons where the experts disagree, the current model is near 50/50, or additional labels would improve coverage.

#### Webcam-like labelled image pool

The calibration images must resemble the frames that FITTED will score in production. Start with approximately **200–400 unique images** of people wearing complete outfits and use them to collect the planned **500–1,000 individual A/B decisions**. Each image should appear in several different pairings.

Prefer images with:

- the full body visible, including shoes;
- a mostly front-facing, neutral pose;
- one person per image;
- consistent crop and resolution;
- clear lighting without heavy filters;
- varied styles, colours, silhouettes, layering and formality;
- varied people, backgrounds, lighting and body proportions; and
- faces blurred or excluded so the task remains outfit-focused.

Do not show likes, prices, brands, captions or popularity cues to raters. Avoid product-only images, flat lays, garment close-ups, heavily obscured outfits and comparisons where framing quality makes the answer obvious.

Construct three pair groups:

1. **Clear contrasts** validate that raters understand the task.
2. **Close comparisons** provide the most useful fashion-preference signal.
3. **Robustness comparisons** test similar outfits across different people, backgrounds and lighting.

Use the prompt: **“Which outfit is better styled as a complete look? Judge the clothing, coordination and fit—not the person, photo quality or brand.”**

Randomise left/right placement and build train, validation and test splits by person, outfit and capture session before constructing pairs. An image or adjacent frame from the same outfit must never cross split boundaries. Important validation and test pairs should receive multiple independent ratings so inter-rater agreement and the realistic model ceiling can be measured.

Reason tags and dimension judgements should be collected only on their selected subsets to keep the primary task fast. `Cannot judge` labels train frame-quality rejection; they do not participate in preference training.

This dataset calibrates a low-capacity pairwise combiner over frozen expert outputs. It is not large enough to train or fully fine-tune a visual encoder from scratch.

### VLM role

For the hackathon, an image-capable VLM analyses one-to-five synchronised frame pairs captured as a final burst. It returns structured component-quality, whole-outfit coordination, body-aware fit, holistic, frame-quality and explanation fields. The fallback's public score uses only the deterministic 45/30/25 dimensions; the holistic value is retained for diagnostics and as a candidate feature for the later learned combiner so it is not double-counted. The application may use the VLM explanation in the final result experience.

The VLM's single `componentQuality` value is a fallback approximation. The final component branch remains the garment-aware calculation above, using per-component style, category importance, detection confidence, and visibility.

Use **Gemini 3.6 Flash** for the hackathon's paired-image VLM fallback. Keep the provider boundary replaceable, but defer broad model comparison until after the hackathon unless Gemini blocks delivery.

The VLM does not continuously process the 30 FPS webcam stream. Live video remains independent; finalisation requests five time-spaced local crops from each browser and sends the available one-to-five complete pairs to Gemini in one labelled request. Only one request per room may be in flight and stale work is discarded. The later learned scorer, not the VLM fallback, is responsible for approximately 1 FPS live scoring. Periodic VLM polling every 2–3 seconds is optional experimentation only.

Streaming-video VLMs are a post-hackathon investigation for continuous commentary, garment movement or long-session memory. They are not required for outfit scoring, where the visual state changes slowly relative to the video frame rate.

### Final FITTED scoring function

The executable runtime contract and its remaining implementation checklist live in
[`docs/specs/scoring-spec.md`](specs/scoring-spec.md).

For a human-labelled pair, the calibration model compares the source-expert predictions.
The three descoped source experts are retained below for reference and marked; the
hackathon combiner uses the remaining four features only:

```text
features(A, B) = [
  instagram(A) - instagram(B),                                   [out of scope]
  depop(A) - depop(B),                                           [out of scope]
  momentum(A) - momentum(B),                                     [out of scope]
  component_quality(A) - component_quality(B),
  outfit_coordination(A) - outfit_coordination(B),
  body_fit(A) - body_fit(B),
  vlm_holistic(A) - vlm_holistic(B)
]
```

The initial combiner can be pairwise logistic regression:

```text
FITTED(A) =
    weight_instagram × instagram(A)      [out of scope: hackathon]
  + weight_depop     × depop(A)          [out of scope: hackathon]
  + weight_momentum  × momentum(A)       [out of scope: hackathon]
  + weight_component × component_quality(A)
  + weight_outfit    × outfit_coordination(A)
  + weight_body_fit  × body_fit(A)
  + weight_vlm       × vlm_holistic(A)

P(A wins) = sigmoid((FITTED(A) - FITTED(B)) / temperature)
```

The terms above are transformed model features, not an assumption that every raw
expert naturally produces a comparable `0..100` value. The trained scoring
artifact must version each expert's fitted centring, scaling, and clipping (or an
equivalent transform), and live inference must apply those exact transforms.
Calibration into a displayed `0..100` FITTED score is a separate output mapping.

The target-audience comparisons learn the expert weights and calibrate the final decision boundary. They are not expected to teach the vision encoder fashion knowledge from scratch.

Each source signal must be evaluated out of sample. Signals that do not improve agreement with held-out target-audience comparisons should receive no weight or be removed.

### Offline and live boundaries

Social-platform information is used only for offline dataset construction and training.

The current hackathon UI uses a clearly labelled seeded demo estimate during the
countdown. The learned live path below remains the intended replacement.

During a live battle:

```text
latest frame from Player A ─┐
                           ├──► quality check and preprocessing
latest frame from Player B ─┘               ↓
                                      visual encoder
                                             ↓
                                  trained experts and weights
                                             ↓
                         provisional score ranges + current edge
                                             ↓
                                smoothed, labelled live estimate
```

The live inference path must not require Instagram, Depop or any other training-data source to be available.

The live video frame rate must remain independent from the inference rate. The system should sample the newest pair of frames, allow only one comparison request in flight, discard stale work, and display the latest valid result.

Start at approximately **one comparison per second**. Increase the rate only if measured inference latency and hardware headroom allow it. The product does not need to classify every camera frame.

Live inference may use a faster and less accurate subset of the final signals. It
produces the provisional ranges defined in the result experience, not the final
winner. Freeze/finalisation invokes the most accurate configured path and is the
only operation that can publish and lock the authoritative exact result.

### Proposed inference output

The next response revision must include an explicit phase. A live response
contains per-player score ranges and a provisional leader; a final response
contains exact scores, a winner or draw, and a finalisation ID. The example below
shows the final variant; the complete union is defined in the scoring spec.

```json
{
  "phase": "final",
  "finalisationId": "string",
  "modelVersion": "string",
  "playerAScore": 72.4,
  "playerBScore": 61.8,
  "winner": "player_a",
  "winProbability": null,
  "breakdown": {
    "playerA": {
      "componentQuality": 70.8,
      "outfitCoordination": 78.1,
      "bodyFit": 69.5,
      "vlmHolistic": 74.0
    },
    "playerB": {
      "componentQuality": 64.2,
      "outfitCoordination": 58.9,
      "bodyFit": 62.7,
      "vlmHolistic": 63.5
    }
  },
  "frameQuality": {
    "playerA": "ok",
    "playerB": "ok"
  },
  "latencyMs": 180
}
```

`winProbability` must only be presented as confidence if it has been calibrated. Frame quality is a separate signal and must not be presented as model confidence.

### Evaluation plan

The first evaluation should use representative held-out image pairs and actual demo hardware.

Measure:

- pairwise agreement with human labels;
- performance on image-disjoint and person-disjoint test data;
- consistency when Player A and Player B are swapped;
- stability across several frames of the same outfit;
- sensitivity to background, lighting, pose, and camera quality;
- median and 95th-percentile inference latency;
- behaviour when one or both outfits cannot be judged.

Exact acceptance thresholds remain **TBD** until a baseline has been measured.

### ML decisions still required

- How exactly is the target audience defined and recruited?
- *(Deferred with the source experts)* What Instagram and Depop data can be obtained lawfully and reliably?
- *(Deferred with the source experts)* Which engagement and demand metrics are available for residual construction?
- *(Deferred with the source experts)* Which confounding variables can be measured for each source?
- What image pool and held-out evaluation set will be used?
- Does DINOv2 Small or SigLIP 2 Base provide the stronger frozen representation on person-disjoint FITTED comparisons?
- *(Deferred with the source experts)* Do Instagram and Depop experts add independent out-of-sample signal?
- *(Deferred with the source experts)* Does visual style momentum add useful signal after the first two experts?
- Is full encoder fine-tuning eventually required?
- What final score mapping, live-band width, and draw threshold meet the measured calibration and continuity targets?
- How frequently should inference run?
- How will the system exclude face and body-type preference while still measuring garment fit and styling proportions?
- How will poor framing or incomplete outfit visibility be detected?
- How will model quality, bias, latency, and reliability be evaluated?

---

## 7. System Design

See [architecture.md](architecture.md) for the current architecture, round flow,
project layout, and key decisions.

Deployment has not been decided. The remaining timing and monitoring work is
listed in [next-steps.md](next-steps.md).

---

## 8. Success Criteria

The hackathon demo is successful when:

- two laptops can create and join the same battle;
- both camera feeds appear and remain usable;
- outfit frames can be analysed;
- the system returns a comparison;
- both users can understand the result;
- the interaction feels responsive; and
- the full flow can be demonstrated reliably.

**TBD:** specific targets for latency, model quality, result stability, connection success rate, and supported browsers/devices.

---

## 9. Out of Scope for the Initial Prototype

- Persistent profiles or battle history.
- Social feeds.
- Matchmaking.
- Global leaderboards.
- Ecommerce or outfit purchasing.
- Virtual try-on.
- Production-scale infrastructure.
- Mobile or narrow-screen layouts. The product is built for the two-laptop demo.

This list may be revisited after the MVP and nice-to-have scope are agreed.

---

## 10. Open Product Questions

- Does a player freeze the battle, does a timer end it, or are both supported?
- What final draw threshold is supported by target-audience labels?
- What makes a battle fun enough to repeat?
- What explanation should accompany a result?
- What language should the product use to avoid presenting subjective taste as objective fact?
- What happens when the model cannot make a reliable comparison?

---

## 11. Decisions Log

See [architecture.md](architecture.md) for the main decisions and rejected
options.
