# Scoring model specification

The server-controlled five-second round, final VLM request, and learned live
score are implemented and tested. The live score is still a single number, not
a calibrated range. See [next-steps.md](../next-steps.md) for unfinished work.

This document defines the scoring API and runtime behaviour. The draw threshold,
display-score calibration, and choice of frozen visual encoder are still open.

## Runtime contract

The implemented schema-version-1 runtime is deliberately limited to the
visual-only baseline. Its three rubric dimensions are finite `0..100` values for
one canonical outfit crop. The scoring artifact fixes their order, non-negative
weights, and pairwise temperature:

```json
{
  "schemaVersion": 1,
  "modelVersion": "visual-baseline-v1",
  "featureNames": [
    "component_quality",
    "outfit_coordination",
    "body_fit"
  ],
  "weights": [0.45, 0.30, 0.25],
  "temperature": 10.0
}
```

Weights must be finite, non-negative, and sum to one. Runtime inputs must exactly
match the artifact's named features; missing experts are not silently replaced or
renormalised. This makes training-serving skew fail loudly.

This baseline head computes:

```text
score(x) = weights · x
P(A wins) = sigmoid((score(A) - score(B)) / temperature)
```

There is no intercept because it cancels in a pairwise difference. The initial
visual-only baseline uses component quality `0.45`, outfit coordination `0.30`,
and body-aware fit `0.25`. A probability is not product-facing confidence until
held-out calibration has been measured.

The final ensemble must not assume that raw visual and VLM experts naturally
share a meaningful `0..100` scale. The same applies to the descoped Instagram,
Depop, and momentum source experts if they are ever rescoped. Before any source
expert is connected, a later artifact schema must record the fitted transform for
each feature (for example centre, scale, and clipping bounds). Training fits the
combiner on those transformed features and inference applies the exact same
transforms. Mapping a model result into a displayed `0..100` FITTED score is a
separate calibration step and is not an input-normalisation shortcut.

## Product scoring modes

FITTED has two deliberately different scoring modes. The live mode optimises for
speed, stability, and entertainment; the final mode optimises for the most
reliable judgement available within the hackathon architecture. A live result is
never presented as the completed verdict.

### Live provisional mode

While the battle is active, the fast scorer may compare the newest valid frame
pair at approximately `1 FPS`. It may use fewer or cheaper signals than the final
path. Fashion-specific garment perception can improve this estimate when it is
available, but a missed garment must not become a zero and garment perception is
not required for the whole-outfit fallback.

The UI displays an integer range for each player rather than an exact decimal,
for example `72–82`, with the persistent label **Live estimate** or
**Provisional**. It may also show `A currently ahead`, `B currently ahead`, or
`Too close to call`; it must not present a live leader as the winner.

Initial hackathon display defaults are:

- smooth the raw live score centre using an exponential moving average with
  `alpha = 0.25` over valid results only;
- start with a band of `centre ± 5` points, clipped to `0..100` and rounded
  outward to whole numbers;
- do not move the visible centre by more than three points per accepted update
  unless a new battle or explicit rescan resets the estimate;
- hold the last valid range when a scoring tick is skipped, inference is busy,
  or the current frame is temporarily unscoreable; and
- mark the held value as stale or analysing rather than fabricating a new score.

These are presentation defaults, not claims of statistical confidence. A range
must not be labelled with a confidence percentage until empirical coverage has
been measured.

### Final authoritative mode

When both players become locally score-ready, the server starts a 5-second
countdown. At zero, or when either player finalises early, stop accepting live
garment updates and capture five current synchronised slots approximately 750 ms
apart using each browser's latest valid outfit crop geometry. A stable buffered
crop is used only when current-frame capture fails. Enter **Analysing final
result** with any one-to-five complete pairs and
invoke the most accurate configured scoring path once. For the hackathon, this
is the paired VLM fallback plus the application-owned deterministic dimension
combiner unless a validated learned scorer replaces it.

The completed response contains exact `0..100` scores, formatted to one decimal,
and the final winner or draw. It is tied to a finalisation ID, broadcast once by
the server-side coordinator, accepted by both clients only for the current
battle, and then locked. Live requests or late responses cannot overwrite it.

The final score is not mathematically clamped to the last live range. Clamping
would make a known-better final judgement inherit an earlier approximation's
error. If the final score falls outside the displayed range, the UI should reveal
the correct result and the event should be recorded as a live-range miss. The
transition may say **Final analysis adjusted the estimate**; it must not hide or
silently alter the final value.

If finalisation times out or returns a not-scoreable result, retain the live range
only as a labelled estimate, show an actionable retry state, and do not declare a
winner. A provisional result never silently becomes authoritative.

### Live-to-final continuity target

The purpose of the live band is to keep the eventual reveal reasonably close
without spending the final model's latency budget throughout the battle. Tune
the smoothing and width on representative webcam pairs against the configured
final scorer, using only earlier live frames to predict each final result.

For the hackathon fixture set, target:

- at least `80%` of per-player final scores landing inside that player's last
  displayed live range;
- at least `75%` agreement between the last provisional leader and the final
  non-draw winner; and
- no unexplained visible jumps greater than three centre points per live update.

Start with the ten-point-wide band above. If the coverage target is missed, widen
it only as far as a sixteen-point total width and retest. If even that cannot
cover the final result reliably, replace live numbers with a qualitative
`currently ahead` / `too close` display rather than showing false precision.
Coverage, absolute live-to-final error, leader reversals, and out-of-range reveals
are evaluation metrics; they are not reasons to falsify the final score.

The next public comparison-response revision should distinguish these states
explicitly rather than overloading one numeric payload:

```ts
type ComparisonScoreState =
  | {
      phase: "live_provisional";
      playerARange: { low: number; high: number };
      playerBRange: { low: number; high: number };
      provisionalLeader: "player_a" | "player_b" | "too_close";
    }
  | {
      phase: "final";
      playerAScore: number;
      playerBScore: number;
      winner: "player_a" | "player_b" | "draw";
      finalisationId: string;
    }
  | {
      phase: "not_scoreable";
      retryable: boolean;
      reason: string;
    };
```

All variants also require the battle, pair/sample, model, and capture-time
identity needed for stale-result rejection. The final and not-scoreable variants
are implemented; `live_provisional` remains planned.

## Paired-image VLM fallback implementation plan

### Purpose and boundary

The paired-image VLM is the first working scoring fallback and the later final
explanation layer. It is not the intended high-frequency live scorer. WebRTC
continues at its normal frame rate, browser-side CV maintains the latest valid
outfit crop geometry, and each browser captures only its own current local video
frame for each final slot. A buffered stable crop is the fallback, not the normal
path. A server-side coordinator pairs A/B submissions and produces one
authoritative result.
When the learned scorer is ready, it should own the approximately `1 FPS` live
score while the VLM runs at freeze/finalisation or when that scorer is
unavailable.

For the hackathon implementation:

- use `gemini-3.6-flash` through the Gemini API;
- send the labelled Player A sequence followed by the labelled Player B sequence
  inline in one request;
- request strict structured output;
- never log image bytes or base64 payloads;
- permit at most one request in flight and never queue old video frames; and
- leave `winProbability` unset until probability calibration is measured.

Google's current documentation confirms that Gemini 3.6 Flash accepts image
inputs and supports schema-constrained structured output:

- [Gemini 3.6 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash)
- [Gemini image understanding](https://ai.google.dev/gemini-api/docs/image-understanding)
- [Gemini structured output](https://ai.google.dev/gemini-api/docs/structured-output)

Model name, media resolution, timeout, and provider must remain configuration
rather than being scattered through application code. Gemini 3.6 Flash is the
decided hackathon fallback so the remaining time is spent on integration and
FITTED-specific verification rather than a broad provider bake-off. The adapter
boundary remains provider-neutral so this decision can be revisited later.

### Request flow

```text
both locally score-ready -> server starts one 5-second round
  -> timer reaches zero or either player finalises early
  -> server requests paired slots at 0, 750, 1500, 2250, and 3000 ms
  -> each browser captures its current local frame with the latest valid crop
     geometry, falling back to its newest stable candidate only if capture fails
  -> coordinator derives identity, keeps the newest submission per role/slot,
     and discards incomplete slots
  -> retain chronological sample identities, assign a burst pair ID, and invoke
     /v1/compare once with one-to-five complete pairs
  -> inference service revalidates type, size, and decodability
  -> VLM adapter sends rubric, A label/images, then B label/images in one request
  -> strict response is parsed into the internal assessment schema
  -> application computes both 45/30/25 visual scores
  -> application applies the provisional draw rule
  -> coordinator broadcasts one authoritative result to both players
  -> both clients accept it only if the battle/finalisation ID is still current
```

The VLM fallback runs only on finalisation by default. The implemented final
burst accepts any one-to-five complete chronological pairs and produces one
burst-level assessment; it never makes five independent Gemini calls. Periodic
VLM sampling every `2–3 seconds` is an optional fallback experiment, not an MVP
requirement. If enabled later, a timer tick while a request is active must be
skipped and never queued.

### Internal VLM assessment

The provider response should be separate from the public API response. Its strict
schema should contain:

```text
playerA/playerB:
  frameQuality: ok | poor | unusable
  componentQuality: 0..100 | null
  outfitCoordination: 0..100 | null
  bodyFit: 0..100 | null
  vlmHolistic: 0..100 | null
  observations: short visible-clothing evidence[]

pair:
  preference: player_a | player_b | draw | cannot_judge
  explanation: concise outfit-focused rationale
```

Numeric fields must be `null` for an unusable player. The public comparison schema
therefore needs an explicit scored/not-scoreable result rather than fabricated
zeroes. Frame quality is never model confidence. The model's direct pair
preference is retained for explanation and evaluation, but the fallback's public
scores and winner are computed by application code from the three dimension
scores. This avoids letting prose or an opaque model-owned aggregate overwrite
the deterministic scoring contract.

`componentQuality` is a fallback-v1 whole-outfit approximation. The final scorer
must instead use garment detections and combine per-component style scores using
category importance, detection confidence, and visibility as specified in the
PRD. `vlmHolistic` is retained as a diagnostic signal and a candidate feature for
the later learned pairwise combiner. It is not added to the fallback's 45/30/25
public score, which would double-count the same visual judgement.

The initial draw threshold may mirror the current UI's `< 2` score-point rule,
but it remains provisional and must be calibrated before the scoring decision is
marked final.

### Prompt contract

The versioned developer prompt must:

- define the same anchored rubric for both players;
- judge clothing, coordination, visible garment fit, and styling proportions;
- prohibit assessment of face, attractiveness, body type, perceived gender,
  wealth, brand prestige, popularity, and image background;
- use image quality only to decide whether the outfit can be judged;
- require visible evidence for each dimension and avoid inventing hidden garment
  details;
- explain `poor`, `unusable`, and `cannot_judge` decisions; and
- identify the inputs consistently as Player A and Player B.

Prompt version, provider model ID, response-schema version, and application
scoring version must be recorded in `modelVersion` or adjacent internal metadata
so results from different configurations can be distinguished during evaluation.

### Failure behaviour

- Missing credentials or model configuration: service remains not ready and
  `/v1/compare` returns `503`.
- Unsupported, empty, oversized, or undecodable input: reject before the provider
  call with a typed `4xx` response.
- Provider refusal, incomplete response, invalid schema, or unusable image: return
  a typed not-scoreable result; do not create a neutral or zero score.
- Timeout: cancel locally, preserve the previous client result, and expose a
  retryable unavailable state. Start with a configurable `12 second` deadline.
- Rate limit or provider `5xx`: do not retry live samples because the next sample
  supersedes them. A frozen/final request may retry once with short capped jitter.
- Late response: discard it when its pair/sample ID is no longer current.
- Logging: record request ID, model/prompt versions, status, latency, and token
  usage, but no image bytes, base64, or full appearance description.

The implemented Node diagnostics distinguish `[scoring] final capture
unavailable` from `[scoring] VLM request started`, `completed`, `rejected`, and
`failed`. This makes it possible to tell from the server terminal whether a
failure occurred before `/v1/compare` or during provider inference without
logging image payloads or appearance descriptions.

### Configuration and ownership

Server-only configuration:

```text
GEMINI_API_KEY
FITTED_SCORING_BACKEND=vlm_fallback
FITTED_VLM_MODEL=gemini-3.6-flash
FITTED_VLM_MEDIA_RESOLUTION=high
FITTED_VLM_TIMEOUT_SECONDS=12
FITTED_VLM_PROMPT_VERSION=v2
FITTED_MAX_BURST_BYTES=15728640
FITTED_ROUND_DURATION_MS=5000
```

The API key must never enter the Next.js client bundle. Each browser owns capture,
candidate selection, and submission of its local frame only. A server-side
coordinator owns room identity, freshness checks, pairing, per-room backpressure,
deduplication, and authoritative result broadcast. The inference service owns the
provider adapter, prompt, schema parsing, deterministic score calculation, and
provider failure mapping. The existing Node room service owns the pairing
coordinator; neither browser owns the comparison, and the Python inference
service remains stateless.

### Implementation phases

#### Phase 1 — Freeze contracts and fixtures

- Define strict internal VLM assessment models, including a not-scoreable path.
- Extend the public response with pair/sample identity and scored/not-scoreable
  states without representing failure as a numeric score.
- Add explicit `live_provisional`, `final`, and `not_scoreable` response states,
  including live ranges and finalisation identity.
- Write the versioned rubric and safety prompt as a separately testable asset.
- Create consented local A/B image fixtures covering valid, poor, unusable,
  close, and clearly different pairs.

#### Phase 2 — Provider adapter

- Add the inference-service-only Google Gen AI dependency and server
  configuration.
- Define a small provider-neutral VLM protocol so tests and future providers
  do not depend directly on the Google SDK.
- Implement the Gemini adapter with labelled chronological image sequences,
  explicit media resolution, strict structured output, and bounded timeout.
- Validate response values after schema parsing and map provider errors into
  explicit domain errors.
- Report configured readiness and provider model in health, with
  provider/prompt/scoring versions in comparison responses.

#### Phase 3 — Fallback scoring engine

- Add a VLM fallback engine behind the existing `InferenceEngine` boundary.
- Calculate player scores with the existing deterministic 45/30/25 function.
- Apply the provisional draw threshold and keep `winProbability` null.
- Return observations and explanation without exposing hidden reasoning.
- Select the backend through configuration; retain explicit `503` behaviour
  when no backend is configured.

#### Phase 4 — Paired browser integration

- Capture each current local video frame with the latest valid outfit crop
  geometry, with newest-stable-candidate fallback and no remote-video capture.
- Pair fresh A/B submissions in a server-side coordinator rather than
  capturing a remote WebRTC element in either browser.
- Add room/player identity, pair/sample IDs, timestamps, collection deadline,
  one-request-per-room backpressure, deduplication, and stale-response rejection.
- Broadcast one authoritative result from the coordinator to both players.
- Preserve the last valid score and show actionable analysing, not-scoreable,
  timeout, and unavailable states.
- Display shared bounded demo scores with a persistent `LIVE ESTIMATE` label,
  no premature winner declaration, and authoritative-final replacement.
- Replace the demo score with smoothed learned live ranges.
- On finalisation, stop live updates, show final analysis in progress, then
  atomically lock the exact server-authoritative scores and winner or draw on
  both clients.
- Record out-of-range final reveals and show an explicit adjusted-estimate
  transition instead of clamping the final score.
- Start one server-authoritative 5-second round when both roles are ready,
  allow either player to finalise early, and cancel on disconnect.
- On finalisation, request five paired slots approximately 750 ms apart and
  score any one-to-five complete pairs in one Gemini request.
- Consider configurable `2–3 second` VLM polling only as an optional
  fallback experiment after the freeze/final path is reliable.

#### Phase 5 — Verification and evaluation

- Unit-test prompt construction, image labelling, schema validation,
  deterministic scoring, error mapping, and refusal/incomplete handling with a
  fake provider.
- API-test configured success, missing configuration, invalid images,
  not-scoreable results, timeouts, and provider failures without network access.
- Coordinator-test one request per room in flight, countdown expiry, early
  finalisation, disconnect cleanup, burst fallback, stale/duplicate rejection,
  and locked-result replay.
- Browser-test the complete countdown, capture, and consistent authoritative
  result flow on both clients.
- Add an opt-in real-provider smoke test that never runs in the default suite.
- Evaluate at least 30–50 consented representative pairs, including swapped
  A/B duplicates and multiple frames of the same outfit.
- Record agreement, swap consistency, score stability, median/P95 latency,
  not-scoreable rate, token use, and approximate request cost.
- Measure live-range coverage, absolute live-to-final error, provisional
  leader agreement, reversals, and visible score jumps on representative battles.
- Tune the live band to the stated coverage target, or replace live numbers
  with the qualitative fallback if the target requires a band wider than sixteen
  points.
- Record Gemini 3.6 Flash results on the frozen evaluation set; defer a
  cross-provider bake-off until after the hackathon unless Gemini blocks delivery.

### Definition of done

The VLM fallback is implemented only when all of the following are true:

- a configured service returns a schema-valid result for two real outfit images;
- an unconfigured service still fails explicitly rather than returning a mock;
- unusable imagery never produces a competitive score;
- score arithmetic is application-owned and deterministic;
- only one paired request per room can be active and stale results cannot update
  the UI;
- both players display the same server-authoritative result;
- live values are visibly provisional ranges and cannot be mistaken for the final
  verdict;
- finalisation prevents late live work from overwriting the locked result;
- a failed finalisation never promotes a provisional range into a winner;
- images and sensitive appearance descriptions are absent from logs;
- unit, API, browser, and opt-in provider smoke tests pass; and
- representative-pair latency, stability, swap, and cost measurements are
  recorded in this specification.

When these conditions are verified, update both this plan and the corresponding
VLM fallback checkbox in `docs/specs/cv-detection.md`. Do not mark the PRD's real
ML inference or pairwise-head items complete solely because the fallback works.

## Required evaluation

- Validate score ranges and reject malformed scoring artifacts.
- Verify that swapping A and B swaps scores and complements win probability.
- Verify identical vectors produce a 50/50 prediction.
- Define a versioned pair-label and expert-feature dataset schema.
- Define and version the per-expert input transformations required before
  source experts are connected.
- Implement non-negative, regularised pairwise logistic training.
- Fit and evaluate the head on person/outfit/session-disjoint data.
- Measure calibration and select the temperature on validation data.
- Compare the visual-only, VLM-only, and available source-expert variants.
- Verify the live-to-final coverage and provisional-leader targets on a
  representative webcam fixture set.
- Decide draw behaviour and a separate displayed-score `0..100` calibration
  from measured results.
- Connect validated expert outputs and the artifact to the inference engine.

The PRD's pairwise-head implementation item remains incomplete until training and
held-out evaluation have both run on representative human labels.
