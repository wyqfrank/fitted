# FITTED architecture

This document describes the system as it works now. See
[next-steps.md](next-steps.md) for unfinished work and [specs/](specs/) for
detailed designs.

## System overview

Each player streams video directly to the other player through WebRTC. The
server handles room state, WebRTC signalling, round timing, and scoring. Camera
video is not relayed through the server.

```text
Laptop A <------ WebRTC video ------> Laptop B
    |                                     |
    +------- local cropped frames --------+
                      |
                      v
        Next.js and Socket.IO server
        - room state and signalling
        - five-second round timer
        - frame pairing
        - result broadcast
                      |
                      v
              Python API service
        - RF-DETR garment detection
        - DINOv2 live scoring
        - Gemini final scoring
```

Pose detection runs in a Web Worker in each browser. It checks framing and crops
the outfit before a frame is sent to the server.

| Path | Purpose |
|---|---|
| `apps/web` | Next.js frontend, Socket.IO server, room logic, and browser CV code |
| `services/inference` | FastAPI service for garment detection and scoring |
| `services/inference/scripts` | Training and evaluation scripts; not part of the deployed package |
| `models/ranker` | Ranker files used by the inference service |
| `scripts/` | Local development scripts |

## Round flow

1. Both players join a room and turn on their cameras.
2. Each browser uses pose detection to check that the outfit is visible.
3. When both players are ready, the server starts a five-second round. The
   server owns the timer. Either player can end the round early, and a disconnect
   cancels it.
4. During the round, each browser sends about one cropped frame per second. The
   server pairs frames from the same moment and sends them to garment detection
   and the live ranker. It allows one request at a time and does not queue old
   frames.
5. At the end of the round, the server asks each browser for five crops, about
   750 ms apart. It sends every complete pair to Gemini in one request.
6. The server sends the final scores and result to both players, then ignores
   late responses.

The server always makes the final comparison. A browser never scores the other
player from the remote WebRTC video.

## Scoring

### Live score

The live path is:

```text
outfit crop -> DINOv2-S -> normalise -> linear model -> percentile -> score from 55 to 85
```

The model file includes 256 calibration quantiles, so the service does not need
a separate score mapping. The linear model has no intercept, which makes the
result consistent when players A and B are swapped.

The current model is `dinov2s-linear-v2`. It uses 384 DINOv2 coordinates and was
trained from 2,000 Gemini comparisons of 128 project photos. It replaced
`dinov2s-pca-linear-v1` on 2026-09-16. See
[ranker-experiments.md](ranker-experiments.md) for results and limitations.

Live scoring uses the same paired frames as garment detection. If garment
detection is off, or only one player is present, the app has no frame pair to
score.

### Final score

Gemini 3.6 Flash receives the available final frame pairs in one request. Its
scores decide the winner. The live score never decides the result.

## Main decisions

| Area | Choice | Reason |
|---|---|---|
| Product | Two-player camera battle | This is the main idea the prototype tests |
| Frontend | Next.js, React, TypeScript, Tailwind, and shadcn/ui | Keeps the UI and signalling host in one stack |
| Video | Peer-to-peer WebRTC | Avoids sending video through a server and keeps latency low |
| NAT traversal | Google STUN, with optional TURN | Some venue networks block direct connections |
| Signalling | Socket.IO | Handles rooms, WebRTC messages, readiness, and timing |
| Round timing | Controlled by the server | Keeps both clients on the same round |
| Frame pairing | Controlled by the Node server | Gives both players one shared result |
| Frame source | Each browser sends its own local crop | Keeps scoring independent of remote video quality |
| Inference | Separate stateless Python service | Keeps ML packages out of the web server |
| Pose checks | MediaPipe in a browser worker | Rejects bad frames before upload and keeps work off the UI thread |
| Crop | Padded outfit crop; feet optional | Keeps the full outfit without rejecting otherwise useful frames |
| Live model | Frozen DINOv2-S with a linear head | Supports roughly one score per second and gives consistent A/B swaps |
| Training cohort | AC and DP raters | Their ratings agreed; the result predicts this group, not universal taste |
| Teacher labels | Gemini comparisons of project photos | Matches the app's image domain and avoids the Fashion144k licence limit |
| Final model | Gemini 3.6 Flash | Already integrated and tested for FITTED |
| Garment model | RF-DETR-Seg Small | Selected as the only live candidate after the slower baseline failed |
| Personal traits | Excluded from scoring | Scores should reflect clothing fit and styling, not faces or body type |
| Result | Live estimate, then exact final score | Only the final server request decides the winner |

## Options we rejected

- Grounding DINO Tiny took about 10.6 seconds per CPU inference, so it is only
  kept as a diagnostic baseline.
- The tested MLP performed worse than the linear model. Its regularisation was
  also incomplete, so the result does not rule out every nonlinear model.
- More PCA dimensions did not improve the original model.
- A projection fitted on Fashion144k performed worse in every tested setup.
- Instagram and Depop models were removed from the hackathon scope.
- StreamingVLM and continuous 30 FPS inference add more complexity than this
  demo needs.
- Full encoder fine-tuning is too expensive for the hackathon. Training uses
  frozen image features.

## Known limits

- The project has no deployment target. Both services currently run locally.
- The live view shows one score, not a calibrated range.
- The current and previous rankers both miss the 95 ms p95 CPU target.
- Latency has not been measured with HTTP, garment detection, and two players
  running together.
- The current live ranker has not been tested in a full two-player battle.
- Model results come from reused development photos, not webcam images from new
  people and outfits.

These items are tracked in [next-steps.md](next-steps.md).
