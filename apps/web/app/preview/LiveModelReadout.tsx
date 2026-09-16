"use client";

import { useMemo } from "react";
import type { FitScoreSample, LiveFitScoreState } from "@/hooks/useLiveFitScore";
import type { LiveFitScore } from "@/lib/fitScore";
import type { CameraStatus } from "@/hooks/useCamera";
import type { DetectorState } from "@/lib/cv/types";

type Props = {
  fit: {
    state: LiveFitScoreState;
    latest: LiveFitScore | null;
    smoothed: number | null;
    history: FitScoreSample[];
    error: string | null;
    modelVersion: string | null;
    scoreable: boolean;
    framingLabel: string;
  };
  cameraStatus: CameraStatus;
  detectorState: DetectorState;
};

const STATE_LABELS: Record<LiveFitScoreState, string> = {
  idle: "IDLE",
  checking: "LOADING MODEL",
  unavailable: "MODEL UNAVAILABLE",
  waiting_for_frame: "NOT SCORING",
  scoring: "SCORING",
  error: "ERROR",
};

function median(values: number[]) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/**
 * The instrument for the one question offline evaluation cannot answer: does
 * this model behave sanely on webcam frames?
 *
 * It was trained and calibrated on 128 curated full-body photos. Live input is
 * a 640px webcam crop taken under the venue's lighting. The model can still
 * return a confident score for unfamiliar input, so this readout makes that
 * failure visible.
 *
 * Two readings matter, and both are on raw scores rather than the smoothed
 * display value, because smoothing would hide exactly what is being tested:
 *
 * - **Spread** — hold still in one outfit. Raw scores should barely move. A
 *   wide spread means the model is reading noise, not the outfit.
 * - **Separation** — step out, change into something clearly different, step
 *   back. The bands should not overlap.
 */
export function LiveModelReadout({ fit, cameraStatus, detectorState }: Props) {
  const stats = useMemo(() => {
    const raws = fit.history.map((sample) => sample.raw);
    if (raws.length < 2) return null;
    const mean = raws.reduce((total, value) => total + value, 0) / raws.length;
    const variance =
      raws.reduce((total, value) => total + (value - mean) ** 2, 0) / (raws.length - 1);
    return {
      n: raws.length,
      mean,
      sd: Math.sqrt(variance),
      min: Math.min(...raws),
      max: Math.max(...raws),
      displayMin: Math.min(...fit.history.map((s) => s.score)),
      displayMax: Math.max(...fit.history.map((s) => s.score)),
      latency: median(fit.history.map((sample) => sample.latencyMs)),
    };
  }, [fit.history]);

  const spark = useMemo(() => {
    if (fit.history.length < 2) return null;
    const raws = fit.history.map((sample) => sample.raw);
    const low = Math.min(...raws);
    const high = Math.max(...raws);
    const span = Math.max(high - low, 1e-6);
    return raws
      .map((value, index) => {
        const x = (index / (raws.length - 1)) * 100;
        const y = 100 - ((value - low) / span) * 100;
        return `${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");
  }, [fit.history]);

  return (
    <aside className="model-readout" aria-live="polite">
      <header>
        <span className={`model-pill is-${fit.state}`}>{STATE_LABELS[fit.state]}</span>
        <code>{fit.modelVersion ?? "—"}</code>
        {cameraStatus !== "ready" ? <span className="model-note">camera {cameraStatus}</span> : null}
      </header>

      {/* Without this line a paused sparkline looks like a rock-steady model
          rather than a player who has wandered out of frame. */}
      <p className={`model-framing ${fit.scoreable ? "is-ok" : ""}`}>
        {detectorState === "loading"
          ? "Starting the fit detector…"
          : fit.scoreable
            ? "Framing OK — scoring"
            : `Not scoring · ${fit.framingLabel}`}
      </p>

      {fit.error ? (
        <p className="model-error">{fit.error}</p>
      ) : (
        <>
          <div className="model-figures">
            <div>
              <b className="tabular-nums">
                {fit.smoothed === null ? "—" : fit.smoothed.toFixed(1)}
              </b>
              <small>SHOWN (smoothed)</small>
            </div>
            <div>
              <b className="tabular-nums">
                {fit.latest ? fit.latest.raw.toFixed(3) : "—"}
              </b>
              <small>RAW MARGIN</small>
            </div>
            <div>
              <b className="tabular-nums">
                {fit.latest ? `${Math.round(fit.latest.percentile * 100)}%` : "—"}
              </b>
              <small>PERCENTILE</small>
            </div>
          </div>

          {spark ? (
            <svg className="model-spark" viewBox="0 0 100 100" preserveAspectRatio="none">
              <polyline points={spark} vectorEffect="non-scaling-stroke" />
            </svg>
          ) : null}

          {stats ? (
            <dl className="model-stats">
              <div>
                <dt>raw spread</dt>
                <dd className="tabular-nums">
                  sd {stats.sd.toFixed(3)} · {stats.min.toFixed(2)} to {stats.max.toFixed(2)}
                </dd>
              </div>
              <div>
                <dt>shown range</dt>
                <dd className="tabular-nums">
                  {stats.displayMin.toFixed(1)} to {stats.displayMax.toFixed(1)}
                </dd>
              </div>
              <div>
                <dt>latency</dt>
                <dd className="tabular-nums">{Math.round(stats.latency)} ms median</dd>
              </div>
              <div>
                <dt>samples</dt>
                <dd className="tabular-nums">{stats.n}</dd>
              </div>
            </dl>
          ) : (
            <p className="model-note">Collecting samples…</p>
          )}

          <p className="model-help">
            Scores only the person crop, and only while your whole fit is in
            frame. Hold still — raw spread should stay tight. Then change into a
            visibly different outfit; the two ranges should not overlap. If they
            do, the score is not reading the outfit.
          </p>
        </>
      )}
    </aside>
  );
}
