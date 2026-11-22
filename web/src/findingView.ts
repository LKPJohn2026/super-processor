// Finding the scenes (mockup 03), and what the Scenes step says when
// Gemini's split did not work (S3 quota, S8 no scenes). Pure functions, so
// the node tests can check them; types only from ./api.
import type { Progress, SegmentList, WizardView } from "./api";

export type FindStageKey = "stills" | "copy" | "upload" | "gemini";
export type FindStageState = "done" | "running" | "waiting" | "skipped";
export type FindStage = { key: FindStageKey | "video"; state: FindStageState; label: string; meta: string };

const ORDER: FindStageKey[] = ["stills", "copy", "upload", "gemini"];

const LABELS: Record<FindStageKey, { done: string; todo: string }> = {
  stills: { done: "Saved stills from the video", todo: "Saving stills from the video" },
  copy: { done: "Made a small silent copy (up to 960 px)", todo: "Making a small silent copy (up to 960 px)" },
  upload: { done: "Uploaded the copy to Gemini", todo: "Uploading the copy to Gemini" },
  gemini: { done: "Gemini found where scenes change", todo: "Gemini is finding where scenes change" },
};

/**
 * The stage the run is at, from the server's progress (wizard.py `_sample`,
 * `_gemini_video` and `_propose_segments` say these words). Null when the
 * page cannot tell (the step has just changed and no new word has come).
 */
export function findingStage(view: Pick<WizardView, "step" | "progress">): FindStageKey | null {
  const progress = view.progress;
  const message = progress?.message ?? "";
  if (view.step === "sampling") return "stills";
  if (view.step !== "proposing") return null;
  if (message.startsWith("Preparing a silent copy")) return "copy";
  if (message.startsWith("Uploading the video to Gemini")) return "upload";
  if (message.startsWith("Gemini is")) return "gemini";
  return null;
}

/** True when Gemini works from the stills because the video could not be used. */
export function fromStills(progress: Progress | null | undefined): boolean {
  return /from the stills/.test(progress?.message ?? "");
}

/** When each stage was first and last seen by this page (ms), for the times beside them. */
export type StageClock = {
  key: FindStageKey | null;
  started: Partial<Record<FindStageKey, number | null>>;
  ended: Partial<Record<FindStageKey, number>>;
};

export const NO_CLOCK: StageClock = { key: null, started: {}, ended: {} };

/**
 * The clock after a look at the run. A stage first seen when the page
 * opened has no known start, except the first one: the run's own elapsed
 * time says when it began.
 */
export function tickClock(
  clock: StageClock,
  key: FindStageKey | null,
  now: number,
  elapsedS?: number,
): StageClock {
  if (!key || key === clock.key) return clock;
  const started = { ...clock.started };
  const ended = { ...clock.ended };
  if (clock.key) ended[clock.key] = now;
  if (!(key in started)) {
    started[key] =
      clock.key !== null ? now : key === "stills" && typeof elapsedS === "number" ? now - elapsedS * 1000 : null;
  }
  return { key, started, ended };
}

/** `0:12`, `1:03`, `12:40`: a stage's time, as the mockup shows it. */
export function shortClock(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function counted(message: string): string {
  const of = /(\d+) of (\d+)/.exec(message);
  if (of) return `${of[1]} of ${of[2]}`;
  const pct = /(\d+)%/.exec(message);
  return pct ? `${pct[1]}%` : "";
}

/**
 * The stage list (mockup 03): done, running and waiting rows built from
 * the run's real progress. When Gemini works from the stills, the copy and
 * the upload become one row that says so.
 */
export function findingStages(
  view: Pick<WizardView, "step" | "progress">,
  clock: StageClock = NO_CLOCK,
  now: number = Date.now(),
): FindStage[] {
  const key = findingStage(view);
  const message = view.progress?.message ?? "";
  // In "proposing" with no word yet, the stills are done and the rest waits.
  const at = key ? ORDER.indexOf(key) : view.step === "proposing" ? 0.5 : 0;
  const stills = fromStills(view.progress);
  const rows: FindStage[] = [];
  ORDER.forEach((stage, index) => {
    const state: FindStageState = index < at ? "done" : index === at ? "running" : "waiting";
    const start = clock.started[stage];
    const end = clock.ended[stage];
    let meta = "";
    if (state === "done" && typeof start === "number" && typeof end === "number") {
      meta = shortClock((end - start) / 1000);
    } else if (state === "running") {
      const count = counted(message);
      meta = count || (typeof start === "number" ? `${shortClock((now - start) / 1000)} so far` : "");
    }
    let label = state === "done" ? LABELS[stage].done : LABELS[stage].todo;
    if (stage === "gemini" && stills) label = `${LABELS.gemini.todo}, from the stills`;
    rows.push({ key: stage, state, label, meta });
  });
  if (!stills) return rows;
  return [
    rows[0],
    {
      key: "video",
      state: "skipped",
      label: "Gemini couldn’t use the video copy, so it works from the stills instead",
      meta: "",
    },
    rows[3],
  ];
}

/** "About 2 min left" when the server can tell; else the time spent. */
export function findingTime(
  progress: Progress | null | undefined,
  sinceS = 0,
): { big: string; forStage: boolean; spent: string } {
  const spent =
    typeof progress?.elapsed_s === "number" ? `${shortClock(progress.elapsed_s + sinceS)} so far` : "";
  if (typeof progress?.eta_s === "number") {
    const left = Math.max(0, progress.eta_s - sinceS);
    const words = left >= 90 ? `${Math.round(left / 60)} min` : `${Math.max(1, Math.round(left))} s`;
    return { big: `About ${words} left`, forStage: true, spent };
  }
  return { big: spent || "Starting…", forStage: false, spent: "" };
}

/** `14:02`, the local time a run started (epoch seconds). */
export function startedAt(epochS: number | null | undefined): string {
  if (typeof epochS !== "number" || !Number.isFinite(epochS)) return "";
  const date = new Date(epochS * 1000);
  return `${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
}

export type ScenesProblem =
  | { kind: "quota"; daily: boolean; retryS: number | null; detail: string }
  | { kind: "failed"; detail: string }
  | { kind: "empty" }
  | null;

/** Seconds from Gemini's "try again in 28 s" / "about 2 min" / "about 1 h". */
export function retryAfterS(text: string): number | null {
  const match = /try again in (?:about )?(\d+(?:\.\d+)?) (s|min|h)\b/.exec(text);
  if (!match) return null;
  const value = Number(match[1]);
  return Math.round(value * (match[2] === "h" ? 3600 : match[2] === "min" ? 60 : 1));
}

/**
 * What went wrong with Gemini's split, from the scene list the server
 * saved (wizard.py `_propose_segments`): its quota used up (S3), another
 * failure, or no scenes at all, when the whole video is the one scene
 * `repair_coverage` makes from an empty answer (S8).
 */
export function scenesProblem(list: SegmentList | null | undefined): ScenesProblem {
  if (!list) return null;
  const error = (list.error ?? "").trim();
  if (error) {
    const match = /^Gemini could not propose segments \((.*)\); the whole video is one\.$/s.exec(error);
    const detail = match ? match[1] : error;
    if (/quota/i.test(detail)) {
      return { kind: "quota", daily: /daily quota/i.test(detail), retryS: retryAfterS(detail), detail };
    }
    return { kind: "failed", detail };
  }
  const only = list.segments.length === 1 ? list.segments[0] : null;
  if (only && only.label === "whole video") return { kind: "empty" };
  return null;
}

/** The note "Ask Gemini again" sends: the editor's own, else a plain request. */
export function askAgainNote(list: SegmentList | null | undefined): string {
  const own = (list?.note ?? "").trim();
  return own || "Find where the scenes change.";
}
