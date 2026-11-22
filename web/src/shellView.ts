// The app shell around each step: the tab title, the connection pill and
// the progress milestones read out to screen readers. Pure functions, so
// the node tests can check them; types only from ./api.
import type { Progress, WizardView } from "./api";

const APP = "Super Processor";

/**
 * The tab title for a step (s9: "Scene 3 of 5 · Previews · Super
 * Processor"). `home` is the All jobs list.
 */
export function documentTitle(view: WizardView | null, home = false): string {
  if (home) return `All jobs · ${APP}`;
  if (!view) return `Connect · ${APP}`;
  const parts: string[] = [];
  switch (view.step) {
    case "intro":
    case "llm_choice":
    case "local_llm_stub":
      parts.push("Welcome");
      break;
    case "setup":
      parts.push("Gemini key", "Welcome");
      break;
    case "pick_file":
      parts.push("Choose a video", "Video");
      break;
    case "sampling":
    case "proposing":
      parts.push("Finding scenes", "Scenes");
      break;
    case "segments":
      parts.push("Check the split", "Scenes");
      break;
    case "analyzing":
    case "preview_segment": {
      const index = view.step === "preview_segment" ? view.segment_preview?.index : view.segment_index;
      const count =
        view.job?.scene_count || view.segment_preview?.count || view.segments?.segments.length || 0;
      const n = (index ?? view.segment_index ?? 0) + 1;
      parts.push(count ? `Scene ${n} of ${count}` : `Scene ${n}`, "Previews");
      break;
    }
    case "ready_to_render":
      parts.push("Render");
      break;
    case "rendering": {
      // 07: "49% · Rendering · Super Processor", once the render reports.
      const pct = view.progress && !view.interrupted ? Math.round(view.progress.pct) : null;
      if (pct !== null && Number.isFinite(pct)) parts.push(`${Math.max(0, Math.min(100, pct))}%`, "Rendering");
      else parts.push("Rendering", "Render");
      break;
    }
    default:
      parts.push("Result");
  }
  if (view.interrupted) parts.unshift("Waiting for you");
  parts.push(APP);
  return parts.join(" · ");
}

/**
 * The key a step is focused under: the H1 takes focus whenever it changes
 * (a new step, another scene's preview, the All jobs list).
 */
export function headingKey(view: WizardView | null, home = false): string {
  if (home) return "home";
  if (!view) return "offline";
  const scene = view.step === "analyzing" || view.step === "preview_segment" ? `:${view.segment_index ?? 0}` : "";
  return `${view.job_id ?? ""}:${view.step}${scene}:${view.interrupted ? "i" : ""}`;
}

export type Connection = "ok" | "warn" | "err";

/** The pill's state from the failed polls in a row since the last answer. */
export function connectionState(failures: number): Connection {
  if (failures <= 0) return "ok";
  return failures === 1 ? "warn" : "err";
}

/** The pill's words, long (desktop) and short (a phone's top bar). */
export function connectionText(state: Connection): { long: string; short: string } {
  if (state === "ok") return { long: "Connected to this computer", short: "Connected" };
  if (state === "warn") return { long: "Reconnecting… work keeps running", short: "Reconnecting…" };
  return { long: "Can’t reach the wizard", short: "Offline" };
}

/** `127.0.0.1:45143` for a banner, from the origin the page talks to. */
export function originHost(origin: string, fallback: string): string {
  try {
    return new URL(origin || fallback).host;
  } catch {
    return origin || fallback;
  }
}

export type Milestone = { key: string; text: string };

/**
 * What to read out for a run's progress, only at milestones (s9): a new
 * stage, or each 10% of the job. Null when nothing new is worth saying.
 * `previous` is the milestone said last.
 */
export function progressMilestone(
  previous: Milestone | null,
  title: string,
  progress: Progress | null | undefined,
): Milestone | null {
  if (!progress) return null;
  const value = typeof progress.overall === "number" ? progress.overall : progress.pct;
  const tens = Math.max(0, Math.min(10, Math.floor((Number.isFinite(value) ? value : 0) / 10)));
  const stage = (progress.message || progress.phase || "").trim();
  // A stage is its words without the counts in them ("Saving still 12 of
  // 60"), so a count going up is not a new stage.
  const stageKey = stageWords(stage);
  const key = `${title}|${stageKey}|${tens}`;
  if (previous && previous.key === key) return null;
  const sameStage = previous ? previous.key.startsWith(`${title}|${stageKey}|`) : false;
  const pct = `${tens * 10}%`;
  const text = sameStage || !stage ? `${title}: ${pct}` : `${title}: ${stage}, ${pct}`;
  return { key, text };
}

function stageWords(text: string): string {
  return text
    .replace(/[0-9]+([.,:][0-9]+)*\s*(%|s|min|h)?/g, "#")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}
