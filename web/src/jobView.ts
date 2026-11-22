// The job around the current step: the stepper, the job's title and the
// scene strip, built from the server's job summary (`view.job`). Pure
// functions of the view, so the node tests can check them. Types only from
// ./api, which the transpile drops.
import type { JobScene, Navigation, StageKey, WizardView } from "./api";

export type StepState = "done" | "current" | "upcoming";

export type StepItem = {
  key: string;
  label: string;
  state: StepState;
  // A short line under the label ("5 scenes", "2 of 5 approved").
  detail: string;
  // The place the step can take the editor back to, when it can now.
  nav: keyof Navigation | null;
};

// Names the editor knows the stages by (see the redesign's stepper).
export const STAGE_LABELS: Record<StageKey, string> = {
  file: "Video",
  segments: "Scenes",
  scenes: "Previews",
  render: "Render",
  result: "Result",
};

const STAGE_NAV: Partial<Record<StageKey, keyof Navigation>> = {
  file: "file",
  segments: "segments",
};

// The rail shown before a job exists, or by a server with no job summary.
const RAIL = ["Setup", "File", "Scenes", "Previews", "Result"] as const;
const RAIL_NAV: Partial<Record<number, keyof Navigation>> = { 0: "setup", 1: "file", 2: "segments" };

export function railIndex(step: WizardView["step"]): number {
  if (step === "intro" || step === "llm_choice" || step === "local_llm_stub" || step === "setup") {
    return 0;
  }
  if (step === "pick_file") return 1;
  if (step === "sampling" || step === "proposing" || step === "segments") return 2;
  if (step === "analyzing" || step === "preview_segment" || step === "ready_to_render") {
    return 3;
  }
  return 4;
}

const KEYS: readonly StageKey[] = ["file", "segments", "scenes", "render", "result"];

/** True when the view carries a job summary the stepper can be drawn from. */
export function hasJobStages(view: WizardView): boolean {
  const stages = view.job?.stages;
  return (
    Array.isArray(stages) &&
    stages.length > 0 &&
    stages.every((stage) => KEYS.includes(stage.key))
  );
}

function plural(count: number, one: string, many: string): string {
  return `${count} ${count === 1 ? one : many}`;
}

function stageDetail(view: WizardView, key: StageKey, state: StepState): string {
  const job = view.job;
  if (!job) return "";
  const count = job.scene_count;
  // The title sits above the stepper; the file needs no second line.
  if (key === "file") return "";
  if (key === "segments") {
    if (state === "done" && count > 0) return plural(count, "scene", "scenes");
    if (state === "current") {
      return view.step === "segments" ? "Check the split" : view.step === "pick_file" ? "" : "Finding scenes…";
    }
    return "";
  }
  if (key === "scenes") {
    if (count > 0 && (state !== "upcoming" || job.approved_count > 0)) {
      return `${job.approved_count} of ${count} approved`;
    }
    return "";
  }
  if (key === "render") {
    if (state === "current") return view.interrupted ? "Waiting for you" : "Rendering…";
    return "";
  }
  if (state === "current" || state === "done") {
    const current = view.versions?.find((version) => version.current);
    return current ? `Version ${current.number}` : state === "current" ? "Review it" : "";
  }
  return "";
}

/**
 * The stepper's items: from the job's stages when the server sends them,
 * else the rail that follows the step. Stages the editor can go back to
 * carry `nav` only while nothing is busy.
 */
export function stepperItems(view: WizardView, busy: boolean): StepItem[] {
  const navigable = (target: keyof Navigation | undefined) =>
    target !== undefined && !busy && !view.busy && view.navigation?.[target] ? target : null;
  if (hasJobStages(view)) {
    return view.job!.stages.map(({ key, status }) => {
      const state: StepState = status === "pending" ? "upcoming" : status;
      return {
        key,
        label: STAGE_LABELS[key],
        state,
        detail: stageDetail(view, key, state),
        nav: navigable(STAGE_NAV[key]),
      };
    });
  }
  const active = railIndex(view.step);
  return RAIL.map((label, index) => ({
    key: label.toLowerCase(),
    label,
    state: index < active ? "done" : index === active ? "current" : "upcoming",
    detail: "",
    nav: navigable(RAIL_NAV[index]),
  }));
}

/** The job's title for the top of the page, or null before a job exists. */
export function jobTitle(view: WizardView): string | null {
  const title = view.job?.title?.trim();
  return title || null;
}

/** `4:32` or `1:02:05`, for a scene's time range. */
export function clockTime(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = String(total % 60).padStart(2, "0");
  return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${secs}` : `${minutes}:${secs}`;
}

export type StripScene = {
  index: number;
  // "3 · At the window"
  name: string;
  range: string;
  status: JobScene["status"];
  // Approved, Reviewing, Preparing…, Waiting.
  statusText: string;
  previewed: boolean;
  stillUrl: string | null;
  // The button that opens it again, or null when it cannot be opened now.
  reopen: { text: string; label: string } | null;
};

/** The scene strip, or an empty list when the job has no scenes to show. */
export function sceneStrip(view: WizardView): StripScene[] {
  const scenes = view.job?.scenes;
  if (!Array.isArray(scenes)) return [];
  return scenes.map((scene) => {
    const number = scene.index + 1;
    const label = scene.label?.trim() || `Scene ${number}`;
    const statusText =
      scene.status === "approved"
        ? "Approved"
        : scene.status === "current"
          ? view.step === "analyzing"
            ? "Preparing…"
            : "Reviewing"
          : "Waiting";
    let reopen: StripScene["reopen"] = null;
    if (scene.reopenable && scene.status !== "current") {
      reopen =
        scene.status === "approved"
          ? { text: "Look again", label: `Look again at scene ${number}, ${label}` }
          : { text: "Go to this scene", label: `Go to scene ${number}, ${label}` };
    }
    return {
      index: scene.index,
      name: `${number} · ${label}`,
      range: `${clockTime(scene.start_s)}–${clockTime(scene.end_s)}`,
      status: scene.status,
      statusText,
      previewed: Boolean(scene.previewed),
      stillUrl: scene.still_url ?? null,
      reopen,
    };
  });
}

/** "2 approved · 3 to go", or "All 5 scenes approved". */
export function stripSummary(view: WizardView): string {
  const job = view.job;
  if (!job || job.scene_count <= 0) return "";
  const approved = Math.min(job.approved_count, job.scene_count);
  if (approved >= job.scene_count) {
    return job.scene_count === 1 ? "The scene is approved" : `All ${job.scene_count} scenes approved`;
  }
  return `${approved} approved · ${job.scene_count - approved} to go`;
}

/**
 * The whole job's percentage to show: never below what this run already
 * showed. `previous` is what the page showed last, with the run it was for;
 * a different run (or a different job) starts again from its own value.
 */
export function steadyOverall(
  previous: { key: string; value: number } | null,
  key: string,
  value: number,
): { key: string; value: number } {
  const clamped = Math.max(0, Math.min(100, value));
  if (previous && previous.key === key && previous.value > clamped) return previous;
  return { key, value: clamped };
}

/**
 * The key the whole job's percentage is kept steady under. Each scene's
 * previews are a run of their own, but they share one band of the job: a
 * scene looked at again must not pull the bar below what it showed. A render
 * or a scan is kept steady per run (a revise renders from zero again).
 */
export function overallKey(view: WizardView): string {
  if (view.step === "analyzing") return `${view.job_id ?? ""}:previews`;
  return `${view.job_id ?? ""}:${view.run?.id ?? view.step}`;
}

// ---------------------------------------------------------------- sidebar

/** A step of the sidebar's stepper: a StepItem that can also have failed. */
export type SideStep = Omit<StepItem, "state"> & { state: StepState | "error" };

const SIDE_LABELS = ["Video", "Scenes", "Previews", "Render", "Result"] as const;
const SIDE_KEYS: readonly StageKey[] = ["file", "segments", "scenes", "render", "result"];
const SIDE_NAV: Partial<Record<number, keyof Navigation>> = { 0: "file", 1: "segments" };

/** The design's five steps, for a step: null before a file is picked (setup). */
export function sideIndex(step: WizardView["step"]): number | null {
  if (step === "intro" || step === "llm_choice" || step === "local_llm_stub" || step === "setup") {
    return null;
  }
  if (step === "pick_file") return 0;
  if (step === "sampling" || step === "proposing" || step === "segments") return 1;
  if (step === "analyzing" || step === "preview_segment") return 2;
  if (step === "ready_to_render" || step === "rendering") return 3;
  return 4;
}

/** True when the page shows the job's sidebar: from picking the file on. */
export function showsSidebar(view: WizardView): boolean {
  return sideIndex(view.step) !== null;
}

/**
 * The sidebar's stepper (mockups 02–09): the job's own stages when the
 * server sends them, else the five steps followed from the step. The
 * current step turns to "error" when `failed` (a step that failed, such as
 * a render that did not finish).
 */
export function sidebarSteps(view: WizardView, busy: boolean, failed = false): SideStep[] {
  const mark = (items: SideStep[]): SideStep[] =>
    failed ? items.map((item) => (item.state === "current" ? { ...item, state: "error" } : item)) : items;
  if (hasJobStages(view)) {
    let items: SideStep[] = stepperItems(view, busy);
    // The server counts "Ready to render" in its scenes stage; the design
    // puts it on Render (mockup 06), once every scene is approved.
    if (view.step === "ready_to_render") {
      items = items.map((item) =>
        item.key === "scenes" && item.state === "current"
          ? { ...item, state: "done" }
          : item.key === "render" && item.state === "upcoming"
            ? { ...item, state: "current", detail: "Choose the size" }
            : item,
      );
    }
    if (items[0]?.key === "file" && !items[0].detail && view.step === "pick_file") {
      items[0] = { ...items[0], detail: "Pick a file" };
    }
    return mark(items);
  }
  const active = sideIndex(view.step) ?? 0;
  const navigable = (target: keyof Navigation | undefined) =>
    target !== undefined && !busy && !view.busy && view.navigation?.[target] ? target : null;
  return mark(
    SIDE_LABELS.map((label, index) => {
      const state: StepState = index < active ? "done" : index === active ? "current" : "upcoming";
      let detail = "";
      if (index === 0 && state === "current") detail = "Pick a file";
      if (index === 1 && state === "current") detail = view.step === "segments" ? "Check the split" : "Finding scenes…";
      if (index === 3 && view.step === "rendering") detail = "Rendering…";
      return { key: SIDE_KEYS[index], label, state, detail, nav: navigable(SIDE_NAV[index]) };
    }),
  );
}

/** The video's length in seconds, from what the server sent, or null. */
export function videoLength(view: WizardView): number | null {
  const fromList = view.segments?.duration_s;
  if (typeof fromList === "number" && fromList > 0) return fromList;
  const segments = view.segments?.segments;
  if (Array.isArray(segments) && segments.length) {
    const end = segments[segments.length - 1].end_s;
    if (end > 0) return end;
  }
  const scenes = view.job?.scenes;
  if (Array.isArray(scenes) && scenes.length) {
    const end = scenes[scenes.length - 1].end_s;
    if (end > 0) return end;
  }
  return null;
}

/**
 * The facts under the file name ("640×480 · 22:01"): only what the server
 * sent. Frame rate and sound are not in the API, so they are not shown.
 */
export function sourceFacts(view: WizardView): string {
  const parts: string[] = [];
  const size = view.looks?.source_size;
  if (Array.isArray(size) && size.length === 2 && size[0] > 0 && size[1] > 0) {
    parts.push(`${size[0]}×${size[1]}`);
  }
  const length = videoLength(view);
  if (length !== null) parts.push(clockTime(length));
  const count = view.job?.scene_count ?? 0;
  if (count > 0) parts.push(plural(count, "scene", "scenes"));
  return parts.join(" · ");
}

/** The job's still for the sidebar: the job list's thumbnail, by job id. */
export function sourceThumbUrl(view: WizardView): string | null {
  const id = view.job?.job_id ?? view.job_id;
  return id ? `/jobs/${encodeURIComponent(id)}/thumb.jpg` : null;
}

/**
 * Stills to try in turn for the sidebar: the job's thumbnail, then the
 * middle scene's still (the job list's thumbnail is the middle still too).
 */
export function sourceStills(view: WizardView): string[] {
  const out: string[] = [];
  const thumb = sourceThumbUrl(view);
  if (thumb) out.push(thumb);
  const fromScenes = (view.job?.scenes ?? []).map((scene) => scene.still_url).filter(Boolean) as string[];
  const fromList = (view.segments?.segments ?? []).map((segment) => segment.still_url).filter(Boolean) as string[];
  const stills = fromScenes.length ? fromScenes : fromList;
  if (stills.length) out.push(stills[Math.floor(stills.length / 2)]);
  return out;
}
