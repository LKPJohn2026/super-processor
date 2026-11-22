// The "All jobs" home: each job's status badge, its button, when it was
// last changed, and when the page opens on the list. Pure functions of the
// rows from GET /api/jobs, so the node tests can check them. Types only from
// ./api, which the transpile drops.
import type { JobEntry, StageKey, WizardStep, WizardView } from "./api";

export type JobTone = "ok" | "working" | "busy" | "waiting" | "failed";

export type JobStatus = { text: string; tone: JobTone };

export type JobAction = {
  // What the button says; its accessible name adds the job's title.
  label: string;
  // "open" posts /api/jobs/open; "show" goes back to the job this window has
  // open, which needs no request; "none" is a disabled button.
  kind: "open" | "show" | "none";
  // Why the button is disabled, shown under it.
  reason: string | null;
};

const STAGE_NAMES: Record<StageKey, string> = {
  file: "Video",
  segments: "Scenes",
  scenes: "Previews",
  render: "Render",
  result: "Result",
};

const WORKING: Partial<Record<WizardStep, string>> = {
  sampling: "Finding scenes",
  proposing: "Finding scenes",
  analyzing: "Making a preview",
  rendering: "Rendering",
};

// Steps before a job has a video: the wizard's own first screens.
const BEFORE_JOB: WizardStep[] = ["intro", "llm_choice", "local_llm_stub", "setup", "pick_file"];

/** A listed job whose files could not be read: no step, and its reason. */
export function unreadable(entry: JobEntry): boolean {
  return entry.step === null && Boolean(entry.error);
}

/**
 * The badge for one job. `pct` is the open job's progress, when a run
 * works on it in this window.
 */
export function jobStatus(entry: JobEntry, pct?: number | null): JobStatus {
  // The backend still counts a job ready to render as on its previews
  // (every scene approved), but what waits for the editor is the render.
  const stage = entry.step === "ready_to_render" ? "Render" : entry.stage ? STAGE_NAMES[entry.stage] : null;
  if (entry.status === "running") {
    const doing = (entry.step && WORKING[entry.step]) || "Working on it";
    const shown = typeof pct === "number" && Number.isFinite(pct) ? ` · ${Math.round(pct)}%` : "";
    return { text: `${doing}${shown}`, tone: "working" };
  }
  if (entry.status === "running_elsewhere") {
    return { text: "Busy in another window", tone: "busy" };
  }
  if (unreadable(entry)) return { text: "Can't be read", tone: "failed" };
  if (entry.interrupted) {
    return { text: `${stage ?? "Work"} · stopped partway`, tone: "waiting" };
  }
  if (entry.status === "result") return { text: "Result ready", tone: "ok" };
  if (entry.status === "failed") return { text: "Didn't finish", tone: "failed" };
  return { text: `${stage ?? "Not started"} · waiting for you`, tone: "waiting" };
}

/**
 * The job's button. `busyHere` is true while a run works in this window:
 * the server then refuses to switch jobs, so other jobs wait.
 */
export function jobAction(entry: JobEntry, busyHere: boolean): JobAction {
  if (entry.open) {
    const watching = entry.status === "running" || busyHere;
    return { label: watching ? "Watch progress" : "Back to this job", kind: "show", reason: null };
  }
  if (entry.status === "running_elsewhere") {
    return {
      label: "Open",
      kind: "none",
      reason: "Another Super Processor window is working on it. You can open it here once that finishes.",
    };
  }
  if (unreadable(entry)) {
    return { label: "Open", kind: "none", reason: "Its files could not be read." };
  }
  if (busyHere || entry.status === "running") {
    return {
      label: "Open",
      kind: "none",
      reason: "Wait for the current run to finish, or stop it, before opening another job.",
    };
  }
  // Opening never starts work ("Resume" is the GPU or Gemini restart on an
  // interrupted job), so a job that waits for the editor says "Open" too.
  return { label: "Open", kind: "open", reason: null };
}

/** Most recently changed first; jobs with no time last, then by id, so the order is stable. */
export function sortJobs(jobs: JobEntry[]): JobEntry[] {
  return [...jobs].sort((a, b) => {
    const left = a.updated_at ?? -Infinity;
    const right = b.updated_at ?? -Infinity;
    if (left !== right) return right - left;
    return a.job_id < b.job_id ? -1 : a.job_id > b.job_id ? 1 : 0;
  });
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** "Changed 5 min ago", from epoch seconds; `nowMs` is Date.now(). */
export function changedText(updatedAt: number | null, nowMs: number): string {
  if (updatedAt === null || !Number.isFinite(updatedAt)) return "";
  const seconds = Math.max(0, nowMs / 1000 - updatedAt);
  if (seconds < 60) return "Changed just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `Changed ${minutes} min ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `Changed ${hours} h ago`;
  const then = new Date(updatedAt * 1000);
  const now = new Date(nowMs);
  const yesterday = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1);
  if (
    then.getFullYear() === yesterday.getFullYear() &&
    then.getMonth() === yesterday.getMonth() &&
    then.getDate() === yesterday.getDate()
  ) {
    return "Changed yesterday";
  }
  const day = `${then.getDate()} ${MONTHS[then.getMonth()]}`;
  return then.getFullYear() === now.getFullYear()
    ? `Changed ${day}`
    : `Changed ${day} ${then.getFullYear()}`;
}

/** The small line under a job's title: its file name and when it changed. */
export function jobMeta(entry: JobEntry, nowMs: number): string {
  const parts = [entry.file_name && entry.file_name !== entry.title ? entry.file_name : null];
  parts.push(changedText(entry.updated_at, nowMs) || null);
  return parts.filter(Boolean).join(" · ");
}

/** True while a run works in this window, so no other job can be opened. */
export function busyHere(view: Pick<WizardView, "busy">, jobs: JobEntry[]): boolean {
  return Boolean(view.busy) || jobs.some((entry) => entry.status === "running");
}

/** The page opens on the job list when it would show the file pick and jobs exist. */
export function startsOnJobs(
  view: Pick<WizardView, "step" | "busy" | "error" | "stopped">,
  jobs: JobEntry[],
): boolean {
  // An error or a stop note on the pick screen is news the editor must see
  // there first.
  return (
    view.step === "pick_file" && !view.busy && !view.error && !view.stopped && jobs.length > 0
  );
}

/**
 * How "Restore a new video" gets to the file pick: "close" when the wizard
 * is there (or before it), "new" to leave the open job as it is on disk and
 * start a new one (POST /api/new, which lands on the file pick, or on setup
 * when there is no Gemini key), else "blocked" while a run works.
 */
export function newVideoRoute(view: Pick<WizardView, "step" | "busy">): "close" | "new" | "blocked" {
  if (BEFORE_JOB.includes(view.step)) return "close";
  if (!view.busy) return "new";
  return "blocked";
}
