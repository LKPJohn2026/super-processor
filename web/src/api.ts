export type WizardStep =
  | "intro"
  | "llm_choice"
  | "local_llm_stub"
  | "setup"
  | "pick_file"
  | "sampling"
  | "proposing"
  | "segments"
  | "analyzing"
  | "preview_segment"
  | "ready_to_render"
  | "rendering"
  | "result"
  | "done";

export type Segment = {
  start_s: number;
  end_s: number;
  label: string;
  note: string;
  contains: string[];
  still_url?: string | null;
  // Its place in the list, as edits by hand name it.
  index?: number;
};

export type SegmentList = {
  segments: Segment[];
  error: string | null;
  note: string;
  // The output scale Gemini chose watching the whole video, and why.
  scale?: number | null;
  scale_reason?: string;
  // Why Gemini worked from stills, when it could not watch the video.
  video_problem?: string;
  // Sent back with each change by hand (and approve), so a page showing an
  // older list is refused instead of changing scenes it did not show.
  revision?: number;
  approved?: boolean;
  // The last scene's end, which is the video's length.
  duration_s?: number | null;
  // The limits a change by hand must keep (3 s, 10 scenes).
  min_segment_s?: number;
  max_segments?: number;
};

export type Progress = {
  phase: string;
  // The running step's own percentage.
  pct: number;
  message: string;
  // The whole job's percentage.
  overall?: number;
  // Seconds left in the running task, when its speed can be measured.
  eta_s?: number | null;
  elapsed_s?: number;
  updated_at?: number;
  // A friendly hint while the step runs, such as undervolting a hot GPU.
  note?: string;
};

/** `42 s`, `3 min 05 s`, or `1 h 07 min`, as the server says it. */
export function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  const pad = (value: number) => String(value).padStart(2, "0");
  if (hours) return `${hours} h ${pad(minutes)} min`;
  if (minutes) return `${minutes} min ${pad(secs)} s`;
  return `${secs} s`;
}

/** Time left, counted down `sinceS` seconds after the view arrived. */
export function progressTiming(progress: Progress, sinceS = 0): string {
  if (typeof progress.eta_s === "number") {
    return `about ${formatDuration(progress.eta_s - sinceS)} left`;
  }
  if (typeof progress.elapsed_s === "number") {
    return `${formatDuration(progress.elapsed_s + sinceS)} so far`;
  }
  return "";
}

/** What the whole video becomes, said on each segment's preview. */
export function outputSizeLine(scale: number): string {
  if (scale === 1) return "The whole video keeps its size; these previews show the clean-up.";
  return `The whole video is upscaled ${scale}×.`;
}

/** Gemini's output scale and its reason, as one line. */
export function scaleChoiceText(scale: number | null | undefined, reason?: string): string {
  if (typeof scale !== "number") return "";
  const what = scale === 1 ? "keep the size (no upscaling)" : `upscale ${scale}×`;
  return `Gemini chose to ${what}${reason ? `: ${reason}` : "."}`;
}

export type ShotLook = Record<
  "deblock" | "denoise" | "contrast" | "brightness" | "saturation" | "gamma" | "grain",
  number
>;

export type ShotCheck = {
  ok: boolean | null;
  problems: string[];
  note: string;
  adjusted?: boolean;
};

export type Region = {
  kind: "faces" | "hands" | "text";
  x0: number;
  y0: number;
  x1: number;
  y1: number;
};

export type LookShot = {
  index: number;
  start_s: number;
  end_s: number;
  label: string;
  contains: string[];
  regions?: Region[];
  strength: number;
  use_upscale?: boolean;
  upscale_override?: boolean | null;
  look: ShotLook;
  reason: string;
  check: ShotCheck;
  shimmer?: number | null;
  before_url: string | null;
  after_url: string | null;
};

export type Looks = {
  scale: number | null;
  error: string | null;
  shots: LookShot[];
  // On the ready screen: 1 is "do not upscale".
  scales?: number[];
  source_size?: [number, number];
  // Gemini's reason while its scale stands.
  scale_reason?: string;
};

export type Navigation = { setup: boolean; file: boolean; segments: boolean };

// The next scene's preview made in the background (``--prefetch`` only).
export type PrefetchNext = { index: number; ready: boolean; running: boolean };

export type SegmentPreview = {
  index: number;
  count: number;
  scale: number | null;
  error: string | null;
  shot: LookShot | null;
  next?: PrefetchNext | null;
};

export type RunKind = "segmenting" | "analyze" | "render";
export type RunOutcome = "running" | "done" | "failed" | "stopped";

/** The run in progress, else the last one, with server times in epoch seconds. */
export type RunRecord = {
  id: string;
  kind: RunKind;
  job_id: string | null;
  started_at: number;
  finished_at: number | null;
  outcome: RunOutcome;
};

export type StageKey = "file" | "segments" | "scenes" | "render" | "result";
export type StageStatus = "done" | "current" | "pending";

export type JobScene = {
  index: number;
  start_s: number;
  end_s: number;
  label: string;
  still_url: string | null;
  status: "approved" | "current" | "pending";
  // Previewed as it is now, so reopening it needs no new analysis.
  previewed: boolean;
  // POST /api/scene may open it now.
  reopenable: boolean;
};

/** The open job, for the stepper and the scene strip. */
export type JobSummary = {
  job_id: string;
  title: string;
  stage: StageKey | null;
  stages: { key: StageKey; status: StageStatus }[];
  scene_count: number;
  approved_count: number;
  current_index: number | null;
  scenes: JobScene[];
  error: string | null;
};

/** One kept result of the job, oldest first. */
export type OutputVersion = {
  number: number;
  name: string;
  url: string;
  kind: "render" | "earlier" | "recovered" | string;
  created_at: string | null;
  note: string | null;
  size: number;
  scale: number | null;
  strength: number | null;
  strength_summary: string | null;
  // The version the job's output.mp4 is now.
  current: boolean;
};

/** One row of GET /api/jobs. */
export type JobEntry = {
  job_id: string;
  title: string;
  file_name: string | null;
  step: WizardStep | null;
  stage: StageKey | null;
  status: "running" | "running_elsewhere" | "result" | "failed" | "paused" | string;
  // Epoch seconds.
  updated_at: number | null;
  // Left on a run step: opening it asks Resume or Stop.
  resumes_work: boolean;
  interrupted: boolean;
  // The job this window has open.
  open: boolean;
  thumb_url: string | null;
  output_url: string | null;
  error: string | null;
};

export type WizardView = {
  step: WizardStep;
  busy?: boolean;
  error: string | null;
  job_id: string | null;
  has_gemini_key: boolean;
  output_ready: boolean;
  output_url: string | null;
  scale: number | null;
  strength: number | null;
  // Every span's strength, for the Result line; `strength` is the latest span's.
  strength_summary?: string | null;
  progress?: Progress | null;
  navigation?: Navigation;
  segment_index?: number;
  segments?: SegmentList | null;
  segment_preview?: SegmentPreview | null;
  looks?: Looks | null;
  seams?: { time_s: number; score: number }[];
  seam_error?: string | null;
  // Left mid-step (a restart, or a job reopened): its work waits for a POST
  // to /resume, or /cancel to stop it; ?run=1 does not start it.
  interrupted?: boolean;
  running_elsewhere?: boolean;
  // A stop can be asked for now (POST /api/cancel).
  cancellable?: boolean;
  // A stop was asked for and the run has not landed it yet ("Stopping…").
  cancel_requested?: boolean;
  // When the editor last stopped a run on this job (ISO time), else null.
  stopped?: string | null;
  run?: RunRecord | null;
  job?: JobSummary | null;
  // Every kept result of this job, oldest first (result and done only).
  versions?: OutputVersion[];
  token?: string;
};

const STEPS: readonly string[] = [
  "intro",
  "llm_choice",
  "local_llm_stub",
  "setup",
  "pick_file",
  "sampling",
  "proposing",
  "segments",
  "analyzing",
  "preview_segment",
  "ready_to_render",
  "rendering",
  "result",
  "done",
];

export function configuredOrigin(): string {
  const params = new URLSearchParams(window.location.search);
  if (params.has("api")) {
    return (params.get("api") ?? "").replace(/\/$/, "");
  }
  if (params.get("live") === "1") {
    const port = params.get("port") || "45143";
    return `http://127.0.0.1:${port}`;
  }
  const fromEnv = import.meta.env.VITE_API_ORIGIN;
  if (typeof fromEnv === "string" && fromEnv.trim()) {
    return fromEnv.trim().replace(/\/$/, "");
  }
  return "";
}

// Connections can overlap (including a failed attempt to connect elsewhere).
// Never send one server's token to another server.
const wizardTokens = new Map<string, string>();

function originKey(origin: string): string {
  return new URL(origin || window.location.origin).origin;
}

function remember(origin: string, view: WizardView): WizardView {
  if (typeof view.token === "string" && view.token) {
    wizardTokens.set(originKey(origin), view.token);
  }
  return view;
}

function tokenFor(origin: string): string {
  return wizardTokens.get(originKey(origin)) ?? "";
}

function isWizardView(value: unknown): value is WizardView {
  if (!value || typeof value !== "object") return false;
  const step = (value as { step?: unknown }).step;
  return typeof step === "string" && STEPS.includes(step);
}

class RequestError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

async function readView(origin: string, response: Response): Promise<WizardView> {
  // Proxies may return HTML (or no body) on an HTTP failure.
  if (!response.ok) {
    const data: unknown = await response.json().catch(() => null);
    const message =
      data && typeof data === "object" && "error" in data
        ? String((data as { error: unknown }).error)
        : `Request failed (${response.status})`;
    throw new RequestError(message, response.status);
  }
  const data: unknown = await response.json();
  if (!isWizardView(data)) {
    throw new Error("The server did not return a wizard state.");
  }
  return remember(origin, data);
}

export async function loadState(origin: string): Promise<WizardView | null> {
  try {
    const response = await fetch(`${origin}/api/state`, {
      headers: { Accept: "application/json" },
      signal: AbortSignal.timeout(2500),
    });
    if (!response.ok) return null;
    const data: unknown = await response.json();
    return isWizardView(data) ? remember(origin, data) : null;
  } catch {
    return null;
  }
}

export async function postAction(
  origin: string,
  action: string,
  fields: Record<string, string>,
): Promise<WizardView> {
  const response = await fetch(`${origin}/api/${action}`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
      "X-Wizard-Token": tokenFor(origin),
    },
    body: JSON.stringify(fields),
  });
  return readView(origin, response);
}

export function uploadVideo(
  origin: string,
  file: File,
  onProgress: (pct: number) => void,
): Promise<WizardView> {
  // fetch() cannot report upload progress, so this one uses XHR. The body is
  // the file itself; the wizard streams it to disk and picks it.
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${origin}/api/upload`);
    xhr.setRequestHeader("X-Wizard-Token", tokenFor(origin));
    xhr.setRequestHeader("X-File-Name", encodeURIComponent(file.name));
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(Math.round((event.loaded / event.total) * 100));
    };
    xhr.onload = () => {
      let data: unknown = null;
      try {
        data = JSON.parse(xhr.responseText);
      } catch {
        // fall through to the status message
      }
      if (xhr.status === 200 && isWizardView(data)) {
        resolve(remember(origin, data));
        return;
      }
      const error =
        data && typeof data === "object" && "error" in data
          ? String((data as { error: unknown }).error)
          : `The upload failed (${xhr.status}).`;
      reject(new Error(error));
    };
    xhr.onerror = () => reject(new Error("The upload failed."));
    xhr.send(file);
  });
}

type Operation = {
  promise: Promise<WizardView>;
  subscribers: Set<(view: WizardView) => void>;
  latest?: WizardView;
};

const inFlight = new Map<string, Operation>();
const POLL_MS = 2000;
const REQUEST_TIMEOUT_MS = 10000;
const RETRY_DELAYS_MS = [500, 1000, 2000];

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

type Route = "render" | "segment" | "analyze";

async function requestWithRetry(origin: string, path: string): Promise<WizardView> {
  for (let attempt = 0; ; attempt += 1) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      return await readView(
        origin,
        await fetch(`${origin}${path}`, {
          headers: { Accept: "application/json", "X-Wizard-Token": tokenFor(origin) },
          signal: controller.signal,
        }),
      );
    } catch (error) {
      const transient =
        controller.signal.aborted || error instanceof TypeError ||
        (error instanceof RequestError &&
          (error.status === 408 || error.status === 429 || error.status >= 500));
      if (!transient || attempt >= RETRY_DELAYS_MS.length) throw error;
    } finally {
      // Also bounds response body reads, not just receiving headers.
      clearTimeout(timer);
    }
    await sleep(RETRY_DELAYS_MS[attempt]);
  }
}

async function runUntilDone(
  origin: string,
  route: Route,
  steps: readonly WizardView["step"][],
  onProgress: (view: WizardView) => void,
): Promise<WizardView> {
  // These starts are idempotent: the backend checks its step and worker lock.
  // Sampling goes on to proposing in the same worker, so both count.
  const start = `/api/${route}?run=1`;
  let view = await requestWithRetry(origin, start);
  let idlePolls = 0;
  // An interrupted step never starts: hand it back so the page can ask.
  while (steps.includes(view.step) && !handsBack(view)) {
    onProgress(view);
    await sleep(POLL_MS);
    view = await requestWithRetry(origin, "/api/state");
    idlePolls = isIdleOnStep(view, steps) ? idlePolls + 1 : 0;
    if (idlePolls >= IDLE_POLLS_BEFORE_RESTART) {
      // The step still waits but nothing works on it (say the worker that
      // ran the step before ended, and none started this one): start it.
      // Idempotent like the first start; an interrupted step never gets here.
      idlePolls = 0;
      view = await requestWithRetry(origin, start);
    }
  }
  return view;
}

// Polls in a row showing a run step that nothing works on before ?run=1 is
// sent again.
const IDLE_POLLS_BEFORE_RESTART = 2;

/** The page must choose (Resume or Stop) or wait on another window: never run it. */
export function handsBack(view: WizardView): boolean {
  return Boolean(view.interrupted || view.running_elsewhere);
}

/**
 * True when the view is on one of `steps` with no worker running and nothing
 * stopping it from starting. An older server that says nothing of `busy`
 * never counts as idle.
 */
export function isIdleOnStep(view: WizardView, steps: readonly WizardView["step"][]): boolean {
  return steps.includes(view.step) && view.busy === false && !handsBack(view);
}

function once(
  origin: string,
  route: Route,
  steps: readonly WizardView["step"][],
  onProgress: (view: WizardView) => void,
  signal?: AbortSignal,
): Promise<WizardView> {
  // Canonical origin includes the port. Neither callbacks nor tokens cross servers.
  const key = `${originKey(origin)}:${route}`;
  let operation = inFlight.get(key);
  if (!operation) {
    const subscribers = new Set<(view: WizardView) => void>();
    const created: Operation = {
      subscribers,
      promise: runUntilDone(origin.replace(/\/$/, ""), route, steps, (view) => {
        created.latest = view;
        for (const subscriber of subscribers) subscriber(view);
      }).finally(() => {
        inFlight.delete(key);
        subscribers.clear();
      }),
    };
    operation = created;
    inFlight.set(key, operation);
  }
  const { subscribers } = operation;
  const unsubscribe = () => subscribers.delete(onProgress);
  if (!signal?.aborted) {
    subscribers.add(onProgress);
    if (operation.latest) onProgress(operation.latest);
    signal?.addEventListener("abort", unsubscribe, { once: true });
  }
  // Unsubscribing an effect must not stop the server's work or its shared poller.
  return operation.promise.finally(() => {
    unsubscribe();
    signal?.removeEventListener("abort", unsubscribe);
  });
}

export type Runner = (
  origin: string,
  onProgress: (view: WizardView) => void,
  signal?: AbortSignal,
) => Promise<WizardView>;

export const runRender: Runner = (origin, onProgress, signal) =>
  once(origin, "render", ["rendering"], onProgress, signal);

export const runSegmenting: Runner = (origin, onProgress, signal) =>
  once(origin, "segment", ["sampling", "proposing"], onProgress, signal);

export const runAnalyze: Runner = (origin, onProgress, signal) =>
  once(origin, "analyze", ["analyzing"], onProgress, signal);

export function outputSrc(origin: string, outputUrl: string | null): string | null {
  if (!outputUrl) return null;
  if (outputUrl.startsWith("http://") || outputUrl.startsWith("https://")) {
    return outputUrl;
  }
  return `${origin}${outputUrl}`;
}

/** `fields` plus the job the page shows, so a stale tab cannot act on another job. */
export function withJob(
  view: Pick<WizardView, "job_id">,
  fields: Record<string, string> = {},
): Record<string, string> {
  return view.job_id && !("job_id" in fields) ? { ...fields, job_id: view.job_id } : { ...fields };
}

/**
 * The body of a stop. `run_id` names the run the page shows working, so a
 * late click never stops a newer run; a step left interrupted has no run
 * working, and the server refuses a `run_id` then, so none is sent.
 */
export function cancelFields(
  view: Pick<WizardView, "job_id" | "busy" | "run">,
): Record<string, string> {
  const fields = withJob(view);
  if (view.busy && view.run && view.run.outcome === "running") fields.run_id = view.run.id;
  return fields;
}

/** Stop the scan or render in progress, or one left interrupted. */
export function cancelRun(origin: string, view: WizardView): Promise<WizardView> {
  return postAction(origin, "cancel", cancelFields(view));
}

/** Let an interrupted step's work start; the runner then starts it with ?run=1. */
export function resumeRun(origin: string, view: WizardView): Promise<WizardView> {
  return postAction(origin, "resume", withJob(view));
}

/** Open an approved scene (or the next to approve) again, by its index. */
export function reopenScene(origin: string, view: WizardView, index: number): Promise<WizardView> {
  return postAction(origin, "scene", withJob(view, { index: String(index) }));
}

export type SegmentEdit =
  | { action: "move"; index: number; at: number }
  | { action: "split"; index: number; at: number }
  | { action: "merge"; index: number }
  | { action: "approve" }
  | { action: "note"; note: string };

/**
 * The body of a change to the segments. `revision` is the list the page
 * shows; a change by hand needs it, and the server refuses an older one.
 * Cut times go to the millisecond, as the server saves them.
 */
export function segmentFields(view: WizardView, edit: SegmentEdit): Record<string, string> {
  const fields: Record<string, string> = { action: edit.action };
  if (edit.action === "move" || edit.action === "split" || edit.action === "merge") {
    fields.index = String(edit.index);
  }
  if (edit.action === "move" || edit.action === "split") {
    fields.at = String(Math.round(edit.at * 1000) / 1000);
  }
  if (edit.action === "note") fields.note = edit.note;
  const revision = view.segments?.revision;
  if (typeof revision === "number") fields.revision = String(revision);
  return withJob(view, fields);
}

export function editSegments(
  origin: string,
  view: WizardView,
  edit: SegmentEdit,
): Promise<WizardView> {
  return postAction(origin, "segments", segmentFields(view, edit));
}

function isJobEntry(value: unknown): value is JobEntry {
  return (
    !!value &&
    typeof value === "object" &&
    typeof (value as { job_id?: unknown }).job_id === "string"
  );
}

/** Every job, most recently changed first; throws when the server cannot say. */
export async function loadJobs(origin: string): Promise<JobEntry[]> {
  const response = await fetch(`${origin}/api/jobs`, {
    headers: { Accept: "application/json" },
    signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
  });
  if (!response.ok) {
    const data: unknown = await response.json().catch(() => null);
    throw new Error(
      data && typeof data === "object" && "error" in data
        ? String((data as { error: unknown }).error)
        : `Could not list the jobs (${response.status}).`,
    );
  }
  const data: unknown = await response.json();
  const jobs = data && typeof data === "object" ? (data as { jobs?: unknown }).jobs : null;
  if (!Array.isArray(jobs)) throw new Error("The server did not return a job list.");
  return jobs.filter(isJobEntry);
}

/**
 * Show another job in this window, where it was left. A job left mid-step
 * opens interrupted; nothing runs until Resume.
 */
export function openJob(origin: string, jobId: string): Promise<WizardView> {
  // `job` is the one to open; any window may switch jobs, so no job_id.
  return postAction(origin, "jobs/open", { job: jobId });
}
