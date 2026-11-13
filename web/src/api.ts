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
  if (scale > 2) {
    return `The whole video is upscaled ${scale}×; previews are 2×, which shows the same restoration at a size any GPU fits.`;
  }
  return `The whole video is upscaled ${scale}×.`;
}

/** Gemini's output scale and its reason, as one line. */
export function scaleChoiceText(scale: number | null | undefined, reason?: string): string {
  if (typeof scale !== "number") return "";
  const what = scale === 1 ? "keep the size (no upscaling)" : `upscale ${scale}×`;
  return `Gemini chose to ${what}${reason ? `: ${reason}` : "."}`;
}

/** Say why a large source is not offered FlashVSR. */
export function sizeNote(looks: Looks): string {
  const limit = looks.max_upscale_long_edge;
  if (!looks.large_source || !limit) return "";
  return (
    `Upscaling is offered only for videos up to ${limit} px on the long edge; ` +
    "larger ones would not fit the GPU and take many hours. This one keeps its size."
  );
}

/** Say so when a small source could go to 4× on a larger GPU. */
export function vramNote(looks: Looks): string {
  const vram = looks.gpu_vram_gb;
  if (looks.scales?.includes(4) || !looks.small_source || typeof vram !== "number") return "";
  const have = vram ? `this one has ${vram} GB` : "none was found";
  return `4× needs an NVIDIA GPU with at least 24 GB of memory; ${have}.`;
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
  // Gemini's reason while its scale stands, and what keeps 4× off the list.
  scale_reason?: string;
  gpu_vram_gb?: number;
  small_source?: boolean;
  large_source?: boolean;
  max_upscale_long_edge?: number;
};

export type Navigation = { setup: boolean; file: boolean; segments: boolean };

export type SegmentPreview = {
  index: number;
  count: number;
  scale: number | null;
  error: string | null;
  shot: LookShot | null;
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
  let view = await requestWithRetry(origin, `/api/${route}?run=1`);
  while (steps.includes(view.step)) {
    onProgress(view);
    await sleep(POLL_MS);
    view = await requestWithRetry(origin, "/api/state");
  }
  return view;
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
