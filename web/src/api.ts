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
};

export type Progress = {
  phase: string;
  pct: number;
  message: string;
};

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

// This run's token from the last state the wizard sent. Posts and ?run=1
// requests must carry it; the wizard refuses them otherwise.
let wizardToken = "";

function remember(view: WizardView): WizardView {
  if (typeof view.token === "string" && view.token) {
    wizardToken = view.token;
  }
  return view;
}

function isWizardView(value: unknown): value is WizardView {
  if (!value || typeof value !== "object") return false;
  const step = (value as { step?: unknown }).step;
  return typeof step === "string" && STEPS.includes(step);
}

async function readView(response: Response): Promise<WizardView> {
  const data: unknown = await response.json();
  if (!response.ok) {
    const message =
      data && typeof data === "object" && "error" in data
        ? String((data as { error: unknown }).error)
        : `Request failed (${response.status})`;
    throw new Error(message);
  }
  if (!isWizardView(data)) {
    throw new Error("The server did not return a wizard state.");
  }
  return remember(data);
}

export async function loadState(origin: string): Promise<WizardView | null> {
  try {
    const response = await fetch(`${origin}/api/state`, {
      headers: { Accept: "application/json" },
      signal: AbortSignal.timeout(2500),
    });
    if (!response.ok) return null;
    const data: unknown = await response.json();
    return isWizardView(data) ? remember(data) : null;
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
      "X-Wizard-Token": wizardToken,
    },
    body: JSON.stringify(fields),
  });
  return readView(response);
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
    xhr.setRequestHeader("X-Wizard-Token", wizardToken);
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
        resolve(remember(data));
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

const inFlight = new Map<string, Promise<WizardView>>();

const POLL_MS = 2000;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

type Route = "render" | "segment" | "analyze";

async function runUntilDone(
  origin: string,
  route: Route,
  steps: readonly WizardView["step"][],
  onProgress: (view: WizardView) => void,
): Promise<WizardView> {
  // The server starts the long pass on a worker thread and answers at once.
  // Poll the state, reporting progress, until the wizard leaves those steps.
  // Sampling goes on to proposing in the same worker, so both count.
  let view = await readView(
    await fetch(`${origin}/api/${route}?run=1`, {
      headers: { Accept: "application/json", "X-Wizard-Token": wizardToken },
    }),
  );
  while (steps.includes(view.step)) {
    onProgress(view);
    await sleep(POLL_MS);
    view = await readView(
      await fetch(`${origin}/api/state`, {
        headers: { Accept: "application/json" },
      }),
    );
  }
  return view;
}

function once(
  origin: string,
  route: Route,
  steps: readonly WizardView["step"][],
  onProgress: (view: WizardView) => void,
): Promise<WizardView> {
  const running = inFlight.get(route);
  if (running) return running;
  const next = runUntilDone(origin, route, steps, onProgress).finally(() => {
    inFlight.delete(route);
  });
  inFlight.set(route, next);
  return next;
}

export type Runner = (
  origin: string,
  onProgress: (view: WizardView) => void,
) => Promise<WizardView>;

export const runRender: Runner = (origin, onProgress) =>
  once(origin, "render", ["rendering"], onProgress);

export const runSegmenting: Runner = (origin, onProgress) =>
  once(origin, "segment", ["sampling", "proposing"], onProgress);

export const runAnalyze: Runner = (origin, onProgress) =>
  once(origin, "analyze", ["analyzing"], onProgress);

export function outputSrc(origin: string, outputUrl: string | null): string | null {
  if (!outputUrl) return null;
  if (outputUrl.startsWith("http://") || outputUrl.startsWith("https://")) {
    return outputUrl;
  }
  return `${origin}${outputUrl}`;
}
