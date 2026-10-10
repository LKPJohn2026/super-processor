export type WizardStep =
  | "intro"
  | "llm_choice"
  | "local_llm_stub"
  | "setup"
  | "pick_file"
  | "analyzing"
  | "overview"
  | "choose_split"
  | "enhance"
  | "rendering"
  | "result"
  | "done";

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
};

const STEPS: readonly string[] = [
  "intro",
  "llm_choice",
  "local_llm_stub",
  "setup",
  "pick_file",
  "analyzing",
  "overview",
  "choose_split",
  "enhance",
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
  return data;
}

export async function loadState(origin: string): Promise<WizardView | null> {
  try {
    const response = await fetch(`${origin}/api/state`, {
      headers: { Accept: "application/json" },
      signal: AbortSignal.timeout(2500),
    });
    if (!response.ok) return null;
    const data: unknown = await response.json();
    return isWizardView(data) ? data : null;
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
    },
    body: JSON.stringify(fields),
  });
  return readView(response);
}

let renderInFlight: Promise<WizardView> | null = null;

const RENDER_POLL_MS = 2000;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function renderUntilDone(origin: string): Promise<WizardView> {
  // The server starts the GPU pass on a worker thread and answers at once.
  // Poll the state until the wizard leaves the rendering step.
  let view = await readView(
    await fetch(`${origin}/api/render?run=1`, {
      headers: { Accept: "application/json" },
    }),
  );
  while (view.step === "rendering") {
    await sleep(RENDER_POLL_MS);
    view = await readView(
      await fetch(`${origin}/api/state`, {
        headers: { Accept: "application/json" },
      }),
    );
  }
  return view;
}

export function runRender(origin: string): Promise<WizardView> {
  if (renderInFlight) return renderInFlight;
  renderInFlight = renderUntilDone(origin).finally(() => {
    renderInFlight = null;
  });
  return renderInFlight;
}

export function outputSrc(origin: string, outputUrl: string | null): string | null {
  if (!outputUrl) return null;
  if (outputUrl.startsWith("http://") || outputUrl.startsWith("https://")) {
    return outputUrl;
  }
  return `${origin}${outputUrl}`;
}
