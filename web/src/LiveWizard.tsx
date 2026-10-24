import { useEffect, useState, type DragEvent, type FormEvent, type ReactNode } from "react";
import {
  outputSrc,
  postAction,
  runAnalyze,
  runRender,
  runSegmenting,
  type LookShot,
  type Looks,
  type Progress,
  type Region,
  type Runner,
  type Segment,
  type SegmentList,
  type Navigation,
  type SegmentPreview,
  uploadVideo,
  type WizardView,
} from "./api";

const RAIL = ["Setup", "File", "Segments", "Previews", "Result"] as const;
// Rail steps the editor can go back (or forward) to, by their index.
const NAV_TARGETS: Record<number, keyof Navigation> = { 0: "setup", 1: "file", 2: "segments" };
const BACK_TEXT: Record<keyof Navigation, string> = {
  setup: "Back to Gemini setup",
  file: "Choose another file",
  segments: "Back to the segments",
};

const RUNNERS: Partial<Record<WizardView["step"], Runner>> = {
  sampling: runSegmenting,
  proposing: runSegmenting,
  analyzing: runAnalyze,
  rendering: runRender,
};

function railIndex(step: WizardView["step"]): number {
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

type LiveWizardProps = {
  origin: string;
  view: WizardView;
  onView: (view: WizardView) => void;
  onDisconnect: () => void;
};

export function LiveWizard({ origin, view, onView, onDisconnect }: LiveWizardProps) {
  const [busy, setBusy] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const active = railIndex(view.step);

  async function send(action: string, fields: Record<string, string>) {
    setBusy(true);
    setLocalError(null);
    try {
      onView(await postAction(origin, action, fields));
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : "Request failed");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    const run = RUNNERS[view.step];
    if (!run) return;
    let cancelled = false;
    setLocalError(null);
    // Each poll while the worker runs carries its progress.
    const progress = (next: WizardView) => {
      if (!cancelled) onView(next);
    };
    run(origin, progress)
      .then((next) => {
        if (!cancelled) onView(next);
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setLocalError(error instanceof Error ? error.message : "The step failed");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [origin, view.step, onView]);

  const message = localError || view.error;

  return (
    <>
      <p className="brand">Super Processor</p>
      <ol className="steps" aria-label="Progress">
        {RAIL.map((label, index) => {
          const target = NAV_TARGETS[index];
          const reachable = target !== undefined && view.navigation?.[target] && !busy;
          return (
            <li key={label} className={index === active ? "active" : undefined}>
              {reachable ? (
                <button
                  type="button"
                  className="step-link"
                  onClick={() => void send("goto", { to: target })}
                >
                  {label}
                </button>
              ) : (
                label
              )}
            </li>
          );
        })}
      </ol>
      <div className="panel" aria-busy={busy}>
        <WizardStepView
          origin={origin}
          view={view}
          busy={busy}
          message={message}
          onSend={send}
          onView={onView}
        />
        <div className="actions">
          <button type="button" className="secondary" onClick={onDisconnect}>
            Disconnect
          </button>
        </div>
      </div>
    </>
  );
}

function WizardStepView({
  origin,
  view,
  busy,
  message,
  onSend,
  onView,
}: {
  origin: string;
  view: WizardView;
  busy: boolean;
  message: string | null;
  onSend: (action: string, fields: Record<string, string>) => Promise<void>;
  onView: (view: WizardView) => void;
}) {
  if (view.step === "intro") {
    return (
      <>
        <h1>Super Processor</h1>
        <p className="lede">
          Enhance the video you already shot. A local GPU model restores and
          upscales it. You choose scale and strength. FFmpeg only trims,
          splices, and copies audio.
        </p>
        <button type="button" disabled={busy} onClick={() => void onSend("intro", {})}>
          Continue
        </button>
        {message ? <Alert text={message} /> : null}
      </>
    );
  }

  if (view.step === "llm_choice") {
    return (
      <>
        <h1>What type of LLM do you have now?</h1>
        <p className="lede">
          Pick how diagnosis should run. Media stays on this machine except
          for Gemini API calls you authorize.
        </p>
        {message ? <Alert text={message} /> : null}
        <button
          type="button"
          className="choice"
          disabled={busy}
          onClick={() => void onSend("llm", { choice: "gemini" })}
        >
          <strong>A. I don’t have a local LLM</strong>
          <span className="muted">Use a Google AI Studio (Gemini) API key</span>
        </button>
        <button
          type="button"
          className="choice"
          disabled={busy}
          onClick={() => void onSend("llm", { choice: "local" })}
        >
          <strong>B. I have a local LLM</strong>
          <span className="muted">Configure a local model (coming later)</span>
        </button>
      </>
    );
  }

  if (view.step === "local_llm_stub") {
    return (
      <>
        <h1>Local LLM setup</h1>
        <p className="lede">
          Local models are not wired yet. Continue with Gemini to finish a job
          today.
        </p>
        <ol>
          <li>Choose a local runtime</li>
          <li>Pull a multimodal model that accepts image frames</li>
          <li>Point Super Processor at that endpoint</li>
        </ol>
        <button
          type="button"
          disabled={busy}
          onClick={() => void onSend("local-llm", { action: "gemini" })}
        >
          Use Gemini instead
        </button>
      </>
    );
  }

  if (view.step === "setup") {
    return <SetupStep busy={busy} hasKey={view.has_gemini_key} message={message} onSend={onSend} />;
  }

  if (view.step === "pick_file") {
    return (
      <PickStep
        origin={origin}
        view={view}
        busy={busy}
        message={message}
        onSend={onSend}
        onView={onView}
      />
    );
  }

  if (view.step === "sampling" || view.step === "proposing") {
    return (
      <Busy
        title="Finding the segments"
        progress={view.progress ?? null}
        message={message}
      >
        FFmpeg saves a small still about every second, on this computer.
        Gemini looks at them and splits the video where the light, place, or
        picture quality changes.
      </Busy>
    );
  }

  if (view.step === "segments") {
    return (
      <SegmentsStep
        origin={origin}
        list={view.segments ?? { segments: [], error: null, note: "" }}
        busy={busy}
        message={message}
        onSend={onSend}
        view={view}
      />
    );
  }

  if (view.step === "analyzing") {
    const count = view.segments?.segments.length;
    return (
      <Busy
        title={`Segment ${(view.segment_index ?? 0) + 1}${count ? ` of ${count}` : ""}`}
        progress={view.progress ?? null}
        message={message}
      >
        Gemini is choosing clean-up, strength, and finishing for this segment.
        A few seconds of it are previewed on the local GPU, and Gemini checks
        the preview for faces, hands, text, and texture that went wrong.
      </Busy>
    );
  }

  if (view.step === "preview_segment") {
    return (
      <PreviewStep
        origin={origin}
        preview={
          view.segment_preview ?? { index: 0, count: 0, scale: null, error: null, shot: null }
        }
        busy={busy}
        message={message}
        onSend={onSend}
        view={view}
      />
    );
  }

  if (view.step === "ready_to_render") {
    return (
      <ReadyStep
        looks={view.looks ?? { scale: null, error: null, shots: [] }}
        busy={busy}
        message={message}
        onSend={onSend}
        view={view}
      />
    );
  }

  if (view.step === "rendering") {
    return (
      <Busy title="Upscaling" progress={view.progress ?? null} message={message}>
        Restoring and upscaling on the local GPU. FFmpeg only trims, splices,
        and copies audio. This can take several minutes.
      </Busy>
    );
  }

  if (view.step === "result") {
    return <ResultStep origin={origin} view={view} busy={busy} message={message} onSend={onSend} />;
  }

  if (view.step === "done") {
    return (
      <>
        <h1>Done</h1>
        <p className="lede">
          Your enhanced file is in the job directory as <code>output.mp4</code>.
        </p>
        {view.output_ready ? (
          <video
            controls
            src={outputSrc(origin, view.output_url) ?? undefined}
            aria-label="Upscaled result"
          />
        ) : null}
        <div className="actions">
          <button
            type="button"
            className="secondary"
            disabled={busy}
            onClick={() => void onSend("new", {})}
          >
            Start a new video
          </button>
        </div>
      </>
    );
  }

  return (
    <>
      <h1>Continue in the classic wizard</h1>
      <p>
        This step ({view.step}) is served by the Python wizard page. Open the
        URL printed by <code>super-processor review</code> to finish it. This
        shell drives every step from introduction to result.
      </p>
      {message ? <Alert text={message} /> : null}
    </>
  );
}

function Busy({
  title,
  progress,
  message,
  children,
}: {
  title: string;
  progress: Progress | null;
  message: string | null;
  children: ReactNode;
}) {
  const pct = progress ? Math.max(0, Math.min(100, Math.round(progress.pct))) : null;
  return (
    <>
      <h1>{title}</h1>
      <p className="lede" role="status">
        {children}
      </p>
      {pct !== null ? (
        <p className="progress">
          <progress max={100} value={pct} aria-label={`${title}: ${pct}%`}>
            {pct}%
          </progress>{" "}
          {pct}% · {progress?.message}
        </p>
      ) : null}
      {message ? <Alert text={message} /> : null}
    </>
  );
}

function Back({
  view,
  to,
  busy,
  onSend,
}: {
  view: WizardView;
  to: (keyof Navigation)[];
  busy: boolean;
  onSend: (action: string, fields: Record<string, string>) => Promise<void>;
}) {
  const shown = to.filter((target) => view.navigation?.[target]);
  if (!shown.length) return null;
  return (
    <div className="actions back">
      {shown.map((target) => (
        <button
          key={target}
          type="button"
          className="link"
          disabled={busy}
          onClick={() => void onSend("goto", { to: target })}
        >
          ← {BACK_TEXT[target]}
        </button>
      ))}
    </div>
  );
}

function Alert({ text }: { text: string }) {
  return (
    <p className="error" role="alert">
      {text}
    </p>
  );
}

function SetupStep({
  busy,
  hasKey,
  message,
  onSend,
}: {
  busy: boolean;
  hasKey: boolean;
  message: string | null;
  onSend: (action: string, fields: Record<string, string>) => Promise<void>;
}) {
  const [apiKey, setApiKey] = useState("");

  function submit(event: FormEvent) {
    event.preventDefault();
    void onSend("setup", { api_key: apiKey });
  }

  return (
    <>
      <h1>Gemini API setup</h1>
      <p>
        {hasKey
          ? "A Gemini API key is already configured on this machine."
          : "Follow these steps, then paste your key below."}
      </p>
      <ol>
        <li>
          Open{" "}
          <a href="https://aistudio.google.com/apikey" target="_blank" rel="noreferrer">
            Google AI Studio
          </a>
        </li>
        <li>Accept the terms if prompted</li>
        <li>Create an API key and copy it</li>
        <li>Paste it here</li>
      </ol>
      {message ? <Alert text={message} /> : null}
      <form onSubmit={submit}>
        <label htmlFor="api-key">
          API key
          <input
            id="api-key"
            name="api_key"
            type="password"
            autoComplete="off"
            value={apiKey}
            onChange={(event) => setApiKey(event.target.value)}
          />
        </label>
        <div className="actions">
          <button type="submit" disabled={busy}>
            Save and continue
          </button>
          {hasKey ? (
            <button
              type="button"
              className="secondary"
              disabled={busy}
              onClick={() => void onSend("setup", { skip: "1" })}
            >
              Use existing key
            </button>
          ) : null}
        </div>
      </form>
    </>
  );
}

function PickStep({
  origin,
  view,
  busy,
  message,
  onSend,
  onView,
}: {
  origin: string;
  view: WizardView;
  busy: boolean;
  message: string | null;
  onSend: (action: string, fields: Record<string, string>) => Promise<void>;
  onView: (view: WizardView) => void;
}) {
  const [path, setPath] = useState("");
  const [over, setOver] = useState(false);
  const [upload, setUpload] = useState<{ name: string; pct: number } | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  function submit(event: FormEvent) {
    event.preventDefault();
    void onSend("pick", { path });
  }

  function send(file: File | undefined) {
    if (!file) return;
    setUploadError(null);
    setUpload({ name: file.name, pct: 0 });
    uploadVideo(origin, file, (pct) => setUpload({ name: file.name, pct }))
      .then(onView)
      .catch((error: unknown) => {
        setUploadError(error instanceof Error ? error.message : "The upload failed.");
      })
      .finally(() => setUpload(null));
  }

  function drop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setOver(false);
    send(event.dataTransfer.files[0]);
  }

  return (
    <>
      <h1>Pick a video file</h1>
      <p className="lede">
        Choose a video on this computer. Gemini splits it into segments, you
        approve a short before/after preview of each, and then you choose
        whether to upscale the whole video on your GPU.
      </p>
      <p className="muted">Up to 1080p and 30 minutes.</p>
      {message ? <Alert text={message} /> : null}
      <div className="actions">
        <button type="button" disabled={busy || upload !== null} onClick={() => void onSend("browse", {})}>
          Choose a file…
        </button>
      </div>
      <label
        className={over ? "dropzone over" : "dropzone"}
        onDragEnter={(event) => {
          event.preventDefault();
          setOver(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => setOver(false)}
        onDrop={drop}
      >
        <strong>…or drop a video here</strong>
        <span className="muted">A dropped file is copied into the jobs folder on this computer.</span>
        <input
          type="file"
          accept="video/*,.mkv,.mts,.ts"
          hidden
          disabled={busy || upload !== null}
          onChange={(event) => send(event.target.files?.[0])}
        />
        {upload ? (
          <span className="progress" role="status">
            <progress max={100} value={upload.pct} aria-label={`Copying ${upload.name}`} />{" "}
            {upload.pct}% · Copying {upload.name}
          </span>
        ) : null}
      </label>
      {uploadError ? <Alert text={uploadError} /> : null}
      <details className="fallback">
        <summary>Type a path instead</summary>
        <form onSubmit={submit}>
          <label htmlFor="video-path">
            Absolute path to video
            <input
              id="video-path"
              name="path"
              type="text"
              value={path}
              autoComplete="off"
              onChange={(event) => setPath(event.target.value)}
              placeholder="/home/you/clip.mp4"
            />
          </label>
          <div className="actions">
            <button type="submit" className="secondary" disabled={busy}>
              Use this path
            </button>
          </div>
        </form>
      </details>
      <Back view={view} to={["setup", "segments"]} busy={busy} onSend={onSend} />
    </>
  );
}

function ResultStep({
  origin,
  view,
  busy,
  message,
  onSend,
}: {
  origin: string;
  view: WizardView;
  busy: boolean;
  message: string | null;
  onSend: (action: string, fields: Record<string, string>) => Promise<void>;
}) {
  const [note, setNote] = useState("");
  const src = outputSrc(origin, view.output_url);

  function submit(event: FormEvent) {
    event.preventDefault();
    void onSend("result", { note });
  }

  return (
    <>
      <h1>Result</h1>
      {view.scale !== null && view.strength !== null ? (
        <p className="muted">
          Scale {view.scale}× · strength {view.strength.toFixed(2)}. Lower
          strength means less invented texture.
        </p>
      ) : null}
      {src ? (
        <video controls src={src} aria-label="Upscaled result">
          <p>
            Your browser cannot play this video. The file is served at{" "}
            <a href={src}>output.mp4</a>.
          </p>
        </video>
      ) : (
        <p>The result file is not ready yet.</p>
      )}
      <SeamNote seams={view.seams ?? []} error={view.seam_error ?? null} />
      <p className="lede">Are you happy, or do you want to say something?</p>
      {message ? <Alert text={message} /> : null}
      <div className="actions">
        <button
          type="button"
          disabled={busy}
          onClick={() => void onSend("result", { mood: "happy" })}
        >
          I am happy
        </button>
      </div>
      <form onSubmit={submit}>
        <label htmlFor="revise-note">
          Tell me what to change
          <textarea
            id="revise-note"
            name="note"
            rows={3}
            value={note}
            onChange={(event) => setNote(event.target.value)}
            placeholder="e.g. reduce artificial detail from 00:30 to 00:40"
          />
        </label>
        <div className="actions">
          <button type="submit" disabled={busy}>
            Revise this range
          </button>
          <button
            type="button"
            className="secondary"
            disabled={busy}
            onClick={() => void onSend("result", { note, restart: "1" })}
          >
            Start over from the segments with this note
          </button>
        </div>
      </form>
      <div className="actions">
        <button
          type="button"
          className="secondary"
          disabled={busy}
          onClick={() => void onSend("new", {})}
        >
          Start a new video
        </button>
      </div>
    </>
  );
}

export function formatTime(seconds: number): string {
  const rounded = Math.round(Math.max(0, seconds) * 10) / 10;
  const minutes = Math.floor(rounded / 60);
  const rest = rounded - minutes * 60;
  const text = Number.isInteger(rest)
    ? String(rest).padStart(2, "0")
    : rest.toFixed(1).padStart(4, "0");
  return `${minutes}:${text}`;
}

function words(items: string[]): string {
  return items.map((item) => item.replace(/_/g, " ")).join(", ");
}

function Notice({ text }: { text: string | null }) {
  if (!text) return null;
  return (
    <p className="notice" role="status">
      {text}
    </p>
  );
}

function Timeline({ segments }: { segments: Segment[] }) {
  const total = segments.reduce((sum, s) => sum + (s.end_s - s.start_s), 0);
  if (total <= 0) return null;
  return (
    <div className="timeline" role="img" aria-label={`${segments.length} segments across the video`}>
      {segments.map((segment, index) => (
        <span
          key={`${segment.start_s}-${segment.end_s}`}
          style={{ width: `${((segment.end_s - segment.start_s) / total) * 100}%` }}
          title={`${index + 1}. ${formatTime(segment.start_s)}–${formatTime(segment.end_s)}`}
        >
          {index + 1}
        </span>
      ))}
    </div>
  );
}

function SegmentsStep({
  origin,
  list,
  busy,
  message,
  onSend,
  view,
}: {
  origin: string;
  list: SegmentList;
  busy: boolean;
  message: string | null;
  onSend: (action: string, fields: Record<string, string>) => Promise<void>;
  view: WizardView;
}) {
  const [note, setNote] = useState("");
  const count = list.segments.length;
  const noun = count === 1 ? "segment" : "segments";

  function propose(event: FormEvent) {
    event.preventDefault();
    void onSend("segments", { action: "note", note });
  }

  return (
    <>
      <h1>Segments</h1>
      <p className="lede">
        {list.error
          ? "Gemini could not split the video, so it is one segment for now. Approve it to go on, or ask Gemini to try again."
          : "Gemini split the video where the picture needs different settings. Approve the segments, or say where the cuts should go and Gemini will try again."}
      </p>
      {message ? <Alert text={message} /> : null}
      <Notice text={list.error} />
      <Timeline segments={list.segments} />
      <p className="muted">
        {count} {noun}
      </p>
      <ol className="shots">
        {list.segments.map((segment, index) => (
          <li key={`${segment.start_s}-${segment.end_s}`} className="shot">
            {segment.still_url ? (
              <img
                src={outputSrc(origin, segment.still_url) ?? undefined}
                alt={`Still from segment ${index + 1}`}
              />
            ) : null}
            <div>
              <h2>
                {index + 1}. {segment.label || "Unlabelled segment"}
              </h2>
              <p className="muted">
                {formatTime(segment.start_s)}–{formatTime(segment.end_s)} ·{" "}
                {(segment.end_s - segment.start_s).toFixed(1)} s
              </p>
              <p>
                {segment.contains.length ? (
                  segment.contains.map((item) => (
                    <span key={item} className="chip">
                      {item.replace(/_/g, " ")}
                    </span>
                  ))
                ) : (
                  <span className="muted">Nothing to protect</span>
                )}
              </p>
            </div>
          </li>
        ))}
      </ol>
      <div className="actions">
        <button
          type="button"
          disabled={busy}
          onClick={() => void onSend("segments", { action: "approve" })}
        >
          Approve {count} {noun}
        </button>
      </div>
      <form onSubmit={propose}>
        <label htmlFor="segments-note">
          Not right? Say what to change
          <input
            id="segments-note"
            type="text"
            value={note}
            placeholder="e.g. cut where we walk outside"
            onChange={(event) => setNote(event.target.value)}
          />
        </label>
        <div className="actions">
          <button type="submit" className="secondary" disabled={busy}>
            Propose again
          </button>
        </div>
      </form>
      <Back view={view} to={["file"]} busy={busy} onSend={onSend} />
    </>
  );
}

const NEUTRAL: Record<string, number> = {
  deblock: 0,
  denoise: 0,
  contrast: 1,
  brightness: 0,
  saturation: 1,
  gamma: 1,
  grain: 0,
};

export function settingsSummary(strength: number, look: Record<string, number>): string {
  const parts = [`strength ${strength.toFixed(2)}`];
  for (const [name, neutral] of Object.entries(NEUTRAL)) {
    const value = look[name] ?? neutral;
    if (Math.abs(value - neutral) > 1e-6) parts.push(`${name} ${value.toFixed(2)}`);
  }
  return parts.join(" · ");
}

function Boxes({ regions }: { regions: Region[] }) {
  return (
    <>
      {regions.map((region, i) => (
        <span
          key={`${region.kind}-${i}`}
          className={`box ${region.kind}`}
          title={region.kind}
          style={{
            left: `${region.x0 * 100}%`,
            top: `${region.y0 * 100}%`,
            width: `${(region.x1 - region.x0) * 100}%`,
            height: `${(region.y1 - region.y0) * 100}%`,
          }}
        />
      ))}
    </>
  );
}

function Protected({ regions, contains }: { regions: Region[]; contains: string[] }) {
  const kinds = [...new Set(regions.map((region) => region.kind))].sort();
  if (kinds.length) {
    return <p className="muted">Protected inside the boxes: {kinds.join(", ")}.</p>;
  }
  const flagged = contains.filter((item) => ["faces", "hands", "text"].includes(item));
  if (!flagged.length) return null;
  return (
    <p className="muted">
      Contains {flagged.join(", ")} but no boxes were placed, so the whole shot
      is held to a lower strength.
    </p>
  );
}

function SeamNote({
  seams,
  error,
}: {
  seams: { time_s: number; score: number }[];
  error: string | null;
}) {
  if (error) return <p className="muted">{error}</p>;
  if (!seams.length) return null;
  const listed = seams.map((seam) => formatTime(seam.time_s)).join(", ");
  const first = seams[0].time_s;
  const example = `smooth ${formatTime(Math.max(0, first - 1))} to ${formatTime(first + 1)}`;
  return (
    <p className="muted">
      Possible seam{seams.length > 1 ? "s" : ""} at {listed}: the picture jumps
      there and the source does not. To re-render across one in a single pass,
      write a note like “{example}”.
    </p>
  );
}

function poster(videoUrl: string | null): string | null {
  // Each preview has a still of the same name, saved for the check.
  return videoUrl ? videoUrl.replace(/\.mp4$/, ".jpg") : null;
}

function checkSummary(check: LookShot["check"]): string {
  let note = (check.note || "").trim();
  if (note && !/[.!?]$/.test(note)) note += ".";
  if (check.ok === null) return "Not checked.";
  if (check.ok) return note ? `Check passed. ${note}` : "Check passed.";
  let text = `Check found: ${words(check.problems) || "a problem"}.`;
  if (note) text += ` ${note}`;
  if (check.adjusted) text += " Settings were adjusted and the preview redone.";
  return text;
}

function LookCard({ origin, shot }: { origin: string; shot: LookShot }) {
  const summary = settingsSummary(shot.strength, shot.look);
  return (
    <>
      <h2>
        {shot.index + 1}. {formatTime(shot.start_s)}–{formatTime(shot.end_s)} ·{" "}
        {shot.label || "Unlabelled segment"}
      </h2>
      {shot.after_url ? (
        <div className="pair">
          <figure>
            <figcaption>Before</figcaption>
            <div className="frame">
              <video
                controls
                muted
                loop
                preload="metadata"
                src={outputSrc(origin, shot.before_url) ?? undefined}
                poster={outputSrc(origin, poster(shot.before_url)) ?? undefined}
                aria-label={`Segment ${shot.index + 1} before`}
              />
              <Boxes regions={shot.regions ?? []} />
            </div>
          </figure>
          <figure>
            <figcaption>After</figcaption>
            <div className="frame">
              <video
                controls
                muted
                loop
                preload="metadata"
                src={outputSrc(origin, shot.after_url) ?? undefined}
                poster={outputSrc(origin, poster(shot.after_url)) ?? undefined}
                aria-label={`Segment ${shot.index + 1} after`}
              />
              <Boxes regions={shot.regions ?? []} />
            </div>
          </figure>
        </div>
      ) : null}
      <p>
        <strong>
          {shot.use_upscale === false ? `No FlashVSR: resized only. ${summary}` : summary}
        </strong>
      </p>
      <Protected regions={shot.regions ?? []} contains={shot.contains} />
      <p>{shot.reason}</p>
      <p className="muted">
        {checkSummary(shot.check)}
        {typeof shot.shimmer === "number"
          ? ` Texture stability: ${shot.shimmer.toFixed(1)}× as jumpy as the source.`
          : ""}
      </p>
    </>
  );
}

function PreviewStep({
  origin,
  preview,
  busy,
  message,
  onSend,
  view,
}: {
  origin: string;
  preview: SegmentPreview;
  busy: boolean;
  message: string | null;
  onSend: (action: string, fields: Record<string, string>) => Promise<void>;
  view: WizardView;
}) {
  const [note, setNote] = useState("");
  const [resample, setResample] = useState(false);
  const last = preview.index + 1 >= preview.count;

  function redo(event: FormEvent) {
    event.preventDefault();
    void onSend("preview", { action: "redo", note, resample: resample ? "1" : "" });
  }

  return (
    <>
      <h1>
        Segment {preview.index + 1} of {preview.count}
      </h1>
      <p className="lede">
        A few seconds of this segment, before and after. Approve it, or tell me
        what to change.
      </p>
      {preview.scale ? (
        <p className="muted">The whole video is upscaled {preview.scale}×.</p>
      ) : null}
      {message ? <Alert text={message} /> : null}
      <Notice text={preview.error} />
      {preview.shot ? <LookCard origin={origin} shot={preview.shot} /> : null}
      <div className="actions">
        <button
          type="button"
          disabled={busy}
          onClick={() => void onSend("preview", { action: "approve" })}
        >
          {last ? "Approve and finish" : "Approve and go to the next segment"}
        </button>
      </div>
      <form onSubmit={redo}>
        <label htmlFor="preview-note">
          Not right? Tell me what to change in this segment
          <input
            id="preview-note"
            type="text"
            value={note}
            placeholder="e.g. skin looks waxy, keep it softer"
            onChange={(event) => setNote(event.target.value)}
          />
        </label>
        <label htmlFor="preview-resample" className="inline">
          <input
            id="preview-resample"
            type="checkbox"
            checked={resample}
            onChange={(event) => setResample(event.target.checked)}
          />
          Preview a different part
        </label>
        <div className="actions">
          <button type="submit" className="secondary" disabled={busy}>
            Redo this segment
          </button>
        </div>
      </form>
      {preview.shot ? (
        <UpscaleToggle shot={preview.shot} busy={busy} onSend={onSend} />
      ) : null}
      <Back view={view} to={["segments"]} busy={busy} onSend={onSend} />
    </>
  );
}

function UpscaleToggle({
  shot,
  busy,
  onSend,
}: {
  shot: LookShot;
  busy: boolean;
  onSend: (action: string, fields: Record<string, string>) => Promise<void>;
}) {
  const on = shot.use_upscale !== false;
  const who = typeof shot.upscale_override === "boolean" ? "You" : "Gemini";
  return (
    <div className="actions">
      <p className="muted">
        {who} chose: this segment {on ? "uses FlashVSR" : "skips FlashVSR (plain resize)"}.
      </p>
      <button
        type="button"
        className="secondary"
        disabled={busy}
        onClick={() => void onSend("preview", { action: "upscale", on: on ? "0" : "1" })}
      >
        {on ? "Skip FlashVSR for this segment" : "Use FlashVSR for this segment"}
      </button>
    </div>
  );
}

function ReadyStep({
  looks,
  busy,
  message,
  onSend,
  view,
}: {
  looks: Looks;
  busy: boolean;
  message: string | null;
  onSend: (action: string, fields: Record<string, string>) => Promise<void>;
  view: WizardView;
}) {
  const scales = [...(looks.scales ?? [])].sort((a, b) => b - a);
  const [scale, setScale] = useState<number | null>(
    looks.scale !== null && scales.includes(looks.scale) ? looks.scale : (scales[0] ?? null),
  );
  const [width, height] = looks.source_size ?? [0, 0];
  return (
    <>
      <h1>Ready to render</h1>
      <p className="lede">
        Every segment is approved. Choose whether to upscale, then render the
        whole video with these settings.
      </p>
      {message ? <Alert text={message} /> : null}
      <ul className="summary">
        {looks.shots.map((shot) => {
          const summary = settingsSummary(shot.strength, shot.look);
          return (
            <li key={shot.index}>
              {formatTime(shot.start_s)}–{formatTime(shot.end_s)} ·{" "}
              {shot.label || "Unlabelled segment"}:{" "}
              {shot.use_upscale === false ? `No FlashVSR: resized only. ${summary}` : summary}
            </li>
          );
        })}
      </ul>
      {scales.length ? (
        <fieldset className="options">
          <legend>Upscale</legend>
          {scales.map((option) => {
            const size = width && height ? ` · ${width * option}×${height * option}` : "";
            return (
              <label key={option} className="option">
                <input
                  type="radio"
                  name="scale"
                  value={option}
                  checked={scale === option}
                  onChange={() => setScale(option)}
                />
                <span>
                  {option === 1 ? (
                    <>
                      <strong>Don&apos;t upscale</strong>
                      {size} · clean-up and finishing only, no FlashVSR
                    </>
                  ) : (
                    <>
                      <strong>Upscale {option}×</strong>
                      {size} · FlashVSR on the GPU
                    </>
                  )}
                </span>
              </label>
            );
          })}
        </fieldset>
      ) : null}
      <div className="actions">
        <button
          type="button"
          disabled={busy}
          onClick={() =>
            void onSend("ready", scale === null ? { action: "render" } : { action: "render", scale: String(scale) })
          }
        >
          Render the whole video
        </button>
      </div>
      <Back view={view} to={["segments"]} busy={busy} onSend={onSend} />
    </>
  );
}
