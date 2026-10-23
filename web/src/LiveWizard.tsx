import { useEffect, useState, type FormEvent, type ReactNode } from "react";
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
  type SegmentList,
  type SegmentPreview,
  type WizardView,
} from "./api";

const RAIL = ["Setup", "File", "Segments", "Previews", "Upscale", "Result"] as const;

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
  if (step === "analyzing" || step === "preview_segment") return 3;
  if (step === "ready_to_render" || step === "rendering") return 4;
  return 5;
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
        {RAIL.map((label, index) => (
          <li key={label} className={index === active ? "active" : undefined}>
            {label}
          </li>
        ))}
      </ol>
      <div className="panel" aria-busy={busy}>
        <WizardStepView
          origin={origin}
          view={view}
          busy={busy}
          message={message}
          onSend={send}
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
}: {
  origin: string;
  view: WizardView;
  busy: boolean;
  message: string | null;
  onSend: (action: string, fields: Record<string, string>) => Promise<void>;
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
    return <PickStep busy={busy} message={message} onSend={onSend} />;
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
        list={view.segments ?? { segments: [], error: null, note: "" }}
        busy={busy}
        message={message}
        onSend={onSend}
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
  busy,
  message,
  onSend,
}: {
  busy: boolean;
  message: string | null;
  onSend: (action: string, fields: Record<string, string>) => Promise<void>;
}) {
  const [path, setPath] = useState("");

  function submit(event: FormEvent) {
    event.preventDefault();
    void onSend("pick", { path });
  }

  return (
    <>
      <h1>Pick a video file</h1>
      <p className="lede">
        Choose a clip on this machine. A local GPU model restores and upscales
        it. The first pass usually covers the whole file.
      </p>
      <p className="muted">Up to 1080p and 30 minutes.</p>
      {message ? <Alert text={message} /> : null}
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
          <button type="submit" disabled={busy}>
            Find the segments
          </button>
        </div>
      </form>
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

function SegmentsStep({
  list,
  busy,
  message,
  onSend,
}: {
  list: SegmentList;
  busy: boolean;
  message: string | null;
  onSend: (action: string, fields: Record<string, string>) => Promise<void>;
}) {
  const [note, setNote] = useState("");

  function propose(event: FormEvent) {
    event.preventDefault();
    void onSend("segments", { action: "note", note });
  }

  return (
    <>
      <h1>Segments</h1>
      <p className="lede">
        Gemini split the video where the picture needs different settings.
        Approve the segments, or say where the cuts should go and Gemini will
        try again.
      </p>
      {message ? <Alert text={message} /> : null}
      {list.error ? <p className="muted">{list.error}</p> : null}
      <ol className="shots">
        {list.segments.map((segment, index) => (
          <li key={`${segment.start_s}-${segment.end_s}`} className="shot">
            <div>
              <h2>
                {index + 1}. {formatTime(segment.start_s)}–{formatTime(segment.end_s)} ·{" "}
                {segment.label || "Unlabelled segment"}
              </h2>
              <p className="muted">Contains: {words(segment.contains) || "nothing flagged"}</p>
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
          Approve segments
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
}: {
  origin: string;
  preview: SegmentPreview;
  busy: boolean;
  message: string | null;
  onSend: (action: string, fields: Record<string, string>) => Promise<void>;
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
      {preview.error ? <p className="muted">{preview.error}</p> : null}
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
    </>
  );
}

function ReadyStep({
  looks,
  busy,
  message,
  onSend,
}: {
  looks: Looks;
  busy: boolean;
  message: string | null;
  onSend: (action: string, fields: Record<string, string>) => Promise<void>;
}) {
  return (
    <>
      <h1>Ready to render</h1>
      <p className="lede">
        Every segment is approved. The whole video is now restored and upscaled
        with these settings.
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
      <div className="actions">
        <button
          type="button"
          disabled={busy}
          onClick={() => void onSend("ready", { action: "render" })}
        >
          Render the whole video
        </button>
      </div>
    </>
  );
}
