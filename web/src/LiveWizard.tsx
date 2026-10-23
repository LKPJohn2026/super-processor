import { useEffect, useState, type FormEvent } from "react";
import {
  outputSrc,
  postAction,
  runPlan,
  runRender,
  runScan,
  type LookShot,
  type Looks,
  type Region,
  type Shot,
  type WizardView,
} from "./api";

const RAIL = ["Setup", "File", "Shots", "Looks", "Upscale", "Result"] as const;

function railIndex(step: WizardView["step"]): number {
  if (step === "intro" || step === "llm_choice" || step === "local_llm_stub" || step === "setup") {
    return 0;
  }
  if (step === "pick_file") return 1;
  if (step === "finding_shots" || step === "shots") return 2;
  if (step === "planning" || step === "looks") return 3;
  if (step === "rendering") return 4;
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
    const runners = { rendering: runRender, finding_shots: runScan, planning: runPlan };
    if (!(view.step in runners)) return;
    let cancelled = false;
    setLocalError(null);
    const run = runners[view.step as keyof typeof runners];
    run(origin)
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

  if (view.step === "finding_shots") {
    return (
      <>
        <h1>Finding the shots</h1>
        <p className="lede" role="status">
          FFmpeg is finding where each shot starts and measuring how blocky,
          noisy, soft, dark, or flat it is. Gemini then labels each shot from
          one still.
        </p>
        {message ? <Alert text={message} /> : null}
      </>
    );
  }

  if (view.step === "shots") {
    return (
      <ShotsStep
        origin={origin}
        shots={view.shots ?? []}
        labelError={view.label_error ?? null}
        busy={busy}
        message={message}
        onSend={onSend}
      />
    );
  }

  if (view.step === "planning") {
    return (
      <>
        <h1>Planning each shot</h1>
        <p className="lede" role="status">
          Gemini is choosing clean-up, strength, and finishing for each shot.
          Each shot then gets a short before/after preview on the local GPU,
          and Gemini checks the previews for faces, hands, text, and texture
          that went wrong.
        </p>
        {message ? <Alert text={message} /> : null}
      </>
    );
  }

  if (view.step === "looks") {
    return (
      <LooksStep
        origin={origin}
        looks={view.looks ?? { scale: null, error: null, shots: [] }}
        busy={busy}
        message={message}
        onSend={onSend}
      />
    );
  }

  if (view.step === "rendering") {
    return (
      <>
        <h1>Upscaling</h1>
        <p className="lede" role="status">
          Restoring and upscaling on the local GPU. FFmpeg only trims, splices,
          and copies audio. This can take several minutes.
        </p>
        {message ? <Alert text={message} /> : null}
      </>
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
        shell drives introduction, Gemini setup, file pick, upscale, and result.
      </p>
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
            Find the shots
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

function ShotsStep({
  origin,
  shots,
  labelError,
  busy,
  message,
  onSend,
}: {
  origin: string;
  shots: Shot[];
  labelError: string | null;
  busy: boolean;
  message: string | null;
  onSend: (action: string, fields: Record<string, string>) => Promise<void>;
}) {
  const [splitAt, setSplitAt] = useState<Record<number, string>>({});

  function split(event: FormEvent, index: number) {
    event.preventDefault();
    void onSend("shots", { action: "split", index: String(index), at: splitAt[index] ?? "" });
  }

  return (
    <>
      <h1>Shots</h1>
      <p className="lede">
        These are the shots found and what looks wrong in each. Merge shots
        that belong together or split one that changes partway, then approve
        the list.
      </p>
      {message ? <Alert text={message} /> : null}
      {labelError ? <p className="muted">{labelError}</p> : null}
      <ol className="shots">
        {shots.map((shot, index) => (
          <li key={`${shot.start_s}-${shot.end_s}`} className="shot">
            {shot.still ? (
              <img
                src={outputSrc(origin, `/${shot.still}`) ?? undefined}
                alt={`Still from shot ${index + 1}`}
              />
            ) : null}
            <div>
              <h2>
                {index + 1}. {formatTime(shot.start_s)}–{formatTime(shot.end_s)} ·{" "}
                {shot.label || "Unlabelled shot"}
              </h2>
              <p>Problems: {words(shot.issues) || "none found"}</p>
              <p className="muted">Contains: {words(shot.contains) || "nothing flagged"}</p>
              <form className="actions" onSubmit={(event) => split(event, index)}>
                {index < shots.length - 1 ? (
                  <button
                    type="button"
                    className="secondary"
                    disabled={busy}
                    onClick={() =>
                      void onSend("shots", { action: "merge", index: String(index) })
                    }
                  >
                    Merge with next
                  </button>
                ) : null}
                <input
                  type="text"
                  aria-label={`Split shot ${index + 1} at`}
                  placeholder={formatTime((shot.start_s + shot.end_s) / 2)}
                  value={splitAt[index] ?? ""}
                  onChange={(event) =>
                    setSplitAt({ ...splitAt, [index]: event.target.value })
                  }
                />
                <button type="submit" className="secondary" disabled={busy}>
                  Split here
                </button>
              </form>
            </div>
          </li>
        ))}
      </ol>
      <div className="actions">
        <button
          type="button"
          disabled={busy}
          onClick={() => void onSend("shots", { action: "approve" })}
        >
          Approve shots and upscale
        </button>
      </div>
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

function LooksStep({
  origin,
  looks,
  busy,
  message,
  onSend,
}: {
  origin: string;
  looks: Looks;
  busy: boolean;
  message: string | null;
  onSend: (action: string, fields: Record<string, string>) => Promise<void>;
}) {
  const [notes, setNotes] = useState<Record<number, string>>({});

  function redo(event: FormEvent, index: number) {
    event.preventDefault();
    void onSend("looks", { action: "redo", index: String(index), note: notes[index] ?? "" });
  }

  return (
    <>
      <h1>Looks</h1>
      <p className="lede">
        Each shot has its own settings and a short before/after preview.
        Approve them all to render the whole video, or tell me what to change
        in one shot.
      </p>
      {looks.scale ? <p className="muted">The whole video is upscaled {looks.scale}×.</p> : null}
      {message ? <Alert text={message} /> : null}
      {looks.error ? <p className="muted">{looks.error}</p> : null}
      <ol className="shots">
        {looks.shots.map((shot) => (
          <li key={shot.index} className="look">
            <h2>
              {shot.index + 1}. {formatTime(shot.start_s)}–{formatTime(shot.end_s)} ·{" "}
              {shot.label || "Unlabelled shot"}
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
                      aria-label={`Shot ${shot.index + 1} before`}
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
                      aria-label={`Shot ${shot.index + 1} after`}
                    />
                    <Boxes regions={shot.regions ?? []} />
                  </div>
                </figure>
              </div>
            ) : null}
            <p>
              <strong>{settingsSummary(shot.strength, shot.look)}</strong>
            </p>
            <Protected regions={shot.regions ?? []} contains={shot.contains} />
            <p>{shot.reason}</p>
            <p className="muted">
              {checkSummary(shot.check)}
              {typeof shot.shimmer === "number"
                ? ` Texture stability: ${shot.shimmer.toFixed(1)}× as jumpy as the source.`
                : ""}
            </p>
            <form onSubmit={(event) => redo(event, shot.index)}>
              <label htmlFor={`note-${shot.index}`}>
                Not right? Tell me what to change in this shot
                <input
                  id={`note-${shot.index}`}
                  type="text"
                  value={notes[shot.index] ?? ""}
                  placeholder="e.g. skin looks waxy, keep it softer"
                  onChange={(event) => setNotes({ ...notes, [shot.index]: event.target.value })}
                />
              </label>
              <div className="actions">
                <button type="submit" className="secondary" disabled={busy}>
                  Redo this shot
                </button>
              </div>
            </form>
          </li>
        ))}
      </ol>
      <div className="actions">
        <button
          type="button"
          disabled={busy}
          onClick={() => void onSend("looks", { action: "approve" })}
        >
          Approve all and render
        </button>
      </div>
    </>
  );
}
