import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type MutableRefObject,
  type ReactNode,
} from "react";
import { outputSrc, type OutputVersion, type WizardView } from "./api";
import { CompareViewer } from "./CompareViewer";
import { resultTitle } from "./runStates";
import {
  NO_RANGE,
  changesSoFar,
  compareScene,
  completeRange,
  currentVersion,
  markRange,
  outputPlace,
  rangeText,
  resultChips,
  resultTag,
  resultTimeline,
  reviseLabel,
  reviseNote,
  reviseProblem,
  savedFacts,
  startOverText,
  type MarkedRange,
  type ResultChipKey,
} from "./resultView";
import { formatTime } from "./sceneEdit";
import {
  downloadName,
  sizeText,
  versionLabel,
  versionMeta,
  versionNote,
  versionsNewestFirst,
  versionsSummary,
} from "./versions";
import { Button, Card, Chip, Field, Icon, LinkButton, Menu, TextArea } from "./ui";

// Resolves true when the server took the action (false: the error shows).
type SendAction = (action: string, fields: Record<string, string>) => Promise<boolean | void>;

type StepProps = {
  origin: string;
  view: WizardView;
  busy: boolean;
  message: string | null;
  onSend: SendAction;
  // The note after a stop, from the shell.
  stopped?: ReactNode;
};

/**
 * 08 Result: the restored video in the viewer with the scene seams under
 * it; "Keep this video", "Fix part of it" (a marked part, fix chips and a
 * note for a revise), the notes so far, the versions, and a More menu with
 * "Restore another video" and "Start over from the scenes…" (behind S7).
 */
export function ResultStep({ origin, view, busy, message, onSend, stopped }: StepProps) {
  const src = outputSrc(origin, view.output_url);
  const [range, setRange] = useState<MarkedRange>(NO_RANGE);
  const [chosen, setChosen] = useState<ResultChipKey[]>([]);
  const [typed, setTyped] = useState("");
  const [problem, setProblem] = useState<string | null>(null);
  const [durationS, setDurationS] = useState<number | null>(null);
  const [confirming, setConfirming] = useState(false);
  const now = useRef(0);
  const moreButton = useRef<HTMLButtonElement | null>(null);
  const seams = view.seams ?? [];
  const chips = resultChips(seams);
  const version = currentVersion(view.versions);
  const timeline = resultTimeline(view, durationS, range);
  const changes = changesSoFar(view.versions);
  const marked = rangeText(range);
  const note = reviseNote(range, chosen, typed, seams);

  const onTime = useCallback((seconds: number) => {
    now.current = seconds;
  }, []);
  const mark = useCallback((which: "start" | "end", seconds: number) => {
    setProblem(null);
    setRange((value) => markRange(value, which, seconds));
  }, []);

  function submit(event: FormEvent) {
    event.preventDefault();
    const said = reviseProblem(range, note);
    if (said) {
      setProblem(said);
      return;
    }
    void onSend("result", { note });
  }

  return (
    <div className="result-step">
      <header className="step-head">
        <p className="overline">Step 5 of 5 · Result</p>
        <h1>{resultTitle(view)}</h1>
      </header>
      {stopped}
      <div className="result-main">
        <div className="result-left col gap16">
          {src ? (
            <CompareViewer
              afterSrc={src}
              afterLabel={resultTag(view.scale, version)}
              sceneName="The restored video"
              positionLabel="Position in the restored video"
              onDuration={setDurationS}
              onTime={onTime}
              onMark={mark}
              timeline={timeline ? <SeamsTimeline timeline={timeline} marked={completeRange(range) ? marked : ""} /> : null}
            />
          ) : (
            <div className="viewer empty">
              <p>The result file is not ready yet.</p>
            </div>
          )}
          {src ? (
            <p className="caption subtle">
              The wizard keeps no copy of the original to play beside it, so this is the restored video on
              its own. The scene previews showed both.
            </p>
          ) : null}
          <SeamNote seams={seams} error={view.seam_error ?? null} />
          {view.scale !== null && view.strength_summary ? (
            <p className="caption subtle">
              {view.scale}× · {view.strength_summary}. Less AI detail means less invented texture.
            </p>
          ) : null}
          <Versions origin={origin} view={view} />
        </div>
        <aside className="result-panel col gap16" aria-label="Keep or fix the result">
          {message ? (
            <p className="error" role="alert">
              {message}
            </p>
          ) : null}
          <Button kind="primary" size="lg" icon="check" disabled={busy} onClick={() => void onSend("result", { mood: "happy" })}>
            Keep this video
          </Button>
          <Card as="section" className="col gap12 fix-card" aria-labelledby="fix-title">
            <form className="col gap12" onSubmit={submit} noValidate>
              <h2 id="fix-title" className="h3">
                Fix part of it
              </h2>
              <div className="row gap8 wrap">
                <Button kind="secondary" size="md" disabled={!src} onClick={() => mark("start", now.current)}>
                  Mark start
                </Button>
                <Button kind="secondary" size="md" disabled={!src} onClick={() => mark("end", now.current)}>
                  Mark end
                </Button>
              </div>
              <p className="caption subtle">
                or press <span className="kbd">I</span> and <span className="kbd">O</span> while it plays
              </p>
              <div className="marked-row" role="status">
                {marked ? (
                  <>
                    <span className="body-s strong grow">{marked}</span>
                    <Button kind="ghost" size="md" className="small" onClick={() => setRange(NO_RANGE)}>
                      Clear<span className="sr-only"> the marked part</span>
                    </Button>
                  </>
                ) : (
                  <span className="caption subtle">No part marked: the note applies where it says.</span>
                )}
              </div>
              <div className="row gap6 wrap" role="group" aria-label="What to fix">
                {chips.map((chip) => (
                  <Chip
                    key={chip.key}
                    selected={chosen.includes(chip.key)}
                    disabled={busy}
                    onToggle={(on) => {
                      setProblem(null);
                      setChosen((value) => (on ? [...value, chip.key] : value.filter((key) => key !== chip.key)));
                    }}
                  >
                    {chip.text}
                  </Chip>
                ))}
              </div>
              <Field
                id="revise-note"
                className="compact"
                label="Anything else? (optional)"
                type="text"
                autoComplete="off"
                value={typed}
                placeholder="e.g. the faces look waxy here"
                error={problem}
                onChange={(event) => {
                  setProblem(null);
                  setTyped(event.target.value);
                }}
              />
              {note ? (
                <p className="caption subtle">
                  Gemini reads: <q>{note}</q>
                </p>
              ) : null}
              <Button type="submit" kind="secondary" size="md" icon="rotate" disabled={busy} className="self-start">
                {reviseLabel(range)}
              </Button>
            </form>
          </Card>
          {changes.length ? (
            <section className="col gap8" aria-labelledby="changes-title">
              <h2 id="changes-title" className="overline">
                Changes so far
              </h2>
              <ul className="changes">
                {changes.map((change) => (
                  <li key={change.version} className="row gap8 caption">
                    <Icon name="message" size={16} color="var(--ink-subtle)" />
                    <span>
                      <b>You:</b> {change.note}
                      <br />
                      <span className="subtle">{change.version}</span>
                    </span>
                  </li>
                ))}
              </ul>
            </section>
          ) : null}
          <Menu
            buttonRef={moreButton}
            disabled={busy}
            items={[
              { label: "Restore another video", icon: "film", onSelect: () => void onSend("new", {}) },
              { label: "Start over from the scenes…", icon: "rotate", danger: true, onSelect: () => setConfirming(true) },
            ]}
          />
        </aside>
      </div>
      <StartOverDialog
        open={confirming}
        view={view}
        busy={busy}
        initialNote={note}
        returnFocus={moreButton}
        onClose={() => setConfirming(false)}
        onConfirm={(text) => {
          void onSend("result", { note: text, restart: "1" }).then((ok) => {
            if (ok !== false) setConfirming(false);
          });
        }}
      />
    </div>
  );
}

function SeamsTimeline({
  timeline,
  marked,
}: {
  timeline: NonNullable<ReturnType<typeof resultTimeline>>;
  marked: string;
}) {
  return (
    <div className="seams">
      <div className="row between gap12 wrap caption seams-head">
        <span>
          Scene seams
          {timeline.seams.length ? (
            <span className="seams-legend">
              <span className="seam-flag-key" aria-hidden="true" /> possible seam
            </span>
          ) : null}
        </span>
        <span>{marked ? `Marked: ${marked}` : ""}</span>
      </div>
      <div className="seam-track" aria-hidden="true">
        {timeline.scenes.map((cut) => (
          <span key={`c${cut.pct}`} className="seam-cut" style={{ left: `${cut.pct}%` }} />
        ))}
        {timeline.seams.map((seam) => (
          <span key={`s${seam.pct}`} className="seam-flag" style={{ left: `${seam.pct}%` }} />
        ))}
        {timeline.marked ? (
          <span
            className="seam-marked"
            style={{ left: `${timeline.marked.left}%`, width: `${timeline.marked.width}%` }}
          />
        ) : null}
      </div>
      <p className="sr-only">{timeline.summary}</p>
    </div>
  );
}

/**
 * S7: a real modal (showModal: the page behind is inert, Esc closes it)
 * that names what starting over deletes and keeps. The wizard needs a note
 * for a start over, so the dialog asks for one.
 */
function StartOverDialog({
  open,
  view,
  busy,
  initialNote,
  returnFocus,
  onClose,
  onConfirm,
}: {
  open: boolean;
  view: WizardView;
  busy: boolean;
  initialNote: string;
  // Focus goes back here once the dialog has closed.
  returnFocus: MutableRefObject<HTMLButtonElement | null>;
  onClose: () => void;
  onConfirm: (note: string) => void;
}) {
  const ref = useRef<HTMLDialogElement | null>(null);
  const [note, setNote] = useState("");
  const [error, setError] = useState<string | null>(null);
  const text = startOverText(view);
  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) {
      setNote(initialNote);
      setError(null);
      dialog.showModal();
    } else if (!open && dialog.open) {
      dialog.close();
      returnFocus.current?.focus();
    }
    // initialNote is read when the dialog opens only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);
  return (
    <dialog
      ref={ref}
      className="confirm-dialog card"
      aria-labelledby="start-over-title"
      aria-describedby="start-over-text"
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
    >
      <form
        className="col gap12"
        onSubmit={(event) => {
          event.preventDefault();
          if (!note.trim()) {
            setError("Say what to do differently, so Gemini doesn’t repeat it.");
            return;
          }
          onConfirm(note.trim());
        }}
        noValidate
      >
        <h2 id="start-over-title" className="h2">
          Start over from the scenes?
        </h2>
        <p id="start-over-text" className="body-s">
          This deletes <b>{text.deletes}</b>, and Gemini finds the scenes again. {text.keeps}
        </p>
        <TextArea
          label="What should be different this time?"
          rows={3}
          value={note}
          error={error}
          placeholder="e.g. fewer scenes, and keep the faces softer"
          onChange={(event) => {
            setError(null);
            setNote(event.target.value);
          }}
        />
        <div className="row gap12 wrap dialog-actions">
          <Button kind="secondary" size="md" onClick={onClose}>
            Keep these scenes
          </Button>
          <Button type="submit" kind="dangerfill" size="md" disabled={busy}>
            Delete and start over
          </Button>
        </div>
      </form>
    </dialog>
  );
}

/**
 * 09 Saved: after "Keep this video". The output with what is known about
 * it, where it is, a download, "Restore another video", and a still of a
 * scene before and after.
 */
export function SavedStep({ origin, view, busy, message, onSend }: StepProps) {
  const src = outputSrc(origin, view.output_url);
  const version = currentVersion(view.versions);
  const [picture, setPicture] = useState<{ width: number; height: number; duration: number } | null>(null);
  const [copied, setCopied] = useState("");
  const name = version ? downloadName(view.job?.title, version) : `${(view.job?.title ?? "Restored video").trim()} – restored.mp4`;
  const facts = savedFacts(picture, version, sizeText);
  const scene = compareScene(view);
  const thumbAt = scene?.atS ?? 0;

  async function copyLink() {
    if (!src) return;
    const link = new URL(src, window.location.href).href;
    try {
      await navigator.clipboard.writeText(link);
      setCopied("Link copied.");
    } catch {
      setCopied(`Could not copy. The link is ${link}`);
    }
  }

  return (
    <div className="saved-step col gap24">
      <header className="saved-head">
        <span className="saved-mark" aria-hidden="true">
          <Icon name="check" size={30} color="#fff" strokeWidth={3} />
        </span>
        <div className="col gap4">
          <h1>Your restored video is ready</h1>
          <p className="muted">
            Saved on this computer, in the wizard’s jobs folder. Any small copy sent to Gemini is deleted after
            about 48 hours.
          </p>
        </div>
      </header>
      {message ? (
        <p className="error" role="alert">
          {message}
        </p>
      ) : null}
      <Card className="col gap16 saved-card">
        <div className="saved-file">
          {src ? (
            <video
              className="saved-thumb"
              src={`${src}#t=${thumbAt.toFixed(1)}`}
              preload="metadata"
              muted
              playsInline
              aria-hidden="true"
              tabIndex={-1}
              onLoadedMetadata={(event) => {
                const video = event.currentTarget;
                setPicture({ width: video.videoWidth, height: video.videoHeight, duration: video.duration });
              }}
            />
          ) : (
            <span className="saved-thumb still-gap" aria-hidden="true" />
          )}
          <div className="col gap6 grow saved-text">
            <h2 className="h2">{name}</h2>
            {facts ? <p className="body-s muted">{facts}</p> : null}
            <div className="saved-path">
              <span className="mono grow">{outputPlace(view.job_id)}</span>
              <Button kind="ghost" size="md" icon="copy" disabled={!src} onClick={() => void copyLink()}>
                Copy link
              </Button>
            </div>
            <p className="caption subtle" role="status">
              {copied}
            </p>
          </div>
        </div>
        <div className="row gap12 wrap">
          {src ? (
            <LinkButton href={src} kind="primary" size="lg" icon="download" download={name}>
              Save a copy…
            </LinkButton>
          ) : null}
          <Button kind="ghost" size="lg" disabled={busy} onClick={() => void onSend("new", {})}>
            Restore another video
          </Button>
        </div>
      </Card>
      {scene && src ? (
        <section className="col gap12" aria-labelledby="pair-title">
          <h2 id="pair-title" className="overline">
            Before and after
          </h2>
          <div className="saved-pair">
            <figure className="col gap6">
              <img src={outputSrc(origin, scene.stillUrl) ?? undefined} alt={`Original, ${scene.label}`} />
              <figcaption className="caption subtle">Original · {scene.label}</figcaption>
            </figure>
            <figure className="col gap6">
              <video
                src={`${src}#t=${scene.atS.toFixed(1)}`}
                preload="metadata"
                muted
                playsInline
                tabIndex={-1}
                aria-label={`Restored, ${scene.label}, at ${formatTime(scene.atS)}`}
              />
              <figcaption className="caption subtle">
                Restored{picture?.width ? ` · ${picture.width}×${picture.height}` : ""} · at {formatTime(scene.atS)}
              </figcaption>
            </figure>
          </div>
        </section>
      ) : null}
      <Versions origin={origin} view={view} />
    </div>
  );
}

/** Possible seams the render found, with a note that would smooth one. */
function SeamNote({ seams, error }: { seams: { time_s: number; score: number }[]; error: string | null }) {
  if (error) return <p className="caption subtle">{error}</p>;
  if (!seams.length) return null;
  const listed = seams.map((seam) => formatTime(seam.time_s)).join(", ");
  return (
    <p className="caption subtle">
      Possible seam{seams.length > 1 ? "s" : ""} at {listed}: the picture jumps there and the source does not.
      Mark the part and pick “Fix a seam” to re-render across it in one pass.
    </p>
  );
}

/**
 * Every kept result of the job, newest first, each with a player that opens
 * in place and a download link. The one the job's output is now is marked.
 */
export function Versions({ origin, view }: { origin: string; view: WizardView }) {
  const [open, setOpen] = useState<number | null>(null);
  const list = versionsNewestFirst(view.versions);
  if (!list.length) return null;
  const now = Date.now();
  return (
    <section className="versions" aria-labelledby="versions-title">
      <h2 id="versions-title" className="overline">
        Versions
      </h2>
      <p className="muted versions-summary">{versionsSummary(view.versions)}</p>
      <ol className="versions-list">
        {list.map((version) => (
          <VersionRow
            key={version.number}
            origin={origin}
            title={view.job?.title ?? null}
            version={version}
            now={now}
            open={open === version.number}
            onToggle={() => setOpen((shown) => (shown === version.number ? null : version.number))}
          />
        ))}
      </ol>
    </section>
  );
}

function VersionRow({
  origin,
  title,
  version,
  now,
  open,
  onToggle,
}: {
  origin: string;
  title: string | null;
  version: OutputVersion;
  now: number;
  open: boolean;
  onToggle: () => void;
}) {
  const src = outputSrc(origin, version.url) ?? undefined;
  const label = versionLabel(version);
  const note = versionNote(version);
  const playerId = `version-player-${version.number}`;
  return (
    <li className={`version-row${version.current ? " current" : ""}`}>
      <div className="version-text">
        <p className="version-name">
          <strong>{label}</strong>
          {version.current ? <span className="badge success">Current</span> : null}
        </p>
        <p className="version-meta">{versionMeta(version, now)}</p>
        {note ? <p className="version-note">{note}</p> : null}
      </div>
      <div className="version-actions">
        <button
          type="button"
          className="btn secondary md"
          aria-expanded={open}
          aria-controls={playerId}
          aria-label={`${open ? "Hide" : "Play"} ${label.toLowerCase()}`}
          onClick={onToggle}
        >
          {open ? "Hide" : "Play"}
        </button>
        <a
          className="btn secondary md"
          href={src}
          download={downloadName(title, version)}
          aria-label={`Download ${label.toLowerCase()}`}
        >
          Download
        </a>
      </div>
      <div id={playerId} className="version-player" hidden={!open}>
        {open ? <video controls preload="metadata" src={src} aria-label={label} /> : null}
      </div>
    </li>
  );
}
