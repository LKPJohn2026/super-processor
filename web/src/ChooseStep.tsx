import { useEffect, useRef, useState, type DragEvent, type FormEvent, type ReactNode, type RefObject } from "react";
import { uploadVideo, type WizardView } from "./api";
import {
  browserChecks,
  checksPass,
  fileSizeText,
  findFailure,
  pickRefusal,
  type FileCheck,
  type VideoMeta,
} from "./pickView";
import { Badge, Banner, Button, Card, Icon, Progress } from "./ui";

type SendAction = (action: string, fields: Record<string, string>) => Promise<boolean | void>;

// The wizard takes these (file_dialog.py VIDEO_EXTENSIONS).
const ACCEPT = "video/*,.mp4,.mov,.mkv,.avi,.webm,.m4v,.mts,.ts";

/** A check row (gen.py `checkrow()`): the icon and the words say the result, not colour alone. */
export function CheckRow({ check }: { check: FileCheck }) {
  return (
    <li className="check-row">
      <Icon
        name={check.ok ? "check-circle" : "x-circle"}
        color={check.ok ? "var(--ok)" : "var(--err)"}
      />
      <span className="col">
        <span className="body-s strong">
          {check.label}
          <span className="sr-only">{check.ok ? ": passed" : ": failed"}</span>
        </span>
        <span className={check.ok ? "caption subtle" : "caption check-bad"}>{check.value}</span>
      </span>
    </li>
  );
}

/** What goes to Gemini, said where the editor decides (mockup 02). */
function PrivacyLine() {
  return (
    <p className="row gap12 privacy-line body-s">
      <Icon name="send" size={18} color="var(--focus)" />
      <span>
        A small silent copy (up to 960 px) goes to Google Gemini to find the scenes. The full-quality video and its
        sound stay on this computer.
      </span>
    </p>
  );
}

type Via = "browse" | "upload" | "path";

/**
 * Step 1 · Choose a video (mockup 02), with "File can't be used" (S4). The
 * system file dialog and a typed path pick the file where it is; a file
 * dropped or chosen in the browser is checked here first (its size and
 * length) and copied to the wizard on "Find the scenes".
 */
export function ChooseStep({
  origin,
  busy,
  message,
  onSend,
  onView,
  stopped,
  back,
}: {
  origin: string;
  busy: boolean;
  message: string | null;
  onSend: SendAction;
  onView: (view: WizardView) => void;
  // The note after a stop, and the links back (from LiveWizard).
  stopped?: ReactNode;
  back?: ReactNode;
}) {
  const [path, setPath] = useState("");
  const [pathOpen, setPathOpen] = useState(false);
  const [over, setOver] = useState(false);
  const [chosen, setChosen] = useState<File | null>(null);
  const [upload, setUpload] = useState<{ name: string; pct: number } | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  // How the last pick was made and what it named, for S4's title and its button.
  const [tried, setTried] = useState<{ via: Via; name: string | null } | null>(null);
  const picker = useRef<HTMLInputElement | null>(null);
  const pathInput = useRef<HTMLInputElement | null>(null);
  const refusedCard = useRef<HTMLDivElement | null>(null);
  // An error put aside by "Choose another", until the next pick is sent.
  const [dismissed, setDismissed] = useState<string | null>(null);

  const said = uploadError ?? message;
  const error = chosen || said === dismissed ? null : said;
  const refusal = pickRefusal(error, tried?.name ?? null);
  const failure = refusal ? null : findFailure(error);
  const working = busy || upload !== null;

  // A screen-level error takes focus once (s9).
  useEffect(() => {
    if (refusal) refusedCard.current?.focus();
  }, [Boolean(refusal), error]);

  useEffect(() => {
    if (pathOpen) pathInput.current?.focus();
  }, [pathOpen]);

  function browse() {
    setUploadError(null);
    setDismissed(null);
    setTried({ via: "browse", name: null });
    void onSend("browse", {});
  }

  function choose(file: File | undefined) {
    if (!file) return;
    setUploadError(null);
    setChosen(file);
  }

  function openPicker() {
    if (picker.current) {
      picker.current.value = "";
      picker.current.click();
    }
  }

  function start(file: File) {
    setUploadError(null);
    setDismissed(null);
    setTried({ via: "upload", name: file.name });
    setUpload({ name: file.name, pct: 0 });
    uploadVideo(origin, file, (pct) => setUpload({ name: file.name, pct }))
      .then((next) => {
        if (next.error) setChosen(null);
        onView(next);
      })
      .catch((reason: unknown) => {
        setChosen(null);
        setUploadError(reason instanceof Error ? reason.message : "The upload failed.");
      })
      .finally(() => setUpload(null));
  }

  function submitPath(event: FormEvent) {
    event.preventDefault();
    setUploadError(null);
    setDismissed(null);
    setTried({ via: "path", name: path.trim() || null });
    void onSend("pick", { path });
  }

  function chooseAnother() {
    setChosen(null);
    setDismissed(said);
    if (tried?.via === "upload") openPicker();
    else if (tried?.via === "path") {
      setPathOpen(true);
      pathInput.current?.focus();
    } else browse();
  }

  function drop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setOver(false);
    if (!working) choose(event.dataTransfer.files[0]);
  }

  const dropArea = (
    <div
      className={over ? "drop-area over" : "drop-area"}
      onDragEnter={(event) => {
        event.preventDefault();
        setOver(true);
      }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={() => setOver(false)}
      onDrop={drop}
    >
      <Icon name="upload" size={28} color="var(--leaf)" />
      <h2 className="h3">Drop a video here</h2>
      <p className="row gap8 body-s muted drop-or">
        or
        <Button kind="secondary" size="md" disabled={working} onClick={browse}>
          Browse this computer…
        </Button>
      </p>
      <p className="caption subtle drop-hint">
        MP4, MOV, MKV, AVI and more · up to 1080p and 30 minutes ·{" "}
        <button
          type="button"
          className="text-link"
          aria-expanded={pathOpen}
          aria-controls="video-path-form"
          onClick={() => setPathOpen((open) => !open)}
        >
          Type a path instead
        </button>{" "}
        ·{" "}
        <button type="button" className="text-link" disabled={working} onClick={openPicker}>
          Choose in the browser (makes a copy)
        </button>
      </p>
      <input
        ref={picker}
        type="file"
        accept={ACCEPT}
        hidden
        tabIndex={-1}
        aria-hidden="true"
        onChange={(event) => choose(event.target.files?.[0])}
      />
    </div>
  );

  return (
    <div className="choose-step">
      <header className="step-head">
        <p className="overline">Step 1 of 5 · Video</p>
        <h1>Choose a video</h1>
        <p className="muted">Old films, VHS transfers and low-resolution clips work best.</p>
      </header>
      {stopped}
      {refusal ? (
        <RefusedCard
          refusal={refusal}
          cardRef={refusedCard}
          busy={working}
          onChooseAnother={chooseAnother}
        />
      ) : failure ? (
        <Banner
          tone="error"
          role="alert"
          title="Finding the scenes didn’t finish"
          actions={
            <Button kind="primary" size="md" disabled={working} onClick={browse}>
              Choose the video again
            </Button>
          }
        >
          {failure}. Pick the video again to start over; it is listed under All jobs too.
        </Banner>
      ) : error ? (
        <Banner tone="error" role="alert" title="That didn’t work">
          {error}
        </Banner>
      ) : null}
      {chosen ? (
        <ChosenFile
          file={chosen}
          upload={upload}
          busy={working}
          onStart={() => start(chosen)}
          onAnother={() => {
            setChosen(null);
            openPicker();
          }}
        />
      ) : (
        dropArea
      )}
      {pathOpen ? (
        <form id="video-path-form" className="path-form" onSubmit={submitPath}>
          <div className="field grow">
            <label htmlFor="video-path">Path to the video on this computer</label>
            <input
              ref={pathInput}
              id="video-path"
              name="path"
              type="text"
              className="input"
              value={path}
              autoComplete="off"
              spellCheck={false}
              onChange={(event) => setPath(event.target.value)}
              placeholder="/home/you/Videos/clip.mp4"
            />
          </div>
          <Button type="submit" kind="secondary" size="md" disabled={working || !path.trim()}>
            Use this path
          </Button>
        </form>
      ) : null}
      {chosen ? null : <PrivacyLine />}
      {back}
    </div>
  );
}

/** S4: the checks that failed, in red with the reason, and the way on. */
function RefusedCard({
  refusal,
  cardRef,
  busy,
  onChooseAnother,
}: {
  refusal: NonNullable<ReturnType<typeof pickRefusal>>;
  cardRef: RefObject<HTMLDivElement | null>;
  busy: boolean;
  onChooseAnother: () => void;
}) {
  return (
    <div ref={cardRef} tabIndex={-1} role="alert" className="card col gap16 refused-card" aria-labelledby="refused-title">
      <div className="row gap12 wrap">
        <h2 id="refused-title" className="h2 grow refused-name">
          {refusal.name ?? "This video"}
        </h2>
        <Badge tone="error">Can’t use this one</Badge>
      </div>
      <ul className="check-list">
        {refusal.checks.map((check) => (
          <CheckRow key={check.key} check={check} />
        ))}
      </ul>
      <div className="row gap12 wrap">
        <Button kind="primary" size="lg" disabled={busy} onClick={onChooseAnother}>
          Choose another video
        </Button>
      </div>
    </div>
  );
}

/**
 * The file card (mockup 02) for a file chosen in the browser: a frame of
 * it, its name and size, and the checks the browser can make. "Find the
 * scenes" copies it to the wizard, which checks it again and starts.
 */
function ChosenFile({
  file,
  upload,
  busy,
  onStart,
  onAnother,
}: {
  file: File;
  upload: { name: string; pct: number } | null;
  busy: boolean;
  onStart: () => void;
  onAnother: () => void;
}) {
  const [src, setSrc] = useState<string | null>(null);
  const [meta, setMeta] = useState<VideoMeta | null>(null);
  const [readable, setReadable] = useState<boolean | null>(null);
  useEffect(() => {
    const url = URL.createObjectURL(file);
    setSrc(url);
    setMeta(null);
    setReadable(null);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  const checks = browserChecks(meta);
  const fits = checksPass(checks);
  const badge =
    readable === null ? (
      <Badge tone="info">Checking…</Badge>
    ) : !fits ? (
      <Badge tone="error">Can’t use this one</Badge>
    ) : (
      <Badge tone="success">Ready</Badge>
    );
  return (
    <Card as="section" className="col gap20 file-card" aria-labelledby="file-card-name">
      <div className="row gap16 file-card-head">
        {src ? (
          <video
            className="file-thumb"
            // A frame a second in, as the still.
            src={`${src}#t=1`}
            preload="metadata"
            muted
            playsInline
            aria-hidden="true"
            tabIndex={-1}
            onLoadedMetadata={(event) => {
              const video = event.currentTarget;
              setMeta({ width: video.videoWidth, height: video.videoHeight, duration: video.duration });
              setReadable(true);
            }}
            onError={() => setReadable(false)}
          />
        ) : (
          <span className="file-thumb file-thumb-gap" aria-hidden="true" />
        )}
        <div className="col gap6 grow file-card-text">
          <div className="row gap8 wrap">
            <h2 id="file-card-name" className="h2 file-name">
              {file.name}
            </h2>
            {badge}
          </div>
          <span className="body-s muted">
            {fileSizeText(file.size)} · copied into the jobs folder when you start
          </span>
          <span className="row gap6 caption subtle">
            <Icon name="volume" size={16} />
            Sound is copied over unchanged
          </span>
        </div>
      </div>
      <div className="divider" />
      {checks.length ? (
        <ul className="check-list check-grid">
          {checks.map((check) => (
            <CheckRow key={check.key} check={check} />
          ))}
        </ul>
      ) : null}
      <p className="caption subtle">
        {readable === false
          ? "This browser can’t read the file’s size and length. The wizard checks it, and its frame rate, when you start."
          : "The wizard checks the frame rate when you start."}
      </p>
      <div className="divider" />
      <PrivacyLine />
      {upload ? (
        <div className="col gap8" role="status">
          <Progress value={upload.pct} label={`Copying ${upload.name}`} />
          <span className="body-s muted">
            Copying {upload.name} to the wizard · {upload.pct}%
          </span>
        </div>
      ) : null}
      <div className="row gap12 wrap">
        <Button kind="primary" size="lg" icon="sparkles" disabled={busy || !fits} onClick={onStart}>
          {upload ? "Copying…" : "Find the scenes"}
        </Button>
        <Button kind="secondary" size="lg" disabled={busy} onClick={onAnother}>
          Choose another
        </Button>
      </div>
    </Card>
  );
}
