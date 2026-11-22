import { useEffect, useState, type ReactNode } from "react";
import { outputSrc, scaleChoiceText, type Looks, type Progress as RunProgress, type WizardView } from "./api";
import { startedAt } from "./findingView";
import { jobTitle } from "./jobView";
import { strengthWord } from "./previewView";
import {
  detailBadge,
  failureTitle,
  initialScale,
  offersWithoutDetail,
  posterUrl,
  readyTitle,
  renderStages,
  sizeChoices,
  timeDone,
  timeLeft,
} from "./renderView";
import { offersRender, readyText, renderFailed, viewTone } from "./runStates";
import { Badge, Banner, Button, Card, Icon, Progress, StageRow } from "./ui";

type SendAction = (action: string, fields: Record<string, string>) => Promise<boolean | void>;

/**
 * Step 4 · Ready to render (mockup 06), and the same step after a render
 * that failed (S5) or was stopped: the size as two choice cards (a real
 * radio group over the server's scale options), the scenes with their AI
 * detail, and Start render.
 */
export function ReadyStep({
  origin,
  view,
  looks,
  busy,
  message,
  onSend,
  stopped,
  back,
}: {
  origin: string;
  view: WizardView;
  looks: Looks;
  busy: boolean;
  message: string | null;
  onSend: SendAction;
  // The note after a stopped render (StoppedNote in LiveWizard).
  stopped?: ReactNode;
  back?: ReactNode;
}) {
  const [scale, setScale] = useState<number | null>(() => initialScale(looks));
  const choices = sizeChoices(looks);
  const failed = renderFailed(view);
  const tone = viewTone(view);
  const text = readyText(view);
  const count = looks.shots.length || view.job?.scene_count || 0;
  const title = failed || tone === "stopped" ? text.title : readyTitle(count);
  const render = (chosen: number | null) =>
    void onSend("ready", chosen === null ? { action: "render" } : { action: "render", scale: String(chosen) });
  const withoutDetail = offersWithoutDetail(looks, scale);
  // Render shows after a failed render too, to try again.
  const offered = offersRender(view);
  const stills = new Map((view.job?.scenes ?? []).map((scene) => [scene.index, scene.still_url]));

  return (
    <div className="ready-step">
      <header className="step-head">
        <p className="overline">Step 4 of 5 · Render</p>
        <h1>{title}</h1>
        {/* After a failure the banner below says what is kept (S5). */}
        {failed && message ? null : (
          <p className="muted">
            {tone === "stopped"
              ? text.lede
              : "Every frame and its timing are kept. Nothing new is added to the story."}
          </p>
        )}
      </header>
      {stopped}
      {failed && message ? (
        <div className="col gap12 render-failed">
          <Banner tone="error" role="alert" title={failureTitle(message)}>
            {text.lede}
          </Banner>
          <div className="row gap12 wrap" hidden={!offered}>
            <Button kind="primary" size="lg" icon="rotate" disabled={busy} onClick={() => render(scale)}>
              Try the render again
            </Button>
            {withoutDetail ? (
              <Button kind="secondary" size="lg" disabled={busy} onClick={() => render(1)}>
                Render without AI detail
              </Button>
            ) : null}
          </div>
          {withoutDetail ? (
            <p className="caption subtle">
              Without AI detail the video keeps its size and gets the clean-up only; the graphics card does far
              less work.
            </p>
          ) : null}
        </div>
      ) : message ? (
        <Banner tone="error" role="alert" title="The render didn’t start">
          {message.charAt(0).toUpperCase() + message.slice(1)}.
        </Banner>
      ) : null}
      <div className="ready-layout">
        <div className="ready-choose">
          {choices.length ? (
            <fieldset className="size-choices" disabled={busy}>
              <legend className="sr-only">Output size</legend>
              {choices.map((choice) => {
                const on = scale === choice.scale;
                const src = choice.image ? outputSrc(origin, choice.image.url) : null;
                return (
                  <label key={choice.scale} className={on ? "choice sel" : "choice"}>
                    <span className="row gap12 choice-head">
                      <input
                        type="radio"
                        name="scale"
                        className="radio"
                        value={choice.scale}
                        checked={on}
                        onChange={() => setScale(choice.scale)}
                      />
                      <span className="h2 grow">{choice.title}</span>
                      {choice.recommended ? <Badge tone="success" icon={false}>Gemini’s pick</Badge> : null}
                    </span>
                    {src ? (
                      <figure className="choice-figure">
                        <img src={src} alt="" loading="lazy" />
                        <figcaption className="caption subtle">{choice.image?.caption}</figcaption>
                      </figure>
                    ) : null}
                    <span className="body-s muted">{choice.spec}</span>
                  </label>
                );
              })}
            </fieldset>
          ) : null}
          {looks.scale_reason ? (
            <p className="caption subtle">{scaleChoiceText(looks.scale, looks.scale_reason)}</p>
          ) : null}
          {back}
        </div>
        <div className="ready-side">
          {looks.shots.length ? (
            <Card as="section" className="col gap12" aria-labelledby="ready-scenes-title">
              <h2 id="ready-scenes-title" className="h3">
                {looks.shots.length === 1 ? "Your scene" : `Your ${looks.shots.length} scenes`}
              </h2>
              <ul className="ready-scenes">
                {looks.shots.map((shot) => {
                  const thumb = outputSrc(origin, stills.get(shot.index) ?? posterUrl(shot.before_url));
                  const badge = detailBadge(shot, strengthWord(shot.strength), scale);
                  return (
                    <li key={shot.index} className="row gap12">
                      {thumb ? <img src={thumb} alt="" loading="lazy" /> : <span className="still-gap" aria-hidden="true" />}
                      <span className="body-s grow">
                        {shot.index + 1} · {shot.label?.trim() || `Scene ${shot.index + 1}`}
                      </span>
                      <Badge tone={badge.tone} icon={false}>
                        {badge.text}
                      </Badge>
                    </li>
                  );
                })}
              </ul>
            </Card>
          ) : null}
          <Card as="section" className="col gap12" aria-label="Start the render">
            <p className="row gap8 body-s">
              <Icon name="lock" size={18} color="var(--ink-muted)" />
              Nothing is uploaded during the render
            </p>
            <p className="row gap8 body-s">
              <Icon name="cpu" size={18} color="var(--ink-muted)" />
              It runs on this computer’s graphics card
            </p>
            {failed || !offered ? null : (
              <Button kind="primary" size="lg" icon="film" disabled={busy} onClick={() => render(scale)}>
                Start render
              </Button>
            )}
            <p className="caption subtle">You can close the tab while it renders.</p>
          </Card>
        </div>
      </div>
    </div>
  );
}

/**
 * Step 4 · Rendering (mockup 07): the time left (only the server's
 * estimate), one bar, the render's stages, "Safe to close this tab", the
 * notify switch and Cancel render.
 */
export function RenderingStep({
  view,
  message,
  stop,
  notify,
}: {
  view: WizardView;
  message: string | null;
  stop?: ReactNode;
  notify?: ReactNode;
}) {
  const progress = view.progress ?? null;
  // The view arrives every two seconds; count the clock between them.
  const [tick, setTick] = useState<{ of: RunProgress | null; since: number }>({ of: null, since: 0 });
  useEffect(() => {
    if (!progress) return undefined;
    const arrived = Date.now();
    const timer = window.setInterval(() => setTick({ of: progress, since: (Date.now() - arrived) / 1000 }), 1000);
    return () => window.clearInterval(timer);
  }, [progress]);
  const since = tick.of === progress ? tick.since : 0;
  const pct = progress ? Math.max(0, Math.min(100, Math.round(progress.pct))) : 0;
  const left = progress ? timeLeft(progress.eta_s, since) : null;
  const done = progress ? timeDone(progress.elapsed_s, since) : "";
  const started = view.run?.kind === "render" ? startedAt(view.run.started_at) : "";
  const big = left ?? (done ? done.replace(/ done$/, " so far") : "Starting…");
  const caption = [started ? `Started ${started}` : "", left ? done : ""].filter(Boolean).join(" · ");
  const stages = renderStages(progress);
  return (
    <div className="render-step">
      <header className="step-head">
        <p className="overline">Step 4 of 5 · Render</p>
        <h1>Restoring {jobTitle(view) ?? "your video"}</h1>
      </header>
      {message ? (
        <Banner tone="error" role="alert" title="Something went wrong">
          {message}
        </Banner>
      ) : null}
      <Card as="section" className="col gap16" aria-label="Progress">
        <div className="row between wrap finding-time">
          <span className="display">{big}</span>
          {caption ? <span className="caption subtle">{caption}</span> : null}
        </div>
        <Progress value={pct} label={`Render: ${pct}%`} />
        <p className="body-s muted">{progress?.message || "Starting the render…"}</p>
        <div className="divider" />
        <ul className="stage-list" aria-label="Stages">
          {stages.map((stage) => (
            <StageRow key={stage.key} state={stage.state} label={stage.label} meta={stage.meta} />
          ))}
        </ul>
      </Card>
      {/* Always in the page, so a note that appears (a hot GPU) is read out. */}
      <div role="status">
        {progress?.note ? (
          <Banner tone="warning" title="Note">
            {progress.note}
          </Banner>
        ) : null}
      </div>
      <Banner tone="info" title="Safe to close this tab">
        The render keeps going on this computer. Open the wizard link again to come back to it.
      </Banner>
      {notify ? <div className="notify-row">{notify}</div> : null}
      {stop}
    </div>
  );
}
