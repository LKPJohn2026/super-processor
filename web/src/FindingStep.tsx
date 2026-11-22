import { useEffect, useRef, useState, type ReactNode } from "react";
import type { Progress as RunProgress, SegmentList, SegmentEdit, WizardView } from "./api";
import {
  NO_CLOCK,
  askAgainNote,
  findingStage,
  findingStages,
  findingTime,
  scenesProblem,
  shortClock,
  startedAt,
  tickClock,
  type StageClock,
} from "./findingView";
import { Banner, Button, Card, Icon, Progress, StageRow } from "./ui";

/**
 * Step 2 · Finding the scenes (mockup 03): the time left (or spent), one
 * bar for this step, the stages as the server reports them, "You can close
 * this tab", the notify switch and Cancel.
 */
export function FindingStep({
  view,
  message,
  stop,
  notify,
}: {
  view: WizardView;
  message: string | null;
  // The Cancel button and its "Stopping" line (StopButton in LiveWizard).
  stop?: ReactNode;
  notify?: ReactNode;
}) {
  const progress = view.progress ?? null;
  const key = findingStage(view);
  const clock = useRef<StageClock>(NO_CLOCK);
  // Only once the run has said how far it is: its elapsed time dates the first stage.
  if (progress) clock.current = tickClock(clock.current, key, Date.now(), progress.elapsed_s);
  // The view arrives every two seconds; count the clock on between them.
  const [tick, setTick] = useState<{ of: RunProgress | null; since: number }>({ of: null, since: 0 });
  useEffect(() => {
    const arrived = Date.now();
    const timer = window.setInterval(() => setTick({ of: progress, since: (Date.now() - arrived) / 1000 }), 1000);
    return () => window.clearInterval(timer);
  }, [progress]);
  const since = tick.of === progress ? tick.since : 0;
  const time = findingTime(progress, since);
  const stages = findingStages(view, clock.current, Date.now());
  const pct = progress ? Math.max(0, Math.min(100, Math.round(progress.pct))) : 0;
  const started = view.run?.kind === "segmenting" ? startedAt(view.run.started_at) : "";
  const caption = [time.stageLeft, started ? `Started ${started}` : ""]
    .filter(Boolean)
    .join(" · ");
  const again = view.step === "proposing" && Boolean(view.segments?.note);
  return (
    <div className="finding-step">
      <header className="step-head">
        <p className="overline">Step 2 of 5 · Scenes</p>
        <h1>{again ? "Gemini is looking at your video again" : "Gemini is watching your video"}</h1>
        <p className="muted">
          It looks for the places where the light, the location or the picture quality changes, so each scene gets
          its own settings.
        </p>
      </header>
      {message ? (
        <Banner tone="error" role="alert" title="Something went wrong">
          {message}
        </Banner>
      ) : null}
      <Card as="section" className="col gap16 finding-card" aria-label="Progress">
        <div className="row between wrap finding-time">
          <span className="display">{time.big}</span>
          {caption ? <span className="caption subtle">{caption}</span> : null}
        </div>
        <Progress value={pct} label={`Finding the scenes: ${pct}%`} />
        <ul className="stage-list" aria-label="Stages">
          {stages.map((stage) => (
            <StageRow key={stage.key} state={stage.state} label={stage.label} meta={stage.meta} />
          ))}
        </ul>
      </Card>
      {/* Always in the page, so a note that appears is read out. */}
      <div role="status">
        {progress?.note ? (
          <Banner tone="warning" title="Note">
            {progress.note}
          </Banner>
        ) : null}
      </div>
      <Banner tone="info" title="You can close this tab">
        This keeps running on this computer. Open the wizard link again to come back to it.
      </Banner>
      {notify}
      {stop}
    </div>
  );
}

function useCountdown(seconds: number | null, key: string): number | null {
  const [left, setLeft] = useState(seconds);
  useEffect(() => {
    setLeft(seconds);
    if (seconds === null) return;
    const ends = Date.now() + seconds * 1000;
    const timer = window.setInterval(() => {
      const now = Math.max(0, Math.ceil((ends - Date.now()) / 1000));
      setLeft(now);
      if (now <= 0) window.clearInterval(timer);
    }, 1000);
    return () => window.clearInterval(timer);
    // The countdown restarts for a new list, not on every poll.
  }, [seconds, key]);
  return left;
}

/**
 * What the Scenes step says when Gemini's split did not work: its quota
 * used up (S3, with the wait Gemini gave), another failure, or no scenes
 * at all (S8). Each offers the way on as a button: ask Gemini again (a
 * note back, as "Propose again" sends), or go on with the whole video as
 * one scene (approve).
 */
export function ScenesProblemCard({
  list,
  locked,
  onEdit,
}: {
  list: SegmentList;
  locked: boolean;
  onEdit: (edit: SegmentEdit) => Promise<boolean>;
}) {
  const problem = scenesProblem(list);
  const left = useCountdown(problem?.kind === "quota" ? problem.retryS : null, `${list.revision ?? ""}`);
  if (!problem) return null;
  const ask = () => void onEdit({ action: "note", note: askAgainNote(list) });
  const whole = () => void onEdit({ action: "approve" });
  if (problem.kind === "quota") {
    const waiting = typeof left === "number" && left > 0;
    return (
      <Card as="section" className="col gap16 problem-card" aria-labelledby="problem-title">
        <Banner
          tone="error"
          title={
            <span id="problem-title">
              {problem.daily ? "Gemini’s daily quota is used up" : "Gemini’s free quota is used up for now"}
            </span>
          }
        >
          {problem.daily
            ? "Google limits how many requests a key can make each day."
            : "Google limits how many requests a key can make each minute."}{" "}
          Gemini said: {problem.detail}.
        </Banner>
        {waiting ? (
          <p className="row gap16 quota-wait">
            <span className="display" aria-hidden="true">
              {shortClock(left)}
            </span>
            <span className="muted">
              <span className="sr-only">About {left} seconds </span>until you can try again
            </span>
          </p>
        ) : null}
        <div className="row gap12 wrap">
          <Button kind="primary" size="lg" icon="rotate" disabled={locked || waiting} onClick={ask}>
            Try again
          </Button>
          <Button kind="secondary" size="lg" disabled={locked} onClick={whole}>
            Continue without Gemini’s split
          </Button>
        </div>
        <p className="caption subtle">
          Without it the whole video is one scene. Later steps still ask Gemini; if it can’t answer, settings are
          estimated from measurements.
        </p>
      </Card>
    );
  }
  if (problem.kind === "failed") {
    return (
      <Card as="section" className="col gap16 problem-card" aria-labelledby="problem-title">
        <Banner tone="warning" title={<span id="problem-title">Gemini couldn’t split the video</span>}>
          {problem.detail}. For now the whole video is one scene.
        </Banner>
        <div className="row gap12 wrap">
          <Button kind="primary" size="lg" icon="rotate" disabled={locked} onClick={ask}>
            Ask Gemini again
          </Button>
          <Button kind="secondary" size="lg" disabled={locked} onClick={whole}>
            Use the whole video as one scene
          </Button>
        </div>
      </Card>
    );
  }
  return (
    <Card as="section" className="col gap16 problem-card empty-card" aria-labelledby="problem-title">
      <Icon name="scissors" size={32} color="var(--ink-subtle)" />
      <h2 id="problem-title" className="h2">
        Gemini didn’t return any scenes
      </h2>
      <p className="muted">This sometimes happens with very short or very dark videos.</p>
      <div className="row gap12 wrap">
        <Button kind="primary" size="lg" icon="rotate" disabled={locked} onClick={ask}>
          Ask Gemini again
        </Button>
        <Button kind="secondary" size="lg" disabled={locked} onClick={whole}>
          Use the whole video as one scene
        </Button>
      </div>
    </Card>
  );
}
