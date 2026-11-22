import { useEffect, useState } from "react";
import { outputSrc, type JobEntry, type WizardView } from "./api";
import { busyHere, jobAction, jobMeta, jobStatus, newVideoRoute, type JobStatus } from "./jobsList";
import { Badge, Banner, Button, Card, Icon, Menu, type BadgeTone } from "./ui";

// The list's tones in the design's badge colours.
const TONE: Record<JobStatus["tone"], BadgeTone> = {
  ok: "success",
  working: "info",
  waiting: "warning",
  busy: "neutral",
  failed: "error",
};

type JobsHomeProps = {
  origin: string;
  view: WizardView;
  // null while the first list loads.
  jobs: JobEntry[] | null;
  loadError: string | null;
  // The job being opened, while its request runs.
  opening: string | null;
  openError: string | null;
  // What the page last did, read out by screen readers.
  status: string;
  // A way back to the open job when it has no row in the list.
  backLabel: string | null;
  onOpen: (entry: JobEntry) => void;
  onShowCurrent: () => void;
  onNewVideo: () => void;
  onRefresh: () => void;
  onDisconnect: () => void;
};

/** Every job on this computer, most recently changed first (mockup s1-jobs). */
export function JobsHome({
  origin,
  view,
  jobs,
  loadError,
  opening,
  openError,
  status,
  backLabel,
  onOpen,
  onShowCurrent,
  onNewVideo,
  onRefresh,
  onDisconnect,
}: JobsHomeProps) {
  // Re-render each minute so "Changed 5 min ago" stays true.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 60_000);
    return () => window.clearInterval(timer);
  }, []);
  useEffect(() => setNow(Date.now()), [jobs]);
  // "Looking for your videos…" only when the list is slow, so a quick load
  // does not flash it.
  const loading = jobs === null && !loadError;
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    if (!loading) {
      setSlow(false);
      return;
    }
    const timer = window.setTimeout(() => setSlow(true), 300);
    return () => window.clearTimeout(timer);
  }, [loading]);

  const route = newVideoRoute(view);
  const here = busyHere(view, jobs ?? []);
  // "" while "Restore a new video" goes back to the file pick.
  const working = opening !== null;
  const pct = view.progress?.overall ?? view.progress?.pct ?? null;

  return (
    <section className="jobs-home" aria-labelledby="jobs-title" aria-busy={jobs === null || working}>
      <div className="jobs-head">
        <div className="col gap4">
          <h1 id="jobs-title" className="h1">
            Your videos
          </h1>
          <p className="muted">Pick up where you left off, or start a new one.</p>
        </div>
        <div className="row gap8 jobs-head-actions">
          <Menu
            srLabel="More: refresh the list, disconnect"
            align="end"
            items={[
              { label: "Refresh the list", icon: "rotate", disabled: jobs === null && !loadError, onSelect: onRefresh },
              { label: "Disconnect from the wizard", icon: "wifi", onSelect: onDisconnect },
            ]}
          />
          <Button icon="film" disabled={route === "blocked" || working} onClick={onNewVideo}>
            Restore a new video
          </Button>
        </div>
      </div>
      {route === "blocked" ? (
        <p className="muted jobs-hint">
          A new video can start once the current step has finished or been stopped.
        </p>
      ) : null}
      <p className="sr-only" role="status" aria-live="polite">
        {status}
      </p>
      {openError ? <Banner tone="error" role="alert" title={openError} /> : null}
      {loading && slow ? <p className="muted">Looking for your videos…</p> : null}
      {loadError ? (
        <Banner
          tone="warning"
          title="Could not list your videos"
          actions={
            <Button kind="secondary" size="md" icon="rotate" onClick={onRefresh}>
              Try again
            </Button>
          }
        >
          {loadError}
        </Banner>
      ) : null}
      {jobs !== null && jobs.length === 0 ? (
        <div className="jobs-empty">
          <Icon name="film" size={28} color="var(--leaf)" />
          <h2 className="h3">No videos yet</h2>
          <p className="muted">
            Each video you restore shows up here, with its result, so you can pick up where you
            left off.
          </p>
        </div>
      ) : null}
      {jobs !== null && jobs.length > 0 ? (
        <Card className="jobs-card">
          <h2 className="overline">Recent jobs</h2>
          <ul className="jobs-list">
            {jobs.map((entry) => {
              const badge = jobStatus(entry, entry.open ? pct : null);
              const action = jobAction(entry, here);
              const thumb = outputSrc(origin, entry.thumb_url);
              const result = entry.status === "result" ? outputSrc(origin, entry.output_url) : null;
              const meta = jobMeta(entry, now);
              const reasonId = `job-reason-${entry.job_id}`;
              const isOpening = opening === entry.job_id;
              return (
                <li key={entry.job_id} className={entry.open ? "job-row open" : "job-row"}>
                  {thumb ? (
                    <img className="job-thumb" src={thumb} alt="" loading="lazy" />
                  ) : (
                    <span className="job-thumb still-gap" aria-hidden="true" />
                  )}
                  <div className="job-text">
                    <h3 className="job-name h3">
                      {entry.title}
                      {entry.open ? <span className="job-here"> · open here</span> : null}
                    </h3>
                    {meta ? <span className="job-meta">{meta}</span> : null}
                    {result ? (
                      <a className="job-result" href={result} target="_blank" rel="noreferrer">
                        Watch the result<span className="sr-only"> of {entry.title} (opens a new tab)</span>
                      </a>
                    ) : null}
                  </div>
                  <Badge tone={TONE[badge.tone]} srPrefix="Status: " className="job-badge">
                    {badge.text}
                  </Badge>
                  <div className="job-action">
                    <Button
                      kind="secondary"
                      size="md"
                      disabled={action.kind === "none" || working}
                      aria-describedby={action.reason ? reasonId : undefined}
                      onClick={() => (action.kind === "show" ? onShowCurrent() : onOpen(entry))}
                    >
                      {isOpening ? "Opening…" : action.label}
                      <span className="sr-only"> {entry.title}</span>
                    </Button>
                    {action.reason ? (
                      <span id={reasonId} className="job-reason">
                        {action.reason}
                      </span>
                    ) : null}
                  </div>
                </li>
              );
            })}
          </ul>
        </Card>
      ) : null}
      {backLabel ? (
        <div className="row gap12 wrap">
          <Button kind="secondary" size="md" icon="chevron-left" onClick={onShowCurrent}>
            {backLabel}
          </Button>
        </div>
      ) : null}
    </section>
  );
}
