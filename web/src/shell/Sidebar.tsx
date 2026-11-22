import { useEffect, useState, type ReactNode } from "react";
import { outputSrc, type WizardView } from "../api";
import { jobTitle, sidebarSteps, sourceFacts, sourceStills, type SideStep } from "../jobView";
import { Icon } from "../ui";

const GO_BACK: Record<string, string> = {
  file: "pick another file",
  segments: "go back to the scenes",
};

/** The job's still, trying each candidate in turn; a plain block when none loads. */
function SourceThumb({ origin, view }: { origin: string; view: WizardView }) {
  const candidates = sourceStills(view)
    .map((url) => outputSrc(origin, url))
    .filter((url): url is string => Boolean(url));
  const key = candidates.join("|");
  const [failed, setFailed] = useState(0);
  useEffect(() => setFailed(0), [key]);
  const src = candidates[failed];
  if (!src) {
    return (
      <span className="src-thumb src-thumb-gap" aria-hidden="true">
        <Icon name="film" size={28} color="var(--ink-subtle)" />
      </span>
    );
  }
  return <img className="src-thumb" src={src} alt="" onError={() => setFailed((value) => value + 1)} />;
}

function StepMarker({ step, number }: { step: SideStep; number: number }) {
  if (step.state === "done") return <Icon name="check" size={14} color="#fff" strokeWidth={3} />;
  if (step.state === "error") return <>!</>;
  return <>{number}</>;
}

const STATE_WORDS: Record<SideStep["state"], string> = {
  done: "done",
  current: "now",
  upcoming: "to come",
  error: "needs attention",
};

/**
 * The job panel on the left (mockups, `sidebar()`): the source still, its
 * name and the facts the server sends, then the five steps with what each
 * one is at. Steps the editor can go back to are buttons.
 */
export function Sidebar({
  origin,
  view,
  busy,
  failed,
  onGoto,
  links,
}: {
  origin: string;
  view: WizardView;
  busy: boolean;
  failed: boolean;
  onGoto: (target: string) => void;
  // Side links for things that exist (the Gemini key settings).
  links?: ReactNode;
}) {
  const steps = sidebarSteps(view, busy, failed);
  const title = jobTitle(view);
  const facts = sourceFacts(view);
  const hasSource = Boolean(view.job_id);
  return (
    <aside className="sidebar" aria-label="This job">
      {hasSource ? (
        <div className="side-source">
          <SourceThumb origin={origin} view={view} />
          <div className="side-source-text">
            {title ? <p className="body-s strong side-name">{title}</p> : null}
            {facts ? <p className="caption subtle">{facts}</p> : null}
          </div>
        </div>
      ) : null}
      <h2 className="overline">This job</h2>
      <nav aria-label="Steps of this job">
        <ol className="side-steps">
          {steps.map((step, index) => (
            <li key={step.key} className={`step ${step.state}`} aria-current={step.state === "current" || step.state === "error" ? "step" : undefined}>
              <span className="marker" aria-hidden="true">
                <StepMarker step={step} number={index + 1} />
              </span>
              <span className="step-text">
                {step.nav ? (
                  <button
                    type="button"
                    className="label step-link"
                    aria-label={`${step.label}, ${STATE_WORDS[step.state]}: ${GO_BACK[step.nav] ?? "go back"}`}
                    onClick={() => onGoto(step.nav!)}
                  >
                    {step.label}
                  </button>
                ) : (
                  <span className="label">
                    {step.label}
                    <span className="sr-only">, {STATE_WORDS[step.state]}</span>
                  </span>
                )}
                {step.detail ? <span className="detail">{step.detail}</span> : null}
              </span>
            </li>
          ))}
        </ol>
      </nav>
      {links ? <div className="side-links">{links}</div> : null}
    </aside>
  );
}
