import type { ReactNode } from "react";
import { Icon } from "./Icon";

/** The design's thin bar (gen.py `progress()`), as a real progressbar. */
export function Progress({ value, label }: { value: number; label: string }) {
  const pct = Math.max(0, Math.min(100, Math.round(value)));
  return (
    <div
      className="progress-bar"
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct}
    >
      <div style={{ width: `${pct}%` }} />
    </div>
  );
}

export type StageState = "done" | "running" | "waiting" | "skipped";

const STATE_WORD: Record<StageState, string> = {
  done: "Done",
  running: "Working",
  waiting: "Waiting",
  skipped: "Skipped",
};

/**
 * One line of a stage list (gen.py `stage()`): an icon, the label and a
 * short meta line. The state is also said in words for screen readers.
 */
export function StageRow({ state, label, meta }: { state: StageState; label: ReactNode; meta?: ReactNode }) {
  const icon =
    state === "done" ? (
      <Icon name="check-circle" color="var(--ok)" />
    ) : state === "running" ? (
      <Icon name="clock" color="var(--focus)" />
    ) : state === "skipped" ? (
      <Icon name="info" color="var(--ink-subtle)" />
    ) : (
      <Icon name="clock" color="var(--line-strong)" />
    );
  return (
    <li className={`stage-row ${state}`}>
      {icon}
      <span className="lbl grow">
        <span className="sr-only">{STATE_WORD[state]}: </span>
        {label}
      </span>
      {meta ? <span className="caption subtle">{meta}</span> : null}
    </li>
  );
}
