// What the page shows around a run: the Stop button, the note after a stop,
// the Resume or Stop choice for an interrupted job, and a render that failed.
// Pure functions of the view, so the node tests can check the decisions.
import type { WizardView } from "./api";

type Step = WizardView["step"];

const SEGMENTING: readonly Step[] = ["sampling", "proposing"];
const RUN_STEPS: readonly Step[] = [...SEGMENTING, "analyzing", "rendering"];

/** The tone a step is shown in. A stop is neither a success nor a failure. */
export type Tone = "working" | "interrupted" | "stopped" | "failed" | "ok";

export function viewTone(view: WizardView): Tone {
  if (view.interrupted && RUN_STEPS.includes(view.step)) return "interrupted";
  if (RUN_STEPS.includes(view.step)) return "working";
  if (view.error) return "failed";
  if (view.stopped) return "stopped";
  return "ok";
}

/**
 * The note to show after the editor stopped a run, on the step it went back
 * to; null while a run works or waits for Resume, and once a later action
 * cleared it. The server clears an error when it lands a stop, so an error
 * shown beside it is a newer one and wins.
 */
export function stoppedNote(view: WizardView): string | null {
  const tone = viewTone(view);
  if (tone !== "stopped") return null;
  const text = (view.stopped ?? "").trim();
  return text || null;
}

export type StopControl = {
  // Show a Stop button at all.
  shown: boolean;
  disabled: boolean;
  label: string;
  // Read out (aria-live) while a stop is on its way.
  status: string | null;
};

/**
 * The Stop button while finding the segments or rendering. It shows only
 * when the server says a stop can be asked for, and stays (disabled, as
 * "Stopping…") once one was asked for and has not landed yet. `posting` is
 * true while this page's own stop request is on its way.
 */
export function stopControl(view: WizardView, posting: boolean): StopControl {
  const render = view.step === "rendering";
  const onStep = render || SEGMENTING.includes(view.step);
  const asked = Boolean(view.cancel_requested);
  const shown = onStep && !view.interrupted && (Boolean(view.cancellable) || asked);
  const stopping = asked || posting;
  return {
    shown,
    disabled: !shown || stopping || !view.cancellable,
    label: stopping ? "Stopping…" : render ? "Stop the render" : "Stop",
    status:
      shown && asked
        ? render
          ? "Stopping the render. The graphics card can take up to a minute to let go."
          : "Stopping. Gemini is told to stop; this takes a moment."
        : null,
  };
}

/** True when the Render button belongs on the page, even after a failed render. */
export function offersRender(view: WizardView): boolean {
  return view.step === "ready_to_render" && !view.interrupted;
}

export type ReadyText = { title: string; lede: string; renderLabel: string };

/** Wording for "Ready to render", after a failed or stopped render too. */
export function readyText(view: WizardView): ReadyText {
  const tone = viewTone(view);
  const count = view.looks?.shots.length ?? view.segments?.segments.length ?? 0;
  const scenes = count
    ? `Your ${count} approved ${count === 1 ? "segment and its settings are" : "segments and their settings are"} kept.`
    : "Your approved segments and their settings are kept.";
  if (tone === "failed") {
    return { title: "The render didn’t finish", lede: scenes, renderLabel: "Try the render again" };
  }
  if (tone === "stopped") {
    return { title: "Render stopped", lede: scenes, renderLabel: "Render the whole video" };
  }
  return {
    title: "Ready to render",
    lede: "Every segment is approved. Check the output size, then render the whole video with these settings.",
    renderLabel: "Render the whole video",
  };
}

/** The result screen's heading: after a stop it shows the earlier video. */
export function resultTitle(view: WizardView): string {
  const tone = viewTone(view);
  if (tone === "stopped") return "Render stopped";
  return "Result";
}

export type InterruptedChoice = {
  title: string;
  lede: string;
  resumeLabel: string;
  stopLabel: string;
  // What Stop does here, said before the editor presses it.
  stopHint: string;
};

/**
 * The Resume or Stop choice for a job left mid-step. Nothing runs until the
 * editor chooses, so reopening a job never spends GPU time or Gemini quota.
 */
export function interruptedChoice(view: WizardView): InterruptedChoice {
  const lead = "Nothing runs until you choose.";
  if (view.step === "rendering") {
    return {
      title: "The render was interrupted",
      lede: `${lead} Resume starts the render again on this computer’s GPU.`,
      resumeLabel: "Resume the render",
      stopLabel: "Stop",
      stopHint: "Stop keeps your approved segments, and your earlier video if there is one.",
    };
  }
  if (view.step === "analyzing") {
    const n = (view.segment_index ?? 0) + 1;
    return {
      title: `Segment ${n} was not looked at yet`,
      lede: `${lead} Resume asks Gemini about segment ${n} and makes its preview.`,
      resumeLabel: "Resume",
      stopLabel: "Stop",
      stopHint: "Stop goes back to your approved segments; nothing is deleted.",
    };
  }
  return {
    title: "Finding the segments was interrupted",
    lede: `${lead} Resume asks Gemini to find the segments again.`,
    resumeLabel: "Resume",
    stopLabel: "Stop",
    stopHint: "Stop goes back to the segments you had, or to picking the file.",
  };
}
