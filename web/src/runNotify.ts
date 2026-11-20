// Desktop notifications when a long run (finding the scenes, the render)
// finishes or fails while the tab is hidden. Only pure decisions live here;
// LiveWizard asks the browser. Permission is asked only from the editor's
// own click on "Notify me when it's done", never on load.
import type { RunKind, WizardView } from "./api";

export type RunNotice = { title: string; body: string; tag: string };

// The runs worth a notification: they take minutes. A scene preview does not.
const LONG_RUNS: readonly RunKind[] = ["segmenting", "render"];

export const NOTIFY_KEY = "super-processor.notify";

/**
 * The notification `next` calls for, given the view shown before it: only
 * when the page saw a long run working and the same run has now finished
 * or failed. A stop is the editor's own doing, so it never notifies, and a
 * run the page never saw working (a job just opened) does not either.
 */
export function runNotice(
  prev: Pick<WizardView, "run"> | null | undefined,
  next: Pick<WizardView, "run" | "error" | "job" | "segments">,
): RunNotice | null {
  const before = prev?.run;
  const after = next.run;
  if (!before || !after || before.id !== after.id) return null;
  if (before.outcome !== "running" || !LONG_RUNS.includes(after.kind)) return null;
  if (after.outcome !== "done" && after.outcome !== "failed") return null;
  const tag = `super-processor-${after.id}`;
  const title = (next.job?.title ?? "").trim();
  const named = title ? `${title}: ` : "";
  if (after.outcome === "failed") {
    const reason = (next.error ?? "").trim() || "Open the tab to see what happened.";
    return {
      title: after.kind === "render" ? "The render failed" : "Finding the scenes failed",
      body: `${named}${reason}`,
      tag,
    };
  }
  if (after.kind === "render") {
    return {
      title: "Your restored video is ready",
      body: `${named}watch it and keep it, or ask for a change.`,
      tag,
    };
  }
  const count = next.segments?.segments.length || next.job?.scene_count || 0;
  const found = count === 1 ? "1 scene found" : count ? `${count} scenes found` : "the scenes are found";
  return {
    title: "The scenes are ready",
    body: `${named}${found}. Check them when you are back.`,
    tag,
  };
}

export type Permission = "unsupported" | "default" | "granted" | "denied";

/** The browser's notification permission, or "unsupported". */
export function notifyPermission(scope: { Notification?: { permission?: string } } | undefined): Permission {
  const api = scope?.Notification;
  if (!api) return "unsupported";
  const value = api.permission;
  return value === "granted" || value === "denied" ? value : "default";
}

type Storage = Pick<globalThis.Storage, "getItem" | "setItem">;

/** The editor's saved choice; off when nothing is saved or storage is blocked. */
export function savedChoice(storage: () => Storage | null | undefined): boolean {
  try {
    return storage()?.getItem(NOTIFY_KEY) === "on";
  } catch {
    return false;
  }
}

/** Save the choice; false when the browser would not keep it. */
export function saveChoice(storage: () => Storage | null | undefined, on: boolean): boolean {
  try {
    const store = storage();
    if (!store) return false;
    store.setItem(NOTIFY_KEY, on ? "on" : "off");
    return true;
  } catch {
    return false;
  }
}

export type NotifyToggle = {
  shown: boolean;
  checked: boolean;
  disabled: boolean;
  // A short line under the toggle, or null.
  hint: string | null;
};

/** The "Notify me when it's done" toggle, from the choice and the permission. */
export function notifyToggle(on: boolean, permission: Permission): NotifyToggle {
  if (permission === "unsupported") return { shown: false, checked: false, disabled: true, hint: null };
  if (permission === "denied") {
    return {
      shown: true,
      checked: false,
      disabled: true,
      hint: "This browser blocks notifications from this page. You can allow them in its site settings.",
    };
  }
  const checked = on && permission === "granted";
  return {
    shown: true,
    checked,
    disabled: false,
    hint: checked ? "You will get a notification if this tab is in the background when it ends." : null,
  };
}

/** Show it now: a notice, the editor asked, the browser allows it, and the tab is hidden. */
export function shouldNotify(
  notice: RunNotice | null,
  on: boolean,
  permission: Permission,
  hidden: boolean,
): notice is RunNotice {
  return notice !== null && on && permission === "granted" && hidden;
}
