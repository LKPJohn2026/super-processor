// The Scene preview step (mockup 05-preview, S6): its heading, the Gemini
// check box, "What this scene gets" in plain words, AI detail, the
// "Something off?" chips and the note they build, and the comparison
// viewer's arithmetic. Pure functions, so the node tests can check them;
// types only from ./api.
import type { LookShot, ShotCheck, ShotLook } from "./api";

/** `4:32` or `1:02:05` (the same as jobView's clockTime, kept here so this file stands alone). */
function clock(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = String(total % 60).padStart(2, "0");
  return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${secs}` : `${minutes}:${secs}`;
}

/** "Step 3 of 5 · Previews · Scene 3 of 5". */
export function previewOverline(index: number, count: number): string {
  const scene = count > 0 ? `Scene ${index + 1} of ${count}` : `Scene ${index + 1}`;
  return `Step 3 of 5 · Previews · ${scene}`;
}

/** The H1: the scene's label, else "Scene 3". */
export function previewTitle(shot: Pick<LookShot, "label"> | null, index: number): string {
  return shot?.label?.trim() || `Scene ${index + 1}`;
}

/** The scene's range beside the H1: "4:32–9:10". */
export function previewRange(shot: Pick<LookShot, "start_s" | "end_s"> | null): string {
  return shot ? `${clock(shot.start_s)}–${clock(shot.end_s)}` : "";
}

type NamedScene = { index: number; label: string };

/**
 * The primary button: "Approve scene 3 — next: Night street", or
 * "— next: Render" on the last scene.
 */
export function approveLabel(index: number, count: number, scenes: readonly NamedScene[]): string {
  const here = `Approve scene ${index + 1}`;
  if (count > 0 && index + 1 >= count) return `${here} — next: Render`;
  const next = scenes.find((scene) => scene.index === index + 1);
  const label = next?.label?.trim() || `scene ${index + 2}`;
  return `${here} — next: ${label}`;
}

/** Gemini's SPAN strength (0 adds no AI detail, 1 all of it) as a word. */
export function strengthWord(strength: number): string {
  if (!(strength > 0)) return "None";
  if (strength <= 0.35) return "Gentle";
  if (strength <= 0.65) return "Medium";
  return "Strong";
}

/** How much of a 0–1 clean-up value: light, medium or strong. */
function amount(value: number): string {
  if (value <= 0.34) return "light";
  if (value <= 0.67) return "medium";
  return "strong";
}

function percent(delta: number): string {
  const rounded = Math.round(delta * 100);
  return `${rounded > 0 ? "+" : ""}${rounded}%`;
}

export type GetsIcon = "layers" | "sparkles" | "eye" | "shield" | "film";
export type GetsItem = { icon: GetsIcon; text: string; amount: string };

const EPS = 1e-6;

/**
 * "What this scene gets", from the scene's real look settings: each
 * clean-up or finishing value away from neutral, and the AI detail (or
 * that it is only resized).
 */
export function sceneGets(
  shot: Pick<LookShot, "strength" | "look" | "use_upscale" | "dpir" | "dpir_on_gpu">,
  scale: number | null,
): GetsItem[] {
  const look: Partial<ShotLook> = shot.look ?? {};
  const items: GetsItem[] = [];
  const deblock = look.deblock ?? 0;
  if (deblock > EPS) items.push({ icon: "layers", text: "Removes blockiness", amount: amount(deblock) });
  const dpir = shot.dpir ?? 0;
  const denoise = look.denoise ?? 0;
  if (dpir > 0) {
    // DPIR replaces the look's denoise; without a GPU the strongest FFmpeg one stands in.
    items.push(
      shot.dpir_on_gpu === false
        ? { icon: "sparkles", text: "Strongest denoise (no GPU for DPIR)", amount: "" }
        : { icon: "sparkles", text: "Strong denoise (DPIR)", amount: `level ${dpir} of 3` },
    );
  } else if (denoise > EPS) {
    items.push({ icon: "sparkles", text: "Reduces noise", amount: amount(denoise) });
  }
  if (scale !== 1) {
    if (shot.use_upscale === false) {
      items.push({ icon: "film", text: "Just resized, no AI detail", amount: scale ? `${scale}×` : "" });
    } else {
      items.push({ icon: "sparkles", text: "AI detail", amount: strengthWord(shot.strength).toLowerCase() });
    }
  }
  const contrast = (look.contrast ?? 1) - 1;
  if (Math.abs(contrast) > EPS) {
    items.push({ icon: "eye", text: contrast > 0 ? "More contrast" : "Less contrast", amount: percent(contrast) });
  }
  const brightness = look.brightness ?? 0;
  if (Math.abs(brightness) > EPS) {
    items.push({ icon: "eye", text: brightness > 0 ? "Brighter" : "Darker", amount: percent(brightness) });
  }
  const saturation = (look.saturation ?? 1) - 1;
  if (Math.abs(saturation) > EPS) {
    items.push({ icon: "eye", text: saturation > 0 ? "More colour" : "Less colour", amount: percent(saturation) });
  }
  const gamma = (look.gamma ?? 1) - 1;
  if (Math.abs(gamma) > EPS) {
    items.push({ icon: "eye", text: gamma > 0 ? "Lifts the shadows" : "Deepens the shadows", amount: percent(gamma) });
  }
  const grain = look.grain ?? 0;
  if (grain > EPS) items.push({ icon: "film", text: "Adds film grain", amount: amount(grain) });
  return items;
}

const PROBLEM_WORDS: Record<string, string> = {
  identity_change: "a face changed",
  bad_anatomy: "hands or bodies look wrong",
  garbled_text: "text is garbled",
  waxy_skin: "skin looks waxy",
  oversharpened: "too sharp",
  halos: "halos round edges",
  fake_texture: "invented texture",
  color_shift: "the colours shifted",
  too_soft: "too soft",
  flicker: "flicker",
};

export type CheckState = { tone: "success" | "warning"; title: string; text: string; checked: boolean };

function sentence(text: string): string {
  const trimmed = text.trim();
  if (!trimmed) return "";
  return /[.!?]$/.test(trimmed) ? trimmed : `${trimmed}.`;
}

/**
 * The Gemini check box from the real check: passed (success), found a
 * problem (warning), or not checked at all (warning, S6).
 */
export function checkState(check: ShotCheck | null | undefined, shimmer?: number | null): CheckState {
  const note = sentence(check?.note ?? "");
  const jumpy =
    typeof shimmer === "number" ? ` Texture is ${shimmer.toFixed(1)}× as jumpy as the source.` : "";
  if (!check || check.ok === null) {
    return {
      tone: "warning",
      title: "Gemini couldn’t check this preview",
      text: `Look closely at faces, hands, text and fine patterns before you approve it.${jumpy}`,
      checked: false,
    };
  }
  if (check.ok) {
    return { tone: "success", title: "Checked by Gemini", text: (note || "No problems found.") + jumpy, checked: true };
  }
  const found = check.problems.map((problem) => PROBLEM_WORDS[problem] ?? problem.replace(/_/g, " "));
  let text = `Found: ${found.join(", ") || "a problem"}.`;
  if (note) text += ` ${note}`;
  if (check.adjusted) text += " Settings were adjusted and the preview redone.";
  return { tone: "warning", title: "Gemini found a problem", text: text + jumpy, checked: true };
}

/** The tag on the restored side: "Restored · 2×", "Resized · 2×", or "Cleaned up". */
export function afterTag(scale: number | null, useUpscale: boolean | undefined): string {
  if (scale === 1) return "Cleaned up";
  const times = typeof scale === "number" ? ` · ${scale}×` : "";
  return useUpscale === false ? `Resized${times}` : `Restored${times}`;
}

export type FixChipKey = "sharp" | "painted" | "waxy" | "flicker" | "dark";
export type FixChip = { key: FixChipKey; text: string; note: string };

/** "Something off?" chips: each only adds words to the note sent to Gemini with "Redo this preview". */
export const FIX_CHIPS: readonly FixChip[] = [
  { key: "sharp", text: "Too sharp", note: "It looks too sharp; make it softer." },
  { key: "painted", text: "Looks painted", note: "It looks painted; keep more of the original texture." },
  { key: "waxy", text: "Skin looks waxy", note: "Skin looks waxy; keep faces natural." },
  { key: "flicker", text: "Flickers", note: "The picture flickers or crawls; make it steadier." },
  { key: "dark", text: "Too dark", note: "It is too dark; brighten it a little." },
];

/** The note sent with "Redo this preview": the chosen chips' words in order, then the typed words. */
export function fixNote(chosen: readonly FixChipKey[], typed: string): string {
  const parts = FIX_CHIPS.filter((chip) => chosen.includes(chip.key)).map((chip) => chip.note);
  const words = typed.trim();
  if (words) parts.push(words);
  return parts.join(" ");
}

/** What "Try Gemini again" (S6) sends: the redo needs words, so it asks for a fresh check. */
export const RECHECK_NOTE = "Gemini could not check the last preview: choose the settings again and check the new preview.";
/** What "Preview a different 3 seconds" sends with resample when nothing is typed. */
export const RESAMPLE_NOTE = "Show a different part of this scene.";

// ------------------------------------------------------------ viewer

export type ViewerMode = "wipe" | "side" | "ab";

/** The wipe's place, 0–100 (% from the left), kept off the very edges. */
export function clampWipe(value: number): number {
  if (!Number.isFinite(value)) return 50;
  return Math.min(100, Math.max(0, value));
}

/** A key on the wipe handle: the new place, or null when the key is not the handle's. */
export function wipeKey(key: string, value: number, big = false): number | null {
  const step = big ? 20 : 5;
  if (key === "ArrowLeft" || key === "ArrowDown") return clampWipe(value - step);
  if (key === "ArrowRight" || key === "ArrowUp") return clampWipe(value + step);
  if (key === "PageDown") return clampWipe(value - 20);
  if (key === "PageUp") return clampWipe(value + 20);
  if (key === "Home") return 0;
  if (key === "End") return 100;
  return null;
}

export type ViewerAction = "play" | "back" | "forward" | "flip" | "zoom" | "markStart" | "markEnd";

/** The viewer's keys (s9): Space plays, ←/→ step a frame, B flips before/after, Z zooms to 100%. */
export function viewerKey(key: string): ViewerAction | null {
  if (key === " " || key === "Spacebar") return "play";
  if (key === "ArrowLeft") return "back";
  if (key === "ArrowRight") return "forward";
  if (key === "b" || key === "B") return "flip";
  if (key === "z" || key === "Z") return "zoom";
  // Only a viewer that marks a range (the Result) acts on these.
  if (key === "i" || key === "I") return "markStart";
  if (key === "o" || key === "O") return "markEnd";
  return null;
}

/** The player's time: "0:01.2 / 0:03.0". */
export function playerTime(current: number, duration: number): string {
  const one = (seconds: number) => {
    const safe = Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
    const tenths = Math.floor(safe * 10 + 1e-6);
    const minutes = Math.floor(tenths / 600);
    const rest = (tenths % 600) / 10;
    return `${minutes}:${rest.toFixed(1).padStart(4, "0")}`;
  };
  return `${one(current)} / ${one(duration)}`;
}

/** A time stepped by `frames` frames, kept inside the clip. */
export function stepTime(current: number, duration: number, frames: number, frameS: number): number {
  const end = Number.isFinite(duration) && duration > 0 ? duration : 0;
  return Math.min(end, Math.max(0, current + frames * frameS));
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

/** The raw numbers, for Advanced settings and the Ready list: "strength 0.40 · deblock 0.60". */
export function settingsSummary(strength: number, look: Partial<Record<string, number>>): string {
  const parts = [`strength ${strength.toFixed(2)}`];
  for (const [name, neutral] of Object.entries(NEUTRAL)) {
    const value = look[name] ?? neutral;
    if (Math.abs(value - neutral) > 1e-6) parts.push(`${name} ${value.toFixed(2)}`);
  }
  return parts.join(" · ");
}

/** Whether the shown preview exists but Gemini did not check it (S6). */
export function previewUnchecked(view: { step: string; segment_preview?: { shot: Pick<LookShot, "after_url" | "check"> | null } | null }): boolean {
  if (view.step !== "preview_segment") return false;
  const shot = view.segment_preview?.shot;
  return Boolean(shot?.after_url) && !checkState(shot?.check).checked;
}
