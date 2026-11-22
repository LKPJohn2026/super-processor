// The Result (08) and Saved (09) steps: pure functions, so the node tests
// can check them; types only from ./api.
import type { OutputVersion, WizardView } from "./api";

/** A part of the result marked on the player (Mark start / Mark end, I / O). */
export type MarkedRange = { start: number | null; end: number | null };

export const NO_RANGE: MarkedRange = { start: null, end: null };

/** `7:30`: whole seconds, for the marked part and the note. */
function clock(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = String(total % 60).padStart(2, "0");
  return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${secs}` : `${minutes}:${secs}`;
}

/**
 * The range after marking its start or end at `time`. A start after the
 * end (or an end before the start) begins the range again from that mark.
 */
export function markRange(range: MarkedRange, which: "start" | "end", time: number): MarkedRange {
  const at = Number.isFinite(time) && time > 0 ? time : 0;
  if (which === "start") {
    return range.end !== null && range.end > at ? { start: at, end: range.end } : { start: at, end: null };
  }
  return range.start !== null && range.start < at ? { start: range.start, end: at } : { start: null, end: at };
}

/** The range with both ends, in whole seconds (the end rounded up), or null. */
export function completeRange(range: MarkedRange): { start: number; end: number } | null {
  if (range.start === null || range.end === null) return null;
  const start = Math.floor(range.start);
  const end = Math.max(start + 1, Math.ceil(range.end));
  return { start, end };
}

/** "7:30 – 7:40", or what is still to mark; "" when nothing is marked. */
export function rangeText(range: MarkedRange): string {
  const whole = completeRange(range);
  if (whole) return `${clock(whole.start)} – ${clock(whole.end)}`;
  if (range.start !== null) return `From ${clock(range.start)} · now mark the end`;
  if (range.end !== null) return `Up to ${clock(range.end)} · now mark the start`;
  return "";
}

/** The button that sends the note: "Re-render 7:30–7:40". */
export function reviseLabel(range: MarkedRange): string {
  const whole = completeRange(range);
  return whole ? `Re-render ${clock(whole.start)}–${clock(whole.end)}` : "Re-render with this note";
}

export type ResultChipKey = "less-detail" | "smoother" | "sharper" | "seam";
export type ResultChip = { key: ResultChipKey; text: string; note: string };

/** The "Fix part of it" chips: each adds words to the note Gemini reads. */
export const RESULT_CHIPS: readonly ResultChip[] = [
  { key: "less-detail", text: "Less AI detail", note: "less AI detail" },
  { key: "smoother", text: "Smoother", note: "smoother, with less flicker" },
  { key: "sharper", text: "Sharper", note: "sharper" },
  { key: "seam", text: "Fix a seam", note: "smooth the seam" },
];

/** The chips offered: "Fix a seam" only when the render found a possible seam. */
export function resultChips(seams: readonly { time_s: number }[] | null | undefined): ResultChip[] {
  return RESULT_CHIPS.filter((chip) => chip.key !== "seam" || Boolean(seams?.length));
}

/**
 * The note sent for "Re-render": the marked part, the chips chosen and the
 * editor's own words. "Fix a seam" with no part marked names the first
 * possible seam, a second either side (as the seam note suggests).
 */
export function reviseNote(
  range: MarkedRange,
  chosen: readonly ResultChipKey[],
  typed: string,
  seams: readonly { time_s: number }[] = [],
): string {
  const whole = completeRange(range);
  const fixes = RESULT_CHIPS.filter((chip) => chosen.includes(chip.key)).map((chip) => {
    if (chip.key !== "seam" || whole || !seams.length) return chip.note;
    const at = seams[0].time_s;
    return `smooth the seam from ${clock(Math.max(0, at - 1))} to ${clock(at + 1)}`;
  });
  const words = typed.trim();
  const parts: string[] = [];
  if (fixes.length) parts.push(fixes.join("; "));
  if (words) parts.push(words);
  if (!parts.length) return "";
  const said = parts.join(". ");
  return whole ? `From ${clock(whole.start)} to ${clock(whole.end)}: ${said}` : said;
}

/** Why the note cannot be sent yet, or null. */
export function reviseProblem(range: MarkedRange, note: string): string | null {
  if (note) return null;
  const whole = completeRange(range);
  return whole
    ? `Say what to change in ${clock(whole.start)}–${clock(whole.end)}: pick a fix or type it.`
    : "Pick a fix or say what to change, and mark the part if it is only part of the video.";
}

export type Timeline = {
  // Where each scene starts (after the first), in percent of the video.
  scenes: { pct: number; time: string }[];
  // The render's possible seams, in percent.
  seams: { pct: number; time: string }[];
  marked: { left: number; width: number } | null;
  // Read out in place of the drawing.
  summary: string;
};

/** The seams timeline under the viewer: scene cuts, possible seams, the marked part. */
export function resultTimeline(
  view: Pick<WizardView, "job" | "seams">,
  durationS: number | null,
  range: MarkedRange,
): Timeline | null {
  const scenes = view.job?.scenes ?? [];
  const lastEnd = scenes.length ? scenes[scenes.length - 1].end_s : 0;
  const length = durationS && durationS > 0 ? durationS : lastEnd;
  if (!(length > 0)) return null;
  const pct = (seconds: number) => Math.max(0, Math.min(100, (seconds / length) * 100));
  const cuts = scenes
    .slice(1)
    .filter((scene) => scene.start_s > 0 && scene.start_s < length)
    .map((scene) => ({ pct: pct(scene.start_s), time: clock(scene.start_s) }));
  const seams = (view.seams ?? [])
    .filter((seam) => seam.time_s >= 0 && seam.time_s <= length)
    .map((seam) => ({ pct: pct(seam.time_s), time: clock(seam.time_s) }));
  const whole = completeRange(range);
  const marked = whole
    ? { left: pct(whole.start), width: Math.max(0.6, pct(Math.min(whole.end, length)) - pct(whole.start)) }
    : null;
  const parts: string[] = [];
  parts.push(cuts.length ? `Scenes change at ${cuts.map((cut) => cut.time).join(", ")}` : "One scene");
  if (seams.length) parts.push(`possible seam${seams.length > 1 ? "s" : ""} at ${seams.map((seam) => seam.time).join(", ")}`);
  if (whole) parts.push(`marked ${clock(whole.start)} to ${clock(whole.end)}`);
  return { scenes: cuts, seams, marked, summary: `${parts.join("; ")}.` };
}

/** "Changes so far": each version a note asked for, oldest first. */
export function changesSoFar(versions: OutputVersion[] | null | undefined): { note: string; version: string }[] {
  return [...(versions ?? [])]
    .sort((a, b) => a.number - b.number)
    .filter((version) => (version.note ?? "").trim())
    .map((version) => ({ note: (version.note ?? "").trim(), version: `Version ${version.number}` }));
}

/** The version the job's output is now, if the server lists it. */
export function currentVersion(versions: OutputVersion[] | null | undefined): OutputVersion | null {
  return (versions ?? []).find((version) => version.current) ?? null;
}

/** The viewer's tag on the result: "Restored · 2× · version 2". */
export function resultTag(scale: number | null | undefined, version: Pick<OutputVersion, "number"> | null): string {
  const base = scale === 1 ? "Cleaned up" : `Restored${typeof scale === "number" ? ` · ${scale}×` : ""}`;
  return version ? `${base} · version ${version.number}` : base;
}

/** What "Start over from the scenes" deletes and keeps (S7), in plain words. */
export function startOverText(view: Pick<WizardView, "job" | "versions">): { deletes: string; keeps: string } {
  const approved = view.job?.approved_count ?? 0;
  const previews =
    approved > 1 ? `all ${approved} approved previews` : approved === 1 ? "the approved preview" : "the previews";
  const kept = currentVersion(view.versions);
  return {
    deletes: `the scene split, ${previews} and their settings`,
    keeps: kept
      ? `Your restored video stays in Versions as version ${kept.number}, and Gemini keeps your note so it doesn’t repeat what you disliked.`
      : "Your restored video is kept in Versions, and Gemini keeps your note so it doesn’t repeat what you disliked.",
  };
}

/** The facts under the saved file's name: only what is known. */
export function savedFacts(
  picture: { width: number; height: number; duration: number } | null,
  version: Pick<OutputVersion, "size" | "scale"> | null,
  sizeText: (bytes: number) => string,
): string {
  const parts: string[] = [];
  if (picture && picture.width > 0 && picture.height > 0) parts.push(`${picture.width}×${picture.height}`);
  if (picture && Number.isFinite(picture.duration) && picture.duration > 0) parts.push(clock(picture.duration));
  const size = version ? sizeText(version.size) : "";
  if (size) parts.push(size);
  if (version && typeof version.scale === "number" && version.scale > 1) parts.push(`${version.scale}× the original size`);
  return parts.join(" · ");
}

/** The output's place, relative to the wizard's jobs folder. */
export function outputPlace(jobId: string | null | undefined): string {
  return jobId ? `jobs/${jobId}/output.mp4` : "output.mp4";
}

/**
 * The scene the before/after stills come from: the middle one with a
 * still, and the moment to take the restored frame from (its middle).
 */
export function compareScene(
  view: Pick<WizardView, "job">,
): { index: number; label: string; stillUrl: string; atS: number } | null {
  const scenes = (view.job?.scenes ?? []).filter((scene) => scene.still_url);
  if (!scenes.length) return null;
  const scene = scenes[Math.floor(scenes.length / 2)];
  return {
    index: scene.index,
    label: scene.label?.trim() || `Scene ${scene.index + 1}`,
    stillUrl: scene.still_url as string,
    atS: (scene.start_s + scene.end_s) / 2,
  };
}
