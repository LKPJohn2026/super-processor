// Changing the scenes by hand on the segments step: move the cut between two
// scenes, merge two neighbours, split one. The checks here give instant
// feedback with the server's own rules (segment_loop.check_coverage): times
// to the millisecond, each scene at least `min_segment_s` long unless it is
// the only one, at most `max_segments` scenes. The server stays the judge;
// these only stop a change it would refuse. Types only from ./api, which the
// transpile drops, so the node tests can load this file on its own.
import type { Segment, SegmentList } from "./api";

export type SceneLimits = { minS: number; maxScenes: number };
export type Range = { low: number; high: number };

// The server's limits, when an older server leaves them out of the view.
export const DEFAULT_LIMITS: SceneLimits = { minS: 3, maxScenes: 10 };

type Span = Pick<Segment, "start_s" | "end_s">;

export function limitsOf(list: Pick<SegmentList, "min_segment_s" | "max_segments">): SceneLimits {
  const minS = list.min_segment_s;
  const maxScenes = list.max_segments;
  return {
    minS: typeof minS === "number" && minS >= 0 ? minS : DEFAULT_LIMITS.minS,
    maxScenes:
      typeof maxScenes === "number" && maxScenes >= 1 ? maxScenes : DEFAULT_LIMITS.maxScenes,
  };
}

// Whole milliseconds, the precision the server saves and compares times at.
const ms = (seconds: number) => Math.round(seconds * 1000);
const seconds = (millis: number) => millis / 1000;

/** m:ss, with tenths when there are any ("4:32", "0:03.5", "62:03"). */
export function formatTime(value: number): string {
  const rounded = Math.round(Math.max(0, value) * 10) / 10;
  const minutes = Math.floor(rounded / 60);
  const rest = rounded - minutes * 60;
  const text = Number.isInteger(rest) ? String(rest).padStart(2, "0") : rest.toFixed(1).padStart(4, "0");
  return `${minutes}:${text}`;
}

/**
 * A time typed by the editor, in seconds: "272", "4:32", "4:32.5" or
 * "1:02:03". Null when it is not one.
 */
export function parseTime(text: string): number | null {
  const parts = text.trim().replace(",", ".").split(":");
  if (parts.length > 3 || parts.some((part) => !/^\d+(\.\d+)?$/.test(part))) return null;
  // Only the last part may have a fraction, and minutes and seconds after
  // a colon stay under 60 ("62:03" is fine: the app writes long videos so).
  if (parts.slice(0, -1).some((part) => part.includes("."))) return null;
  if (parts.slice(1).some((part) => Number(part) >= 60)) return null;
  const total = parts.reduce((sum, part) => sum * 60 + Number(part), 0);
  return Number.isFinite(total) ? total : null;
}

// Range ends said so that the shown time is itself allowed.
const shownLow = (value: number) => Math.ceil(value * 10 - 1e-6) / 10;
const shownHigh = (value: number) => Math.floor(value * 10 + 1e-6) / 10;
function rangeText(range: Range): string {
  return `${formatTime(shownLow(range.low))} and ${formatTime(shownHigh(range.high))}`;
}

const secondsText = (value: number) => `${Number(value.toFixed(3))} s`;

/**
 * Where the cut after scene `index` may go: far enough from the cuts on
 * either side that both scenes keep the minimum length. Null when there is
 * no such cut, or no room to move it.
 */
export function cutRange(segments: Span[], index: number, minS: number): Range | null {
  if (!Number.isInteger(index) || index < 0 || index >= segments.length - 1) return null;
  const low = ms(segments[index].start_s) + ms(minS);
  const high = ms(segments[index + 1].end_s) - ms(minS);
  return low <= high ? { low: seconds(low), high: seconds(high) } : null;
}

/** Where scene `index` may be split so both halves keep the minimum length. */
export function splitRange(segments: Span[], index: number, minS: number): Range | null {
  const scene = segments[index];
  if (!scene) return null;
  const low = ms(scene.start_s) + Math.max(ms(minS), 1);
  const high = ms(scene.end_s) - Math.max(ms(minS), 1);
  return low <= high ? { low: seconds(low), high: seconds(high) } : null;
}

export function clamp(value: number, range: Range): number {
  return Math.min(range.high, Math.max(range.low, value));
}

/** Why moving the cut after scene `index` to `at` would be refused, or null. */
export function checkMove(
  segments: Span[],
  index: number,
  at: number,
  limits: SceneLimits,
): string | null {
  if (!Number.isFinite(at)) return "Say where the cut goes, as a time.";
  if (!Number.isInteger(index) || index < 0 || index >= segments.length - 1) {
    return "There is no cut after that scene.";
  }
  const range = cutRange(segments, index, limits.minS);
  const scenes = `scenes ${index + 1} and ${index + 2}`;
  if (!range) {
    return `The cut between ${scenes} cannot move: both scenes need at least ${secondsText(limits.minS)}.`;
  }
  const atMs = ms(at);
  if (atMs < ms(range.low) || atMs > ms(range.high)) {
    return `Each scene needs at least ${secondsText(limits.minS)}, so the cut between ${scenes} can go between ${rangeText(range)}.`;
  }
  return checkCoverage(withCutAt(segments, index, at), limits);
}

/** Why merging scene `index` with the next one would be refused, or null. */
export function checkMerge(
  segments: Span[],
  index: number,
  limits: SceneLimits = DEFAULT_LIMITS,
): string | null {
  if (!Number.isInteger(index) || index < 0 || index >= segments.length) {
    return "Pick one of the scenes shown.";
  }
  if (index === segments.length - 1) {
    return `Scene ${index + 1} is the last one, so there is no next scene to merge it with.`;
  }
  return checkCoverage(withMerge(segments, index), limits);
}

/**
 * Why the server would refuse this whole list, or null: the same rules as
 * segment_loop.check_coverage (at least one scene, at most `maxScenes`,
 * each starting where the one before ends, from 0, and none under `minS`
 * unless it is the only one), compared to the millisecond. Run on the list
 * a change would make, so a short scene elsewhere is caught here too.
 */
export function checkCoverage(segments: Span[], limits: SceneLimits): string | null {
  if (!segments.length) return "There must be at least one scene.";
  if (segments.length > limits.maxScenes) {
    return `There can be at most ${limits.maxScenes} scenes.`;
  }
  let reach = 0;
  for (const segment of segments) {
    if (ms(segment.start_s) !== reach) {
      return "Each scene must start where the one before it ends. Reload the page to see the scenes as they are.";
    }
    reach = ms(segment.end_s);
  }
  if (segments.length > 1) {
    const short = segments.findIndex((segment) => ms(segment.end_s) - ms(segment.start_s) < ms(limits.minS));
    if (short >= 0) {
      return `Scene ${short + 1} would be shorter than ${secondsText(limits.minS)}. Merge it with a scene next to it, or move its cut, first.`;
    }
  }
  return null;
}

/** The scenes with scene `index` and the next one made one. */
export function withMerge<T extends Span>(segments: T[], index: number): T[] {
  if (index < 0 || index >= segments.length - 1) return segments;
  const merged = { ...segments[index], end_s: segments[index + 1].end_s };
  return [...segments.slice(0, index), merged, ...segments.slice(index + 2)];
}

/** The scenes with scene `index` split in two at `at`. */
export function withSplit<T extends Span>(segments: T[], index: number, at: number): T[] {
  const scene = segments[index];
  if (!scene) return segments;
  return [
    ...segments.slice(0, index),
    { ...scene, end_s: at },
    { ...scene, start_s: at },
    ...segments.slice(index + 1),
  ];
}

/** Why splitting scene `index` at `at` would be refused, or null. */
export function checkSplit(
  segments: Span[],
  index: number,
  at: number,
  limits: SceneLimits,
): string | null {
  if (!Number.isInteger(index) || index < 0 || index >= segments.length) {
    return "Pick one of the scenes shown.";
  }
  if (segments.length >= limits.maxScenes) {
    return `There can be at most ${limits.maxScenes} scenes. Merge two scenes first, then split this one.`;
  }
  const range = splitRange(segments, index, limits.minS);
  if (!range) {
    return `Scene ${index + 1} is too short to split: each half needs at least ${secondsText(limits.minS)}.`;
  }
  if (!Number.isFinite(at)) return "Say where to split, as a time like 1:30.";
  const atMs = ms(at);
  if (atMs < ms(range.low) || atMs > ms(range.high)) {
    return `Split scene ${index + 1} between ${rangeText(range)}, so each half is at least ${secondsText(limits.minS)}.`;
  }
  return checkCoverage(withSplit(segments, index, at), limits);
}

/** Whether scene `index` can be split at all, ignoring where. */
export function canSplit(segments: Span[], index: number, limits: SceneLimits): string | null {
  const range = splitRange(segments, index, limits.minS);
  return checkSplit(segments, index, range ? range.low : Number.NaN, limits);
}

/** A good first guess for a split: the middle of the scene, in tenths. */
export function splitSuggestion(segments: Span[], index: number, minS: number): number | null {
  const range = splitRange(segments, index, minS);
  if (!range) return null;
  const scene = segments[index];
  return clamp(Math.round(((scene.start_s + scene.end_s) / 2) * 10) / 10, range);
}

/** The scenes as they would be with the cut after `index` at `at` (for drawing a drag). */
export function withCutAt<T extends Span>(segments: T[], index: number, at: number): T[] {
  if (index < 0 || index >= segments.length - 1) return segments;
  return segments.map((segment, i) =>
    i === index ? { ...segment, end_s: at } : i === index + 1 ? { ...segment, start_s: at } : segment,
  );
}

/**
 * The cut's new time for a key on its handle, kept in its range: arrows
 * move it a second, Page Up / Page Down ten, Home and End to the ends.
 * Null for any other key.
 */
export function nudgeCut(at: number, key: string, range: Range): number | null {
  const steps: Record<string, number> = {
    ArrowLeft: -1,
    ArrowDown: -1,
    ArrowRight: 1,
    ArrowUp: 1,
    PageDown: -10,
    PageUp: 10,
  };
  if (key === "Home") return range.low;
  if (key === "End") return range.high;
  const step = steps[key];
  if (step === undefined) return null;
  return clamp(seconds(ms(at) + step * 1000), range);
}

/** How far, in pixels, a pointer must move on a cut before it is a drag and not a click. */
export const DRAG_THRESHOLD_PX = 4;

/** Whether a pointer that went down at `startX` and is now at `x` is dragging. */
export function isDrag(startX: number, x: number, threshold = DRAG_THRESHOLD_PX): boolean {
  return Math.abs(x - startX) >= threshold;
}

/**
 * Where a dragged cut is: its time when the pointer went down, moved by
 * how far the pointer has gone along a track `width` wide showing `total`
 * seconds, in tenths and kept in `range`. Relative, so grabbing a handle
 * off its centre does not make the cut jump; no movement gives back
 * `startAt` exactly.
 */
export function dragTime(
  startAt: number,
  startX: number,
  x: number,
  width: number,
  total: number,
  range: Range,
): number {
  if (!(width > 0) || !(total > 0) || x === startX) return startAt;
  const at = Math.round((startAt + ((x - startX) / width) * total) * 10) / 10;
  return clamp(at, range);
}

/** The time under a pointer at `x` on a timeline drawn from `left`, `width` wide, in tenths. */
export function timeAt(x: number, left: number, width: number, duration: number): number {
  if (!(width > 0) || !(duration > 0)) return 0;
  const share = Math.min(1, Math.max(0, (x - left) / width));
  return Math.round(share * duration * 10) / 10;
}

export function scenesText(count: number): string {
  return `${count} ${count === 1 ? "scene" : "scenes"}`;
}

export function cutName(index: number): string {
  return `Cut between scenes ${index + 1} and ${index + 2}`;
}

/** What a screen reader says for a cut handle. */
export function cutValueText(index: number, at: number): string {
  return `${cutName(index)}, at ${formatTime(at)}`;
}

// The server's words for a change made from a page showing older scenes
// (wizard._STALE_SEGMENTS). It answers 400 with only this text, so the page
// knows the refusal by it.
const STALE_TEXT = "the scenes changed since this page showed them";

/**
 * Whether a refused change means the page shows old scenes: another window
 * changed them (the revision moved on), or opened another job (409).
 */
export function isStaleScenes(message: string, status?: number): boolean {
  return status === 409 || message.toLowerCase().includes(STALE_TEXT);
}

export const STALE_SCENES_TEXT =
  "The scenes were changed in another window, so this page now shows them as they are. Look them over, then make your change again.";

export const STALE_SCENES_UNLOADED_TEXT =
  "The scenes were changed in another window, and this page could not load them again. Retry the connection or reload the page, then make your change again.";
