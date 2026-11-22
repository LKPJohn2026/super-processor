// The Scenes step (mockup 04-scenes): its heading, how Gemini saw the
// video, what each scene asks to be handled gently, the "Suggest different
// cuts" chips and the note they build, and the timeline's ruler. Pure
// functions, so the node tests can check them; types only from ./api.
import type { Segment, SegmentList } from "./api";

type Span = Pick<Segment, "start_s" | "end_s">;

/**
 * The H1: how many scenes Gemini found, or that its split did not work.
 * Only Gemini's first split (revision 0) is said to be Gemini's: any later
 * list may have been changed by hand.
 */
export function scenesHeading(list: Pick<SegmentList, "segments" | "error" | "revision">): string {
  const count = list.segments.length;
  if (list.error) return count === 1 ? "The whole video is one scene for now" : "Check the scenes";
  if (!count) return "No scenes yet";
  const scenes = `${count} ${count === 1 ? "scene" : "scenes"}`;
  return list.revision ? `Your video has ${scenes}` : `Gemini found ${scenes}`;
}

export type SourceNote = { kind: "video" | "stills"; text: string };

/**
 * How Gemini looked at the video for this split. It works from the stills
 * when the video could not be used, and the server keeps why
 * (`video_problem`); with no such reason and a split that worked, it
 * watched the video. Null when the split failed (its card says so).
 */
export function sourceNote(list: Pick<SegmentList, "segments" | "error" | "video_problem">): SourceNote | null {
  const problem = list.video_problem?.trim();
  if (problem) return { kind: "stills", text: `Gemini worked from stills: ${problem}` };
  if (list.error || !list.segments.length) return null;
  return { kind: "video", text: "Gemini watched the whole video" };
}

const CONTAINS_WORDS: Record<string, string> = {
  faces: "Faces",
  hands: "Hands",
  text: "Text",
  fine_pattern: "Fine pattern",
};

/** The badges for what a scene contains: "Faces — handled gently". */
export function protectLabels(contains: readonly string[]): string[] {
  const seen = new Set<string>();
  const labels: string[] = [];
  for (const item of contains) {
    if (seen.has(item)) continue;
    seen.add(item);
    const word = CONTAINS_WORDS[item] ?? item.replace(/_/g, " ").replace(/^./, (first) => first.toUpperCase());
    labels.push(`${word} — handled gently`);
  }
  return labels;
}

export type CutChipKey = "merge" | "split" | "fewer" | "more";
export type CutChip = { key: CutChipKey; text: string; note: string };

/**
 * The "Suggest different cuts" chips for these scenes. Each one only adds
 * words to the note sent to Gemini with "Ask Gemini to re-split": merge
 * the two neighbours that are shortest together, split the longest scene
 * (when it is long enough to make two), fewer or more scenes (within the
 * server's limit).
 */
export function cutChips(segments: Span[], limits: { minS: number; maxScenes: number }): CutChip[] {
  const chips: CutChip[] = [];
  const count = segments.length;
  const length = (segment: Span) => segment.end_s - segment.start_s;
  if (count >= 2) {
    let best = 0;
    for (let index = 1; index < count - 1; index += 1) {
      const together = length(segments[index]) + length(segments[index + 1]);
      if (together < length(segments[best]) + length(segments[best + 1])) best = index;
    }
    chips.push({
      key: "merge",
      text: `Merge ${best + 1} and ${best + 2}`,
      note: `Merge scenes ${best + 1} and ${best + 2}.`,
    });
  }
  if (count >= 1 && count < limits.maxScenes) {
    let longest = 0;
    segments.forEach((segment, index) => {
      if (length(segment) > length(segments[longest])) longest = index;
    });
    if (length(segments[longest]) >= 2 * limits.minS) {
      chips.push({
        key: "split",
        text: `Split scene ${longest + 1}`,
        note: `Split scene ${longest + 1} into two.`,
      });
    }
  }
  if (count >= 2) chips.push({ key: "fewer", text: "Fewer scenes", note: "Use fewer scenes." });
  if (count < limits.maxScenes) chips.push({ key: "more", text: "More scenes", note: "Use more scenes." });
  return chips;
}

/** The chips chosen after `key` is toggled: fewer and more rule each other out. */
export function toggleChip(chosen: readonly CutChipKey[], key: CutChipKey, on: boolean): CutChipKey[] {
  const rest = chosen.filter((item) => item !== key);
  if (!on) return rest;
  const other: CutChipKey | null = key === "fewer" ? "more" : key === "more" ? "fewer" : null;
  return [...rest.filter((item) => item !== other), key];
}

/** The note sent to Gemini: the chosen chips' words in the chips' order, then the typed words. */
export function composeNote(chips: readonly CutChip[], chosen: readonly CutChipKey[], typed: string): string {
  const parts = chips.filter((chip) => chosen.includes(chip.key)).map((chip) => chip.note);
  const words = typed.trim();
  if (words) parts.push(words);
  return parts.join(" ");
}

export type RulerMark = { at: number; share: number; edge: "start" | "end" | null };

/**
 * The times under the timeline: 0:00, each cut far enough from the last
 * mark shown and from the end that the labels do not run into each other,
 * and the video's length.
 */
export function rulerMarks(segments: Span[], minShare = 0.07): RulerMark[] {
  const total = segments.length ? segments[segments.length - 1].end_s : 0;
  if (!(total > 0)) return [];
  const marks: RulerMark[] = [{ at: 0, share: 0, edge: "start" }];
  let last = 0;
  for (const segment of segments.slice(0, -1)) {
    const share = segment.end_s / total;
    if (share - last < minShare || 1 - share < minShare) continue;
    marks.push({ at: segment.end_s, share, edge: null });
    last = share;
  }
  marks.push({ at: total, share: 1, edge: "end" });
  return marks;
}
