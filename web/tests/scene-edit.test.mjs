import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

// Same as run-states.test.mjs: the installed compiler, loaded as a data: module.
// sceneEdit.ts only imports types, which the transpile drops.
const source = await readFile(new URL("../src/sceneEdit.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
});
assert.ok(!/from\s+["']\.\/api["']/.test(outputText), "sceneEdit.ts must import only types");
const {
  DEFAULT_LIMITS,
  DRAG_THRESHOLD_PX,
  STALE_SCENES_TEXT,
  STALE_SCENES_UNLOADED_TEXT,
  canSplit,
  checkCoverage,
  checkMerge,
  checkMove,
  checkSplit,
  cutRange,
  cutValueText,
  dragTime,
  formatTime,
  isDrag,
  isStaleScenes,
  limitsOf,
  nudgeCut,
  parseTime,
  scenesText,
  splitRange,
  splitSuggestion,
  timeAt,
  withCutAt,
  withMerge,
  withSplit,
} = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);

const limits = { minS: 3, maxScenes: 10 };
// Cuts at the given times over a clip ending at the last one.
function scenes(...edges) {
  return edges.slice(0, -1).map((start, i) => ({ start_s: start, end_s: edges[i + 1] }));
}

test("limitsOf reads the server's limits and falls back to 3 s and 10 scenes", () => {
  assert.deepEqual(limitsOf({ min_segment_s: 4, max_segments: 6 }), { minS: 4, maxScenes: 6 });
  assert.deepEqual(limitsOf({}), DEFAULT_LIMITS);
  assert.deepEqual(DEFAULT_LIMITS, limits);
});

test("a cut moves only as far as leaves both scenes 3 s", () => {
  const list = scenes(0, 10, 20, 30);
  assert.deepEqual(cutRange(list, 0, 3), { low: 3, high: 17 });
  assert.deepEqual(cutRange(list, 1, 3), { low: 13, high: 27 });
  assert.equal(cutRange(list, 2, 3), null, "no cut after the last scene");
  assert.equal(cutRange(scenes(0, 2.5, 5), 0, 3), null, "no room to move");
  assert.equal(checkMove(list, 0, 17, limits), null, "exactly 3 s is allowed");
  assert.equal(checkMove(list, 0, 3, limits), null);
  assert.match(checkMove(list, 0, 17.001, limits), /at least 3 s.*between 0:03 and 0:17/);
  assert.match(checkMove(list, 0, 2.999, limits), /scenes 1 and 2/);
  assert.match(checkMove(list, 2, 25, limits), /no cut after/);
  assert.match(checkMove(list, 0, Number.NaN, limits), /as a time/);
});

test("times compare to the millisecond, as the server saves them", () => {
  const list = scenes(0, 10.0004, 20);
  // 3.0004 rounds to 3.000, which the server takes.
  assert.equal(checkMove(list, 0, 16.9996, limits), null);
  assert.match(checkMove(list, 0, 17.0006, limits), /at least/);
  // The range said is one the editor can type back.
  assert.match(checkMove(scenes(0, 10, 20.05), 0, 30, limits), /between 0:03 and 0:17/);
});

test("merge needs a next scene", () => {
  const list = scenes(0, 10, 20, 30);
  assert.equal(checkMerge(list, 0), null);
  assert.equal(checkMerge(list, 1), null);
  assert.match(checkMerge(list, 2), /last one/);
  assert.match(checkMerge(list, 5), /Pick one/);
});

test("split keeps both halves 3 s and the scenes at 10 or fewer", () => {
  const list = scenes(0, 10, 15, 30);
  assert.deepEqual(splitRange(list, 0, 3), { low: 3, high: 7 });
  assert.equal(splitRange(list, 1, 3), null, "5 s cannot make two 3 s halves");
  assert.equal(checkSplit(list, 0, 5, limits), null);
  assert.equal(checkSplit(list, 0, 3, limits), null);
  assert.match(checkSplit(list, 0, 2, limits), /between 0:03 and 0:07/);
  assert.match(checkSplit(list, 1, 12, limits), /too short to split/);
  assert.equal(canSplit(list, 0, limits), null);
  assert.match(canSplit(list, 1, limits), /too short/);
  const ten = scenes(...Array.from({ length: 11 }, (_, i) => i * 10));
  assert.equal(ten.length, 10);
  assert.match(checkSplit(ten, 0, 5, limits), /at most 10 scenes/);
  assert.match(canSplit(ten, 0, limits), /Merge two scenes first/);
  assert.equal(checkSplit(ten.slice(0, 9), 0, 5, limits), null, "nine can still be split");
});

test("a split is first offered in the middle of the scene", () => {
  const list = scenes(0, 10, 15, 30);
  assert.equal(splitSuggestion(list, 0, 3), 5);
  assert.equal(splitSuggestion(list, 2, 3), 22.5);
  assert.equal(splitSuggestion(list, 1, 3), null);
});

test("keys nudge a cut within its range", () => {
  const range = { low: 3, high: 17 };
  assert.equal(nudgeCut(10, "ArrowRight", range), 11);
  assert.equal(nudgeCut(10, "ArrowUp", range), 11);
  assert.equal(nudgeCut(10, "ArrowLeft", range), 9);
  assert.equal(nudgeCut(10, "ArrowDown", range), 9);
  assert.equal(nudgeCut(10, "PageUp", range), 17, "clamped to the range");
  assert.equal(nudgeCut(10, "PageDown", range), 3);
  assert.equal(nudgeCut(10, "Home", range), 3);
  assert.equal(nudgeCut(10, "End", range), 17);
  assert.equal(nudgeCut(10.3, "ArrowRight", range), 11.3);
  assert.equal(nudgeCut(10, "a", range), null);
  assert.equal(nudgeCut(10, "Enter", range), null);
});

test("withCutAt draws a cut being moved without changing the others", () => {
  const list = scenes(0, 10, 20, 30);
  const moved = withCutAt(list, 1, 14);
  assert.deepEqual(moved, scenes(0, 10, 14, 30));
  assert.deepEqual(list, scenes(0, 10, 20, 30), "the list given is left alone");
  assert.equal(withCutAt(list, 2, 25), list);
});

test("timeAt turns a pointer position into a time on the timeline", () => {
  assert.equal(timeAt(150, 100, 200, 60), 15);
  assert.equal(timeAt(50, 100, 200, 60), 0);
  assert.equal(timeAt(400, 100, 200, 60), 60);
  assert.equal(timeAt(133, 100, 200, 60), 9.9);
  assert.equal(timeAt(150, 100, 0, 60), 0);
});

test("times are written and read the way the editor sees them", () => {
  assert.equal(formatTime(272), "4:32");
  assert.equal(formatTime(3.5), "0:03.5");
  assert.equal(formatTime(3723), "62:03");
  assert.equal(parseTime("4:32"), 272);
  assert.equal(parseTime(" 4:32.5 "), 272.5);
  assert.equal(parseTime("90"), 90);
  assert.equal(parseTime("1,5"), 1.5);
  assert.equal(parseTime("62:03"), 3723);
  assert.equal(parseTime("1:02:03"), 3723);
  assert.equal(parseTime("4:75"), null);
  assert.equal(parseTime("4.5:10"), null);
  assert.equal(parseTime(""), null);
  assert.equal(parseTime("abc"), null);
  assert.equal(parseTime("-3"), null);
  assert.equal(parseTime(formatTime(17.4)), 17.4);
});

test("a cut handle says which cut it is and where", () => {
  assert.equal(cutValueText(1, 272), "Cut between scenes 2 and 3, at 4:32");
  assert.equal(scenesText(1), "1 scene");
  assert.equal(scenesText(4), "4 scenes");
});

test("a refused change from a page with old scenes is known by its text or a 409", () => {
  const server =
    "the scenes changed since this page showed them; look at them again before changing them";
  assert.equal(isStaleScenes(server, 400), true);
  assert.equal(isStaleScenes("this job is no longer open in this window", 409), true);
  assert.equal(isStaleScenes("there is no cut after that scene", 400), false);
  assert.equal(isStaleScenes("Request failed"), false);
  assert.match(STALE_SCENES_TEXT, /another window/);
});

test("a click on a cut is not a drag and keeps its exact time", () => {
  const range = { low: 3, high: 27 };
  assert.equal(isDrag(100, 100), false);
  assert.equal(isDrag(100, 100 + DRAG_THRESHOLD_PX - 1), false);
  assert.equal(isDrag(100, 100 - DRAG_THRESHOLD_PX), true);
  // No movement gives the time back untouched, even off the tenths.
  assert.equal(dragTime(12.345, 150, 150, 375, 300, range), 12.345);
  assert.equal(dragTime(12.345, 150, 150, 375, 300, { low: 13, high: 20 }), 12.345);
});

test("a drag moves the cut by how far the pointer went, not to it", () => {
  const range = { low: 3, high: 27 };
  // 200 px showing 30 s: 20 px is 3 s, wherever the handle was grabbed.
  assert.equal(dragTime(10, 500, 520, 200, 30, range), 13);
  assert.equal(dragTime(10, 500, 480, 200, 30, range), 7);
  assert.equal(dragTime(10, 500, 1000, 200, 30, range), 27, "kept in range");
  assert.equal(dragTime(10, 500, 0, 200, 30, range), 3);
  assert.equal(dragTime(10, 500, 520, 0, 30, range), 10, "no track, no move");
});

test("the whole list a change makes is checked, as the server does", () => {
  assert.equal(checkCoverage(scenes(0, 10, 20), limits), null);
  assert.equal(checkCoverage(scenes(0, 1), limits), null, "one short scene alone is fine");
  assert.match(checkCoverage([], limits), /at least one/);
  assert.match(checkCoverage(scenes(0, 10, 12, 30), limits), /Scene 2 would be shorter than 3 s/);
  assert.match(checkCoverage([{ start_s: 0, end_s: 5 }, { start_s: 6, end_s: 20 }], limits), /start where/);
  assert.match(checkCoverage(scenes(...Array.from({ length: 12 }, (_, i) => i * 10)), limits), /at most 10/);
  // A short scene elsewhere is caught before anything is sent.
  const short = scenes(0, 10, 20, 22, 40);
  assert.match(checkMove(short, 0, 12, limits), /Scene 3/);
  assert.match(checkMerge(short, 0, limits), /Scene 2 would be/);
  assert.equal(checkMerge(short, 2, limits), null, "merging the short scene fixes it");
  assert.equal(checkMerge(short, 1, limits), null, "so does merging into it");
  assert.match(checkSplit(short, 3, 30, limits), /Scene 3/);
});

test("merge and split make the lists the server would", () => {
  const list = scenes(0, 10, 20, 30);
  assert.deepEqual(withMerge(list, 1), scenes(0, 10, 30));
  assert.deepEqual(withMerge(list, 2), list, "no next scene");
  assert.deepEqual(withSplit(list, 0, 4), scenes(0, 4, 10, 20, 30));
});

test("when the scenes cannot be loaded again the page does not say it shows them", () => {
  assert.doesNotMatch(STALE_SCENES_UNLOADED_TEXT, /shows them/);
  assert.match(STALE_SCENES_TEXT, /shows them as they are/);
});
