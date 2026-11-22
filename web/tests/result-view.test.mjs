import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

// Same as shell-view.test.mjs: resultView.ts only imports types.
const source = await readFile(new URL("../src/resultView.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
});
assert.ok(!/from\s+["']\.\/api["']/.test(outputText), "resultView.ts must import only types");
const {
  NO_RANGE,
  changesSoFar,
  compareScene,
  completeRange,
  currentVersion,
  markRange,
  outputPlace,
  rangeText,
  resultChips,
  resultTag,
  resultTimeline,
  reviseLabel,
  reviseNote,
  reviseProblem,
  savedFacts,
  startOverText,
} = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);

const version = (number, extra = {}) => ({
  number,
  name: `v00${number}.mp4`,
  url: `/jobs/j/outputs/v00${number}.mp4`,
  kind: "render",
  created_at: null,
  note: null,
  size: 0,
  scale: null,
  strength: null,
  strength_summary: null,
  current: false,
  ...extra,
});

const scene = (index, start_s, end_s, extra = {}) => ({
  index,
  start_s,
  end_s,
  label: `Scene ${index}`,
  still_url: `/s${index}.jpg`,
  status: "approved",
  previewed: true,
  reopenable: false,
  ...extra,
});

test("marking a range from the player's time", () => {
  let range = markRange(NO_RANGE, "start", 450.4);
  assert.deepEqual(range, { start: 450.4, end: null });
  assert.equal(rangeText(range), "From 7:30 · now mark the end");
  assert.equal(completeRange(range), null);
  range = markRange(range, "end", 459.2);
  assert.deepEqual(completeRange(range), { start: 450, end: 460 });
  assert.equal(rangeText(range), "7:30 – 7:40");
  assert.equal(reviseLabel(range), "Re-render 7:30–7:40");
  // A start after the end begins again from that mark.
  assert.deepEqual(markRange(range, "start", 500), { start: 500, end: null });
  // An end before the start, likewise.
  assert.deepEqual(markRange({ start: 100, end: null }, "end", 50), { start: null, end: 50 });
  assert.equal(rangeText({ start: null, end: 50 }), "Up to 0:50 · now mark the start");
  assert.equal(rangeText(NO_RANGE), "");
  assert.equal(reviseLabel(NO_RANGE), "Re-render with this note");
  // The same second marked twice still covers a second.
  assert.deepEqual(completeRange({ start: 10.2, end: 10.4 }), { start: 10, end: 11 });
  assert.deepEqual(markRange(NO_RANGE, "start", Number.NaN), { start: 0, end: null });
});

test("the note sent for a revise: range, chips and words", () => {
  const range = { start: 450, end: 460 };
  assert.equal(reviseNote(range, ["less-detail", "smoother"], " whiskers shimmer "),
    "From 7:30 to 7:40: less AI detail; smoother, with less flicker. whiskers shimmer");
  assert.equal(reviseNote(NO_RANGE, [], "make it 4x"), "make it 4x");
  assert.equal(reviseNote(range, [], ""), "");
  // "Fix a seam" with nothing marked names the first seam, a second either side.
  assert.equal(reviseNote(NO_RANGE, ["seam"], "", [{ time_s: 83.4 }]), "smooth the seam from 1:22 to 1:24");
  assert.equal(reviseNote(range, ["seam"], "", [{ time_s: 83.4 }]), "From 7:30 to 7:40: smooth the seam");
  assert.match(reviseProblem(range, ""), /7:30–7:40/);
  assert.ok(reviseProblem(NO_RANGE, ""));
  assert.equal(reviseProblem(NO_RANGE, "sharper"), null);
  assert.deepEqual(resultChips([]).map((chip) => chip.key), ["less-detail", "smoother", "sharper"]);
  assert.deepEqual(resultChips([{ time_s: 3 }]).map((chip) => chip.key), ["less-detail", "smoother", "sharper", "seam"]);
});

test("the seams timeline: scene cuts, possible seams, the marked part", () => {
  const view = { job: { scenes: [scene(0, 0, 25), scene(1, 25, 75), scene(2, 75, 100)] }, seams: [{ time_s: 50, score: 3 }] };
  const line = resultTimeline(view, null, { start: 10, end: 20 });
  assert.deepEqual(line.scenes.map((cut) => cut.pct), [25, 75]);
  assert.deepEqual(line.seams.map((seam) => seam.pct), [50]);
  assert.deepEqual(line.marked, { left: 10, width: 10 });
  assert.equal(line.summary, "Scenes change at 0:25, 1:15; possible seam at 0:50; marked 0:10 to 0:20.");
  // The player's length wins over the scenes'.
  assert.deepEqual(resultTimeline(view, 200, NO_RANGE).scenes.map((cut) => cut.pct), [12.5, 37.5]);
  assert.equal(resultTimeline({ job: null, seams: [] }, null, NO_RANGE), null);
  assert.equal(resultTimeline({ job: { scenes: [scene(0, 0, 9)] } }, null, NO_RANGE).summary, "One scene.");
});

test("changes so far, the current version and the viewer's tag", () => {
  const versions = [version(2, { note: "less AI detail 2:00–2:10", current: true }), version(1), version(3, { note: "  " })];
  assert.deepEqual(changesSoFar(versions), [{ note: "less AI detail 2:00–2:10", version: "Version 2" }]);
  assert.deepEqual(changesSoFar(null), []);
  assert.equal(currentVersion(versions).number, 2);
  assert.equal(currentVersion([version(1)]), null);
  assert.equal(resultTag(2, versions[0]), "Restored · 2× · version 2");
  assert.equal(resultTag(1, null), "Cleaned up");
  assert.equal(resultTag(null, null), "Restored");
});

test("start over names what it deletes and keeps", () => {
  const text = startOverText({ job: { approved_count: 5 }, versions: [version(2, { current: true })] });
  assert.equal(text.deletes, "the scene split, all 5 approved previews and their settings");
  assert.match(text.keeps, /version 2/);
  assert.match(startOverText({ job: { approved_count: 1 }, versions: [] }).deletes, /the approved preview /);
});

test("the saved file's facts and place, and the before/after scene", () => {
  const sizeText = (bytes) => (bytes ? `${Math.round(bytes / 1e6)} MB` : "");
  assert.equal(
    savedFacts({ width: 1280, height: 960, duration: 1321 }, version(2, { size: 820e6, scale: 2 }), sizeText),
    "1280×960 · 22:01 · 820 MB · 2× the original size",
  );
  assert.equal(savedFacts(null, null, sizeText), "");
  assert.equal(savedFacts({ width: 0, height: 0, duration: Number.NaN }, version(1, { scale: 1 }), sizeText), "");
  assert.equal(outputPlace("abc"), "jobs/abc/output.mp4");
  const view = { job: { scenes: [scene(0, 0, 10), scene(1, 10, 30), scene(2, 30, 40, { still_url: null })] } };
  assert.deepEqual(compareScene(view), { index: 1, label: "Scene 1", stillUrl: "/s1.jpg", atS: 20 });
  assert.equal(compareScene({ job: null }), null);
});
