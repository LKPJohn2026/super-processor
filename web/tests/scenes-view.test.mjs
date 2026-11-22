import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

// Same as shell-view.test.mjs: scenesView.ts only imports types.
const source = await readFile(new URL("../src/scenesView.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
});
assert.ok(!/from\s+["']\.\/api["']/.test(outputText), "scenesView.ts must import only types");
const { composeNote, cutChips, protectLabels, rulerMarks, scenesHeading, sourceNote, toggleChip } = await import(
  `data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`
);

const seg = (start_s, end_s) => ({ start_s, end_s, label: "", note: "", contains: [] });
const FIVE = [seg(0, 48), seg(48, 272), seg(272, 550), seg(550, 940), seg(940, 1321)];
const LIMITS = { minS: 3, maxScenes: 10 };

test("the heading says how many scenes Gemini found, or that its split did not work", () => {
  assert.equal(scenesHeading({ segments: FIVE, error: null }), "Gemini found 5 scenes");
  assert.equal(scenesHeading({ segments: [seg(0, 9)], error: null }), "Gemini found 1 scene");
  assert.equal(scenesHeading({ segments: [seg(0, 9)], error: "quota" }), "The whole video is one scene for now");
  assert.equal(scenesHeading({ segments: [], error: null }), "No scenes yet");
});

test("the source note says whether Gemini watched the video or worked from stills, and why", () => {
  assert.deepEqual(sourceNote({ segments: FIVE, error: null, video_problem: "" }), {
    kind: "video",
    text: "Gemini watched the whole video",
  });
  assert.deepEqual(sourceNote({ segments: FIVE, error: null, video_problem: "the upload failed" }), {
    kind: "stills",
    text: "Gemini worked from stills: the upload failed",
  });
  // A failed split has its own card.
  assert.equal(sourceNote({ segments: [seg(0, 9)], error: "quota" }), null);
});

test("what a scene contains reads as handled gently, once each", () => {
  assert.deepEqual(protectLabels(["faces", "fine_pattern", "faces", "text"]), [
    "Faces — handled gently",
    "Fine pattern — handled gently",
    "Text — handled gently",
  ]);
  assert.deepEqual(protectLabels([]), []);
});

test("the chips name the shortest neighbours and the longest scene", () => {
  const chips = cutChips(FIVE, LIMITS);
  assert.deepEqual(
    chips.map((chip) => chip.text),
    ["Merge 1 and 2", "Split scene 4", "Fewer scenes", "More scenes"],
  );
  // One scene: nothing to merge or have fewer of.
  assert.deepEqual(
    cutChips([seg(0, 20)], LIMITS).map((chip) => chip.key),
    ["split", "more"],
  );
  // Too short to split; at the limit, no more scenes.
  assert.deepEqual(
    cutChips([seg(0, 5)], LIMITS).map((chip) => chip.key),
    ["more"],
  );
  const ten = Array.from({ length: 10 }, (_, i) => seg(i * 10, i * 10 + 10));
  assert.deepEqual(
    cutChips(ten, LIMITS).map((chip) => chip.key),
    ["merge", "fewer"],
  );
});

test("fewer and more rule each other out; the note keeps the chips' order, then the typed words", () => {
  assert.deepEqual(toggleChip(["more"], "fewer", true), ["fewer"]);
  assert.deepEqual(toggleChip(["fewer", "merge"], "more", true), ["merge", "more"]);
  assert.deepEqual(toggleChip(["merge", "split"], "merge", false), ["split"]);
  const chips = cutChips(FIVE, LIMITS);
  assert.equal(
    composeNote(chips, ["split", "merge"], "  cut where we walk outside "),
    "Merge scenes 1 and 2. Split scene 4 into two. cut where we walk outside",
  );
  assert.equal(composeNote(chips, [], ""), "");
});

test("the ruler shows 0:00, the cuts with room for their label, and the length", () => {
  assert.deepEqual(
    rulerMarks(FIVE).map((mark) => mark.at),
    [0, 272, 550, 940, 1321],
  );
  assert.deepEqual(rulerMarks([]), []);
  assert.deepEqual(rulerMarks([seg(0, 9)]).map((mark) => mark.edge), ["start", "end"]);
});
