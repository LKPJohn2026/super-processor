import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

// Same as shell-view.test.mjs: previewView.ts only imports types.
const source = await readFile(new URL("../src/previewView.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
});
assert.ok(!/from\s+["']\.\/api["']/.test(outputText), "previewView.ts must import only types");
const {
  afterTag,
  approveLabel,
  checkState,
  clampWipe,
  fixNote,
  playerTime,
  previewOverline,
  previewRange,
  previewTitle,
  regionLabel,
  sceneGets,
  stepTime,
  strengthWord,
  viewerKey,
  wipeKey,
} = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);

const NEUTRAL = { deblock: 0, denoise: 0, contrast: 1, brightness: 0, saturation: 1, gamma: 1, grain: 0 };
const SCENES = [
  { index: 0, label: "Opening titles" },
  { index: 1, label: "Kittens indoors" },
  { index: 2, label: "At the window" },
  { index: 3, label: "Night street" },
  { index: 4, label: "" },
];

test("the heading names the step, the scene and its range", () => {
  assert.equal(previewOverline(2, 5), "Step 3 of 5 · Previews · Scene 3 of 5");
  assert.equal(previewTitle({ label: " At the window " }, 2), "At the window");
  assert.equal(previewTitle({ label: "" }, 2), "Scene 3");
  assert.equal(previewTitle(null, 0), "Scene 1");
  assert.equal(previewRange({ start_s: 272, end_s: 550 }), "4:32–9:10");
  assert.equal(previewRange(null), "");
});

test("the approve button names the next scene, or Render on the last one", () => {
  assert.equal(approveLabel(2, 5, SCENES), "Approve scene 3 — next: Night street");
  assert.equal(approveLabel(3, 5, SCENES), "Approve scene 4 — next: scene 5");
  assert.equal(approveLabel(4, 5, SCENES), "Approve scene 5 — next: Render");
  assert.equal(approveLabel(0, 3, []), "Approve scene 1 — next: scene 2");
});

test("strength reads as a word", () => {
  assert.equal(strengthWord(0), "None");
  assert.equal(strengthWord(0.3), "Gentle");
  assert.equal(strengthWord(0.5), "Medium");
  assert.equal(strengthWord(0.8), "Strong");
});

test("what the scene gets comes from its real look, in plain words", () => {
  const items = sceneGets(
    {
      strength: 0.3,
      use_upscale: true,
      look: { ...NEUTRAL, deblock: 0.5, denoise: 0.2, contrast: 1.05, gamma: 0.9 },
      regions: [{ kind: "faces", x0: 0, y0: 0, x1: 0.5, y1: 0.5 }],
      contains: ["faces"],
    },
    2,
  );
  assert.deepEqual(
    items.map((item) => [item.text, item.amount]),
    [
      ["Removes blockiness", "medium"],
      ["Reduces noise", "light"],
      ["AI detail", "gentle"],
      ["More contrast", "+5%"],
      ["Deepens the shadows", "-10%"],
      ["Faces handled gently", ""],
    ],
  );
});

test("a scene only resized says so, and a kept size has no AI detail line", () => {
  const resized = sceneGets({ strength: 0.5, use_upscale: false, look: NEUTRAL, contains: [] }, 2);
  assert.deepEqual(resized.map((item) => item.text), ["Just resized, no AI detail"]);
  assert.deepEqual(sceneGets({ strength: 0.5, look: NEUTRAL, contains: [] }, 1), []);
  const unboxed = sceneGets({ strength: 0.5, look: NEUTRAL, contains: ["text"], regions: [] }, 2);
  assert.equal(unboxed.at(-1).text, "Held to a lower strength (no boxes placed)");
});

test("the check box says passed, found, or not checked", () => {
  assert.deepEqual(checkState({ ok: true, problems: [], note: "Fur stays natural" }), {
    tone: "success",
    title: "Checked by Gemini",
    text: "Fur stays natural.",
    checked: true,
  });
  const found = checkState({ ok: false, problems: ["waxy_skin", "halos"], note: "", adjusted: true }, 1.5);
  assert.equal(found.tone, "warning");
  assert.equal(
    found.text,
    "Found: skin looks waxy, halos round edges. Settings were adjusted and the preview redone. Texture is 1.5× as jumpy as the source.",
  );
  const unchecked = checkState({ ok: null, problems: [], note: "" });
  assert.equal(unchecked.checked, false);
  assert.equal(unchecked.title, "Gemini couldn’t check this preview");
  assert.equal(checkState(null).checked, false);
});

test("labels for boxes and the restored side", () => {
  assert.equal(regionLabel({ kind: "text" }), "Text — handled gently");
  assert.equal(afterTag(2, true), "Restored · 2×");
  assert.equal(afterTag(2, false), "Resized · 2×");
  assert.equal(afterTag(1, true), "Cleaned up");
  assert.equal(afterTag(null, undefined), "Restored");
});

test("the fix chips only add words to the note, in their order", () => {
  assert.equal(fixNote([], ""), "");
  assert.equal(fixNote(["dark", "sharp"], " keep the grain "), "It looks too sharp; make it softer. It is too dark; brighten it a little. keep the grain");
});

test("the viewer's keys, wipe and clock", () => {
  assert.equal(viewerKey(" "), "play");
  assert.equal(viewerKey("b"), "flip");
  assert.equal(viewerKey("ArrowLeft"), "back");
  assert.equal(viewerKey("Z"), "zoom");
  assert.equal(viewerKey("x"), null);
  assert.equal(wipeKey("ArrowRight", 50), 55);
  assert.equal(wipeKey("ArrowLeft", 2), 0);
  assert.equal(wipeKey("End", 10), 100);
  assert.equal(wipeKey("a", 10), null);
  assert.equal(clampWipe(140), 100);
  assert.equal(clampWipe(Number.NaN), 50);
  assert.equal(playerTime(1.25, 3), "0:01.2 / 0:03.0");
  assert.equal(playerTime(Number.NaN, 63.5), "0:00.0 / 1:03.5");
  assert.equal(stepTime(1, 3, 1, 0.04), 1.04);
  assert.equal(stepTime(0.01, 3, -1, 0.04), 0);
  assert.equal(stepTime(2.99, 3, 1, 0.04), 3);
});
