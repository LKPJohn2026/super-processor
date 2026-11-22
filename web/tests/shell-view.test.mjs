import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

// Same as job-view.test.mjs: shellView.ts only imports types.
const source = await readFile(new URL("../src/shellView.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
});
assert.ok(!/from\s+["']\.\/api["']/.test(outputText), "shellView.ts must import only types");
const { connectionState, connectionText, documentTitle, headingKey, originHost, progressMilestone } = await import(
  `data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`
);

const view = (step, extra = {}) => ({ step, error: null, job_id: "abcdef12", ...extra });

test("the tab title names the step, the scene and the app", () => {
  assert.equal(
    documentTitle(view("preview_segment", { segment_preview: { index: 2, count: 5 }, job: { scene_count: 5 } })),
    "Scene 3 of 5 · Previews · Super Processor",
  );
  assert.equal(documentTitle(view("pick_file")), "Choose a video · Video · Super Processor");
  assert.equal(documentTitle(view("rendering")), "Rendering · Render · Super Processor");
  assert.equal(documentTitle(view("rendering", { interrupted: true })), "Waiting for you · Rendering · Render · Super Processor");
  assert.equal(documentTitle(view("result"), true), "All jobs · Super Processor");
  assert.equal(documentTitle(null), "Connect · Super Processor");
});

test("the heading key changes with the step and the scene, not with progress", () => {
  const a = headingKey(view("analyzing", { segment_index: 1, progress: { pct: 10 } }));
  assert.equal(a, headingKey(view("analyzing", { segment_index: 1, progress: { pct: 60 } })));
  assert.notEqual(a, headingKey(view("analyzing", { segment_index: 2 })));
  assert.notEqual(a, headingKey(view("preview_segment", { segment_index: 1 })));
  assert.equal(headingKey(view("result"), true), "home");
});

test("the pill warns after one failed poll and is offline after two", () => {
  assert.equal(connectionState(0), "ok");
  assert.equal(connectionState(1), "warn");
  assert.equal(connectionState(3), "err");
  assert.equal(connectionText("ok").long, "Connected to this computer");
  assert.equal(connectionText("err").short, "Offline");
});

test("the host the banner names", () => {
  assert.equal(originHost("http://127.0.0.1:45143", "http://x.test"), "127.0.0.1:45143");
  assert.equal(originHost("", "http://localhost:5190"), "localhost:5190");
});

test("progress is read out at each stage and each 10%, not on every poll", () => {
  const p = (pct, message, overall) => ({ phase: "rendering", pct, message, overall });
  const first = progressMilestone(null, "Render", p(5, "Upscaling on the GPU", 31));
  assert.equal(first.text, "Render: Upscaling on the GPU, 30%");
  assert.equal(progressMilestone(first, "Render", p(6, "Upscaling on the GPU", 33)), null);
  const next = progressMilestone(first, "Render", p(20, "Upscaling on the GPU", 41));
  assert.equal(next.text, "Render: 40%");
  // A count in the words is not a new stage.
  const a = progressMilestone(null, "Finding the scenes", p(10, "Saving still 3 of 60", 12));
  assert.equal(progressMilestone(a, "Finding the scenes", p(12, "Saving still 9 of 60", 14)), null);
  const b = progressMilestone(a, "Finding the scenes", p(40, "Uploading the video to Gemini", 15));
  assert.equal(b.text, "Finding the scenes: Uploading the video to Gemini, 10%");
  assert.equal(progressMilestone(null, "Render", null), null);
});
