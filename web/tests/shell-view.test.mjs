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
const { connectionState, connectionText, documentTitle, headingKey, originHost, progressMilestone, stageLabel } = await import(
  `data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`
);

const view = (step, extra = {}) => ({ step, error: null, job_id: "abcdef12", ...extra });

test("the tab title names the step, the scene and the app", () => {
  assert.equal(
    documentTitle(view("preview_segment", { segment_preview: { index: 2, count: 5 }, job: { scene_count: 5 } })),
    "Scene 3 of 5 · Previews · Super Processor",
  );
  assert.equal(documentTitle(view("pick_file")), "Choose a video · Video · Super Processor");
  assert.equal(documentTitle(view("result")), "Result · Super Processor");
  assert.equal(documentTitle(view("done")), "Saved · Result · Super Processor");
  assert.equal(documentTitle(view("rendering")), "Rendering · Render · Super Processor");
  assert.equal(documentTitle(view("rendering", { interrupted: true })), "Waiting for you · Render · Super Processor");
  assert.equal(
    documentTitle(view("rendering", { progress: { phase: "rendering", pct: 49.4, message: "m" } })),
    "49% · Rendering · Super Processor",
  );
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
  // The number read out is the bar's (pct), not the whole job's.
  const first = progressMilestone(null, "Render", p(31, "Upscaling on the GPU: 9% of the work", 5));
  assert.equal(first.text, "Render: Upscaling on the GPU, 30%");
  assert.equal(progressMilestone(first, "Render", p(33, "Upscaling on the GPU: 12% of the work", 6)), null);
  const next = progressMilestone(first, "Render", p(41, "Upscaling on the GPU: 20% of the work", 20));
  assert.equal(next.text, "Render: 40%");
  // A count in the words is not a new stage.
  const a = progressMilestone(null, "Finding the scenes", p(12, "Saving still 3 of 60", 10));
  assert.equal(a.text, "Finding the scenes: Saving still, 10%");
  assert.equal(progressMilestone(a, "Finding the scenes", p(14, "Saving still 9 of 60", 12)), null);
  const b = progressMilestone(a, "Finding the scenes", p(15, "Uploading the video to Gemini", 40));
  assert.equal(b.text, "Finding the scenes: Uploading the video to Gemini, 10%");
  assert.equal(progressMilestone(null, "Render", null), null);
});

test("a stage is named without its own counts, so the bar's number is the only one", () => {
  assert.equal(stageLabel("Upscaling on the GPU: 9% of the work"), "Upscaling on the GPU");
  assert.equal(stageLabel("Saving still 3 of 60"), "Saving still");
  assert.equal(stageLabel("Joining the scenes"), "Joining the scenes");
});
