import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

// Same as api.test.mjs: the installed compiler, loaded as a data: module.
// runStates.ts only imports types, which the transpile drops.
const source = await readFile(new URL("../src/runStates.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
});
assert.ok(!/from\s+["']\.\/api["']/.test(outputText), "runStates.ts must import only types");
const runStates = await import(
  `data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`
);
const {
  interruptedChoice,
  offersRender,
  readyText,
  resultTitle,
  stopControl,
  stoppedNote,
  viewTone,
} = runStates;

const view = (step, extra = {}) => ({ step, error: null, ...extra });
const RENDER_STOPPED = "You stopped the render, so there is no result yet. Press Render to start it again; every approved scene is kept.";
const FAILED_RENDER = { id: "r1", kind: "render", job_id: "j", started_at: 1, finished_at: 2, outcome: "failed" };
const KEPT = "You stopped the render. The video below is the one you had before, with its settings.";

test("a stopped view is not shown as success", () => {
  for (const [step, note] of [
    ["ready_to_render", RENDER_STOPPED],
    ["result", KEPT],
    ["segments", "You stopped Gemini. These are the segments from before."],
    ["pick_file", "You stopped finding the segments. Pick the file again to start over."],
  ]) {
    const stopped = view(step, { stopped: note });
    assert.equal(viewTone(stopped), "stopped", step);
    assert.notEqual(viewTone(stopped), "ok", step);
    assert.equal(stoppedNote(stopped), note, step);
  }
  assert.equal(resultTitle(view("result", { stopped: KEPT })), "Render stopped");
  assert.equal(resultTitle(view("result")), "Result");
  assert.equal(readyText(view("ready_to_render", { stopped: RENDER_STOPPED })).title, "Render stopped");
});

test("no stopped note while a run works or waits, or once it was cleared", () => {
  assert.equal(stoppedNote(view("rendering", { stopped: KEPT, busy: true })), null);
  assert.equal(stoppedNote(view("sampling", { stopped: KEPT, interrupted: true })), null);
  assert.equal(stoppedNote(view("result", { stopped: null })), null);
  assert.equal(stoppedNote(view("result", { stopped: "   " })), null);
  // A newer error wins over an old stop.
  const both = view("ready_to_render", { stopped: RENDER_STOPPED, error: "out of memory", run: FAILED_RENDER });
  assert.equal(viewTone(both), "failed");
  assert.equal(stoppedNote(both), null);
});

test("ready_to_render with an error still offers Render", () => {
  const failed = view("ready_to_render", {
    error: "The render failed, so there is no result yet: CUDA out of memory",
    run: FAILED_RENDER,
    looks: { scale: 2, error: null, shots: [{}, {}, {}] },
  });
  assert.equal(offersRender(failed), true);
  assert.equal(viewTone(failed), "failed");
  const text = readyText(failed);
  assert.equal(text.renderLabel, "Try the render again");
  assert.match(text.lede, /3 approved scenes/);
  assert.equal(readyText(view("ready_to_render")).renderLabel, "Render the whole video");
  assert.equal(offersRender(view("ready_to_render", { stopped: RENDER_STOPPED })), true);
  assert.equal(offersRender(view("result")), false);
});

test("another error on ready_to_render keeps Ready to render, with Render offered", () => {
  const refusal = "scene 2 still has changes to preview";
  // No run, or a run that is not a failed render: not a failure.
  for (const run of [
    null,
    { ...FAILED_RENDER, kind: "analyze" },
    { ...FAILED_RENDER, outcome: "done" },
    { ...FAILED_RENDER, outcome: "stopped" },
  ]) {
    const refused = view("ready_to_render", { error: refusal, run, looks: { scale: 2, error: null, shots: [{}, {}] } });
    assert.notEqual(viewTone(refused), "failed", JSON.stringify(run));
    const text = readyText(refused);
    assert.equal(text.title, "Ready to render");
    assert.equal(text.renderLabel, "Render the whole video");
    assert.equal(offersRender(refused), true);
    // The error is newer than an old stop, so the stop note goes.
    assert.equal(stoppedNote({ ...refused, stopped: RENDER_STOPPED }), null);
  }
  // A failed render is a failure.
  const failed = view("ready_to_render", { error: "CUDA out of memory", run: FAILED_RENDER });
  assert.equal(viewTone(failed), "failed");
  assert.equal(readyText(failed).title, "The render didn’t finish");
  assert.equal(offersRender(failed), true);
  // Other steps keep treating an error as a failure.
  assert.equal(viewTone(view("result", { error: "x" })), "failed");
});

test("Stop shows only when the server says it can stop", () => {
  for (const step of ["sampling", "proposing", "rendering"]) {
    const live = stopControl(view(step, { busy: true, cancellable: true }), false);
    assert.equal(live.shown, true, step);
    assert.equal(live.disabled, false, step);
    assert.equal(live.status, null, step);
    assert.equal(stopControl(view(step, { busy: true, cancellable: false }), false).shown, false, step);
  }
  assert.equal(stopControl(view("rendering", { cancellable: true }), false).label, "Stop the render");
  assert.equal(stopControl(view("sampling", { cancellable: true }), false).label, "Stop");
  // Analysis cannot be stopped while it runs, and an interrupted step has its own choice.
  assert.equal(stopControl(view("analyzing", { busy: true, cancellable: false }), false).shown, false);
  assert.equal(stopControl(view("rendering", { interrupted: true, cancellable: true }), false).shown, false);
});

test("once a stop is asked for, Stop stays as a disabled Stopping…", () => {
  const asked = stopControl(view("rendering", { busy: true, cancellable: false, cancel_requested: true }), false);
  assert.equal(asked.shown, true);
  assert.equal(asked.disabled, true);
  assert.equal(asked.label, "Stopping…");
  assert.match(asked.status, /Stopping/);
  // While this page's own request is on its way.
  const posting = stopControl(view("sampling", { busy: true, cancellable: true }), true);
  assert.equal(posting.disabled, true);
  assert.equal(posting.label, "Stopping…");
});

test("an interrupted job offers Resume and says what Stop does", () => {
  const render = interruptedChoice(view("rendering", { interrupted: true }));
  assert.match(render.lede, /^Nothing runs until you choose\./);
  assert.equal(render.resumeLabel, "Resume the render");
  assert.match(render.stopHint, /approved scenes/);
  assert.match(interruptedChoice(view("analyzing", { interrupted: true, segment_index: 2 })).title, /Scene 3/);
  assert.match(interruptedChoice(view("proposing", { interrupted: true })).title, /Finding the scenes/);
  assert.equal(viewTone(view("sampling", { interrupted: true })), "interrupted");
});
