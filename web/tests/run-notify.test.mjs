import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

// Same as run-states.test.mjs: the installed compiler, loaded as a data: module.
// runNotify.ts only imports types, which the transpile drops.
const source = await readFile(new URL("../src/runNotify.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
});
assert.ok(!/from\s+["']\.\/api["']/.test(outputText), "runNotify.ts must import only types");
const {
  NOTIFY_KEY,
  notifyPermission,
  notifyToggle,
  runNotice,
  saveChoice,
  savedChoice,
  shouldNotify,
} = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);

const run = (kind, outcome, id = "r1") => ({
  id,
  kind,
  job_id: "abcdef12",
  started_at: 1,
  finished_at: outcome === "running" ? null : 2,
  outcome,
});
const view = (step, runRecord, extra = {}) => ({
  step,
  error: null,
  run: runRecord,
  job: { title: "Cat" },
  ...extra,
});

test("a render that finishes notifies once it was seen working", () => {
  const notice = runNotice(view("rendering", run("render", "running")), view("result", run("render", "done")));
  assert.deepEqual(notice, {
    title: "Your restored video is ready",
    body: "Cat: watch it and keep it, or ask for a change.",
    tag: "super-processor-r1",
  });
});

test("finding the scenes notifies with how many were found", () => {
  const before = view("proposing", run("segmenting", "running"));
  const after = view("segments", run("segmenting", "done"), { segments: { segments: [{}, {}, {}] } });
  assert.equal(runNotice(before, after).title, "The scenes are ready");
  assert.equal(runNotice(before, after).body, "Cat: 3 scenes found. Check them when you are back.");
  const one = view("segments", run("segmenting", "done"), { segments: { segments: [{}] }, job: null });
  assert.equal(runNotice(before, one).body, "1 scene found. Check them when you are back.");
});

test("a failed run notifies with its error", () => {
  const failed = runNotice(
    view("rendering", run("render", "running")),
    view("ready_to_render", run("render", "failed"), { error: "Out of GPU memory" }),
  );
  assert.equal(failed.title, "The render failed");
  assert.equal(failed.body, "Cat: Out of GPU memory");
  const scan = runNotice(
    view("sampling", run("segmenting", "running")),
    view("pick_file", run("segmenting", "failed"), { job: null }),
  );
  assert.equal(scan.title, "Finding the scenes failed");
  assert.equal(scan.body, "Open the tab to see what happened.");
});

test("no notification for a stop, a short run, or a run never seen working", () => {
  const working = view("rendering", run("render", "running"));
  // The editor stopped it.
  assert.equal(runNotice(working, view("ready_to_render", run("render", "stopped"))), null);
  // Still going.
  assert.equal(runNotice(working, working), null);
  // A scene preview takes seconds.
  assert.equal(
    runNotice(view("analyzing", run("analyze", "running")), view("preview_segment", run("analyze", "done"))),
    null,
  );
  // The page opened on a finished job, or another job's run.
  assert.equal(runNotice(null, view("result", run("render", "done"))), null);
  assert.equal(runNotice(view("result", run("render", "done")), view("result", run("render", "done"))), null);
  assert.equal(runNotice(working, view("result", run("render", "done", "r2"))), null);
  // A restart lost the run record: the job asks Resume or Stop instead.
  assert.equal(runNotice(working, view("rendering", null, { interrupted: true })), null);
  assert.equal(runNotice(view("rendering", undefined), view("result", run("render", "done"))), null);
});

test("it shows only when asked for, allowed, and the tab is hidden", () => {
  const notice = { title: "t", body: "b", tag: "x" };
  assert.equal(shouldNotify(notice, true, "granted", true), true);
  assert.equal(shouldNotify(notice, true, "granted", false), false);
  assert.equal(shouldNotify(notice, false, "granted", true), false);
  assert.equal(shouldNotify(notice, true, "default", true), false);
  assert.equal(shouldNotify(notice, true, "denied", true), false);
  assert.equal(shouldNotify(notice, true, "unsupported", true), false);
  assert.equal(shouldNotify(null, true, "granted", true), false);
});

test("the browser's permission, or unsupported", () => {
  assert.equal(notifyPermission(undefined), "unsupported");
  assert.equal(notifyPermission({}), "unsupported");
  assert.equal(notifyPermission({ Notification: { permission: "granted" } }), "granted");
  assert.equal(notifyPermission({ Notification: { permission: "denied" } }), "denied");
  assert.equal(notifyPermission({ Notification: { permission: "default" } }), "default");
  assert.equal(notifyPermission({ Notification: {} }), "default");
});

test("the toggle is hidden without notifications and says when they are blocked", () => {
  assert.equal(notifyToggle(true, "unsupported").shown, false);
  const blocked = notifyToggle(true, "denied");
  assert.equal(blocked.shown, true);
  assert.equal(blocked.checked, false);
  assert.equal(blocked.disabled, true);
  assert.match(blocked.hint, /blocks notifications/);
  // Saved as on, but the permission was never given (or was reset): off.
  assert.equal(notifyToggle(true, "default").checked, false);
  assert.equal(notifyToggle(false, "granted").checked, false);
  const on = notifyToggle(true, "granted");
  assert.equal(on.checked, true);
  assert.equal(on.disabled, false);
  assert.match(on.hint, /in the background/);
  assert.equal(notifyToggle(false, "default").hint, null);
});

test("the choice is kept in storage, and blocked storage is off, not an error", () => {
  const map = new Map();
  const store = { getItem: (key) => map.get(key) ?? null, setItem: (key, value) => map.set(key, value) };
  assert.equal(savedChoice(() => store), false);
  assert.equal(saveChoice(() => store, true), true);
  assert.equal(map.get(NOTIFY_KEY), "on");
  assert.equal(savedChoice(() => store), true);
  saveChoice(() => store, false);
  assert.equal(savedChoice(() => store), false);
  const throwing = () => {
    throw new Error("SecurityError");
  };
  assert.equal(savedChoice(throwing), false);
  assert.equal(saveChoice(throwing, true), false);
  const broken = { getItem: throwing, setItem: throwing };
  assert.equal(savedChoice(() => broken), false);
  assert.equal(saveChoice(() => broken, true), false);
  assert.equal(savedChoice(() => null), false);
  assert.equal(saveChoice(() => null, true), false);
});
