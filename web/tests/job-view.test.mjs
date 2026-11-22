import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

// Same as run-states.test.mjs: the installed compiler, loaded as a data: module.
// jobView.ts only imports types, which the transpile drops.
const source = await readFile(new URL("../src/jobView.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
});
assert.ok(!/from\s+["']\.\/api["']/.test(outputText), "jobView.ts must import only types");
const {
  clockTime,
  hasJobStages,
  showsSidebar,
  sidebarSteps,
  sourceFacts,
  sourceStills,
  jobTitle,
  overallKey,
  sceneStrip,
  steadyOverall,
  stepperItems,
  stripSummary,
} = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);

const KEYS = ["file", "segments", "scenes", "render", "result"];
const stages = (current, done = []) =>
  KEYS.map((key) => ({ key, status: key === current ? "current" : done.includes(key) ? "done" : "pending" }));
const scene = (index, status, extra = {}) => ({
  index,
  start_s: index * 60,
  end_s: index * 60 + 48,
  label: `Scene label ${index}`,
  still_url: `/jobs/abcdef12/segments/still-${index}.jpg`,
  status,
  previewed: status === "approved",
  reopenable: false,
  ...extra,
});
const job = (extra = {}) => ({
  job_id: "abcdef12",
  title: "Private Life of a Cat",
  stage: "scenes",
  stages: stages("scenes", ["file", "segments"]),
  scene_count: 3,
  approved_count: 1,
  current_index: 1,
  scenes: [scene(0, "approved", { reopenable: true }), scene(1, "current"), scene(2, "pending")],
  error: null,
  ...extra,
});
const view = (step, extra = {}) => ({ step, error: null, job_id: "abcdef12", ...extra });

test("the stepper follows the job's stages when the server sends them", () => {
  const items = stepperItems(
    view("preview_segment", { job: job(), navigation: { setup: true, file: true, segments: true } }),
    false,
  );
  assert.deepEqual(items.map((i) => i.label), ["Video", "Scenes", "Previews", "Render", "Result"]);
  assert.deepEqual(items.map((i) => i.state), ["done", "done", "current", "upcoming", "upcoming"]);
  assert.equal(items[1].detail, "3 scenes");
  assert.equal(items[2].detail, "1 of 3 approved");
  assert.equal(items[0].nav, "file");
  assert.equal(items[1].nav, "segments");
  assert.equal(items[2].nav, null);
});

test("no step can be gone to while the page or the server is busy", () => {
  const nav = { setup: true, file: true, segments: true };
  for (const [busy, serverBusy] of [[true, false], [false, true]]) {
    const items = stepperItems(view("preview_segment", { job: job(), navigation: nav, busy: serverBusy }), busy);
    assert.ok(items.every((i) => i.nav === null));
  }
});

test("without a job summary the old rail follows the step", () => {
  for (const extra of [{}, { job: null }, { job: job({ stages: [] }) }, { job: job({ stages: [{ key: "x", status: "done" }] }) }]) {
    const v = view("segments", { navigation: { setup: true, file: true, segments: false }, ...extra });
    assert.equal(hasJobStages(v), false);
    const items = stepperItems(v, false);
    assert.deepEqual(items.map((i) => i.label), ["Setup", "File", "Scenes", "Previews", "Result"]);
    assert.deepEqual(items.map((i) => i.state), ["done", "done", "current", "upcoming", "upcoming"]);
    assert.deepEqual(items.map((i) => i.nav), ["setup", "file", null, null, null]);
  }
});

test("a reopened scene keeps the approved count, so the stepper never goes back", () => {
  // Scene 1 reopened after three were approved: the server keeps approved_count.
  const reopened = job({
    approved_count: 3,
    current_index: 0,
    stages: stages("scenes", ["file", "segments", "scenes"]),
    scenes: [scene(0, "current"), scene(1, "approved", { reopenable: true }), scene(2, "approved", { reopenable: true })],
  });
  const items = stepperItems(view("preview_segment", { job: reopened }), false);
  assert.equal(items[2].state, "current");
  assert.equal(items[2].detail, "3 of 3 approved");
  assert.equal(stripSummary(view("preview_segment", { job: reopened })), "All 3 scenes approved");
});

test("the scene strip names each scene, its time and its state", () => {
  const strip = sceneStrip(view("preview_segment", { job: job() }));
  assert.deepEqual(strip.map((s) => s.name), ["1 · Scene label 0", "2 · Scene label 1", "3 · Scene label 2"]);
  assert.deepEqual(strip.map((s) => s.statusText), ["Approved", "Reviewing", "Waiting"]);
  assert.equal(strip[1].range, "1:00–1:48");
  assert.equal(strip[0].stillUrl, "/jobs/abcdef12/segments/still-0.jpg");
  assert.equal(strip[0].previewed, true);
  assert.equal(stripSummary(view("preview_segment", { job: job() })), "1 approved · 2 to go");
  const analyzing = sceneStrip(view("analyzing", { job: job() }));
  assert.equal(analyzing[1].statusText, "Preparing…");
});

test("only scenes the server marks reopenable get a button, never the current one", () => {
  const strip = sceneStrip(
    view("preview_segment", {
      job: job({
        approved_count: 2,
        current_index: 0,
        scenes: [
          scene(0, "current", { reopenable: true }),
          scene(1, "approved", { reopenable: true }),
          scene(2, "pending", { reopenable: true }),
        ],
      }),
    }),
  );
  assert.equal(strip[0].reopen, null);
  assert.deepEqual(strip[1].reopen, { text: "Look again", label: "Look again at scene 2, Scene label 1" });
  assert.deepEqual(strip[2].reopen, { text: "Go to this scene", label: "Go to scene 3, Scene label 2" });
  const none = sceneStrip(view("analyzing", { job: job({ scenes: [scene(0, "approved")] }) }));
  assert.equal(none[0].reopen, null);
});

test("no job, no strip and no title", () => {
  assert.deepEqual(sceneStrip(view("pick_file")), []);
  assert.equal(stripSummary(view("pick_file")), "");
  assert.equal(jobTitle(view("pick_file")), null);
  assert.equal(jobTitle(view("segments", { job: job({ title: "  " }) })), null);
  assert.equal(jobTitle(view("segments", { job: job() })), "Private Life of a Cat");
});

test("the whole job's percentage never goes back within one run", () => {
  let shown = steadyOverall(null, "job:run1", 40);
  shown = steadyOverall(shown, "job:run1", 35);
  assert.equal(shown.value, 40);
  shown = steadyOverall(shown, "job:run1", 52);
  assert.equal(shown.value, 52);
  shown = steadyOverall(shown, "job:run2", 10);
  assert.equal(shown.value, 10, "a new run starts from its own value");
  assert.equal(steadyOverall(null, "k", 140).value, 100);
  assert.equal(overallKey(view("rendering", { run: { id: "r9" } })), "abcdef12:r9");
  assert.equal(overallKey(view("rendering")), "abcdef12:rendering");
  // Every scene's previews share one key, whatever run made them.
  assert.equal(overallKey(view("analyzing", { run: { id: "r1" } })), "abcdef12:previews");
  assert.equal(overallKey(view("analyzing", { run: { id: "r2" } })), "abcdef12:previews");
});

test("clock times for scene ranges", () => {
  assert.equal(clockTime(0), "0:00");
  assert.equal(clockTime(272.9), "4:32");
  assert.equal(clockTime(3725), "1:02:05");
});

test("the sidebar shows from picking the file on, not during setup", () => {
  assert.equal(showsSidebar(view("setup", { job_id: null })), false);
  assert.equal(showsSidebar(view("intro", { job_id: null })), false);
  assert.equal(showsSidebar(view("pick_file", { job_id: null })), true);
  assert.equal(showsSidebar(view("result")), true);
});

test("the sidebar's steps follow the step before a job summary exists", () => {
  const items = sidebarSteps(view("pick_file", { job_id: null }), false);
  assert.deepEqual(items.map((i) => i.label), ["Video", "Scenes", "Previews", "Render", "Result"]);
  assert.deepEqual(items.map((i) => i.state), ["current", "upcoming", "upcoming", "upcoming", "upcoming"]);
  assert.equal(items[0].detail, "Pick a file");
  const ready = sidebarSteps(view("ready_to_render", { job_id: null }), false);
  assert.deepEqual(ready.map((i) => i.state), ["done", "done", "done", "current", "upcoming"]);
});

test("the sidebar's steps come from the job's stages, and a failed step is marked", () => {
  const shown = view("ready_to_render", {
    job: job({ stage: "render", stages: stages("render", ["file", "segments", "scenes"]) }),
  });
  assert.deepEqual(sidebarSteps(shown, false).map((i) => i.state), ["done", "done", "done", "current", "upcoming"]);
  assert.deepEqual(sidebarSteps(shown, false, true).map((i) => i.state), ["done", "done", "done", "error", "upcoming"]);
  assert.equal(sidebarSteps(shown, false, true)[3].detail, "Didn’t finish");
});

test("Ready to render is the Render step, as the server still counts it in its scenes", () => {
  const shown = view("ready_to_render", {
    job: job({ approved_count: 3, stages: stages("scenes", ["file", "segments"]) }),
  });
  const items = sidebarSteps(shown, false);
  assert.deepEqual(items.map((i) => i.state), ["done", "done", "done", "current", "upcoming"]);
  assert.equal(items[3].detail, "Choose the size");
});

test("the source facts say only what the server sent", () => {
  assert.equal(sourceFacts(view("pick_file", { job_id: null })), "");
  assert.equal(
    sourceFacts(view("ready_to_render", { looks: { scale: 2, error: null, shots: [], source_size: [640, 480] }, job: job() })),
    "640×480 · 2:48 · 3 scenes",
  );
  assert.equal(
    sourceFacts(view("segments", { segments: { segments: [], error: null, note: "", duration_s: 1321 } })),
    "22:01",
  );
});

test("the sidebar's still is the job's thumbnail, then the middle scene's still", () => {
  assert.deepEqual(sourceStills(view("pick_file", { job_id: null })), []);
  assert.deepEqual(sourceStills(view("preview_segment", { job: job() })), [
    "/jobs/abcdef12/thumb.jpg",
    "/jobs/abcdef12/segments/still-1.jpg",
  ]);
});

test("the Result step says the version, and Kept once the video is kept", () => {
  const versions = [{ number: 1, current: false }, { number: 2, current: true }];
  const done = job({ stage: "result", stages: stages("result", ["file", "segments", "scenes", "render"]) });
  const last = (step) => sidebarSteps(view(step, { job: done, versions }), false).at(-1).detail;
  assert.equal(last("result"), "Version 2");
  assert.equal(last("done"), "Kept · version 2");
  assert.equal(sidebarSteps(view("done", { job: done, versions }), false).at(-1).state, "done");
});
