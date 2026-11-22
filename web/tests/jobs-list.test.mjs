import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

// Same as run-states.test.mjs: the installed compiler, loaded as a data: module.
// jobsList.ts only imports types, which the transpile drops.
const source = await readFile(new URL("../src/jobsList.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
});
assert.ok(!/from\s+["']\.\/api["']/.test(outputText), "jobsList.ts must import only types");
const {
  busyHere,
  changedText,
  jobAction,
  jobMeta,
  jobStatus,
  newVideoRoute,
  sortJobs,
  startsOnJobs,
} = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);

const entry = (extra = {}) => ({
  job_id: "abcdef12",
  title: "Private Life of a Cat",
  file_name: "PrivateL1947.mp4",
  step: "segments",
  stage: "segments",
  status: "paused",
  updated_at: 1_700_000_000,
  resumes_work: false,
  interrupted: false,
  open: false,
  thumb_url: null,
  output_url: null,
  error: null,
  ...extra,
});

test("jobStatus: the badge says where each job is, in the mockup's words", () => {
  assert.deepEqual(jobStatus(entry({ status: "result", stage: "result", step: "result" })), {
    text: "Result ready",
    tone: "ok",
  });
  assert.deepEqual(jobStatus(entry()), { text: "Scenes · waiting for you", tone: "waiting" });
  assert.deepEqual(jobStatus(entry({ status: "running", step: "rendering", stage: "render" }), 49.4), {
    text: "Rendering · 49%",
    tone: "working",
  });
  assert.equal(jobStatus(entry({ status: "running", step: "sampling" })).text, "Finding scenes");
  assert.deepEqual(jobStatus(entry({ status: "running_elsewhere", interrupted: false })), {
    text: "Busy in another window",
    tone: "busy",
  });
  assert.deepEqual(jobStatus(entry({ step: "rendering", stage: "render", resumes_work: true, interrupted: true })), {
    text: "Render · stopped partway",
    tone: "waiting",
  });
  assert.deepEqual(jobStatus(entry({ status: "failed" })), { text: "Didn't finish", tone: "failed" });
  assert.deepEqual(jobStatus(entry({ step: null, stage: null, error: "bad json" })), {
    text: "Can't be read",
    tone: "failed",
  });
  // A step's own error on a readable job is not an unreadable job.
  assert.equal(jobStatus(entry({ error: "Gemini said no" })).text, "Scenes · waiting for you");
  // All scenes approved: the backend's stage is still "scenes", the wait is for the render.
  assert.equal(jobStatus(entry({ stage: "scenes", step: "ready_to_render" })).text, "Render · waiting for you");
  // An interrupted job that also has a kept result still says it stopped partway.
  assert.equal(
    jobStatus(entry({ status: "result", stage: "render", interrupted: true })).text,
    "Render · stopped partway",
  );
});

test("jobAction: open, continue, back to the open job, or why it cannot open", () => {
  assert.deepEqual(jobAction(entry({ status: "result" }), false), { label: "Open", kind: "open", reason: null });
  // A job waiting for the editor only opens: "Resume" means starting GPU or Gemini work.
  assert.deepEqual(jobAction(entry(), false), { label: "Open", kind: "open", reason: null });
  // An interrupted job opens to the Resume or Stop choice: the button only opens it.
  assert.deepEqual(jobAction(entry({ interrupted: true, resumes_work: true }), false), {
    label: "Open",
    kind: "open",
    reason: null,
  });
  assert.equal(jobAction(entry({ open: true }), false).kind, "show");
  assert.equal(jobAction(entry({ open: true }), false).label, "Back to this job");
  assert.equal(jobAction(entry({ open: true, status: "running" }), true).label, "Watch progress");
  const elsewhere = jobAction(entry({ status: "running_elsewhere" }), false);
  assert.equal(elsewhere.kind, "none");
  assert.match(elsewhere.reason, /another Super Processor window/i);
  // The server refuses to switch jobs while a run works in this window.
  const waiting = jobAction(entry({ status: "result" }), true);
  assert.equal(waiting.kind, "none");
  assert.match(waiting.reason, /current run/);
  assert.equal(jobAction(entry({ step: null, error: "broken" }), false).kind, "none");
});

test("sortJobs: newest first, jobs with no time last, stable by id, input untouched", () => {
  const jobs = [
    entry({ job_id: "b0000001", updated_at: 10 }),
    entry({ job_id: "c0000001", updated_at: null }),
    entry({ job_id: "a0000001", updated_at: 30 }),
    entry({ job_id: "a0000002", updated_at: null }),
    entry({ job_id: "d0000001", updated_at: 30 }),
  ];
  const before = jobs.map((job) => job.job_id);
  assert.deepEqual(
    sortJobs(jobs).map((job) => job.job_id),
    ["a0000001", "d0000001", "b0000001", "a0000002", "c0000001"],
  );
  assert.deepEqual(jobs.map((job) => job.job_id), before);
});

test("changedText and jobMeta: short, plain times", () => {
  const now = new Date(2026, 10, 20, 15, 0, 0).getTime();
  const at = (date) => date.getTime() / 1000;
  assert.equal(changedText(null, now), "");
  assert.equal(changedText(now / 1000 - 20, now), "Changed just now");
  // A clock a little ahead of the page is still "just now".
  assert.equal(changedText(now / 1000 + 5, now), "Changed just now");
  assert.equal(changedText(now / 1000 - 5 * 60, now), "Changed 5 min ago");
  assert.equal(changedText(now / 1000 - 3 * 3600, now), "Changed 3 h ago");
  assert.equal(changedText(at(new Date(2026, 10, 19, 9, 0)), now), "Changed yesterday");
  assert.equal(changedText(at(new Date(2026, 10, 12, 9, 0)), now), "Changed 12 Nov");
  assert.equal(changedText(at(new Date(2025, 0, 3, 9, 0)), now), "Changed 3 Jan 2025");
  assert.equal(jobMeta(entry({ updated_at: now / 1000 - 120 }), now), "PrivateL1947.mp4 · Changed 2 min ago");
  // The file name is not repeated when it is the title.
  assert.equal(
    jobMeta(entry({ title: "clip.mp4", file_name: "clip.mp4", updated_at: null }), now),
    "",
  );
});

test("startsOnJobs and busyHere: when the page opens on the list", () => {
  const jobs = [entry()];
  assert.equal(startsOnJobs({ step: "pick_file", error: null }, jobs), true);
  assert.equal(startsOnJobs({ step: "pick_file", error: null }, []), false);
  assert.equal(startsOnJobs({ step: "segments", error: null }, jobs), false);
  assert.equal(startsOnJobs({ step: "pick_file", error: "could not read it" }, jobs), false);
  assert.equal(startsOnJobs({ step: "pick_file", error: null, busy: true }, jobs), false);
  // A stop note that landed on the file pick is shown there, not behind the list.
  assert.equal(startsOnJobs({ step: "pick_file", error: null, stopped: "2026-11-20T10:00:00" }, jobs), false);
  assert.equal(startsOnJobs({ step: "pick_file", error: null, stopped: null }, jobs), true);
  assert.equal(busyHere({}, jobs), false);
  assert.equal(busyHere({ busy: true }, jobs), true);
  assert.equal(busyHere({}, [entry({ status: "running" })]), true);
  // Another window's run does not stop this one switching jobs.
  assert.equal(busyHere({}, [entry({ status: "running_elsewhere" })]), false);
});

test("newVideoRoute: a new video leaves the open job where it was", () => {
  // Already on the file pick (or before it): just close the list.
  assert.equal(newVideoRoute({ step: "pick_file" }), "close");
  assert.equal(newVideoRoute({ step: "setup" }), "close");
  // From any job step: POST /api/new, which keeps that job's saved place on
  // disk (unlike /api/goto to=file, which would move the job back to the pick).
  for (const step of ["segments", "preview_segment", "ready_to_render", "result", "done"]) {
    assert.equal(newVideoRoute({ step }), "new", step);
  }
  // No "file" navigation is needed: /new goes to setup when there is no key.
  assert.equal(newVideoRoute({ step: "result", navigation: { setup: false, file: false, segments: false } }), "new");
  assert.equal(newVideoRoute({ step: "result", busy: true }), "blocked");
  assert.equal(newVideoRoute({ step: "rendering", busy: true }), "blocked");
});
