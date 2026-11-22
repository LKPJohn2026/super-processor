import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

async function load(name) {
  const source = await readFile(new URL(`../src/${name}.ts`, import.meta.url), "utf8");
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
  });
  assert.ok(!/from\s+["']\.\/api["']/.test(outputText), `${name}.ts must import only types`);
  return import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);
}

const pick = await load("pickView");
const finding = await load("findingView");

test("a refused pick becomes check rows, the failed one with its reason", () => {
  const long = pick.pickRefusal("video is 45.2 minutes long; the limit is 30 minutes", "/v/wedding-1998.mp4");
  assert.equal(long.name, "wedding-1998.mp4");
  assert.deepEqual(long.checks.map((c) => [c.key, c.ok]), [["size", true], ["length", false]]);
  assert.match(long.checks[1].value, /^45:12 — up to 30 min\. Trim/);
  const big = pick.pickRefusal("video is 3840x2160; the limit is 1080p (1920x1080 in either orientation)");
  assert.deepEqual(big.checks.map((c) => [c.key, c.ok]), [["size", false]]);
  assert.match(big.checks[0].value, /^3840×2160/);
  const vfr = pick.pickRefusal("VFR/unsupported video timing: only verified constant frame rate (CFR) is supported.");
  assert.deepEqual(vfr.checks.map((c) => [c.key, c.ok]), [["size", true], ["length", true], ["timing", false]]);
  assert.equal(pick.pickRefusal("file not found: /x/a.mp4").name, "a.mp4");
  assert.equal(pick.pickRefusal("clip.avi could not be read as a video; it may be damaged").name, "clip.avi");
  assert.equal(pick.pickRefusal("this file has no video stream").checks[0].key, "picture");
  assert.equal(pick.pickRefusal("only video files can be uploaded (.mp4, .mov)").checks[0].key, "type");
  assert.equal(pick.pickRefusal("Finding the segments failed: the stills are missing"), null);
  assert.equal(pick.pickRefusal(null), null);
});

test("a failed scene search is told apart from a refused file", () => {
  assert.equal(pick.findFailure("Finding the segments failed: the stills are missing"), "the stills are missing");
  assert.equal(pick.findFailure("video is 45.2 minutes long; the limit is 30 minutes"), null);
});

test("the browser's checks follow the wizard's limits", () => {
  assert.deepEqual(pick.browserChecks(null), []);
  const ok = pick.browserChecks({ width: 640, height: 480, duration: 1321 });
  assert.deepEqual(ok.map((c) => [c.key, c.ok, c.value]), [
    ["size", true, "640×480 — up to 1080p"],
    ["length", true, "22:01 — up to 30 min"],
  ]);
  assert.equal(pick.checksPass(ok), true);
  // Portrait 1080x1920 fits; 2560x1440 does not; 31 min does not.
  assert.equal(pick.browserChecks({ width: 1080, height: 1920, duration: 60 })[0].ok, true);
  assert.equal(pick.browserChecks({ width: 2560, height: 1440, duration: 60 })[0].ok, false);
  const long = pick.browserChecks({ width: 640, height: 480, duration: 31 * 60 });
  assert.equal(long[1].ok, false);
  assert.equal(pick.checksPass(long), false);
  // A length the browser cannot read (Infinity) is left to the wizard.
  assert.equal(pick.browserChecks({ width: 640, height: 480, duration: Infinity }).length, 1);
  assert.equal(pick.fileSizeText(94_000_000), "94 MB");
  assert.equal(pick.fileSizeText(1_400_000_000), "1.4 GB");
});

const view = (step, message, extra = {}) => ({ step, progress: message === null ? null : { phase: step, pct: 50, message, ...extra } });

test("the stage list follows the server's own progress words", () => {
  const states = (v) => finding.findingStages(v).map((s) => `${s.key}:${s.state}`);
  assert.deepEqual(states(view("sampling", "Saving stills from the video: 7 of 22")), [
    "stills:running", "copy:waiting", "upload:waiting", "gemini:waiting",
  ]);
  assert.equal(finding.findingStages(view("sampling", "Saving stills from the video: 7 of 22"))[0].meta, "7 of 22");
  assert.deepEqual(states(view("proposing", "Preparing a silent copy for Gemini: 40%")), [
    "stills:done", "copy:running", "upload:waiting", "gemini:waiting",
  ]);
  assert.deepEqual(states(view("proposing", "Uploading the video to Gemini: 10%")), [
    "stills:done", "copy:done", "upload:running", "gemini:waiting",
  ]);
  assert.deepEqual(states(view("proposing", "Gemini is watching the video and proposing segments")), [
    "stills:done", "copy:done", "upload:done", "gemini:running",
  ]);
  // A word left over from the stills: they are done, nothing else is said to run.
  assert.deepEqual(states(view("proposing", "Saving stills from the video: 22 of 22")), [
    "stills:done", "copy:waiting", "upload:waiting", "gemini:waiting",
  ]);
});

test("when Gemini falls back to the stills, the list says so", () => {
  const rows = finding.findingStages(view("proposing", "Gemini is proposing segments from the stills"));
  assert.deepEqual(rows.map((s) => `${s.key}:${s.state}`), ["stills:done", "video:skipped", "gemini:running"]);
  assert.match(rows[2].label, /from the stills$/);
});

test("stage times come only from what the page saw", () => {
  let clock = finding.tickClock(finding.NO_CLOCK, "stills", 10_000, 4);
  assert.equal(clock.started.stills, 6_000);
  clock = finding.tickClock(clock, "copy", 20_000);
  clock = finding.tickClock(clock, "copy", 25_000);
  const rows = finding.findingStages(view("proposing", "Preparing a silent copy for Gemini"), clock, 32_000);
  assert.equal(rows[0].meta, "0:14");
  assert.equal(rows[1].meta, "0:12 so far");
  // A stage first seen mid-way has no start: no time is made up.
  const late = finding.tickClock(finding.NO_CLOCK, "upload", 1_000, 30);
  assert.equal(late.started.upload, null);
});

test("time left only when the server measures it, else the time spent", () => {
  assert.deepEqual(finding.findingTime({ eta_s: 130, elapsed_s: 63 }), { big: "About 2 min left", forStage: true, spent: "1:03 so far" });
  assert.equal(finding.findingTime({ eta_s: 40 }, 10).big, "About 30 s left");
  assert.deepEqual(finding.findingTime({ eta_s: null, elapsed_s: 63 }), { big: "1:03 so far", forStage: false, spent: "" });
  assert.equal(finding.findingTime(null).big, "Starting…");
  assert.match(finding.startedAt(1_795_324_000), /^\d\d:\d\d$/);
  assert.equal(finding.startedAt(null), "");
});

test("Gemini's failed split is told as quota (S3), another failure, or no scenes (S8)", () => {
  const whole = [{ start_s: 0, end_s: 60, label: "whole video", note: "", contains: [] }];
  const quota = finding.scenesProblem({
    segments: whole,
    note: "",
    error: "Gemini could not propose segments (quota used up; try again in 28 s); the whole video is one.",
  });
  assert.deepEqual(quota, { kind: "quota", daily: false, retryS: 28, detail: "quota used up; try again in 28 s" });
  const daily = finding.scenesProblem({ segments: whole, note: "", error: "Gemini could not propose segments (daily quota used up; try again in about 3 h); the whole video is one." });
  assert.equal(daily.daily, true);
  assert.equal(daily.retryS, 10800);
  assert.deepEqual(
    finding.scenesProblem({ segments: whole, note: "", error: "Gemini could not propose segments (timed out); the whole video is one." }),
    { kind: "failed", detail: "timed out" },
  );
  assert.deepEqual(finding.scenesProblem({ segments: whole, note: "", error: null }), { kind: "empty" });
  assert.equal(finding.scenesProblem({ segments: [{ ...whole[0], label: "Opening" }], note: "", error: null }), null);
  assert.equal(finding.retryAfterS("try again in about 2 min"), 120);
  assert.equal(finding.askAgainNote({ note: " cut at the wedding " }), "cut at the wedding");
  assert.equal(finding.askAgainNote({ note: "" }), "Find where the scenes change.");
});
