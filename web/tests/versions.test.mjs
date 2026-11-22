import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

// Same as run-states.test.mjs: the installed compiler, loaded as a data: module.
// versions.ts only imports types, which the transpile drops.
const source = await readFile(new URL("../src/versions.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
});
assert.ok(!/from\s+["']\.\/api["']/.test(outputText), "versions.ts must import only types");
const {
  downloadName,
  keptText,
  originText,
  prefetchNote,
  sizeText,
  versionLabel,
  versionMeta,
  versionNote,
  versionsNewestFirst,
  versionsSummary,
} = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);

const NOW = Date.parse("2026-11-20T12:00:00Z");
const version = (number, extra = {}) => ({
  number,
  name: `v${String(number).padStart(3, "0")}.mp4`,
  url: `/jobs/abcdef12/outputs/v${String(number).padStart(3, "0")}.mp4`,
  kind: "render",
  created_at: "2026-11-20T11:55:00+00:00",
  note: null,
  size: 820_000_000,
  scale: 2,
  strength: 0.3,
  strength_summary: null,
  current: false,
  ...extra,
});

test("a version is labelled by its number", () => {
  assert.equal(versionLabel(version(1)), "Version 1");
  assert.equal(versionLabel(version(12)), "Version 12");
});

test("when a version was kept reads like the jobs list", () => {
  assert.equal(keptText("2026-11-20T11:59:30+00:00", NOW), "Kept just now");
  assert.equal(keptText("2026-11-20T11:55:00+00:00", NOW), "Kept 5 min ago");
  assert.equal(keptText("2026-11-20T09:00:00+00:00", NOW), "Kept 3 h ago");
  const older = "2026-11-02T08:05:00+00:00";
  const at = new Date(Date.parse(older));
  const hhmm = `${String(at.getHours()).padStart(2, "0")}:${String(at.getMinutes()).padStart(2, "0")}`;
  assert.equal(keptText(older, NOW), `Kept ${at.getDate()} Nov, ${hhmm}`);
  assert.match(keptText("2024-03-01T12:00:00+00:00", NOW), /^Kept \d+ (Feb|Mar) 2024, \d\d:\d\d$/);
  // A clock that runs behind the server never says "in the future".
  assert.equal(keptText("2026-11-20T12:05:00+00:00", NOW), "Kept just now");
});

test("a version without a time (found on disk) says nothing about when", () => {
  assert.equal(keptText(null, NOW), "");
  assert.equal(keptText("", NOW), "");
  assert.equal(keptText("not a time", NOW), "");
});

test("sizes read in MB and GB", () => {
  assert.equal(sizeText(820_000_000), "820 MB");
  assert.equal(sizeText(1_140_000_000), "1.1 GB");
  assert.equal(sizeText(400_000), "under 1 MB");
  assert.equal(sizeText(0), "");
  assert.equal(sizeText(Number.NaN), "");
});

test("the meta line joins when, where from, scale and size", () => {
  assert.equal(versionMeta(version(2), NOW), "Kept 5 min ago · 2× · 820 MB");
  assert.equal(
    versionMeta(version(1, { kind: "recovered", created_at: "", scale: null, size: 0 }), NOW),
    "Found in the job folder",
  );
  assert.equal(originText("earlier"), "Kept from before");
  assert.equal(originText("render"), "");
});

test("a version's note is the editor's words, quoted; the first render says so", () => {
  assert.equal(versionNote(version(2, { note: " less AI detail 2:00–2:10 " })), "“less AI detail 2:00–2:10”");
  assert.equal(versionNote(version(1)), "First render");
  assert.equal(versionNote(version(3, { note: "   " })), "");
  assert.equal(versionNote(version(1, { kind: "recovered" })), "");
});

test("the list shows the newest first and says which one is current", () => {
  const list = [version(1), version(3, { current: true }), version(2)];
  assert.deepEqual(versionsNewestFirst(list).map((v) => v.number), [3, 2, 1]);
  // The server's list is left as it was.
  assert.deepEqual(list.map((v) => v.number), [1, 3, 2]);
  assert.deepEqual(versionsNewestFirst(undefined), []);
  assert.equal(versionsSummary(list), "3 versions kept · the result above is version 3");
  assert.equal(versionsSummary([version(1)]), "1 version kept");
  assert.equal(versionsSummary([]), "");
});

test("a download is named after the job and the version", () => {
  assert.equal(downloadName("Private Life of a Cat", version(2)), "Private Life of a Cat – version 2.mp4");
  assert.equal(downloadName(null, version(1)), "Restored video – version 1.mp4");
  assert.equal(downloadName("../a/b:c", version(1)), "a b c – version 1.mp4");
  assert.equal(downloadName("PrivateL1947.mp4", version(2)), "PrivateL1947 – version 2.mp4");
});

test("the background preview note shows only for the next scene", () => {
  const preview = (next) => ({ index: 1, count: 5, next });
  assert.equal(prefetchNote(preview(undefined)), null);
  assert.equal(prefetchNote(preview(null)), null);
  assert.equal(
    prefetchNote(preview({ index: 2, ready: false, running: true })),
    "Getting scene 3 ready in the background, so it opens sooner.",
  );
  assert.equal(
    prefetchNote(preview({ index: 2, ready: true, running: false })),
    "The preview of scene 3 is ready: it opens as soon as you approve this one.",
  );
  // Ended without a preview (it failed or was dropped): nothing to say.
  assert.equal(prefetchNote(preview({ index: 2, ready: false, running: false })), null);
  // For another scene, or past the last one.
  assert.equal(prefetchNote(preview({ index: 4, ready: true, running: false })), null);
  assert.equal(prefetchNote({ index: 4, count: 5, next: { index: 5, ready: true, running: false } }), null);
});
