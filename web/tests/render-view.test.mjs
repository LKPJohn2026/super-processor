import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

// Same as shell-view.test.mjs: renderView.ts only imports types.
const source = await readFile(new URL("../src/renderView.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
});
assert.ok(!/from\s+["']\.\/api["']/.test(outputText), "renderView.ts must import only types");
const {
  detailBadge,
  failureTitle,
  initialScale,
  offersWithoutDetail,
  posterUrl,
  readyTitle,
  renderStages,
  sizeChoices,
  timeDone,
  timeLeft,
} = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);

const shot = (index, extra = {}) => ({
  index,
  label: `S${index + 1}`,
  before_url: `/previews/shot_00${index}_r1_before.mp4?g=1`,
  after_url: `/previews/shot_00${index}_r1_after.mp4?g=1`,
  strength: 0.3,
  ...extra,
});
const looks = (extra = {}) => ({ scale: 2, error: null, shots: [shot(0), shot(1), shot(2)], scales: [1, 2], source_size: [640, 480], ...extra });

test("the preview's still has the clip's name", () => {
  assert.equal(posterUrl("/previews/a_after.mp4?g=3"), "/previews/a_after.jpg?g=3");
  assert.equal(posterUrl(null), null);
});

test("the size cards say the real size, biggest first, with a real still", () => {
  const [big, same] = sizeChoices(looks());
  assert.equal(big.title, "Sharper and bigger");
  assert.equal(big.spec, "2× · 1280×960 · AI detail where you approved it");
  // Previews were made at 2×: the restored still shows what this card gives.
  assert.equal(big.image.url, "/previews/shot_001_r1_after.jpg?g=1");
  assert.equal(same.title, "Same size, cleaned up");
  assert.equal(same.spec, "640×480 · clean-up only, no AI detail");
  assert.equal(same.image.url, "/previews/shot_001_r1_before.jpg?g=1");
  assert.equal(big.recommended, false);
});

test("a card is Gemini's pick only while its reason stands, and no size is made up", () => {
  const [big, same] = sizeChoices(looks({ scale_reason: "soft source", source_size: [0, 0] }));
  assert.equal(big.recommended, true);
  assert.equal(same.recommended, false);
  assert.equal(big.spec, "2× · AI detail where you approved it");
  assert.equal(sizeChoices(looks({ shots: [] }))[0].image, null);
});

test("the size chosen first is the plan's", () => {
  assert.equal(initialScale(looks()), 2);
  assert.equal(initialScale(looks({ scale: 1 })), 1);
  assert.equal(initialScale(looks({ scale: 4 })), 2);
  assert.equal(initialScale(looks({ scales: [] })), null);
});

test("the heading counts the approved scenes", () => {
  assert.equal(readyTitle(5), "All 5 scenes approved — choose the size");
  assert.equal(readyTitle(1), "The scene is approved — choose the size");
});

test("each scene's AI detail follows the chosen size", () => {
  assert.deepEqual(detailBadge({}, "Gentle", 2), { text: "AI detail: Gentle", tone: "info" });
  assert.deepEqual(detailBadge({ use_upscale: false }, "Gentle", 2), { text: "Just resize", tone: "neutral" });
  assert.deepEqual(detailBadge({}, "Gentle", 1), { text: "Clean-up only", tone: "neutral" });
  assert.deepEqual(detailBadge({}, "None", 2), { text: "No AI detail", tone: "neutral" });
});

test("Render without AI detail is the 1× option, only when offered and not chosen", () => {
  assert.equal(offersWithoutDetail(looks(), 2), true);
  assert.equal(offersWithoutDetail(looks(), 1), false);
  assert.equal(offersWithoutDetail(looks({ scales: [2] }), 2), false);
});

test("the time left is the server's estimate only", () => {
  assert.equal(timeLeft(null), null);
  assert.equal(timeLeft(190), "About 3 min left");
  assert.equal(timeLeft(70, 20), "Less than a minute left");
  assert.equal(timeLeft(3 * 3600 + 7 * 60), "About 3 h 07 min left");
  assert.equal(timeDone(125), "2 min done");
  assert.equal(timeDone(undefined), "");
});

test("the render's stages follow the server's steps", () => {
  const states = (p) => renderStages(p).map((s) => s.state);
  assert.deepEqual(states({ pct: 5, message: "Upscaling on the GPU" }), ["running", "waiting", "waiting"]);
  assert.equal(renderStages({ pct: 5, message: "Upscaling on the GPU" })[0].meta, "starting");
  assert.deepEqual(states({ pct: 47, message: "Upscaling on the GPU: 49% of the work" }), ["running", "waiting", "waiting"]);
  assert.deepEqual(states({ pct: 90, message: "Checking for seams" }), ["done", "running", "waiting"]);
  assert.deepEqual(states({ pct: 95, message: "Keeping a copy of this version" }), ["done", "done", "running"]);
  // Keeping the earlier result's copy comes first, at 2%.
  assert.deepEqual(states({ pct: 2, message: "Keeping a copy of this version" }), ["running", "waiting", "waiting"]);
});

test("S5's title gives the server's reason once", () => {
  assert.equal(
    failureTitle("The render failed, so there is no result yet: the graphics card ran out of memory."),
    "The render stopped: the graphics card ran out of memory",
  );
  assert.equal(failureTitle("disk full"), "The render stopped: disk full");
});
