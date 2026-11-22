import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

// Same as shell-view.test.mjs: welcomeView.ts only imports types.
const source = await readFile(new URL("../src/welcomeView.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
});
assert.ok(!/from\s+["']\.\/api["']/.test(outputText), "welcomeView.ts must import only types");
const { isWelcomeStep, keyCheck, keyError, welcomeStart } = await import(
  `data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`
);

test("the intro, LLM choice and key steps are one Welcome & setup screen", () => {
  for (const step of ["intro", "llm_choice", "local_llm_stub", "setup"]) {
    assert.equal(isWelcomeStep(step), true, step);
  }
  for (const step of ["pick_file", "segments", "rendering", "result"]) {
    assert.equal(isWelcomeStep(step), false, step);
  }
});

test("Choose a video saves a typed key, keeps a found one, or asks for one", () => {
  assert.deepEqual(welcomeStart(false, "  abc  "), { kind: "send", action: "setup", fields: { api_key: "abc" } });
  // A typed key replaces the one found.
  assert.deepEqual(welcomeStart(true, "new"), { kind: "send", action: "setup", fields: { api_key: "new" } });
  assert.deepEqual(welcomeStart(true, ""), { kind: "send", action: "setup", fields: { skip: "1" } });
  assert.deepEqual(welcomeStart(true, "   "), { kind: "send", action: "setup", fields: { skip: "1" } });
  assert.deepEqual(welcomeStart(false, ""), { kind: "needs_key" });
});

test("the key check says found or not, never more than the server knows", () => {
  assert.equal(keyCheck(true).ok, true);
  assert.match(keyCheck(true).detail, /Found on this computer/);
  assert.equal(keyCheck(false).ok, false);
  assert.match(keyCheck(false).detail, /Not set up yet/);
});

test("only a setup refusal goes under the key field", () => {
  assert.equal(keyError({ step: "setup", error: "API key is empty" }), "API key is empty");
  assert.equal(keyError({ step: "setup", error: null }), null);
  assert.equal(keyError({ step: "llm_choice", error: "pick how you want to run the LLM" }), null);
});
