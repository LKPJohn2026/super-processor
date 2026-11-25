import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

// The same transpile-and-import as api.test.mjs: the real api.ts, no browser.
const source = await readFile(new URL("../src/api.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
});
const moduleText = outputText.replaceAll("import.meta.env", "globalThis.__viteEnv");
let moduleId = 0;

function mockGlobal(t, name, value) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, name);
  Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, name, previous);
    else delete globalThis[name];
  });
}

class MemoryStorage {
  items = new Map();
  getItem(key) { return this.items.has(key) ? this.items.get(key) : null; }
  setItem(key, value) { this.items.set(key, String(value)); }
}

/** A page opened at `href`, with fetch recorded; returns the fresh module. */
async function page(t, href, { fetcher, storage = new MemoryStorage(), env = {} } = {}) {
  const url = new URL(href);
  const replaced = [];
  const location = {
    origin: url.origin, pathname: url.pathname, search: url.search, hash: url.hash,
  };
  mockGlobal(t, "window", {
    location,
    history: {
      state: null,
      replaceState(_state, _title, next) {
        replaced.push(next);
        const moved = new URL(next, url.origin);
        location.search = moved.search;
        location.hash = moved.hash;
      },
    },
  });
  mockGlobal(t, "sessionStorage", storage);
  mockGlobal(t, "__viteEnv", env);
  const calls = [];
  t.mock.method(globalThis, "fetch", async (target, options = {}) => {
    calls.push({ url: String(target), options });
    return fetcher ? fetcher(String(target), options) : Response.json({ step: "intro", error: null });
  });
  const api = await import(
    `data:text/javascript;base64,${Buffer.from(moduleText).toString("base64")}#${moduleId++}`
  );
  return { api, calls, replaced, storage };
}

test("only a wizard on this computer is accepted as an address", async (t) => {
  const { api } = await page(t, "https://lkpjohn2026.github.io/super-processor/");
  for (const good of [
    ["http://127.0.0.1:45143", "http://127.0.0.1:45143"],
    ["http://127.0.0.1:8000/", "http://127.0.0.1:8000"],
    ["http://localhost:1", "http://localhost:1"],
    ["http://[::1]:9/", "http://[::1]:9"],
    [" http://127.0.0.1:5/?token=abc ", "http://127.0.0.1:5"],
  ]) {
    assert.equal(api.loopbackOrigin(good[0]), good[1], good[0]);
  }
  for (const bad of [
    "https://evil.example",
    "https://attacker.example",
    "http://evil.test:45143",
    "http://box.test:8000",
    "http://127.0.0.1.evil.test",
    "http://localhost.evil.test:1",
    "http://localhost@evil.test",
    "http://user@127.0.0.1:1",
    "http://user:pw@localhost:1",
    "javascript:alert(1)",
    "file:///etc/passwd",
    "127.0.0.1:45143",
    "",
  ]) {
    assert.equal(api.loopbackOrigin(bad), null, bad);
  }
});

test("a pasted launch URL gives the origin and its token; others are refused", async (t) => {
  const { api } = await page(t, "http://localhost:5173/");
  assert.deepEqual(api.parseWizardAddress("http://127.0.0.1:41234/?token=abc"), {
    origin: "http://127.0.0.1:41234", token: "abc",
  });
  assert.deepEqual(api.parseWizardAddress("http://127.0.0.1:41234/#token=xyz"), {
    origin: "http://127.0.0.1:41234", token: "xyz",
  });
  assert.deepEqual(api.parseWizardAddress("http://localhost:41234"), {
    origin: "http://localhost:41234", token: "",
  });
  assert.match(api.parseWizardAddress("  ").error, /Enter the URL/);
  assert.equal(api.parseWizardAddress("https://evil.example/?token=abc").error, api.LOOPBACK_ONLY);
});

test("a hostile ?api= link is ignored, with a reason, and never contacted", async (t) => {
  for (const hostile of [
    "https://attacker.example",
    "http://evil.test:45143",
    "http://127.0.0.1.evil.test",
    "http://localhost@evil.test",
    "javascript:alert(1)",
  ]) {
    const { api, calls } = await page(
      t, `https://lkpjohn2026.github.io/super-processor/?api=${encodeURIComponent(hostile)}`,
    );
    const origin = api.configuredOrigin();
    assert.equal(origin, "", hostile);
    assert.match(api.configuredOriginProblem(), /Only a wizard on this computer/, hostile);
    // What the welcome screen would send the key with goes nowhere else.
    await api.loadState(origin);
    await api.postAction(origin, "setup", { api_key: "AIza-not-a-real-key" });
    for (const { url } of calls) {
      assert.ok(url.startsWith("/api/"), `${hostile}: fetched ${url}`);
    }
  }
  const { api } = await page(t, "http://localhost:5173/?live=1&port=99999");
  assert.equal(api.configuredOrigin(), "");
  assert.match(api.configuredOriginProblem(), /not a port/);
  const env = await page(t, "http://localhost:5173/", { env: { VITE_API_ORIGIN: "https://evil.example" } });
  assert.equal(env.api.configuredOrigin(), "");
  assert.match(env.api.configuredOriginProblem(), /VITE_API_ORIGIN/);
  const fine = await page(t, "http://localhost:5173/?api=http://127.0.0.1:41234");
  assert.equal(fine.api.configuredOrigin(), "http://127.0.0.1:41234");
  assert.equal(fine.api.configuredOriginProblem(), null);
});

test("the launch token is taken once, dropped from the address bar, and sent", async (t) => {
  for (const href of [
    "http://localhost:5173/?token=launch-T&view=1",
    "http://localhost:5173/?view=1#token=launch-T",
  ]) {
    const storage = new MemoryStorage();
    const { api, calls, replaced } = await page(t, href, {
      storage,
      fetcher: async () => Response.json({ step: "result", error: null, job_id: "j" }),
    });
    const origin = api.configuredOrigin();
    api.adoptLaunchToken(origin);
    assert.deepEqual(replaced, ["/?view=1"], href);
    assert.equal(window.location.search, "?view=1");
    // A second look (React StrictMode) finds nothing left to take.
    api.adoptLaunchToken(origin);
    assert.equal(replaced.length, 1);
    await api.loadState(origin);
    await api.loadJobs(origin).catch(() => {});
    await api.postAction(origin, "intro", {});
    for (const call of calls) {
      assert.equal(call.options.headers["X-Wizard-Token"], "launch-T", call.url);
    }
    // Media can't send a header: the wizard's paths carry the token as ?t=.
    assert.equal(api.outputSrc(origin, "/output.mp4"), "/output.mp4?t=launch-T");
    assert.equal(
      api.outputSrc(origin, "/jobs/abc/output.mp4?v=3#t=1.0"),
      "/jobs/abc/output.mp4?v=3&t=launch-T#t=1.0",
    );
    // Another host never gets it.
    assert.equal(api.outputSrc(origin, "https://cdn.test/x.jpg"), "https://cdn.test/x.jpg");
    // A reload in this tab still has it, from sessionStorage.
    const reloaded = await page(t, "http://localhost:5173/?view=1", { storage });
    assert.equal(reloaded.api.outputSrc("", "/output.mp4"), "/output.mp4?t=launch-T");
  }
});

test("tokens stay with their own wizard, and a 401 is told apart from offline", async (t) => {
  const { api, calls } = await page(t, "http://localhost:5173/", {
    fetcher: async (url) =>
      url.startsWith("http://127.0.0.1:2")
        ? Response.json({ error: "This wizard needs its token." }, { status: 401 })
        : Response.json({ step: "intro", error: null }),
  });
  api.setToken("http://127.0.0.1:1", "token-one");
  assert.equal(api.outputSrc("http://127.0.0.1:2", "/output.mp4"), "http://127.0.0.1:2/output.mp4");
  assert.equal(await api.loadState("http://127.0.0.1:2"), null);
  assert.equal(api.needsToken("http://127.0.0.1:2"), true);
  assert.equal(calls[0].options.headers["X-Wizard-Token"], undefined);
  assert.ok(await api.loadState("http://127.0.0.1:1"));
  assert.equal(api.needsToken("http://127.0.0.1:1"), false);
  assert.equal(calls[1].options.headers["X-Wizard-Token"], "token-one");
});

test("blocked storage leaves the token in memory only", async (t) => {
  const blocked = {
    getItem() { throw new Error("blocked"); },
    setItem() { throw new Error("blocked"); },
  };
  const { api } = await page(t, "http://localhost:5173/#token=mem", { storage: blocked });
  api.adoptLaunchToken(api.configuredOrigin());
  assert.equal(api.outputSrc("", "/output.mp4"), "/output.mp4?t=mem");
});

test("the app the wizard serves keeps the token out of media URLs", async (t) => {
  const { api } = await page(t, "http://127.0.0.1:41000/", { env: { MODE: "package" } });
  api.setToken("", "served-T");
  // Same origin: the HttpOnly cookie covers media, so no ?t= in any link.
  assert.equal(api.outputSrc("", "/jobs/abc/output.mp4"), "/jobs/abc/output.mp4");
  assert.equal(api.outputSrc("", "/output.mp4?v=2#t=1"), "/output.mp4?v=2#t=1");
  assert.equal(
    api.outputSrc("http://127.0.0.1:41000", "/output.mp4"),
    "http://127.0.0.1:41000/output.mp4",
  );
  // Another wizard still needs the token in the URL (its cookie is not ours).
  api.setToken("http://127.0.0.1:41001", "other-T");
  assert.equal(
    api.outputSrc("http://127.0.0.1:41001", "/output.mp4"),
    "http://127.0.0.1:41001/output.mp4?t=other-T",
  );
  // The Vite and Pages shells (other modes) keep ?t=.
  const shell = await page(t, "http://127.0.0.1:41000/", { env: { MODE: "development" } });
  shell.api.setToken("", "served-T");
  assert.equal(shell.api.outputSrc("", "/output.mp4"), "/output.mp4?t=served-T");
});
