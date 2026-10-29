import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

// Use the installed compiler, not a second TS runner or a browser download.
const source = await readFile(new URL("../src/api.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
});
let moduleId = 0;
function mockGlobal(t, name, value) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, name);
  Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, name, previous);
    else delete globalThis[name];
  });
}
async function fixture(t, fetcher) {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  mockGlobal(t, "window", { location: { origin: "http://ui.test", search: "" } });
  t.mock.method(globalThis, "fetch", fetcher);
  // Fresh module state isolates token and in-flight maps between tests.
  return import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}#${moduleId++}`);
}
const state = (step, extra = {}) => ({ step, error: null, ...extra });
const response = (step, extra) => Response.json(state(step, extra));
const flush = async () => {
  for (let i = 0; i < 40; i += 1) await Promise.resolve();
};
async function advance(t, ms) {
  t.mock.timers.tick(ms);
  await flush();
}

for (const [runner, initial, intermediate, final] of [
  ["runSegmenting", "sampling", "proposing", "segments"],
  ["runAnalyze", "analyzing", "analyzing", "preview_segment"],
  ["runRender", "rendering", "rendering", "result"],
]) {
  test(`${runner}: StrictMode cleanup and re-subscription keep progress, one start`, async (t) => {
    const calls = [];
    const queue = [initial, intermediate, final];
    const api = await fixture(t, async (url) => {
      calls.push(url);
      return response(queue.shift());
    });
    const old = [], current = [], concurrent = [];
    const subscription = new AbortController();
    const first = api[runner]("http://a.test", (v) => old.push(v.step), subscription.signal);
    // React StrictMode setup -> cleanup -> setup, even before the first response.
    subscription.abort();
    const second = api[runner]("http://a.test/", (v) => current.push(v.step));
    await flush();
    const third = api[runner]("http://a.test", (v) => concurrent.push(v.step));
    assert.deepEqual(concurrent, [initial], "late subscriber gets last progress immediately");
    await advance(t, 2000);
    await advance(t, 2000);
    const results = await Promise.all([first, second, third]);
    assert.deepEqual(old, []);
    assert.deepEqual(current, [initial, intermediate]);
    assert.deepEqual(concurrent, current);
    assert.ok(results.every((v) => v.step === final));
    assert.equal(calls.filter((url) => url.includes("run=1")).length, 1);
  });
}

test("a step-change effect cleanup can re-subscribe after progress was delivered", async (t) => {
  const queue = ["sampling", "proposing", "segments"];
  const api = await fixture(t, async () => response(queue.shift()));
  const old = [], current = [];
  const subscription = new AbortController();
  const first = api.runSegmenting("http://a.test", (v) => old.push(v.step), subscription.signal);
  await flush();
  subscription.abort();
  const second = api.runSegmenting("http://a.test", (v) => current.push(v.step));
  await advance(t, 2000);
  await advance(t, 2000);
  await Promise.all([first, second]);
  assert.deepEqual(old, ["sampling"]);
  assert.deepEqual(current, ["sampling", "proposing"]);
});

test("operations and delayed token responses stay scoped by origin (including port) and route", async (t) => {
  const calls = [];
  let releaseA;
  const delayedA = new Promise((resolve) => { releaseA = resolve; });
  const api = await fixture(t, async (url, options) => {
    calls.push([url, options?.headers?.["X-Wizard-Token"]]);
    if (url === "http://a.test:1001/api/state") return delayedA;
    if (url === "http://a.test:1002/api/state") return response("intro", { token: "token-B" });
    return response(url.includes("analyze") ? "preview_segment" : "result");
  });
  const loadingA = api.loadState("http://a.test:1001");
  await api.loadState("http://a.test:1002");
  releaseA(response("intro", { token: "token-A" }));
  await loadingA;
  await Promise.all([
    api.runRender("http://a.test:1001", () => {}),
    api.runRender("http://a.test:1002", () => {}),
    api.runAnalyze("http://a.test:1001", () => {}),
    api.postAction("http://a.test:1002", "intro", {}),
    api.postAction("http://unknown.test", "intro", {}),
  ]);
  assert.deepEqual(calls.filter(([url]) => !url.endsWith("/state")), [
    ["http://a.test:1001/api/render?run=1", "token-A"],
    ["http://a.test:1002/api/render?run=1", "token-B"],
    ["http://a.test:1001/api/analyze?run=1", "token-A"],
    ["http://a.test:1002/api/intro", "token-B"],
    ["http://unknown.test/api/intro", ""],
  ]);
});

test("concurrent servers never share progress or results", async (t) => {
  const api = await fixture(t, async (url) => {
    const server = new URL(url).origin;
    return response(url.includes("run=1") ? "rendering" : "result", { job_id: server });
  });
  const progressA = [], progressB = [];
  const a = api.runRender("http://a.test", (view) => progressA.push(view.job_id));
  const b = api.runRender("http://b.test", (view) => progressB.push(view.job_id));
  await flush();
  await advance(t, 2000);
  const [resultA, resultB] = await Promise.all([a, b]);
  assert.deepEqual(progressA, ["http://a.test"]);
  assert.deepEqual(progressB, ["http://b.test"]);
  assert.equal(resultA.job_id, "http://a.test");
  assert.equal(resultB.job_id, "http://b.test");
});

test("upload uses the destination token, even after connecting to another server", async (t) => {
  const api = await fixture(t, async (url) => response("pick_file", {
    token: url.startsWith("http://a.test") ? "token-A" : "token-B",
  }));
  await api.loadState("http://a.test");
  await api.loadState("http://b.test");
  let xhr;
  class XHR {
    headers = {};
    upload = {};
    status = 200;
    responseText = JSON.stringify(state("sampling", { token: "token-A2" }));
    constructor() { xhr = this; }
    open(method, url) { this.method = method; this.url = url; }
    setRequestHeader(name, value) { this.headers[name] = value; }
    send() { this.onload(); }
  }
  mockGlobal(t, "XMLHttpRequest", XHR);
  await api.uploadVideo("http://a.test", { name: "my clip.mp4" }, () => {});
  assert.equal(xhr.url, "http://a.test/api/upload");
  assert.equal(xhr.headers["X-Wizard-Token"], "token-A");
  assert.equal(xhr.headers["X-File-Name"], "my%20clip.mp4");
  assert.equal(xhr.headers["Content-Type"], "application/octet-stream");
});

test("same-origin proxy and explicit UI origin share work and tokens", async (t) => {
  let count = 0;
  const api = await fixture(t, async (url, options) => {
    if (url.endsWith("/state")) return response("intro", { token: "proxy-token" });
    count += 1;
    assert.equal(options.headers["X-Wizard-Token"], "proxy-token");
    return response("result");
  });
  await api.loadState("");
  await Promise.all([api.runRender("", () => {}), api.runRender("http://ui.test", () => {})]);
  assert.equal(count, 1);
});

test("transient start failures retry with bounded exponential backoff", async (t) => {
  let attempts = 0;
  const api = await fixture(t, async () => {
    attempts += 1;
    if (attempts === 1) throw new TypeError("offline");
    if (attempts === 2) return new Response("proxy unavailable", { status: 503 });
    if (attempts === 3) return new Response("busy", { status: 429 });
    return response("result");
  });
  const result = api.runRender("http://a.test", () => {});
  await flush();
  assert.equal(attempts, 1);
  await advance(t, 499);
  assert.equal(attempts, 1);
  await advance(t, 1);
  assert.equal(attempts, 2);
  await advance(t, 1000);
  assert.equal(attempts, 3);
  await advance(t, 2000);
  assert.equal((await result).step, "result");
  assert.equal(attempts, 4);
});

test("poll failures recover without starting work again; success resets retry budget", async (t) => {
  let starts = 0, polls = 0;
  const api = await fixture(t, async (url) => {
    if (url.includes("run=1")) { starts += 1; return response("rendering"); }
    polls += 1;
    if (polls === 1 || polls === 3) throw new TypeError("offline");
    return response(polls < 4 ? "rendering" : "result");
  });
  const seen = [];
  const result = api.runRender("http://a.test", (v) => seen.push(v.step));
  await flush();
  await advance(t, 2000);
  await advance(t, 500);
  await advance(t, 2000);
  await advance(t, 500);
  assert.equal((await result).step, "result");
  assert.equal(starts, 1);
  assert.deepEqual(seen, ["rendering", "rendering"]);
});

test("retry exhaustion rejects, clears operation, and permits explicit recovery", async (t) => {
  let attempts = 0, online = false;
  const api = await fixture(t, async () => {
    attempts += 1;
    if (!online) throw new TypeError("offline");
    return response("result");
  });
  const result = api.runRender("http://a.test", () => {});
  const rejected = assert.rejects(result, /offline/);
  await flush();
  for (const delay of [500, 1000, 2000]) await advance(t, delay);
  await rejected;
  await advance(t, 60000);
  assert.equal(attempts, 4, "retries stop after three backoffs");
  online = true;
  assert.equal((await api.loadState("http://a.test")).step, "result");
  assert.equal((await api.runRender("http://a.test", () => {})).step, "result");
});

for (const phase of ["headers", "body"]) {
  test(`hung response ${phase} times out, retries, and eventually rejects`, async (t) => {
    let attempts = 0, aborted = 0;
    const api = await fixture(t, async (_url, { signal }) => {
      attempts += 1;
      const hanging = new Promise((_resolve, reject) => {
        signal.addEventListener("abort", () => {
          aborted += 1;
          reject(new DOMException("Timed out", "AbortError"));
        }, { once: true });
      });
      return phase === "headers" ? hanging : { ok: true, json: () => hanging };
    });
    const rejected = assert.rejects(api.runRender("http://a.test", () => {}), /Timed out/);
    await flush();
    for (const delay of [500, 1000, 2000]) {
      await advance(t, 10000);
      await advance(t, delay);
    }
    await advance(t, 10000);
    await rejected;
    assert.equal(attempts, 4);
    assert.equal(aborted, 4);
  });
}

for (const [name, makeResponse] of [
  ["forbidden", () => Response.json({ error: "wrong token" }, { status: 403 })],
  ["invalid state", () => Response.json({ step: "unknown" })],
  ["invalid JSON", () => new Response("not JSON")],
]) {
  test(`${name} is not retried`, async (t) => {
    let attempts = 0;
    const api = await fixture(t, async () => { attempts += 1; return makeResponse(); });
    await assert.rejects(api.runRender("http://a.test", () => {}));
    await advance(t, 60000);
    assert.equal(attempts, 1);
  });
}
