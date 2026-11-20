import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

// Use the installed compiler, not a second TS runner or a browser download.
const source = await readFile(new URL("../src/api.ts", import.meta.url), "utf8");
const { outputText: transpiled } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
});
// Vite replaces import.meta.env at build time; a data: module has no env, so
// stand in a global the test controls, read when configuredOrigin() runs.
const outputText = transpiled.replaceAll("import.meta.env", "globalThis.__viteEnv");
assert.ok(outputText !== transpiled, "api.ts no longer reads import.meta.env");
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
// Resolves to "pending" if the promise has not settled yet: with mocked timers a
// wrongly retried request would otherwise hang the test instead of failing it.
async function outcome(promise) {
  const result = promise.then(() => "fulfilled", () => "rejected");
  await flush();
  return Promise.race([result, Promise.resolve("pending")]);
}
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

test("an interrupted step is handed back at once instead of polled forever", async (t) => {
  const calls = [];
  const queue = [
    state("rendering", { interrupted: true }),
    state("rendering", { interrupted: false, busy: true }),
    state("result"),
  ];
  const api = await fixture(t, async (url) => {
    calls.push(url);
    return Response.json(queue.shift());
  });
  const seen = [];
  const first = await outcome(api.runRender("http://a.test", (v) => seen.push(v.step)));
  assert.equal(first, "fulfilled", "an interrupted render must not wait on polls");
  assert.deepEqual(calls, ["http://a.test/api/render?run=1"]);
  assert.deepEqual(seen, []);
  // After Resume the page runs it again, as a fresh start.
  const resumed = api.runRender("http://a.test", (v) => seen.push(v.step));
  await flush();
  await advance(t, 2000);
  await advance(t, 2000);
  assert.equal(await outcome(resumed), "fulfilled");
  assert.equal((await resumed).step, "result");
  assert.deepEqual(seen, ["rendering"]);
  assert.equal(calls.filter((url) => url.includes("run=1")).length, 2);
});

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
    const run = api.runRender("http://a.test", () => {});
    assert.equal(await outcome(run), "rejected", "no retry delay before failing");
    await assert.rejects(run);
    await advance(t, 60000);
    assert.equal(attempts, 1);
  });
}

test("progress text counts the time left down between polls", async (t) => {
  const api = await fixture(t, async () => Response.json({}));
  assert.equal(api.formatDuration(42.4), "42 s");
  assert.equal(api.formatDuration(185), "3 min 05 s");
  assert.equal(api.formatDuration(4020), "1 h 07 min");
  assert.equal(api.formatDuration(-3), "0 s");
  // Rounding happens before splitting, so 59.6 s never reads "60 s".
  assert.equal(api.formatDuration(59.6), "1 min 00 s");
  assert.equal(api.formatDuration(3599.6), "1 h 00 min");
  const running = { phase: "rendering", pct: 50, message: "m", eta_s: 250, elapsed_s: 10 };
  assert.equal(api.progressTiming(running), "about 4 min 10 s left");
  assert.equal(api.progressTiming(running, 10), "about 4 min 00 s left");
  // Nothing to measure (a Gemini call): the time spent counts up instead.
  const waiting = { ...running, eta_s: null };
  assert.equal(api.progressTiming(waiting, 5), "15 s so far");
  assert.equal(api.progressTiming({ phase: "x", pct: 1, message: "" }), "");
});

test("scale lines say what Gemini chose", async (t) => {
  const api = await fixture(t, async () => Response.json({}));
  assert.equal(api.scaleChoiceText(2, "soft SD source"), "Gemini chose to upscale 2×: soft SD source");
  assert.equal(api.scaleChoiceText(1), "Gemini chose to keep the size (no upscaling).");
  assert.equal(api.scaleChoiceText(null, "x"), "");
  assert.match(api.outputSizeLine(1), /keeps its size/);
  assert.equal(api.outputSizeLine(2), "The whole video is upscaled 2×.");
  // An empty reason ends the sentence instead of leaving a dangling colon.
  assert.equal(api.scaleChoiceText(2, ""), "Gemini chose to upscale 2×.");
});


test("the API origin comes from ?api, then ?live, then VITE_API_ORIGIN, else same origin", async (t) => {
  const api = await fixture(t, async () => Response.json({}));
  mockGlobal(t, "__viteEnv", { VITE_API_ORIGIN: " https://gpu.example:9000/ " });
  const from = (search) => {
    window.location.search = search;
    return api.configuredOrigin();
  };
  assert.equal(from("?api=http://box.test:8000/"), "http://box.test:8000");
  // An explicit empty ?api= means the page's own origin, even with an env origin.
  assert.equal(from("?api="), "");
  assert.equal(from("?api=http://box.test&live=1"), "http://box.test");
  assert.equal(from("?live=1"), "http://127.0.0.1:45143");
  assert.equal(from("?live=1&port=5001"), "http://127.0.0.1:5001");
  assert.equal(from(""), "https://gpu.example:9000");
  globalThis.__viteEnv = { VITE_API_ORIGIN: "   " };
  assert.equal(from(""), "");
  globalThis.__viteEnv = {};
  assert.equal(from("?live=0"), "");
});

test("media URLs from the server are prefixed with the API origin unless absolute", async (t) => {
  const api = await fixture(t, async () => Response.json({}));
  assert.equal(api.outputSrc("http://a.test", null), null);
  assert.equal(api.outputSrc("http://a.test", ""), null);
  assert.equal(api.outputSrc("http://a.test", "/output.mp4"), "http://a.test/output.mp4");
  assert.equal(api.outputSrc("", "/previews/000.mp4"), "/previews/000.mp4");
  assert.equal(api.outputSrc("http://a.test", "https://cdn.test/x.jpg"), "https://cdn.test/x.jpg");
  assert.equal(api.outputSrc("http://a.test", "http://b.test/x.jpg"), "http://b.test/x.jpg");
});

test("loadState treats an unreachable, failing or non-wizard server as no wizard", async (t) => {
  const replies = [
    () => { throw new TypeError("offline"); },
    () => Response.json({ step: "intro", token: "bad" }, { status: 500 }),
    () => Response.json({ step: "nonsense", token: "bad" }),
    () => new Response("<html>not the wizard</html>"),
    () => Response.json(null),
  ];
  const calls = [];
  const api = await fixture(t, async (url, options) => {
    calls.push([url, options]);
    const reply = replies.shift();
    return reply ? reply() : response("intro");
  });
  for (let i = 0; i < 5; i += 1) assert.equal(await api.loadState("http://a.test"), null);
  assert.equal(calls[0][0], "http://a.test/api/state");
  assert.ok(calls[0][1].signal instanceof AbortSignal, "the probe is time-bounded");
  // None of the rejected replies may leave its token behind.
  await api.postAction("http://a.test", "intro", {});
  assert.equal(calls.at(-1)[1].headers["X-Wizard-Token"], "");
});

test("postAction POSTs the fields as JSON once and reports the server's error", async (t) => {
  const calls = [];
  const replies = [
    () => response("llm_choice", { token: "t1" }),
    () => Response.json({ error: "That key was rejected." }, { status: 400 }),
    () => new Response("<html>Bad gateway</html>", { status: 502 }),
  ];
  const api = await fixture(t, async (url, options) => {
    calls.push([url, options]);
    return replies.shift()();
  });
  const view = await api.postAction("http://a.test", "intro", { choice: "gemini" });
  assert.equal(view.step, "llm_choice");
  const [url, options] = calls[0];
  assert.equal(url, "http://a.test/api/intro");
  assert.equal(options.method, "POST");
  assert.equal(options.headers["Content-Type"], "application/json");
  assert.deepEqual(JSON.parse(options.body), { choice: "gemini" });
  await assert.rejects(api.postAction("http://a.test", "setup", {}), {
    message: "That key was rejected.",
  });
  // The token from the first reply is sent with later actions.
  assert.equal(calls[1][1].headers["X-Wizard-Token"], "t1");
  // An action is not idempotent, so even a 502 is reported rather than retried.
  await assert.rejects(api.postAction("http://a.test", "setup", {}), {
    message: "Request failed (502)",
  });
  await advance(t, 60000);
  assert.equal(calls.length, 3);
});

test("408 and 5xx are retried as transient; other 4xx fail at once", async (t) => {
  for (const [status, expected] of [[408, 2], [500, 2], [404, 1], [409, 1]]) {
    let attempts = 0;
    const api = await fixture(t, async () => {
      attempts += 1;
      return attempts === 1 ? Response.json({ error: "x" }, { status }) : response("result");
    });
    const run = api.runRender("http://a.test", () => {});
    if (expected === 1) {
      assert.equal(await outcome(run), "rejected", `${status} was retried`);
      await assert.rejects(run, { message: "x" });
    } else {
      assert.equal(await outcome(run), "pending", `${status} was not retried`);
      await advance(t, 500);
      assert.equal(await outcome(run), "fulfilled", `${status}`);
      assert.equal((await run).step, "result");
    }
    assert.equal(attempts, expected, `${status}`);
    t.mock.timers.reset();
  }
});

function fakeXhr(t, behave) {
  const made = [];
  class XHR {
    headers = {};
    upload = {};
    constructor() { made.push(this); }
    open(method, url) { this.method = method; this.url = url; }
    setRequestHeader(name, value) { this.headers[name] = value; }
    send(body) { this.body = body; behave(this); }
  }
  mockGlobal(t, "XMLHttpRequest", XHR);
  return made;
}

test("upload reports whole-percent progress and adopts the token it returns", async (t) => {
  const api = await fixture(t, async (url, options) => {
    if (url.endsWith("/state")) return response("pick_file", { token: "token-A" });
    return response("segments", { token: options.headers["X-Wizard-Token"] });
  });
  await api.loadState("http://a.test");
  const file = { name: "clip.mp4" };
  fakeXhr(t, (xhr) => {
    xhr.upload.onprogress({ lengthComputable: false, loaded: 5, total: 0 });
    xhr.upload.onprogress({ lengthComputable: true, loaded: 1, total: 3 });
    xhr.upload.onprogress({ lengthComputable: true, loaded: 3, total: 3 });
    xhr.status = 200;
    xhr.responseText = JSON.stringify(state("sampling", { token: "token-A2" }));
    xhr.onload();
  });
  const seen = [];
  const view = await api.uploadVideo("http://a.test", file, (pct) => seen.push(pct));
  assert.equal(view.step, "sampling");
  assert.deepEqual(seen, [33, 100]);
  // The upload's reply rotated the token; the next request must carry it.
  const next = await api.postAction("http://a.test", "segments", {});
  assert.equal(next.token, "token-A2");
});

test("a failed upload says why, without trusting a non-wizard reply", async (t) => {
  const api = await fixture(t, async () => response("intro"));
  for (const [status, text, message] of [
    [413, JSON.stringify({ error: "The file is too large." }), "The file is too large."],
    [502, "<html>Bad gateway</html>", "The upload failed (502)."],
    [200, JSON.stringify({ step: "nonsense" }), "The upload failed (200)."],
    [200, "not JSON", "The upload failed (200)."],
  ]) {
    fakeXhr(t, (xhr) => {
      xhr.status = status;
      xhr.responseText = text;
      xhr.onload();
    });
    await assert.rejects(api.uploadVideo("http://a.test", { name: "a.mp4" }, () => {}), {
      message,
    });
  }
  const made = fakeXhr(t, (request) => request.onerror());
  const file = { name: "b.mp4" };
  await assert.rejects(api.uploadVideo("http://a.test", file, () => {}), {
    message: "The upload failed.",
  });
  assert.equal(made[0].method, "POST");
  assert.equal(made[0].body, file, "the file itself is the body");
});

test("a run step nothing works on is started again after two idle polls", async (t) => {
  const calls = [];
  // The first start lands while the earlier worker still holds the step, so
  // nothing starts; polls then show "analyzing" with no worker.
  const queue = [
    state("analyzing", { busy: true }),
    state("analyzing", { busy: false }),
    state("analyzing", { busy: false }),
    state("analyzing", { busy: true }),
    state("preview_segment", { busy: false }),
  ];
  const api = await fixture(t, async (url) => {
    calls.push(url);
    return Response.json(queue.shift());
  });
  const run = api.runAnalyze("http://a.test", () => {});
  await flush();
  await advance(t, 2000);
  assert.equal(calls.filter((url) => url.includes("run=1")).length, 1, "one idle poll is not enough");
  await advance(t, 2000);
  assert.deepEqual(calls, [
    "http://a.test/api/analyze?run=1",
    "http://a.test/api/state",
    "http://a.test/api/state",
    "http://a.test/api/analyze?run=1",
  ]);
  await advance(t, 2000);
  assert.equal((await run).step, "preview_segment");
  assert.equal(calls.filter((url) => url.includes("run=1")).length, 2);
});

test("idle polls must be in a row, and an older server without busy is never restarted", async (t) => {
  const calls = [];
  const queue = [
    state("rendering", { busy: true }),
    state("rendering", { busy: false }),
    state("rendering", { busy: true }),
    state("rendering", { busy: false }),
    state("rendering"),
    state("rendering"),
    state("result"),
  ];
  const api = await fixture(t, async (url) => {
    calls.push(url);
    return Response.json(queue.shift());
  });
  const run = api.runRender("http://a.test", () => {});
  await flush();
  for (let i = 0; i < 6; i += 1) await advance(t, 2000);
  assert.equal((await run).step, "result");
  assert.equal(calls.filter((url) => url.includes("run=1")).length, 1);
});

test("an interrupted step never sends ?run=1 again, even when idle", async (t) => {
  const calls = [];
  // Interrupted after a stop elsewhere / a restart: busy is false, but the
  // page must ask Resume or Stop instead of starting GPU or Gemini work.
  const queue = [
    state("rendering", { busy: true }),
    state("rendering", { busy: false }),
    state("rendering", { busy: false, interrupted: true }),
  ];
  const api = await fixture(t, async (url) => {
    calls.push(url);
    return Response.json(queue.shift() ?? state("rendering", { busy: false, interrupted: true }));
  });
  const run = api.runRender("http://a.test", () => {});
  await flush();
  await advance(t, 2000);
  await advance(t, 2000);
  assert.equal(await outcome(run), "fulfilled");
  assert.equal((await run).interrupted, true);
  await advance(t, 20000);
  assert.deepEqual(calls, [
    "http://a.test/api/render?run=1",
    "http://a.test/api/state",
    "http://a.test/api/state",
  ]);
  assert.equal(api.isIdleOnStep(state("rendering", { busy: false, interrupted: true }), ["rendering"]), false);
  assert.equal(api.isIdleOnStep(state("rendering", { busy: false, running_elsewhere: true }), ["rendering"]), false);
  assert.equal(api.isIdleOnStep(state("rendering", { busy: false }), ["rendering"]), true);
  assert.equal(api.isIdleOnStep(state("result", { busy: false }), ["rendering"]), false);
});

test("helpers POST the right bodies, with the job the page shows", async (t) => {
  const calls = [];
  const api = await fixture(t, async (url, options) => {
    calls.push([url.replace("http://a.test", ""), JSON.parse(options.body)]);
    return response("segments");
  });
  const running = state("rendering", {
    job_id: "job1",
    busy: true,
    run: { id: "run9", kind: "render", job_id: "job1", started_at: 1, finished_at: null, outcome: "running" },
  });
  await api.cancelRun("http://a.test", running);
  // Interrupted: no run works, and the server refuses a run_id then.
  await api.cancelRun("http://a.test", { ...running, busy: false, interrupted: true });
  await api.cancelRun("http://a.test", { ...running, run: { ...running.run, outcome: "done" } });
  await api.resumeRun("http://a.test", running);
  const preview = state("preview_segment", { job_id: "job1" });
  await api.reopenScene("http://a.test", preview, 2);
  const segments = state("segments", { job_id: "job1", segments: { segments: [], error: null, note: "", revision: 4 } });
  await api.editSegments("http://a.test", segments, { action: "move", index: 1, at: 12.34567 });
  await api.editSegments("http://a.test", segments, { action: "split", index: 0, at: 5 });
  await api.editSegments("http://a.test", segments, { action: "merge", index: 3 });
  await api.editSegments("http://a.test", segments, { action: "approve" });
  await api.editSegments("http://a.test", segments, { action: "note", note: "fewer scenes" });
  await api.openJob("http://a.test", "job2");
  assert.deepEqual(calls, [
    ["/api/cancel", { job_id: "job1", run_id: "run9" }],
    ["/api/cancel", { job_id: "job1" }],
    ["/api/cancel", { job_id: "job1" }],
    ["/api/resume", { job_id: "job1" }],
    ["/api/scene", { index: "2", job_id: "job1" }],
    ["/api/segments", { action: "move", index: "1", at: "12.346", revision: "4", job_id: "job1" }],
    ["/api/segments", { action: "split", index: "0", at: "5", revision: "4", job_id: "job1" }],
    ["/api/segments", { action: "merge", index: "3", revision: "4", job_id: "job1" }],
    ["/api/segments", { action: "approve", revision: "4", job_id: "job1" }],
    ["/api/segments", { action: "note", note: "fewer scenes", revision: "4", job_id: "job1" }],
    // Any window may switch jobs: only the job to open is sent.
    ["/api/jobs/open", { job: "job2" }],
  ]);
  // withJob keeps a job_id the caller set and adds none without a job.
  assert.deepEqual(api.withJob({ job_id: "job1" }, { job_id: "other", a: "1" }), { job_id: "other", a: "1" });
  assert.deepEqual(api.withJob({ job_id: null }, { a: "1" }), { a: "1" });
});

test("a 409 from a stale tab is reported, not retried", async (t) => {
  let count = 0;
  const api = await fixture(t, async () => {
    count += 1;
    return Response.json({ error: "this job is no longer open in this window" }, { status: 409 });
  });
  await assert.rejects(api.resumeRun("http://a.test", state("rendering", { job_id: "old" })), {
    message: "this job is no longer open in this window",
  });
  await advance(t, 60000);
  assert.equal(count, 1);
});

test("loadJobs lists the jobs and says why when it cannot", async (t) => {
  const replies = [
    () => Response.json({ jobs: [{ job_id: "a1", title: "Holiday", interrupted: true }, { nope: 1 }] }),
    () => Response.json({ error: "the jobs folder cannot be read" }, { status: 400 }),
    () => new Response("<html>", { status: 502 }),
    () => Response.json({ step: "intro" }),
  ];
  const urls = [];
  const api = await fixture(t, async (url) => {
    urls.push(url);
    return replies.shift()();
  });
  const jobs = await api.loadJobs("http://a.test");
  assert.deepEqual(jobs, [{ job_id: "a1", title: "Holiday", interrupted: true }]);
  assert.equal(urls[0], "http://a.test/api/jobs");
  await assert.rejects(api.loadJobs("http://a.test"), { message: "the jobs folder cannot be read" });
  await assert.rejects(api.loadJobs("http://a.test"), { message: "Could not list the jobs (502)." });
  await assert.rejects(api.loadJobs("http://a.test"), { message: "The server did not return a job list." });
});

test("a view with every new key is still a wizard view, and so is one without them", async (t) => {
  const full = state("preview_segment", {
    run: null, cancellable: false, cancel_requested: false, stopped: null, interrupted: false,
    running_elsewhere: false, versions: [], job: null,
    segment_preview: { index: 0, count: 2, scale: 2, error: null, shot: null, next: { index: 1, ready: false, running: true } },
  });
  const replies = [full, state("intro")];
  const api = await fixture(t, async () => Response.json(replies.shift()));
  assert.deepEqual(await api.loadState("http://a.test"), full);
  assert.equal((await api.loadState("http://a.test")).step, "intro");
});
