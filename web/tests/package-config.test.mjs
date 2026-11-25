import assert from "node:assert/strict";
import { mkdir, mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { loadConfigFromFile } from "vite";
import { bundleProblems } from "../scripts/check-bundle.mjs";

const web = fileURLToPath(new URL("..", import.meta.url));

async function configFor(mode, command = "build") {
  const loaded = await loadConfigFromFile({ command, mode, isPreview: false }, join(web, "vite.config.ts"), web, "silent");
  assert.ok(loaded);
  return loaded.config;
}

test("the package build writes the served app into the Python package", async () => {
  const config = await configFor("package");
  assert.equal(config.base, "/");
  assert.equal(resolve(config.build.outDir), resolve(web, "../src/super_processor/web_dist"));
  assert.equal(config.build.emptyOutDir, true);
  // No data: fonts, so the wizard's CSP needs only font-src 'self'.
  assert.equal(config.build.assetsInlineLimit, 0);
});

test("the Pages build and the dev server keep their own settings", async () => {
  const pages = await configFor("production");
  assert.equal(pages.base, "/super-processor/");
  assert.equal(pages.build?.outDir, undefined);
  const dev = await configFor("development", "serve");
  assert.equal(dev.base, "/");
  assert.ok(dev.server.proxy["/api"]);
});

async function sources(dir) {
  const found = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) found.push(...(await sources(path)));
    else if (/\.(ts|tsx)$/.test(entry.name)) found.push(path);
  }
  return found;
}

// Model and server text reaches the page only as React text, which React
// escapes. The one raw-HTML use draws the app's own icon constants.
test("no source writes server text as HTML", async () => {
  const raw = [];
  for (const file of await sources(join(web, "src"))) {
    const text = await readFile(file, "utf8");
    for (const match of text.matchAll(/dangerouslySetInnerHTML=\{([^\n]*)\}|\.(?:inner|outer)HTML\s*=|insertAdjacentHTML|document\.write/g)) {
      raw.push(`${file.slice(web.length).replace(/\\/g, "/")}: ${match[0]}`);
    }
  }
  assert.deepEqual(raw, ["src/ui/Icon.tsx: dangerouslySetInnerHTML={{ __html: ICON_PATHS[name] }}"]);
  const icons = await readFile(join(web, "src/ui/iconPaths.ts"), "utf8");
  // Plain string constants: no template holes, imports or function calls.
  assert.doesNotMatch(icons, /\$\{|\bimport\b|\bfetch\b|\bwindow\b/);
});

async function fakeBundle(t, { html, files }) {
  const root = await mkdtemp(join(tmpdir(), "super-processor-bundle-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, "assets"));
  await writeFile(join(root, "index.html"), html);
  for (const [name, text] of Object.entries(files)) await writeFile(join(root, "assets", name), text);
  return root;
}

const goodHtml =
  '<!doctype html><html><head><script type="module" crossorigin src="/assets/index-a1.js"></script>' +
  '<link rel="stylesheet" crossorigin href="/assets/index-b2.css"></head><body><div id="root"></div></body></html>';

test("check:package accepts a bundle that fits the wizard's CSP", async (t) => {
  const root = await fakeBundle(t, {
    html: goodHtml,
    files: { "index-a1.js": 'const x={data:null};', "index-b2.css": "@font-face{src:url(/assets/f-c3.woff2)}", "f-c3.woff2": "" },
  });
  assert.deepEqual(bundleProblems(root), []);
});

test("check:package refuses inline code, data: URLs, other sites and missing files", async (t) => {
  const root = await fakeBundle(t, {
    html:
      '<html><head><script>alert(1)</script><style>p{}</style><script src="https://cdn.example/x.js"></script>' +
      '<link rel="stylesheet" href="/assets/gone.css"></head><body onload="x()"><p style="color:red"></p></body></html>',
    files: { "a.css": "@font-face{src:url(data:font/woff2;base64,AAAA)}", "b.js": 'img.src="data:image/png;base64,AAAA"', "c.map": "" },
  });
  const problems = bundleProblems(root).join("\n");
  for (const expected of [
    "inline <script>",
    "<style> element",
    "style= attribute",
    "on…= handler",
    "loads from another site",
    "loads no /assets/*.js",
    "missing /assets/gone.css",
    "assets/a.css has a data: URL",
    "assets/b.js has a data: URL",
    "assets/c.map: the wizard has no type for it",
  ]) {
    assert.match(problems, new RegExp(expected.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&")));
  }
});

test("check:package names a missing build", () => {
  assert.match(bundleProblems(join(tmpdir(), "super-processor-no-such-bundle"))[0], /run npm run build:package/);
});
