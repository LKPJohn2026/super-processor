import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import ts from "typescript";
import { createServer, preview } from "vite";

const source = await readFile(new URL("../src/frameGuard.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
});
const guard = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);

test("isFramed: a page in another page's frame does not start", () => {
  const self = {};
  assert.equal(guard.isFramed({ top: self, self }), false);
  assert.equal(guard.isFramed({ top: {}, self }), true);
  // A cross-origin parent that throws on access counts as framed.
  const hostile = { self, get top() { throw new Error("SecurityError"); } };
  assert.equal(guard.isFramed(hostile), true);
});

test("main.tsx checks for a frame before mounting the app", async () => {
  const main = await readFile(new URL("../src/main.tsx", import.meta.url), "utf8");
  const check = main.indexOf("isFramed(window)");
  assert.ok(check > 0, "main.tsx no longer checks for a frame");
  assert.ok(check < main.indexOf("createRoot(root)"), "the frame check must come before mounting");
});

test("dev and preview servers refuse to be framed", async (t) => {
  const outDir = await mkdtemp(join(tmpdir(), "super-processor-frame-"));
  await writeFile(join(outDir, "index.html"), "<html>frame regression</html>");
  t.after(() => rm(outDir, { recursive: true, force: true }));
  for (const mode of ["dev", "preview"]) {
    const options = {
      configFile: "vite.config.ts",
      logLevel: "silent",
      server: { host: "127.0.0.1", port: 0 },
      preview: { host: "127.0.0.1", port: 0, open: false },
      build: { outDir },
    };
    const vite = mode === "dev" ? await createServer(options) : await preview(options);
    if (mode === "dev") await vite.listen();
    try {
      const origin = `http://127.0.0.1:${vite.httpServer.address().port}`;
      const page = await fetch(`${origin}${mode === "dev" ? "/" : "/super-processor/"}`);
      assert.equal(page.status, 200, mode);
      assert.equal(page.headers.get("X-Frame-Options"), "DENY", mode);
      assert.match(page.headers.get("Content-Security-Policy") ?? "", /frame-ancestors 'none'/, mode);
      await page.arrayBuffer();
    } finally {
      await vite.close();
    }
  }
});
