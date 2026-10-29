import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { createServer as httpServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createServer, preview } from "vite";

// Exercise the real config and actual HTTP forwarding, including range headers.
test("dev and preview proxy API, output, stills and previews to VITE_PROXY_TARGET", async (t) => {
  const target = httpServer((request, response) => {
    const isVideo = request.url.endsWith(".mp4");
    response.writeHead(isVideo ? 206 : 200, {
      "Content-Type": isVideo ? "video/mp4" : request.url.endsWith(".jpg") ? "image/jpeg" : "application/json",
      ...(isVideo ? { "Content-Range": "bytes 0-3/4", "Accept-Ranges": "bytes" } : {}),
    });
    response.end(JSON.stringify({ path: request.url, range: request.headers.range, host: request.headers.host }));
  });
  await new Promise((resolve) => target.listen(0, "127.0.0.1", resolve));
  t.after(() => new Promise((resolve) => target.close(resolve)));
  const targetOrigin = `http://127.0.0.1:${target.address().port}`;
  const previous = process.env.VITE_PROXY_TARGET;
  process.env.VITE_PROXY_TARGET = targetOrigin;
  t.after(() => {
    if (previous === undefined) delete process.env.VITE_PROXY_TARGET;
    else process.env.VITE_PROXY_TARGET = previous;
  });
  // Preview needs an output directory, not a full build for proxy checks.
  const outDir = await mkdtemp(join(tmpdir(), "super-processor-proxy-"));
  await writeFile(join(outDir, "index.html"), "<html>proxy regression</html>");
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
    const http = vite.httpServer;
    try {
      const origin = `http://127.0.0.1:${http.address().port}`;
      for (const path of ["/api/state", "/output.mp4", "/segment_stills/000.jpg", "/shot_stills/000.jpg", "/previews/000.mp4"]) {
        const result = await fetch(`${origin}${path}`, { headers: { Range: "bytes=0-3" } });
        assert.equal(result.status, path.endsWith(".mp4") ? 206 : 200, `${mode}: ${path}`);
        assert.match(result.headers.get("Content-Type"), path.endsWith(".mp4") ? /video\/mp4/ : path.endsWith(".jpg") ? /image\/jpeg/ : /application\/json/);
        assert.deepEqual(await result.json(), {
          path, range: "bytes=0-3", host: new URL(targetOrigin).host,
        });
        if (path.endsWith(".mp4")) assert.equal(result.headers.get("Content-Range"), "bytes 0-3/4");
      }
    } finally {
      await vite.close();
    }
  }
});
