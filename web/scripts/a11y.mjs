import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import AxeBuilder from "@axe-core/playwright";
import { chromium } from "playwright";

const root = join(fileURLToPath(new URL("..", import.meta.url)), "dist");
const base = "/super-processor";
const types = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".json": "application/json",
};

function fileFor(urlPath) {
  const stripped = urlPath.startsWith(base) ? urlPath.slice(base.length) : urlPath;
  const relative = stripped.replace(/^\/+/, "") || "index.html";
  const candidate = normalize(join(root, relative));
  if (!candidate.startsWith(root)) return null;
  return candidate;
}

const server = createServer(async (request, response) => {
  const url = new URL(request.url ?? "/", "http://127.0.0.1");
  let path = fileFor(url.pathname);
  try {
    if (!path) throw new Error("bad path");
    let body;
    try {
      body = await readFile(path);
    } catch {
      path = join(root, "index.html");
      body = await readFile(path);
    }
    response.writeHead(200, {
      "Content-Type": types[extname(path)] ?? "application/octet-stream",
    });
    response.end(body);
  } catch {
    response.writeHead(404);
    response.end("not found");
  }
});

await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const address = server.address();
const port = typeof address === "object" && address ? address.port : 0;
const pageUrl = `http://127.0.0.1:${port}${base}/`;

const browser = await chromium.launch();
const context = await browser.newContext();
const page = await context.newPage();
const failures = [];
try {
  await page.goto(pageUrl, { waitUntil: "networkidle" });
  await page.getByRole("heading", { level: 1, name: /Restore the video/ }).waitFor();
  const offline = page.getByRole("status");
  const statusText = await offline.innerText();
  if (!/No live wizard is connected/i.test(statusText)) {
    failures.push(`unexpected status: ${statusText}`);
  }
  const results = await new AxeBuilder({ page }).analyze();
  for (const violation of results.violations) {
    failures.push(
      `${violation.id}: ${violation.help} (${violation.nodes.length} nodes)`,
    );
  }
  await page.locator("body").focus();
  const focused = [];
  for (let index = 0; index < 8; index += 1) {
    await page.keyboard.press("Tab");
    const name = await page.evaluate(() => {
      const el = document.activeElement;
      if (!el || el === document.body) return "";
      return el.innerText || el.getAttribute("aria-label") || el.id || el.tagName;
    });
    focused.push(name.replace(/\s+/g, " ").trim());
  }
  if (!focused.some((name) => /Connect to local wizard|Local wizard URL|Skip to content/i.test(name))) {
    failures.push(`tab order did not reach a control: ${focused.join(" | ")}`);
  }
  await page.getByRole("button", { name: "Next step" }).click();
  await page.getByRole("heading", { level: 2, name: "LLM choice" }).waitFor();
} finally {
  await browser.close();
  server.close();
}

if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}
console.log(`axe and keyboard check passed: ${pageUrl}`);
