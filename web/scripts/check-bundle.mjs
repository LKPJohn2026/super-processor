// Checks the app `npm run build:package` wrote into the Python package, so it
// loads under the wizard's Content-Security-Policy (see review.py): files
// only from its own address, no inline script or style, no data: URLs.
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

export function bundleProblems(root) {
  const problems = [];
  const indexPath = join(root, "index.html");
  if (!existsSync(indexPath)) return [`${indexPath} is missing; run npm run build:package`];
  const html = readFileSync(indexPath, "utf8");
  for (const tag of html.matchAll(/<script\b([^>]*)>/gi)) {
    if (!/\bsrc\s*=/.test(tag[1])) problems.push("index.html has an inline <script>");
  }
  if (/<style\b/i.test(html)) problems.push("index.html has a <style> element");
  if (/\sstyle\s*=/i.test(html)) problems.push("index.html has a style= attribute");
  if (/\son[a-z]+\s*=/i.test(html)) problems.push("index.html has an on…= handler");
  if (/(?:src|href)\s*=\s*["']?(?:https?:)?\/\//i.test(html)) problems.push("index.html loads from another site");
  const referenced = [...html.matchAll(/(?:src|href)\s*=\s*["']([^"']+)["']/gi)].map((m) => m[1]);
  if (!referenced.some((ref) => /^\/assets\/[^/]+\.js$/.test(ref))) problems.push("index.html loads no /assets/*.js");
  for (const ref of referenced) {
    if (!ref.startsWith("/assets/")) {
      problems.push(`index.html references ${ref}, not a file under /assets/`);
      continue;
    }
    const file = join(root, ref.slice(1));
    if (!existsSync(file) || !statSync(file).isFile()) problems.push(`index.html references missing ${ref}`);
  }
  const assets = join(root, "assets");
  const names = existsSync(assets) ? readdirSync(assets) : [];
  for (const name of names) {
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(name)) problems.push(`assets/${name}: the wizard will not serve this name`);
    if (!/\.(js|css|woff2?|svg|png|ico)$/.test(name)) problems.push(`assets/${name}: the wizard has no type for it`);
    if (name.endsWith(".css")) {
      const css = readFileSync(join(assets, name), "utf8");
      if (/url\(\s*["']?data:/i.test(css)) problems.push(`assets/${name} has a data: URL`);
      if (/url\(\s*["']?(?:https?:)?\/\//i.test(css) || /@import/i.test(css)) {
        problems.push(`assets/${name} loads from another site`);
      }
    }
    if (name.endsWith(".js")) {
      const js = readFileSync(join(assets, name), "utf8");
      if (/["'`]data:[a-z]+\/[a-z0-9.+-]+[;,]/i.test(js)) problems.push(`assets/${name} has a data: URL`);
    }
  }
  return problems;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const root = process.argv[2] ?? fileURLToPath(new URL("../../src/super_processor/web_dist", import.meta.url));
  const problems = bundleProblems(root);
  if (problems.length) {
    for (const problem of problems) console.error(`check:package: ${problem}`);
    process.exit(1);
  }
  console.log(`check:package: ${root} is fine`);
}
