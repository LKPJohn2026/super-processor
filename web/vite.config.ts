import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const proxyTarget = process.env.VITE_PROXY_TARGET || "http://127.0.0.1:45143";
const proxy = Object.fromEntries(
  ["/api", "/output.mp4", "/segment_stills", "/shot_stills", "/previews", "/jobs/"].map(
    (prefix) => [prefix, { target: proxyTarget, changeOrigin: true }],
  ),
);
// Another site must not frame the shell and steer its clicks (the static
// Pages build, which cannot send headers, checks for a frame in main.tsx).
export const frameGuard = {
  "X-Frame-Options": "DENY",
  "Content-Security-Policy": "frame-ancestors 'none'",
};

// `npm run build:package` writes the app that `super-processor review`
// serves at its own address. That folder is not in git; CI and the release
// build it and put it in every wheel and the sdist.
export const PACKAGE_OUT_DIR = fileURLToPath(new URL("../src/super_processor/web_dist", import.meta.url));

export default defineConfig(({ command, isPreview, mode }) =>
  mode === "package"
    ? {
        base: "/",
        plugins: [react()],
        build: {
          outDir: PACKAGE_OUT_DIR,
          // The folder is outside web/, so Vite only clears it when asked.
          emptyOutDir: true,
          // Fonts as files, not data: URLs, so the wizard's CSP can stay
          // font-src 'self'.
          assetsInlineLimit: 0,
        },
      }
    : {
        // `vite preview` also reports command "serve", but it serves the
        // production build, whose asset URLs carry the Pages base.
        base: command === "serve" && !isPreview ? "/" : "/super-processor/",
        plugins: [react()],
        server: { proxy, headers: frameGuard },
        preview: { proxy, headers: frameGuard },
      },
);
