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

export default defineConfig(({ command, isPreview }) => ({
  // `vite preview` also reports command "serve", but it serves the production
  // build, whose asset URLs carry the Pages base.
  base: command === "serve" && !isPreview ? "/" : "/super-processor/",
  plugins: [react()],
  server: { proxy, headers: frameGuard },
  preview: { proxy, headers: frameGuard },
}));
