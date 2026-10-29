import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const proxyTarget = process.env.VITE_PROXY_TARGET || "http://127.0.0.1:45143";
const proxy = Object.fromEntries(
  ["/api", "/output.mp4", "/segment_stills", "/shot_stills", "/previews"].map(
    (prefix) => [prefix, { target: proxyTarget, changeOrigin: true }],
  ),
);

export default defineConfig(({ command }) => ({
  base: command === "serve" ? "/" : "/super-processor/",
  plugins: [react()],
  server: { proxy },
  preview: { proxy },
}));
