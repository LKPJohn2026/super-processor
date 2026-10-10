import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const proxyTarget = process.env.VITE_PROXY_TARGET || "http://127.0.0.1:45143";

export default defineConfig(({ command }) => ({
  base: command === "serve" ? "/" : "/super-processor/",
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: proxyTarget, changeOrigin: true },
      "/output.mp4": { target: proxyTarget, changeOrigin: true },
    },
  },
  preview: {
    proxy: {
      "/api": { target: proxyTarget, changeOrigin: true },
      "/output.mp4": { target: proxyTarget, changeOrigin: true },
    },
  },
}));
