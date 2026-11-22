import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { FRAMED_NOTICE, isFramed } from "./frameGuard";
// Self-hosted fonts (no Google Fonts requests): the design's three faces.
import "@fontsource/source-sans-3/400.css";
import "@fontsource/source-sans-3/600.css";
import "@fontsource/source-sans-3/700.css";
import "@fontsource/source-serif-4/600.css";
import "@fontsource/jetbrains-mono/400.css";
import "./styles.css";

const root = document.getElementById("root");
if (!root) {
  throw new Error("missing root");
}

if (isFramed(window)) {
  // Not mounted, so nothing reads the address, its token or the wizard.
  root.textContent = FRAMED_NOTICE;
} else {
  createRoot(root).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}
