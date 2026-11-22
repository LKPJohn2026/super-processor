// The Welcome & setup screen (mockup 01-welcome): one screen in place of
// the intro, the LLM choice and the Gemini key steps. Pure functions, so
// the node tests can check them; types only from ./api.
import type { WizardView } from "./api";

/** The server's steps the Welcome & setup screen stands for. */
export function isWelcomeStep(step: WizardView["step"]): boolean {
  return step === "intro" || step === "llm_choice" || step === "local_llm_stub" || step === "setup";
}

export type WelcomeStart =
  | { kind: "send"; action: "setup"; fields: Record<string, string> }
  | { kind: "needs_key" };

/**
 * What "Choose a video" sends. The server's /setup moves on to the file
 * pick from any of the welcome steps: with a typed key it saves that key
 * first, with a key already found it keeps it (`skip`). Without either
 * there is nothing to send: the page asks for a key.
 */
export function welcomeStart(hasKey: boolean, typedKey: string): WelcomeStart {
  const key = typedKey.trim();
  if (key) return { kind: "send", action: "setup", fields: { api_key: key } };
  if (hasKey) return { kind: "send", action: "setup", fields: { skip: "1" } };
  return { kind: "needs_key" };
}

export const NEEDS_KEY_TEXT = "Paste your Gemini key first. It is free to make one in Google AI Studio.";

export type KeyCheck = { ok: boolean; label: string; detail: string };

/**
 * The Gemini row of "What you need". The server says only whether a key
 * is found (it may come from the environment as well as the saved file),
 * so the words claim no more than that.
 */
export function keyCheck(hasKey: boolean): KeyCheck {
  return hasKey
    ? { ok: true, label: "Google Gemini key", detail: "Found on this computer." }
    : { ok: false, label: "Google Gemini key", detail: "Not set up yet. Paste a key to start." };
}

/**
 * Whether the key field shows the server's refusal: the error came back on
 * the setup step (an empty key, a key that could not be saved). Other
 * errors show in a banner.
 */
export function keyError(view: Pick<WizardView, "step" | "error">): string | null {
  return view.step === "setup" && view.error ? view.error : null;
}
