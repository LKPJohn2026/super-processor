// Step 4 · Render (mockups 06, 07 and S5): the size choice, the scene
// summary, the render's time left and stages. Pure functions of the view,
// so the node tests can check them; types only from ./api.
import type { LookShot, Looks, Progress } from "./api";

/** The still saved beside each preview clip (same name, .jpg). */
export function posterUrl(videoUrl: string | null | undefined): string | null {
  return videoUrl ? videoUrl.replace(/\.mp4(\?|$)/, ".jpg$1") : null;
}

export type SizeChoice = {
  scale: number;
  title: string;
  // "2× · 1280×960 · AI detail where you approved it"
  spec: string;
  // Gemini chose this size, and said why.
  recommended: boolean;
  image: { url: string; caption: string } | null;
};

/** The shot whose stills stand for the whole video: the middle previewed one. */
function sampleShot(shots: readonly LookShot[]): LookShot | null {
  const previewed = shots.filter((shot) => shot.after_url && shot.before_url);
  return previewed.length ? previewed[Math.floor((previewed.length - 1) / 2)] : null;
}

/**
 * The output sizes to choose from (06), biggest first: "Sharper and
 * bigger" for each upscale, "Same size, cleaned up" for 1. A card's image
 * is a real preview still: the restored one only where it shows what the
 * card gives (previews are made at the plan's scale), else the original.
 */
export function sizeChoices(looks: Looks): SizeChoice[] {
  const scales = [...new Set(looks.scales ?? [])].sort((a, b) => b - a);
  const [width, height] = looks.source_size ?? [0, 0];
  const upscales = scales.filter((scale) => scale > 1).length;
  const shot = sampleShot(looks.shots);
  const scene = shot ? `Scene ${shot.index + 1}` : "";
  return scales.map((scale) => {
    const size = width > 0 && height > 0 ? `${width * scale}×${height * scale}` : "";
    const restoredShown = shot !== null && looks.scale === scale;
    const still = shot ? posterUrl(restoredShown ? shot.after_url : shot.before_url) : null;
    const image = still
      ? {
          url: still,
          caption: restoredShown
            ? `${scene}, as its preview restored it`
            : `${scene}, the original`,
        }
      : null;
    const recommended = Boolean(looks.scale_reason) && looks.scale === scale;
    if (scale === 1) {
      return {
        scale,
        title: "Same size, cleaned up",
        spec: [size, "clean-up only, no AI detail"].filter(Boolean).join(" · "),
        recommended,
        image,
      };
    }
    return {
      scale,
      title: upscales > 1 ? `Sharper and bigger (${scale}×)` : "Sharper and bigger",
      spec: [`${scale}×`, size, "AI detail where you approved it"].filter(Boolean).join(" · "),
      recommended,
      image,
    };
  });
}

/** The size chosen first: the plan's when it is offered, else the biggest. */
export function initialScale(looks: Looks): number | null {
  const scales = [...(looks.scales ?? [])].sort((a, b) => b - a);
  if (looks.scale !== null && scales.includes(looks.scale)) return looks.scale;
  return scales[0] ?? null;
}

/** The heading of "Ready to render" (06) when nothing went wrong. */
export function readyTitle(count: number): string {
  if (count === 1) return "The scene is approved — choose the size";
  if (count > 1) return `All ${count} scenes approved — choose the size`;
  return "Every scene is approved — choose the size";
}

export type DetailBadge = { text: string; tone: "info" | "neutral" };

/**
 * A scene's AI detail at the chosen size, from its settings: none at the
 * same size (SPAN does not run at 1×), "Just resize" where it was turned
 * off, else Gemini's strength as a word (`word`: previewView's strengthWord).
 */
export function detailBadge(
  shot: Pick<LookShot, "use_upscale">,
  word: string,
  scale: number | null,
): DetailBadge {
  if (scale === 1) return { text: "Clean-up only", tone: "neutral" };
  if (shot.use_upscale === false) return { text: "Just resize", tone: "neutral" };
  if (word === "None") return { text: "No AI detail", tone: "neutral" };
  return { text: `AI detail: ${word}`, tone: "info" };
}

/** True when "Render without AI detail" (S5) can be offered: 1× is offered and not chosen. */
export function offersWithoutDetail(looks: Looks, chosen: number | null): boolean {
  return (looks.scales ?? []).includes(1) && chosen !== 1;
}

/** "About 3 min left", from the server's estimate only; null without one. */
export function timeLeft(etaS: number | null | undefined, sinceS = 0): string | null {
  if (typeof etaS !== "number" || !Number.isFinite(etaS)) return null;
  const left = Math.max(0, etaS - sinceS);
  if (left < 60) return "Less than a minute left";
  const minutes = Math.round(left / 60);
  if (minutes < 60) return `About ${minutes} min left`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return `About ${hours} h${rest ? ` ${String(rest).padStart(2, "0")} min` : ""} left`;
}

/** "2 min done", "40 s done": the time the render has run. */
export function timeDone(elapsedS: number | null | undefined, sinceS = 0): string {
  if (typeof elapsedS !== "number" || !Number.isFinite(elapsedS)) return "";
  const total = Math.max(0, Math.round(elapsedS + sinceS));
  if (total < 60) return `${total} s done`;
  const minutes = Math.floor(total / 60);
  if (minutes < 60) return `${minutes} min done`;
  return `${Math.floor(minutes / 60)} h ${String(minutes % 60).padStart(2, "0")} min done`;
}

export type RenderStage = { key: "upscale" | "seams" | "keep"; label: string; state: "done" | "running" | "waiting"; meta: string };

// The server's render step: 5–90% restores and upscales, 90% checks the
// seams, 95% keeps a copy of the new version (only where it must copy).
const UPSCALE_FROM = 5;
const SEAMS_AT = 90;
const KEEP_AT = 95;

/**
 * The render's stages as the server reports them (07). The progress names
 * no scene, so the list is the render's own stages, not one per scene.
 */
export function renderStages(progress: Pick<Progress, "pct" | "message"> | null | undefined): RenderStage[] {
  const pct = progress ? progress.pct : 0;
  const message = progress?.message ?? "";
  const at = message.startsWith("Checking for seams")
    ? 1
    : pct >= KEEP_AT
      ? 2
      : pct >= SEAMS_AT
        ? 1
        : 0;
  const state = (index: number): RenderStage["state"] => (index < at ? "done" : index === at ? "running" : "waiting");
  return [
    {
      key: "upscale",
      label: "Restore and upscale every scene",
      state: state(0),
      // The line above the list says how far, in the server's words.
      meta: at === 0 && pct <= UPSCALE_FROM ? "starting" : "",
    },
    { key: "seams", label: "Check the joins between scenes", state: state(1), meta: "" },
    { key: "keep", label: "Save it as a new version", state: state(2), meta: "" },
  ];
}

/**
 * S5's banner title, from the server's error: "The render stopped: the
 * graphics card ran out of memory", without saying "failed" twice.
 */
export function failureTitle(error: string): string {
  const reason = error
    .trim()
    .replace(/^the (?:render|upscale) failed(?:, so there is no result yet)?:\s*/i, "")
    .replace(/\.$/, "");
  return reason ? `The render stopped: ${reason}` : "The render stopped";
}
