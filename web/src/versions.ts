// The kept results of a job (the Result and Done steps) and the note about
// a scene preview made in the background. Pure functions of the view, so the
// node tests can check the wording.
import type { OutputVersion, SegmentPreview } from "./api";

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** "Version 2". */
export function versionLabel(version: Pick<OutputVersion, "number">): string {
  return `Version ${version.number}`;
}

/** When a version was kept, from the server's ISO time; `nowMs` is Date.now(). */
export function keptText(createdAt: string | null, nowMs: number): string {
  if (!createdAt) return "";
  const at = Date.parse(createdAt);
  if (!Number.isFinite(at)) return "";
  const seconds = Math.max(0, (nowMs - at) / 1000);
  if (seconds < 60) return "Kept just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `Kept ${minutes} min ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `Kept ${hours} h ago`;
  const then = new Date(at);
  const now = new Date(nowMs);
  const day = `${then.getDate()} ${MONTHS[then.getMonth()]}`;
  const time = `${String(then.getHours()).padStart(2, "0")}:${String(then.getMinutes()).padStart(2, "0")}`;
  return then.getFullYear() === now.getFullYear()
    ? `Kept ${day}, ${time}`
    : `Kept ${day} ${then.getFullYear()}, ${time}`;
}

/** "820 MB", "1.1 GB"; empty for an unknown size. */
export function sizeText(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "";
  const mb = bytes / 1_000_000;
  if (mb < 1) return "under 1 MB";
  if (mb < 1000) return `${Math.round(mb)} MB`;
  return `${(mb / 1000).toFixed(1)} GB`;
}

/** Where a version came from, when it was not a plain render. */
export function originText(kind: OutputVersion["kind"]): string {
  if (kind === "earlier") return "Kept from before";
  if (kind === "recovered") return "Found in the job folder";
  return "";
}

/** The small line under a version: when, scale, size. */
export function versionMeta(version: OutputVersion, nowMs: number): string {
  return [
    keptText(version.created_at, nowMs),
    originText(version.kind),
    typeof version.scale === "number" ? `${version.scale}×` : "",
    sizeText(version.size),
  ]
    .filter(Boolean)
    .join(" · ");
}

/** What asked for this version, in the editor's words. */
export function versionNote(version: Pick<OutputVersion, "note" | "number" | "kind">): string {
  const note = (version.note ?? "").trim();
  if (note) return `“${note}”`;
  return version.number === 1 && version.kind === "render" ? "First render" : "";
}

/** The file name a download is saved as: "Cat – version 2.mp4". */
export function downloadName(title: string | null | undefined, version: Pick<OutputVersion, "number">): string {
  // Characters most file systems refuse, and leading dots, are dropped.
  // A title that is the picked file's name loses its video extension.
  const base = (title ?? "")
    .replace(/\.(mp4|m4v|mov|mkv|avi|webm|mpe?g|wmv|ts)$/i, "")
    .replace(/[\\/:*?"<>|\u0000-\u001f]/g, " ")
    .replace(/^\.+/, "")
    .trim();
  const name = (base || "Restored video").slice(0, 80).trim();
  return `${name} – version ${version.number}.mp4`;
}

/** Newest first, as the list shows them; the server sends oldest first. */
export function versionsNewestFirst(versions: OutputVersion[] | null | undefined): OutputVersion[] {
  return [...(versions ?? [])].sort((a, b) => b.number - a.number);
}

/** The heading over the list: "3 versions kept · you are watching version 3". */
export function versionsSummary(versions: OutputVersion[] | null | undefined): string {
  const list = versions ?? [];
  if (!list.length) return "";
  const current = list.find((version) => version.current);
  const count = list.length === 1 ? "1 version kept" : `${list.length} versions kept`;
  return current ? `${count} · the result above is version ${current.number}` : count;
}

/**
 * The small note while the next scene's preview is made in the background
 * (`review --prefetch` only, so usually absent). Null when there is none, or
 * it is for another scene, or it ended without a preview.
 */
export function prefetchNote(preview: Pick<SegmentPreview, "index" | "count" | "next">): string | null {
  const next = preview.next;
  if (!next || next.index !== preview.index + 1 || next.index >= preview.count) return null;
  const scene = `scene ${next.index + 1}`;
  if (next.running) return `Getting ${scene} ready in the background, so it opens sooner.`;
  if (next.ready) return `The preview of ${scene} is ready: it opens as soon as you approve this one.`;
  return null;
}
