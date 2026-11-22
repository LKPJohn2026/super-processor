// Choose a video (mockup 02) and "File can't be used" (S4). Pure functions,
// so the node tests can check them; types only, nothing from ./api.

/** The wizard's input limits (wizard.py MAX_INPUT_*, segments.py MAX_DURATION_S). */
export const MAX_LONG_EDGE = 1920;
export const MAX_SHORT_EDGE = 1080;
export const MAX_DURATION_S = 30 * 60;

export type CheckKey = "file" | "type" | "picture" | "size" | "length" | "timing";

/** One check row (gen.py `checkrow()`): what was checked, passed or not, and what was found. */
export type FileCheck = { key: CheckKey; label: string; ok: boolean; value: string };

/** A refused pick, as S4 shows it. `name` is the file's name when known. */
export type Refusal = { name: string | null; checks: FileCheck[] };

const SIZE_OK: FileCheck = { key: "size", label: "Size", ok: true, value: "Fits within 1080p" };
const LENGTH_OK: FileCheck = { key: "length", label: "Length", ok: true, value: "Up to 30 min" };
const TRIM = "Trim it into parts of 30 min or less and pick one.";

function baseName(path: string): string {
  const parts = path.trim().replace(/\\/g, "/").split("/");
  return parts[parts.length - 1] || path.trim();
}

/** `22:01` or `1:05:09` for a length in seconds. */
export function clockText(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (value: number) => String(value).padStart(2, "0");
  return h ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

/** `94 MB`, `1.4 GB`, for a file's size in bytes. */
export function fileSizeText(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "";
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${Math.round(bytes / 1e6)} MB`;
  if (bytes >= 1e3) return `${Math.round(bytes / 1e3)} kB`;
  return `${bytes} B`;
}

/**
 * The pick's refusal as check rows, from the server's error (wizard.py
 * `_pick_file` and `input_limit_error`, which check the picture, then the
 * size, then the length, then the timing: a later check failing means the
 * earlier ones passed). Null for an error that is not about the file.
 * `tried` is the name or path the page sent, when it knows it.
 */
export function pickRefusal(error: string | null | undefined, tried?: string | null): Refusal | null {
  if (!error) return null;
  const text = error.trim();
  const name = tried ? baseName(tried) : null;
  let match = /^file not found: (.+)$/.exec(text);
  if (match) {
    return {
      name: baseName(match[1]),
      checks: [{ key: "file", label: "File", ok: false, value: `Not found at ${match[1]}` }],
    };
  }
  match = /^only video files can be uploaded \((.+)\)$/.exec(text);
  if (match) {
    return {
      name,
      checks: [{ key: "type", label: "File type", ok: false, value: `Not a video file. These work: ${match[1]}` }],
    };
  }
  if (text === "this file has no video stream") {
    return { name, checks: [{ key: "picture", label: "Picture", ok: false, value: "This file has no video in it" }] };
  }
  match = /^(.+) could not be read as a video; (.+)$/.exec(text);
  if (match) {
    return {
      name: match[1],
      checks: [{ key: "picture", label: "Picture", ok: false, value: `Could not be read as a video: ${match[2]}` }],
    };
  }
  if (text === "could not read the video resolution") {
    return { name, checks: [{ key: "size", label: "Size", ok: false, value: "The picture size could not be read" }] };
  }
  match = /^video is (\d+)x(\d+); the limit is 1080p/.exec(text);
  if (match) {
    return {
      name,
      checks: [
        {
          key: "size",
          label: "Size",
          ok: false,
          value: `${match[1]}×${match[2]} — up to 1080p (1920×1080 either way). Make a smaller copy and pick that.`,
        },
      ],
    };
  }
  if (text === "could not read the video duration") {
    return {
      name,
      checks: [SIZE_OK, { key: "length", label: "Length", ok: false, value: "The length could not be read" }],
    };
  }
  match = /^video is ([\d.]+) minutes long; the limit is 30 minutes$/.exec(text);
  if (match) {
    return {
      name,
      checks: [
        SIZE_OK,
        {
          key: "length",
          label: "Length",
          ok: false,
          value: `${clockText(Number(match[1]) * 60)} — up to 30 min. ${TRIM}`,
        },
      ],
    };
  }
  if (/^VFR\/unsupported video timing/.test(text)) {
    return {
      name,
      checks: [
        SIZE_OK,
        LENGTH_OK,
        {
          key: "timing",
          label: "Steady frame rate",
          ok: false,
          value:
            "The frame rate is not steady (or could not be checked). Convert it to a constant frame rate, then pick the copy.",
        },
      ],
    };
  }
  return null;
}

/** What went wrong while the scenes were found (the page is back on the pick), else null. */
export function findFailure(error: string | null | undefined): string | null {
  const match = /^Finding the segments failed: (.+)$/s.exec((error ?? "").trim());
  return match ? match[1] : null;
}

export type VideoMeta = { width: number; height: number; duration: number };

/**
 * The checks the browser can make on a file chosen in it, before it is
 * copied to the wizard: its picture size and length (read from the file
 * with a video element). The wizard checks again, and the frame rate.
 * Without `meta` (the browser could not read the file) there is nothing
 * to say yet.
 */
export function browserChecks(meta: VideoMeta | null): FileCheck[] {
  if (!meta) return [];
  const checks: FileCheck[] = [];
  if (meta.width > 0 && meta.height > 0) {
    const long = Math.max(meta.width, meta.height);
    const short = Math.min(meta.width, meta.height);
    const ok = long <= MAX_LONG_EDGE && short <= MAX_SHORT_EDGE;
    checks.push({
      key: "size",
      label: "Size",
      ok,
      value: `${meta.width}×${meta.height} — up to 1080p${ok ? "" : ". Make a smaller copy and pick that."}`,
    });
  }
  if (Number.isFinite(meta.duration) && meta.duration > 0) {
    const ok = meta.duration <= MAX_DURATION_S;
    checks.push({
      key: "length",
      label: "Length",
      ok,
      value: `${clockText(meta.duration)} — up to 30 min${ok ? "" : `. ${TRIM}`}`,
    });
  }
  return checks;
}

/** True when no check failed. */
export function checksPass(checks: FileCheck[]): boolean {
  return checks.every((check) => check.ok);
}
