import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type PointerEvent,
  type ReactNode,
} from "react";
import type { Region } from "./api";
import {
  clampWipe,
  playerTime,
  regionLabel,
  stepTime,
  viewerKey,
  wipeKey,
  type ViewerMode,
} from "./previewView";
import { Icon, Segmented } from "./ui";

// Until the clip's own frame rate is seen (requestVideoFrameCallback).
const DEFAULT_FRAME_S = 1 / 25;
// Further apart than this, the original is moved to the restored one's time.
const DRIFT_S = 0.06;
const STAGE_H = 470;

const MODES: { value: ViewerMode; label: string }[] = [
  { value: "wipe", label: "Wipe" },
  { value: "side", label: "Side by side" },
  { value: "ab", label: "A/B" },
];

type FrameCallback = (now: number, metadata: { mediaTime: number }) => void;
type FrameVideo = HTMLVideoElement & { requestVideoFrameCallback?: (callback: FrameCallback) => number };

/** Keys typed into a field, or pressed on a control, are that control's. */
function keyOwner(target: EventTarget | null): "field" | "control" | null {
  if (!(target instanceof HTMLElement)) return null;
  if (target.isContentEditable || target.closest("input, textarea, select")) return "field";
  if (target.closest("button, a, summary, [role=radio], [role=slider], [role=checkbox], [role=switch]")) return "control";
  return null;
}

/**
 * The comparison viewer (mockup 05, gen.py `viewer()`): the real before and
 * after previews kept in step, with one play/pause, one scrubber and frame
 * steps; Wipe (a draggable, keyboard handle), Side by side or A/B; a 100%
 * view; the protect boxes with their labels. Keys (s9): Space plays, ←/→
 * step a frame, B flips before/after, Z zooms to 100%; with `onMark`, I
 * and O mark the start and end of a part at the player's time.
 *
 * Without `beforeSrc` (the Result: the wizard serves no copy of the
 * original) it plays the restored video alone, with the same controls but
 * no Wipe, Side by side or A/B.
 */
export function CompareViewer({
  beforeSrc,
  afterSrc,
  beforePoster,
  afterPoster,
  regions,
  afterLabel,
  sceneName,
  onDuration,
  onTime,
  onMark,
  timeline,
  positionLabel = "Position in the preview",
}: {
  beforeSrc?: string | null;
  afterSrc: string;
  beforePoster?: string;
  afterPoster?: string;
  regions: Region[];
  afterLabel: string;
  sceneName: string;
  onDuration?: (seconds: number) => void;
  // The player's time, as it changes.
  onTime?: (seconds: number) => void;
  // I / O: the start or end of a part, at the player's time.
  onMark?: (which: "start" | "end", seconds: number) => void;
  // Drawn between the picture and the controls (the Result's seams).
  timeline?: ReactNode;
  positionLabel?: string;
}) {
  const single = !beforeSrc;
  const root = useRef<HTMLDivElement | null>(null);
  const picture = useRef<HTMLDivElement | null>(null);
  const before = useRef<HTMLVideoElement | null>(null);
  const after = useRef<HTMLVideoElement | null>(null);
  const frameS = useRef(DEFAULT_FRAME_S);
  const [mode, setMode] = useState<ViewerMode>("wipe");
  const [wipe, setWipe] = useState(50);
  const [showAfter, setShowAfter] = useState(true);
  const [zoom, setZoom] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [size, setSize] = useState<[number, number] | null>(null);
  const dragging = useRef(false);

  // A new preview starts from the beginning, paused.
  useEffect(() => {
    setPlaying(false);
    setTime(0);
    setDuration(0);
    frameS.current = DEFAULT_FRAME_S;
  }, [afterSrc, beforeSrc]);

  const sync = useCallback(() => {
    const a = after.current;
    const b = before.current;
    if (!a || !b) return;
    if (Math.abs(b.currentTime - a.currentTime) > DRIFT_S) b.currentTime = a.currentTime;
  }, []);

  // While playing: follow the restored clip's time, and keep the original on it.
  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    const tick = () => {
      const a = after.current;
      if (a) setTime(a.currentTime);
      sync();
      frame = window.requestAnimationFrame(tick);
    };
    frame = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(frame);
  }, [playing, sync]);

  // The clip's frame length, from the gaps between frames shown.
  useEffect(() => {
    const a = after.current as FrameVideo | null;
    if (!a?.requestVideoFrameCallback || !playing) return;
    let last: number | null = null;
    let alive = true;
    const seen: FrameCallback = (_now, metadata) => {
      if (!alive) return;
      if (last !== null) {
        const gap = metadata.mediaTime - last;
        if (gap > 0.005 && gap < 0.2 && gap < frameS.current + 0.002) frameS.current = gap;
      }
      last = metadata.mediaTime;
      a.requestVideoFrameCallback?.(seen);
    };
    a.requestVideoFrameCallback(seen);
    return () => {
      alive = false;
    };
  }, [playing]);

  const togglePlay = useCallback(() => {
    const a = after.current;
    const b = before.current;
    if (!a || (!b && !single)) return;
    if (a.paused) {
      if (b) b.currentTime = a.currentTime;
      void a.play().catch(() => setPlaying(false));
      if (b) void b.play().catch(() => undefined);
      setPlaying(true);
    } else {
      a.pause();
      if (b) {
        b.pause();
        b.currentTime = a.currentTime;
      }
      setPlaying(false);
      setTime(a.currentTime);
    }
  }, [single]);

  const seek = useCallback(
    (to: number) => {
      const a = after.current;
      const b = before.current;
      if (!a || (!b && !single)) return;
      a.currentTime = to;
      if (b) b.currentTime = to;
      setTime(to);
    },
    [single],
  );

  const step = useCallback(
    (frames: number) => {
      const a = after.current;
      const b = before.current;
      if (!a || (!b && !single)) return;
      a.pause();
      b?.pause();
      setPlaying(false);
      seek(stepTime(a.currentTime, a.duration || duration, frames, frameS.current));
    },
    [duration, seek, single],
  );

  const flip = useCallback(() => {
    if (mode !== "ab") {
      setMode("ab");
      setShowAfter(false);
    } else {
      setShowAfter((value) => !value);
    }
  }, [mode]);

  // The viewer's keys work anywhere on the step, except while typing in a
  // field; on a focused control (a button, the wipe handle) Space and the
  // arrows stay that control's.
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) return;
      const action = viewerKey(event.key);
      if (!action) return;
      if ((action === "markStart" || action === "markEnd") && !onMark) return;
      if (action === "flip" && single) return;
      const owner = keyOwner(event.target);
      if (owner === "field") return;
      if (owner === "control" && (action === "play" || action === "back" || action === "forward")) return;
      if (!root.current?.isConnected) return;
      // A modal dialog (the start-over confirm) makes the page behind it inert.
      if (root.current.closest("[inert]") || document.querySelector("dialog[open]")) return;
      event.preventDefault();
      if (action === "play") togglePlay();
      else if (action === "back") step(-1);
      else if (action === "forward") step(1);
      else if (action === "flip") flip();
      else if (action === "markStart") onMark?.("start", after.current?.currentTime ?? 0);
      else if (action === "markEnd") onMark?.("end", after.current?.currentTime ?? 0);
      else setZoom((value) => !value);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [togglePlay, step, flip, onMark, single]);

  useEffect(() => {
    onTime?.(time);
  }, [time, onTime]);

  // Entering the 100% view: start at the middle of the picture.
  useEffect(() => {
    const stage = picture.current;
    if (!zoom || !stage) return;
    stage.scrollLeft = (stage.scrollWidth - stage.clientWidth) / 2;
    stage.scrollTop = (stage.scrollHeight - stage.clientHeight) / 2;
  }, [zoom, mode]);

  function wipeFrom(event: PointerEvent<HTMLElement>) {
    const frame = (event.currentTarget as HTMLElement).closest(".vframe") ?? event.currentTarget;
    const box = frame.getBoundingClientRect();
    if (box.width > 0) setWipe(clampWipe(((event.clientX - box.left) / box.width) * 100));
  }

  const ratio = size ? size[0] / size[1] : 4 / 3;
  const frameStyle = {
    "--ar": String(ratio),
    "--stage-h": `${STAGE_H}px`,
    ...(zoom && size ? { width: `${size[0]}px` } : null),
  } as CSSProperties;
  const stacked = single || mode !== "side";
  const tag = (text: string, side: "left" | "right") => <span className={`vtag ${side}`}>{text}</span>;
  const boxes = regions.map((region, index) => (
    <span
      key={`${region.kind}-${index}`}
      className={`protect ${region.kind}${region.y0 < 0.1 ? " inside" : ""}`}
      style={{
        left: `${region.x0 * 100}%`,
        top: `${region.y0 * 100}%`,
        width: `${(region.x1 - region.x0) * 100}%`,
        height: `${(region.y1 - region.y0) * 100}%`,
      }}
    >
      <span>{regionLabel(region)}</span>
    </span>
  ));
  // The two videos keep their place in the tree in every mode, so changing
  // mode never reloads them or loses the time.
  const beforeFrameStyle: CSSProperties = { ...frameStyle };
  if (mode === "wipe") beforeFrameStyle.clipPath = `inset(0 ${100 - wipe}% 0 0)`;
  if (mode === "ab" && showAfter) beforeFrameStyle.visibility = "hidden";

  const stage = (
    <>
      <div className="vframe v-after" style={frameStyle}>
        <video
          ref={after}
          src={afterSrc}
          poster={afterPoster}
          muted
          loop
          playsInline
          preload="auto"
          aria-hidden="true"
          tabIndex={-1}
          onLoadedMetadata={(event) => {
            const video = event.currentTarget;
            if (video.videoWidth && video.videoHeight) setSize([video.videoWidth, video.videoHeight]);
            setDuration(video.duration || 0);
            if (video.duration) onDuration?.(video.duration);
          }}
          onTimeUpdate={(event) => {
            if (!playing) setTime(event.currentTarget.currentTime);
          }}
          onPause={() => setPlaying(false)}
        />
        {stacked ? null : tag(afterLabel, "left")}
        {stacked ? null : boxes}
      </div>
      {single ? null : (
      <div className="vframe v-before" style={beforeFrameStyle}>
        <video
          ref={before}
          src={beforeSrc}
          poster={beforePoster}
          muted
          loop
          playsInline
          preload="auto"
          aria-hidden="true"
          tabIndex={-1}
        />
        {stacked ? null : tag("Original", "left")}
        {stacked ? null : boxes}
      </div>
      )}
      {single ? (
        <div className="vframe v-overlay" style={frameStyle}>
          {boxes}
          {tag(afterLabel, "left")}
        </div>
      ) : stacked ? (
        <div
          className={mode === "wipe" ? "vframe v-overlay wiping" : "vframe v-overlay"}
          style={frameStyle}
          onPointerDown={(event) => {
            if (mode !== "wipe" || event.button !== 0) return;
            dragging.current = true;
            event.currentTarget.setPointerCapture(event.pointerId);
            wipeFrom(event);
          }}
          onPointerMove={(event) => {
            if (dragging.current) wipeFrom(event);
          }}
          onPointerUp={() => {
            dragging.current = false;
          }}
          onPointerCancel={() => {
            dragging.current = false;
          }}
        >
          {boxes}
          {mode === "wipe" ? (
            <>
              <span className="wipe-line" style={{ left: `${wipe}%` }} aria-hidden="true" />
              <div
                className="wipe-handle"
                role="slider"
                tabIndex={0}
                aria-label="Wipe: original on the left, restored on the right"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(wipe)}
                aria-valuetext={`${Math.round(wipe)}% original`}
                style={{ left: `${wipe}%` }}
                onKeyDown={(event) => {
                  const next = wipeKey(event.key, wipe, event.shiftKey);
                  if (next === null) return;
                  event.preventDefault();
                  event.stopPropagation();
                  setWipe(next);
                }}
              >
                <Icon name="chevron-left" size={16} />
                <Icon name="chevron-right" size={16} />
              </div>
              {wipe > 12 ? tag("Original", "left") : null}
              {wipe < 88 ? tag(afterLabel, "right") : null}
            </>
          ) : (
            tag(showAfter ? afterLabel : "Original", "left")
          )}
        </div>
      ) : null}
    </>
  );

  return (
    <div
      ref={root}
      className="viewer"
      role="group"
      aria-label={single ? sceneName : `Before and after: ${sceneName}`}
    >
      <div
        ref={picture}
        className={["picture", mode === "side" ? "side" : null, zoom ? "zoomed" : null].filter(Boolean).join(" ")}
      >
        {stage}
      </div>
      {timeline}
      <p className="sr-only" role="status">
        {mode === "ab" ? (showAfter ? `Showing ${afterLabel}` : "Showing the original") : ""}
      </p>
      <div className="vcontrols">
        <button type="button" className="vplay" aria-label={playing ? "Pause" : "Play"} onClick={togglePlay}>
          {playing ? <span className="pause-bars" aria-hidden="true" /> : <Icon name="play" size={16} />}
        </button>
        <span className="row gap4">
          <button type="button" className="vstep" aria-label="Back one frame" onClick={() => step(-1)}>
            <Icon name="chevron-left" size={20} />
          </button>
          <button type="button" className="vstep" aria-label="Forward one frame" onClick={() => step(1)}>
            <Icon name="chevron-right" size={20} />
          </button>
        </span>
        <span className="mono vtime" aria-hidden="true">
          {playerTime(time, duration)}
        </span>
        <input
          className="vscrub"
          type="range"
          min={0}
          max={duration || 0}
          step={0.01}
          value={Math.min(time, duration || 0)}
          aria-label={positionLabel}
          aria-valuetext={playerTime(time, duration)}
          style={{ "--played": `${duration ? (time / duration) * 100 : 0}%` } as CSSProperties}
          onChange={(event) => seek(Number(event.target.value))}
        />
        {single ? null : <Segmented label="Compare" dark options={MODES} value={mode} onChange={setMode} />}
        <button
          type="button"
          className="vzoom"
          aria-pressed={zoom}
          aria-label="Show at 100%"
          onClick={() => setZoom((value) => !value)}
        >
          <Icon name="eye" size={16} />
          100%
        </button>
      </div>
    </div>
  );
}
