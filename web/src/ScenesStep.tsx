import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
  type PointerEvent,
  type ReactNode,
} from "react";
import { outputSrc, scaleChoiceText, type Segment, type SegmentEdit, type SegmentList, type WizardView } from "./api";
import { ScenesProblemCard } from "./FindingStep";
import { scenesProblem } from "./findingView";
import {
  canSplit,
  checkMerge,
  checkMove,
  checkSplit,
  cutName,
  cutRange,
  cutValueText,
  dragTime,
  formatTime,
  isDrag,
  limitsOf,
  nudgeCut,
  parseTime,
  scenesText,
  splitRange,
  splitSuggestion,
  withCutAt,
  type Range,
  type SceneLimits,
} from "./sceneEdit";
import {
  composeNote,
  cutChips,
  protectLabels,
  rulerMarks,
  scenesHeading,
  sourceNote,
  toggleChip,
  type CutChipKey,
} from "./scenesView";
import { Badge, Banner, Button, Card, Chip, Field, Icon, LinkButton } from "./ui";

type EditScenes = (edit: SegmentEdit) => Promise<boolean>;

// Room for one time under the timeline ("22:01" in 13 px mono, and a gap).
const RULER_LABEL_PX = 56;

/**
 * The scenes along the video, with a handle on each cut. A handle is a
 * slider: drag it, or focus it and use the arrow keys (Page Up / Page Down
 * for ten seconds, Home / End for as far as it goes), then Enter to save
 * or Esc to put it back. Moving focus away saves it too.
 */
function CutTimeline({
  origin,
  segments,
  limits,
  locked,
  onMove,
  onRefused,
}: {
  origin: string;
  segments: Segment[];
  limits: SceneLimits;
  locked: boolean;
  onMove: (index: number, at: number) => Promise<boolean>;
  onRefused: (text: string) => void;
}) {
  const track = useRef<HTMLDivElement | null>(null);
  // The cut being moved and where it is now, before the server has it.
  const [draft, setDraft] = useState<{ index: number; at: number } | null>(null);
  const saving = useRef(false);
  // A pointer down on a handle: where it went down and the cut's time
  // then. It becomes a drag only once it has moved a few pixels, so a
  // click or tap just focuses the handle and changes nothing.
  const gesture = useRef<{
    index: number;
    pointerId: number;
    startX: number;
    startAt: number;
    width: number;
    dragging: boolean;
  } | null>(null);
  // The track's width, so the ruler keeps room for each time it shows.
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const element = track.current;
    if (!element || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => setWidth(element.getBoundingClientRect().width));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const total = segments.length ? segments[segments.length - 1].end_s : 0;
  if (total <= 0) return null;
  const shown = draft ? withCutAt(segments, draft.index, draft.at) : segments;
  const share = (value: number) => `${(value / total) * 100}%`;

  async function save(index: number, at: number) {
    if (saving.current) return;
    const before = segments[index]?.end_s;
    if (before === undefined || Math.round(before * 1000) === Math.round(at * 1000)) {
      setDraft(null);
      return;
    }
    const refused = checkMove(segments, index, at, limits);
    if (refused) {
      setDraft(null);
      onRefused(refused);
      return;
    }
    saving.current = true;
    try {
      await onMove(index, at);
    } finally {
      saving.current = false;
      setDraft(null);
    }
  }

  function key(event: KeyboardEvent<HTMLDivElement>, index: number) {
    if (locked) return;
    const at = draft?.index === index ? draft.at : segments[index].end_s;
    if (event.key === "Enter") {
      event.preventDefault();
      if (draft?.index === index) void save(index, draft.at);
      return;
    }
    if (event.key === "Escape") {
      if (draft) event.preventDefault();
      setDraft(null);
      return;
    }
    const range = cutRange(segments, index, limits.minS);
    if (!range) {
      if (nudgeCut(at, event.key, { low: at, high: at }) !== null) {
        event.preventDefault();
        onRefused(checkMove(segments, index, at, limits) ?? "This cut cannot move.");
      }
      return;
    }
    const next = nudgeCut(at, event.key, range);
    if (next === null) return;
    event.preventDefault();
    setDraft({ index, at: next });
  }

  // Where the cut being dragged is, or null when the pointer has not moved
  // far enough yet to be a drag.
  function dragged(event: PointerEvent<HTMLDivElement>, range: Range): number | null {
    const current = gesture.current;
    if (!current || current.pointerId !== event.pointerId) return null;
    if (!current.dragging && !isDrag(current.startX, event.clientX)) return null;
    current.dragging = true;
    return dragTime(current.startAt, current.startX, event.clientX, current.width, total, range);
  }

  return (
    <div className="cut-timeline">
      <div className="cut-track" ref={track}>
        {shown.map((segment, index) => {
          const still = segment.still_url ? outputSrc(origin, segment.still_url) : null;
          return (
            <span
              key={index}
              className={["cut-block", index === 0 ? "first" : null, index === shown.length - 1 ? "last" : null]
                .filter(Boolean)
                .join(" ")}
              aria-hidden="true"
              style={{ left: share(segment.start_s), width: share(segment.end_s - segment.start_s) }}
            >
              {still ? <img src={still} alt="" loading="lazy" /> : null}
              <span className="cut-number">{index + 1}</span>
            </span>
          );
        })}
        {shown.slice(0, -1).map((segment, index) => {
          const range = cutRange(segments, index, limits.minS);
          const moving = draft?.index === index;
          return (
            <div
              key={index}
              role="slider"
              tabIndex={0}
              className={`cut-handle${moving ? " moving" : ""}`}
              style={{ left: share(segment.end_s) }}
              aria-label={cutName(index)}
              aria-valuemin={range?.low ?? segment.end_s}
              aria-valuemax={range?.high ?? segment.end_s}
              aria-valuenow={segment.end_s}
              aria-valuetext={cutValueText(index, segment.end_s)}
              aria-describedby="cut-help"
              aria-disabled={locked || !range ? true : undefined}
              onKeyDown={(event) => key(event, index)}
              onBlur={() => {
                if (moving && draft) void save(index, draft.at);
              }}
              onPointerDown={(event) => {
                if (locked || !range || event.button !== 0) return;
                event.preventDefault();
                event.currentTarget.focus();
                event.currentTarget.setPointerCapture(event.pointerId);
                gesture.current = {
                  index,
                  pointerId: event.pointerId,
                  startX: event.clientX,
                  // A cut moved with the keys and not saved yet drags on from there.
                  startAt: moving && draft ? draft.at : segments[index].end_s,
                  width: track.current?.getBoundingClientRect().width ?? 0,
                  dragging: false,
                };
              }}
              onPointerMove={(event) => {
                if (!range || gesture.current?.index !== index) return;
                const at = dragged(event, range);
                if (at !== null) setDraft({ index, at });
              }}
              onPointerUp={(event) => {
                const current = gesture.current;
                if (!range || !current || current.index !== index || current.pointerId !== event.pointerId) return;
                const at = dragged(event, range);
                gesture.current = null;
                if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                  event.currentTarget.releasePointerCapture(event.pointerId);
                }
                // A click or tap only focuses the cut: nothing is sent.
                if (!current.dragging || at === null) return;
                void save(index, at);
              }}
              onPointerCancel={() => {
                const current = gesture.current;
                gesture.current = null;
                if (current?.dragging) setDraft(null);
              }}
            >
              <span className="cut-time" aria-hidden="true">
                Drag to move this cut · {formatTime(segment.end_s)}
              </span>
            </div>
          );
        })}
      </div>
      <div className="cut-scale" aria-hidden="true">
        {rulerMarks(shown, width > 0 ? Math.min(0.5, RULER_LABEL_PX / width) : undefined).map((mark) => (
          <span
            key={`${mark.edge ?? "cut"}-${mark.at}`}
            className={mark.edge ? `mark ${mark.edge}` : "mark"}
            style={mark.edge ? undefined : { left: share(mark.at) }}
          >
            {formatTime(mark.at)}
          </span>
        ))}
      </div>
      <p id="cut-help" className="caption subtle cut-help">
        Drag a cut, or focus it and use the arrow keys (Page Up / Page Down: 10 s). Enter saves, Esc puts it back.
        To type a time, use Edit on a scene.
      </p>
    </div>
  );
}

/** A time typed for a change to one scene: where to split it, or where its cut goes. */
function TimeForm({
  id,
  title,
  label,
  submitText,
  initial,
  range,
  locked,
  closeIfUnchanged = false,
  check,
  onSubmit,
  onClose,
}: {
  id: string;
  title: string;
  label: string;
  submitText: string;
  initial: number | null;
  range: Range | null;
  locked: boolean;
  // When the time is left as it was shown, close without sending anything.
  closeIfUnchanged?: boolean;
  check: (at: number) => string | null;
  onSubmit: (at: number) => Promise<boolean>;
  onClose: () => void;
}) {
  const [text, setText] = useState(initial === null ? "" : formatTime(initial));
  const [problem, setProblem] = useState<string | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (closeIfUnchanged && initial !== null && text.trim() === formatTime(initial)) {
      onClose();
      return;
    }
    const at = parseTime(text);
    const refused = at === null ? "Write the time like 1:30, or in seconds like 90." : check(at);
    setProblem(refused);
    if (refused || at === null) return;
    if (await onSubmit(at)) onClose();
  }

  return (
    <form className="split-form" onSubmit={submit} aria-label={title}>
      <label htmlFor={`${id}-at`}>
        {label}
        <input
          id={`${id}-at`}
          type="text"
          inputMode="decimal"
          autoComplete="off"
          spellCheck={false}
          value={text}
          placeholder="1:30"
          aria-describedby={`${id}-hint`}
          aria-invalid={problem ? true : undefined}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Escape") onClose();
          }}
          autoFocus
        />
      </label>
      <p id={`${id}-hint`} className="muted split-hint">
        {range
          ? `Minutes and seconds, between ${formatTime(Math.ceil(range.low * 10 - 1e-6) / 10)} and ${formatTime(Math.floor(range.high * 10 + 1e-6) / 10)}.`
          : null}
      </p>
      {problem ? (
        <p className="error" role="alert">
          {problem}
        </p>
      ) : null}
      <div className="row gap8 wrap">
        <Button type="submit" kind="primary" size="md" disabled={locked}>
          {submitText}
        </Button>
        <Button kind="secondary" size="md" onClick={onClose}>
          Cancel
        </Button>
      </div>
    </form>
  );
}


/**
 * Step 2 · Scenes (mockup 04): what Gemini found and how it saw the video,
 * the timeline with a handle on each cut, a row per scene (with Edit for
 * merge, split and move cut), and on the right "Suggest different cuts"
 * (chips and words sent to Gemini) above "Looks right — preview scenes".
 */
export function ScenesStep({
  origin,
  list,
  busy,
  message,
  onEdit,
  view,
  stopped,
  back,
}: {
  origin: string;
  list: SegmentList;
  busy: boolean;
  message: string | null;
  onEdit: EditScenes;
  view: WizardView;
  // The "stopped" note (StoppedNote in LiveWizard) and the way back.
  stopped?: ReactNode;
  back?: ReactNode;
}) {
  const [note, setNote] = useState("");
  const [chosen, setChosen] = useState<CutChipKey[]>([]);
  // What the last change by hand did, or why it could not be made.
  const [said, setSaid] = useState<{ text: string; refused: boolean } | null>(null);
  // The scene whose Edit actions are shown, and a time form open there.
  const [expanded, setExpanded] = useState<number | null>(null);
  const [open, setOpen] = useState<{ kind: "split" | "move"; index: number } | null>(null);
  // The scene to focus once the list shows a merge or a split.
  const [focusScene, setFocusScene] = useState<number | null>(null);
  const sceneList = useRef<HTMLOListElement | null>(null);
  const segments = list.segments;
  const count = segments.length;
  const limits = limitsOf(list);
  const locked = busy || Boolean(view.busy);
  // A list the server could not read cannot be changed (its error shows).
  const editable = count > 0 && typeof list.revision === "number";
  const problem = scenesProblem(list);
  const source = sourceNote(list);
  const scale = scaleChoiceText(list.scale, list.scale_reason);
  const chips = cutChips(segments, limits);

  useEffect(() => {
    if (focusScene === null) return;
    sceneList.current?.querySelector<HTMLElement>(`[data-scene-title="${focusScene}"]`)?.focus();
    setFocusScene(null);
  }, [focusScene, list.revision]);

  // A new list (a change, or a reload) closes what was open on the old one,
  // and the chips name its scenes afresh.
  useEffect(() => {
    setOpen(null);
    setExpanded(null);
    setChosen([]);
  }, [list.revision]);

  function propose(event: FormEvent) {
    event.preventDefault();
    void onEdit({ action: "note", note: composeNote(chips, chosen, note) });
  }

  function refused(text: string) {
    setSaid({ text, refused: true });
  }

  async function edit(change: SegmentEdit, done: string, focus: number | null) {
    setSaid(null);
    if (await onEdit(change)) {
      setSaid({ text: done, refused: false });
      if (focus !== null) setFocusScene(focus);
      return true;
    }
    return false;
  }

  function merge(index: number) {
    const problem = checkMerge(segments, index, limits);
    if (problem) return refused(problem);
    void edit(
      { action: "merge", index },
      `Scenes ${index + 1} and ${index + 2} are one scene now. ${scenesText(count - 1)}.`,
      index,
    );
  }

  function openSplit(index: number) {
    const problem = canSplit(segments, index, limits);
    if (problem) return refused(problem);
    setSaid(null);
    setOpen(open?.kind === "split" && open.index === index ? null : { kind: "split", index });
  }

  function openMove(index: number) {
    if (!cutRange(segments, index, limits.minS)) {
      return refused(checkMove(segments, index, segments[index].end_s + 1, limits) ?? "This cut cannot move.");
    }
    setSaid(null);
    setOpen(open?.kind === "move" && open.index === index ? null : { kind: "move", index });
  }

  function move(index: number, at: number) {
    return edit({ action: "move", index, at }, `${cutName(index)} moved to ${formatTime(at)}.`, null);
  }

  const left = (
    <div className="scenes-main col gap20">
      <header className="step-head">
        <p className="overline">Step 2 of 5 · Scenes</p>
        <h1>{scenesHeading(list)}</h1>
        <p className="muted">
          {list.error
            ? "Approve it to go on, or ask Gemini to try again."
            : "It cut where the light, place or picture quality changes, so each scene gets its own settings."}{" "}
          {source ? (
            <span className={`source-note ${source.kind}`}>
              <Icon name={source.kind === "video" ? "sparkles" : "alert"} size={16} />
              {source.text}
            </span>
          ) : null}
        </p>
        {scale ? <p className="caption subtle scale-note">{scale}</p> : null}
      </header>
      {stopped}
      {message ? (
        <Banner tone="error" role="alert" title="Something went wrong">
          {message}
        </Banner>
      ) : null}
      <ScenesProblemCard list={list} locked={locked} onEdit={onEdit} />
      {!problem && list.error ? (
        <Banner tone="warning" title="Note">
          {list.error}
        </Banner>
      ) : null}
      {editable ? (
        <Card as="section" tight className="col gap12 cut-editor" aria-labelledby="cut-editor-title">
          <div className="row between wrap gap8">
            <h2 id="cut-editor-title" className="body-s strong">
              Timeline
            </h2>
            <span className="caption subtle">
              Each scene at least {limits.minS} s · up to {limits.maxScenes} scenes
            </span>
          </div>
          <CutTimeline
            origin={origin}
            segments={segments}
            limits={limits}
            locked={locked}
            onRefused={refused}
            onMove={move}
          />
        </Card>
      ) : null}
      <p className={said?.refused ? "error edit-status" : "muted edit-status"} role="status">
        {said ? said.text : ""}
      </p>
      <section aria-labelledby="scene-rows-title">
        <h2 id="scene-rows-title" className="sr-only">
          {scenesText(count)}
        </h2>
        <ol className="scene-rows" ref={sceneList}>
          {segments.map((segment, index) => (
            <SceneRow
              key={index}
              origin={origin}
              segment={segment}
              index={index}
              count={count}
              editable={editable}
              locked={locked}
              expanded={expanded === index}
              onExpand={(on) => {
                setExpanded(on ? index : null);
                if (!on && open?.index === index) setOpen(null);
              }}
              onMerge={() => merge(index)}
              onSplit={() => openSplit(index)}
              onMove={() => openMove(index)}
              openKind={open?.index === index ? open.kind : null}
            >
              {editable && open?.kind === "split" && open.index === index ? (
                <div id={`split-${index}-form`}>
                  <TimeForm
                    id={`split-${index}`}
                    title={`Split scene ${index + 1}`}
                    label={`Split scene ${index + 1} at`}
                    submitText="Split here"
                    initial={splitSuggestion(segments, index, limits.minS)}
                    range={splitRange(segments, index, limits.minS)}
                    locked={locked}
                    check={(at) => checkSplit(segments, index, at, limits)}
                    onClose={() => setOpen(null)}
                    onSubmit={(at) =>
                      edit(
                        { action: "split", index, at },
                        `Scene ${index + 1} is two scenes now, cut at ${formatTime(at)}. ${scenesText(count + 1)}.`,
                        index + 1,
                      )
                    }
                  />
                </div>
              ) : null}
              {editable && open?.kind === "move" && open.index === index ? (
                <div id={`move-${index}-form`}>
                  <TimeForm
                    id={`move-${index}`}
                    title={cutName(index)}
                    label={`Move the cut between scenes ${index + 1} and ${index + 2} to`}
                    submitText="Move cut"
                    initial={segment.end_s}
                    closeIfUnchanged
                    range={cutRange(segments, index, limits.minS)}
                    locked={locked}
                    check={(at) => checkMove(segments, index, at, limits)}
                    onClose={() => setOpen(null)}
                    onSubmit={(at) => move(index, at)}
                  />
                </div>
              ) : null}
            </SceneRow>
          ))}
        </ol>
      </section>
      {back}
    </div>
  );

  // When the split did not work its card offers the ways on.
  if (problem) return <div className="scenes-step">{left}</div>;

  return (
    <div className="scenes-step">
      {left}
      <div className="scenes-side col gap16">
        <Card as="section" className="col gap12" aria-labelledby="suggest-title">
          <h2 id="suggest-title" className="h3">
            Suggest different cuts
          </h2>
          <form className="col gap12" onSubmit={propose}>
            {chips.length ? (
              <div className="row gap8 wrap" role="group" aria-label="Ask Gemini to">
                {chips.map((chip) => (
                  <Chip
                    key={chip.key}
                    selected={chosen.includes(chip.key)}
                    disabled={locked}
                    onToggle={(on) => setChosen((value) => toggleChip(value, chip.key, on))}
                    // "Split scene 4" is also a scene row's button; this one only asks Gemini.
                    label={`Ask Gemini: ${chip.note}`}
                  >
                    {chip.text}
                  </Chip>
                ))}
              </div>
            ) : null}
            <Field
              id="segments-note"
              label="Or say it in words (optional)"
              type="text"
              autoComplete="off"
              value={note}
              placeholder="e.g. cut where we walk outside"
              onChange={(event) => setNote(event.target.value)}
            />
            <Button type="submit" kind="secondary" size="md" icon="rotate" disabled={locked}>
              Ask Gemini to re-split
            </Button>
          </form>
        </Card>
        <div className="grow" />
        <div className="col gap8 scenes-approve">
          <Button kind="primary" size="lg" disabled={locked} onClick={() => void onEdit({ action: "approve" })}>
            Looks right — preview scenes
          </Button>
          <p className="caption subtle">Next: a 3-second before/after of each scene.</p>
        </div>
      </div>
    </div>
  );
}

function SceneRow({
  origin,
  segment,
  index,
  count,
  editable,
  locked,
  expanded,
  openKind,
  onExpand,
  onMerge,
  onSplit,
  onMove,
  children,
}: {
  origin: string;
  segment: Segment;
  index: number;
  count: number;
  editable: boolean;
  locked: boolean;
  expanded: boolean;
  openKind: "split" | "move" | null;
  onExpand: (on: boolean) => void;
  onMerge: () => void;
  onSplit: () => void;
  onMove: () => void;
  children: ReactNode;
}) {
  const number = index + 1;
  const still = segment.still_url ? outputSrc(origin, segment.still_url) : null;
  const badges = protectLabels(segment.contains);
  const last = index === count - 1;
  const panel = `scene-${index}-edits`;
  return (
    <li className="scene-row">
      <div className="scene-row-main">
        {still ? (
          <img className="scene-thumb" src={still} alt="" loading="lazy" />
        ) : (
          <span className="scene-thumb still-gap" aria-hidden="true" />
        )}
        <div className="col gap4 grow scene-row-text">
          <h3 className="body-s strong" tabIndex={-1} data-scene-title={index}>
            {number} · {segment.label || "Unlabelled scene"}
          </h3>
          <span className="caption subtle">
            {formatTime(segment.start_s)}–{formatTime(segment.end_s)} · {formatTime(segment.end_s - segment.start_s)}
          </span>
        </div>
        {badges.length ? (
          <ul className="row gap6 wrap scene-badges" aria-label="Handled gently">
            {badges.map((label) => (
              <li key={label}>
                <Badge tone="info" icon={false}>
                  {label}
                </Badge>
              </li>
            ))}
          </ul>
        ) : null}
        <div className="row gap4 scene-row-actions">
          {still ? (
            <LinkButton href={still} kind="ghost" size="md" icon="eye" newTab>
              View still<span className="sr-only"> of scene {number} (opens in a new tab)</span>
            </LinkButton>
          ) : null}
          {editable ? (
            <Button
              kind="ghost"
              size="md"
              icon="more"
              aria-label={`Edit scene ${number}`}
              aria-expanded={expanded}
              aria-controls={expanded ? panel : undefined}
              onClick={() => onExpand(!expanded)}
            >
              Edit
            </Button>
          ) : null}
        </div>
      </div>
      {editable && expanded ? (
        <div id={panel} className="scene-edits col gap8">
          <div className="row gap8 wrap">
            {!last ? (
              <Button
                kind="secondary"
                size="md"
                disabled={locked}
                aria-label={`Merge scenes ${number} and ${number + 1}`}
                onClick={onMerge}
              >
                Merge with {number + 1}
              </Button>
            ) : null}
            <Button
              kind="secondary"
              size="md"
              icon="scissors"
              disabled={locked}
              aria-expanded={openKind === "split"}
              aria-controls={openKind === "split" ? `split-${index}-form` : undefined}
              aria-label={`Split scene ${number}`}
              onClick={onSplit}
            >
              Split…
            </Button>
            {!last ? (
              <Button
                kind="secondary"
                size="md"
                disabled={locked}
                aria-expanded={openKind === "move"}
                aria-controls={openKind === "move" ? `move-${index}-form` : undefined}
                aria-label={`Move the cut between scenes ${number} and ${number + 1}`}
                onClick={onMove}
              >
                Move cut…
              </Button>
            ) : null}
          </div>
          {children}
        </div>
      ) : null}
    </li>
  );
}
