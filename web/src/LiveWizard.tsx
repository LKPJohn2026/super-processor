import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
  type MutableRefObject,
  type PointerEvent,
  type ReactNode,
} from "react";
import {
  cancelRun,
  editSegments,
  loadJobs,
  loadState,
  openJob,
  outputSizeLine,
  outputSrc,
  postAction,
  progressTiming,
  reopenScene,
  resumeRun,
  scaleChoiceText,
  runAnalyze,
  runRender,
  runSegmenting,
  type JobEntry,
  type LookShot,
  type OutputVersion,
  type Looks,
  type Progress,
  type Region,
  type Runner,
  type Segment,
  type SegmentList,
  type Navigation,
  type SegmentEdit,
  type SegmentPreview,
  withJob,
  type WizardView,
} from "./api";
import {
  interruptedChoice,
  offersRender,
  renderFailed,
  readyText,
  resultTitle,
  stopControl,
  stoppedNote,
} from "./runStates";
import {
  jobTitle,
  overallKey,
  sceneStrip,
  showsSidebar,
  steadyOverall,
  stripSummary,
} from "./jobView";
import {
  connectionState,
  documentTitle,
  headingKey,
  originHost,
  progressMilestone,
  type Milestone,
} from "./shellView";
import { AppShell } from "./shell/AppShell";
import { Sidebar } from "./shell/Sidebar";
import { TopBar } from "./shell/TopBar";
import { Banner, Button, Icon, Toggle } from "./ui";
import { newVideoRoute, sortJobs, startsOnJobs } from "./jobsList";
import {
  notifyPermission,
  notifyToggle,
  runNotice,
  saveChoice,
  savedChoice,
  shouldNotify,
} from "./runNotify";
import {
  downloadName,
  prefetchNote,
  versionLabel,
  versionMeta,
  versionNote,
  versionsNewestFirst,
  versionsSummary,
} from "./versions";
import { JobsHome } from "./JobsHome";
import { WelcomeStep } from "./Welcome";
import { ChooseStep } from "./ChooseStep";
import { FindingStep, ScenesProblemCard } from "./FindingStep";
import { scenesProblem } from "./findingView";
import { isWelcomeStep } from "./welcomeView";
import {
  STALE_SCENES_TEXT,
  STALE_SCENES_UNLOADED_TEXT,
  canSplit,
  scenesText,
  checkMerge,
  checkMove,
  checkSplit,
  cutName,
  cutRange,
  cutValueText,
  dragTime,
  formatTime,
  isDrag,
  isStaleScenes,
  limitsOf,
  nudgeCut,
  parseTime,
  splitRange,
  splitSuggestion,
  withCutAt,
  type Range,
  type SceneLimits,
} from "./sceneEdit";

const BACK_TEXT: Record<keyof Navigation, string> = {
  setup: "Back to Gemini setup",
  file: "Choose another file",
  segments: "Back to the scenes",
};

// How often, and how many times while none has started, the page looks
// for the next scene's background preview.
const PREFETCH_LOOK_MS = 2000;
const PREFETCH_WAITING_LOOKS = 6;

const RUNNERS: Partial<Record<WizardView["step"], Runner>> = {
  sampling: runSegmenting,
  proposing: runSegmenting,
  analyzing: runAnalyze,
  rendering: runRender,
};

type LiveWizardProps = {
  origin: string;
  view: WizardView;
  onView: (view: WizardView) => void;
  onDisconnect: () => void;
};

export function LiveWizard({ origin, view, onView, onDisconnect }: LiveWizardProps) {
  const [busy, setBusy] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const [runFailed, setRunFailed] = useState(false);
  const [recovery, setRecovery] = useState(0);
  // Failed polls in a row (the connection pill): 0 since the last answer.
  const [failures, setFailures] = useState(0);
  // Progress milestones, read out once each (s9).
  const [announce, setAnnounce] = useState("");
  const milestone = useRef<Milestone | null>(null);
  const connection = useRef<AbortController | null>(null);
  // An error to keep when the step it was said for renders: a refused
  // change reloads the job, which may land on another step, and the step
  // change must not wipe the reason the page changed. Used once.
  const carriedError = useRef<{ step: string; text: string } | null>(null);
  // The whole job's percentage last shown, kept here rather than in Busy so
  // it survives the steps between two runs (a scene looked at again must not
  // pull the bar back). Cleared when the job goes back to its scenes list.
  const overallShown = useRef<SteadyOverall | null>(null);
  if (view.step === "segments" || view.step === "pick_file") overallShown.current = null;
  // Set after a scene is opened again: once its step has rendered, focus
  // moves to that step's heading (the button that was pressed is gone).
  const [headingFocus, setHeadingFocus] = useState(0);
  // The "All jobs" home: "loading" while the page decides whether to open
  // on it (it does when it would show the file pick and jobs exist).
  const [home, setHome] = useState<"loading" | "shown" | "off">(() =>
    view.step === "pick_file" && !view.busy && !view.error && !view.stopped ? "loading" : "off",
  );
  const [jobs, setJobs] = useState<JobEntry[] | null>(null);
  const [jobsError, setJobsError] = useState<string | null>(null);
  const [jobsReload, setJobsReload] = useState(0);
  const [opening, setOpening] = useState<string | null>(null);
  const [openError, setOpenError] = useState<string | null>(null);
  const [jobsStatus, setJobsStatus] = useState("");
  const latestView = useRef(view);
  latestView.current = view;
  // "Notify me when it's done": off until the editor turns it on, and the
  // browser is asked only from that click (see NotifyToggle).
  const [notifyOn, setNotifyOn] = useState(() => savedChoice(browserStorage));
  const shownView = useRef<WizardView | null>(null);
  const notified = useRef(new Set<string>());
  useEffect(() => {
    const before = shownView.current;
    shownView.current = view;
    const notice = runNotice(before, view);
    if (!notice || notified.current.has(notice.tag)) return;
    if (!shouldNotify(notice, notifyOn, notifyPermission(window), document.hidden)) return;
    notified.current.add(notice.tag);
    try {
      const shown = new Notification(notice.title, { body: notice.body, tag: notice.tag });
      shown.onclick = () => {
        window.focus();
        shown.close();
      };
    } catch {
      // Some browsers (Chrome on Android) only notify from a service worker.
    }
  }, [view, notifyOn]);

  function changeNotify(on: boolean) {
    setNotifyOn(on);
    saveChoice(browserStorage, on);
  }
  const homeOn = home !== "off";
  useEffect(() => {
    if (!homeOn) return;
    let cancelled = false;
    let inFlight = false;
    const load = () => {
      if (inFlight) return;
      inFlight = true;
      loadJobs(origin)
        .then((list) => {
          if (cancelled) return;
          const sorted = sortJobs(list);
          setJobs(sorted);
          setJobsError(null);
          setHome((current) =>
            current === "loading" ? (startsOnJobs(latestView.current, sorted) ? "shown" : "off") : current,
          );
        })
        .catch((error: unknown) => {
          if (cancelled) return;
          setJobsError(error instanceof Error ? error.message : "the server did not answer");
          // Opening on the list is only a convenience: without it, the file pick.
          setHome((current) => (current === "loading" ? "off" : current));
        })
        .finally(() => {
          inFlight = false;
        });
    };
    load();
    // Running jobs change status on their own, here or in another window.
    const timer = window.setInterval(load, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [homeOn, origin, jobsReload]);

  function showHome() {
    setOpenError(null);
    setJobsStatus("");
    setHome("shown");
    setJobsReload((value) => value + 1);
    setHeadingFocus((value) => value + 1);
  }

  function leaveHome() {
    setOpenError(null);
    setHome("off");
    setHeadingFocus((value) => value + 1);
  }

  // Set synchronously, so two clicks in one tick send one request.
  const switching = useRef(false);

  async function openFromList(entry: JobEntry) {
    if (switching.current) return;
    switching.current = true;
    const current = connection.current;
    setOpening(entry.job_id);
    setOpenError(null);
    setJobsStatus(`Opening ${entry.title}…`);
    try {
      const next = await openJob(origin, entry.job_id);
      if (current?.signal.aborted) return;
      // The opened job starts clean: nothing from the last one carries over.
      carriedError.current = null;
      overallShown.current = null;
      setLocalError(null);
      onView(next);
      setJobsStatus(`Opened ${entry.title}.`);
      leaveHome();
    } catch (error) {
      if (current?.signal.aborted) return;
      const text = error instanceof Error ? error.message : "the server did not answer";
      setOpenError(`Could not open ${entry.title}: ${text}`);
      setJobsStatus(`Could not open ${entry.title}.`);
      setJobsReload((value) => value + 1);
    } finally {
      switching.current = false;
      setOpening(null);
    }
  }

  async function newVideo() {
    const route = newVideoRoute(view);
    if (route === "close") {
      leaveHome();
      return;
    }
    if (route === "blocked" || switching.current) return;
    switching.current = true;
    const current = connection.current;
    setOpening("");
    setOpenError(null);
    try {
      // /new leaves the open job as it is on disk (its place is kept), and
      // lands on the file pick, or on setup when there is no Gemini key.
      const next = await postAction(origin, "new", withJob(view));
      if (current?.signal.aborted) return;
      carriedError.current = null;
      overallShown.current = null;
      setLocalError(null);
      onView(next);
      if (next.error || (next.step !== "pick_file" && next.step !== "setup")) {
        setOpenError(next.error ?? "Could not start a new video.");
      } else {
        leaveHome();
      }
    } catch (error) {
      if (current?.signal.aborted) return;
      setOpenError(error instanceof Error ? error.message : "Could not start a new video.");
    } finally {
      switching.current = false;
      setOpening(null);
    }
  }

  useEffect(() => {
    if (!headingFocus) return;
    const heading = document.querySelector<HTMLElement>("#main h1");
    if (heading) {
      heading.tabIndex = -1;
      heading.focus();
    }
  }, [headingFocus]);

  useEffect(() => {
    const current = new AbortController();
    connection.current = current;
    return () => current.abort();
  }, [origin]);

  async function recover() {
    const current = connection.current;
    setBusy(true);
    const next = await loadState(origin);
    // A disconnected/replaced server must not update the new connection's view.
    if (current?.signal.aborted) return;
    setBusy(false);
    if (!next) {
      setFailures((value) => Math.max(value + 1, 2));
      setLocalError("Still unable to reach the wizard. Check its printed URL, then retry or disconnect.");
      return;
    }
    setFailures(0);
    carriedError.current = null;
    onView(next);
    setRunFailed(false);
    setLocalError(null);
    setRecovery((value) => value + 1);
  }

  // A POST refused because the page is out of date (409: another window
  // opened another job, or the run it names already ended): load the job as
  // it is now and say the server's reason on that view. False when the job
  // could not be loaded again.
  async function showCurrent(said: string): Promise<boolean> {
    const current = connection.current;
    const next = await loadState(origin);
    if (current?.signal.aborted) return true;
    if (!next) return false;
    carriedError.current = { step: next.step, text: said };
    onView(next);
    setLocalError(said);
    return true;
  }

  // Every request from the page goes through here: one at a time, the
  // answer shown, a refusal said.
  async function perform(request: (shown: WizardView) => Promise<WizardView>): Promise<boolean> {
    setBusy(true);
    setLocalError(null);
    carriedError.current = null;
    try {
      onView(await request(view));
      setFailures(0);
      return true;
    } catch (error) {
      // A request that never reached the server (fetch rejects with a TypeError).
      if (error instanceof TypeError) setFailures((value) => value + 1);
      const text = error instanceof Error ? error.message : "Request failed";
      const status = (error as { status?: unknown } | null)?.status;
      if (status !== 409 || !(await showCurrent(text))) setLocalError(text);
      return false;
    } finally {
      setBusy(false);
    }
  }

  // Every POST names the job this page shows, so a tab left open on an
  // older job is refused (409) instead of acting on the current one.
  function send(action: string, fields: Record<string, string>): Promise<boolean> {
    return perform((shown) => postAction(origin, action, withJob(shown, fields)));
  }

  // Load the job as it is now (the "Working in another window" screen).
  async function checkAgain(): Promise<boolean> {
    const current = connection.current;
    setBusy(true);
    setLocalError(null);
    const next = await loadState(origin);
    if (current?.signal.aborted) return false;
    setBusy(false);
    if (!next) {
      setLocalError("Could not reach the wizard. Check that it still runs, then try again.");
      return false;
    }
    carriedError.current = null;
    onView(next);
    return true;
  }

  // A change to the segments (or approving them) names the list the page
  // shows. When another window changed them first the server refuses it;
  // the page then shows the scenes as they are now and says why.
  async function editScenes(edit: SegmentEdit): Promise<boolean> {
    setBusy(true);
    setLocalError(null);
    carriedError.current = null;
    try {
      const next = await editSegments(origin, view, edit);
      onView(next);
      // A change the rules refuse comes back as the view's error.
      return !next.error;
    } catch (error) {
      const text = error instanceof Error ? error.message : "Request failed";
      const status = (error as { status?: unknown } | null)?.status;
      if (isStaleScenes(text, typeof status === "number" ? status : undefined)) {
        const said = status === 409 ? text : STALE_SCENES_TEXT;
        // The page could not load the scenes again, so it must not say
        // it shows them as they are.
        if (!(await showCurrent(said))) setLocalError(status === 409 ? text : STALE_SCENES_UNLOADED_TEXT);
      } else {
        setLocalError(text);
      }
      return false;
    } finally {
      setBusy(false);
    }
  }

  // The next scene's preview made in the background (--prefetch): the view
  // only changes when the page asks, so while it is on its way the page
  // looks again every two seconds. A few looks while none has started yet,
  // for as long as one runs; never with ?run=1, so nothing starts from here.
  const busyNow = useRef(busy);
  busyNow.current = busy;
  const shownPreview = view.step === "preview_segment" ? view.segment_preview : null;
  const prefetchNext = shownPreview?.next;
  const prefetchWait =
    !shownPreview || prefetchNext === undefined
      ? null
      : prefetchNext === null
        ? "waiting"
        : prefetchNext.running
          ? "running"
          : null;
  const previewIndex = shownPreview?.index;
  useEffect(() => {
    if (!prefetchWait) return;
    let cancelled = false;
    let timer = 0;
    let looks = 0;
    const look = async () => {
      if (prefetchWait === "waiting" && looks >= PREFETCH_WAITING_LOOKS) return;
      looks += 1;
      if (!busyNow.current) {
        const before = latestView.current;
        const next = await loadState(origin);
        if (cancelled) return;
        // A request the page sent meanwhile has the newer word.
        if (next && !busyNow.current && latestView.current === before) onView(next);
      }
      if (!cancelled) timer = window.setTimeout(() => void look(), PREFETCH_LOOK_MS);
    };
    timer = window.setTimeout(() => void look(), PREFETCH_LOOK_MS);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [origin, prefetchWait, previewIndex, view.job_id, onView]);

  useEffect(() => {
    const run = RUNNERS[view.step];
    setRunFailed(false);
    // A new step starts clean: an error from the last one (say a stop that
    // came too late) must not show on the step the run went on to, except
    // the reason a refused change took the page to this step.
    const carried = carriedError.current;
    carriedError.current = null;
    setLocalError(carried && carried.step === view.step ? carried.text : null);
    // An interrupted step waits for Resume or Stop (see Interrupted).
    if (!run || view.interrupted) return;
    let cancelled = false;
    const subscription = new AbortController();
    // Each poll while the worker runs carries its progress.
    const progress = (next: WizardView) => {
      if (cancelled) return;
      setFailures(0);
      onView(next);
    };
    run(origin, progress, subscription.signal)
      .then((next) => {
        if (cancelled) return;
        setFailures(0);
        onView(next);
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setRunFailed(true);
          setFailures((value) => Math.max(value, 2));
          const detail = error instanceof Error ? error.message : "The step failed";
          setLocalError(
            `Progress updates stopped: ${detail}. Work may still be running on the server. Retry the connection to refresh its state.`,
          );
        }
      });
    return () => {
      cancelled = true;
      subscription.abort();
    };
  }, [origin, view.step, view.interrupted, onView, recovery]);

  const lost = failures > 0;
  // Whether this page can reach the wizard, for the top bar's pill: a
  // quiet look at /api/state every 10 s (its answer is not applied, so it
  // never races the step's own requests). Skipped while the tab is hidden.
  useEffect(() => {
    let cancelled = false;
    const look = async () => {
      if (document.hidden) return;
      const next = await loadState(origin);
      if (cancelled) return;
      setFailures((value) => (next ? 0 : value + 1));
    };
    // Once a look has failed, look sooner, so the page is usable again soon
    // after the wizard comes back.
    const timer = window.setInterval(() => void look(), lost ? 4_000 : 10_000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [origin, lost]);

  // Read out a run's progress at milestones only: each stage, each 10%.
  const runTitle =
    view.interrupted || !RUNNERS[view.step]
      ? null
      : view.step === "rendering"
        ? "Render"
        : view.step === "analyzing"
          ? `Scene ${(view.segment_index ?? 0) + 1}`
          : "Finding the scenes";
  useEffect(() => {
    if (!runTitle) {
      milestone.current = null;
      return;
    }
    const next = progressMilestone(milestone.current, runTitle, view.progress);
    if (!next) return;
    milestone.current = next;
    setAnnounce(next.text);
  }, [runTitle, view.progress]);

  const message = localError || view.error;
  const conn = connectionState(failures);
  const homeShown = home !== "off";
  const failedStep = Boolean(view.error) && (renderFailed(view) || view.run?.outcome === "failed");
  const offline = runFailed || conn === "err";
  // While the wizard can't be reached (S2) the step stays on screen, dimmed,
  // with its actions turned off; what was typed into its fields stays (they
  // are not disabled), to send once it is back.
  const locked = busy || offline;

  const topBar = (
    <TopBar
      jobTitle={homeShown ? null : jobTitle(view)}
      connection={conn}
      onAllJobs={homeShown ? null : showHome}
    />
  );

  if (homeShown) {
    const listed = jobs?.some((entry) => entry.open) ?? false;
    return (
      <AppShell
        topBar={topBar}
        title={documentTitle(view, true)}
        headingKey={headingKey(view, true)}
        announce={announce}
        wide
      >
        <JobsHome
          origin={origin}
          view={view}
          jobs={jobs}
          loadError={jobsError}
          opening={opening}
          openError={openError}
          status={jobsStatus}
          backLabel={view.job_id && !listed && home === "shown" ? "Back to this job" : null}
          onOpen={(entry) => void openFromList(entry)}
          onShowCurrent={leaveHome}
          onNewVideo={() => void newVideo()}
          onRefresh={() => setJobsReload((value) => value + 1)}
          onDisconnect={onDisconnect}
        />
      </AppShell>
    );
  }

  const setupLink = Boolean(view.navigation?.setup) && !locked && !view.busy;
  const sidebar = showsSidebar(view) ? (
    <Sidebar
      origin={origin}
      view={view}
      busy={locked}
      failed={failedStep}
      onGoto={(target) => void send("goto", { to: target })}
      links={
        setupLink ? (
          <button type="button" className="side-link" onClick={() => void send("goto", { to: "setup" })}>
            <Icon name="key" size={18} />
            Gemini key settings
          </button>
        ) : null
      }
    />
  ) : null;

  return (
    <AppShell
      topBar={topBar}
      sidebar={sidebar}
      title={documentTitle(view)}
      headingKey={headingKey(view)}
      announce={announce}
      busy={busy}
      wide={!sidebar}
    >
      {offline ? (
        <Banner
          tone="error"
          role="alert"
          title={`Can’t reach the wizard at ${originHost(origin, window.location.origin)}`}
          actions={
            <>
              <Button kind="primary" size="md" icon="rotate" disabled={busy} onClick={() => void recover()}>
                {busy ? "Reconnecting…" : "Try again"}
              </Button>
              <Button kind="secondary" size="md" onClick={onDisconnect}>
                Change address
              </Button>
            </>
          }
        >
          It may have stopped. Your work is safe — anything that was running keeps running. Anything you typed
          stays on this page: send it once the wizard is back.
        </Banner>
      ) : null}
      <div
        className={["step-body", isWelcomeStep(view.step) ? "welcome-body" : null, offline ? "offline-dim" : null]
          .filter(Boolean)
          .join(" ")}
      >
        <WizardStepView
          origin={origin}
          view={view}
          busy={locked}
          message={message}
          onSend={send}
          onRequest={perform}
          onCheckAgain={checkAgain}
          onEditScenes={editScenes}
          onView={onView}
          overallShown={overallShown}
          notify={<NotifyToggle on={notifyOn} onChange={changeNotify} />}
        />
        <SceneStrip
          origin={origin}
          view={view}
          busy={locked}
          onRequest={perform}
          onReopened={() => setHeadingFocus((value) => value + 1)}
        />
      </div>
    </AppShell>
  );
}

// Resolves true when the server took the action (false: the error shows).
type SendAction = (action: string, fields: Record<string, string>) => Promise<boolean | void>;
type EditScenes = (edit: SegmentEdit) => Promise<boolean>;
// A request made with one of api.ts's helpers, given the view the page shows.
type RequestAction = (request: (shown: WizardView) => Promise<WizardView>) => Promise<boolean>;
type SteadyOverall = { key: string; value: number };

function browserStorage(): Storage | null {
  return typeof window === "undefined" ? null : window.localStorage;
}

/**
 * "Notify me when it's done", while the scenes are found or the render runs.
 * Turning it on asks the browser's permission, from this click only. Hidden
 * where the browser has no notifications.
 */
function NotifyToggle({ on, onChange }: { on: boolean; onChange: (on: boolean) => void }) {
  // The permission is read on every paint (the view polls), so a change made
  // in the browser's settings shows up; `asked` repaints after a request.
  const [, setAsked] = useState(0);
  const [refused, setRefused] = useState(false);
  const permission = notifyPermission(typeof window === "undefined" ? undefined : window);
  const toggle = notifyToggle(on, permission);
  if (!toggle.shown) return null;

  async function change(checked: boolean) {
    if (!checked) {
      onChange(false);
      return;
    }
    let answer: string = Notification.permission;
    if (answer === "default") {
      try {
        // Older Safari answers through a callback, not a promise.
        answer = await new Promise<string>((resolve) => {
          const asked = Notification.requestPermission(resolve);
          if (asked) void asked.then(resolve, () => resolve("default"));
        });
      } catch {
        answer = "default";
      }
    }
    setAsked((value) => value + 1);
    setRefused(answer !== "granted");
    onChange(answer === "granted");
  }

  const hint = toggle.hint ?? (refused ? "Notifications stay off: the browser was not allowed to show them." : null);
  return (
    <div className="notify-toggle">
      <Toggle
        id="notify-done"
        checked={toggle.checked}
        disabled={toggle.disabled}
        hintId="notify-hint"
        onChange={(checked) => void change(checked)}
      >
        <Icon name="bell" size={18} />
        Notify me when it’s done
      </Toggle>
      <p id="notify-hint" className="muted notify-hint" role="status">
        {hint}
      </p>
    </div>
  );
}

/** Every scene of the job with its state; approved scenes can be opened again. */
function SceneStrip({
  origin,
  view,
  busy,
  onRequest,
  onReopened,
}: {
  origin: string;
  view: WizardView;
  busy: boolean;
  onRequest: RequestAction;
  onReopened: () => void;
}) {
  const scenes = sceneStrip(view);
  const list = useRef<HTMLOListElement | null>(null);
  const current = scenes.find((scene) => scene.status === "current")?.index;
  // On a narrow screen the strip scrolls sideways: keep the scene being
  // looked at in sight, without scrolling the page.
  useEffect(() => {
    const strip = list.current;
    if (!strip || current === undefined || strip.scrollWidth <= strip.clientWidth) return;
    const pill = strip.querySelector<HTMLElement>(`[data-scene="${current}"]`);
    if (pill) strip.scrollLeft = Math.max(0, pill.offsetLeft - strip.clientWidth / 2 + pill.offsetWidth / 2);
  }, [current, scenes.length]);
  const summary = stripSummary(view);
  const problem = view.job?.error;
  if (!scenes.length) {
    return problem ? (
      <p className="muted strip-problem">Part of this job could not be read: {problem}</p>
    ) : null;
  }
  const locked = busy || Boolean(view.busy);
  async function reopen(index: number) {
    // Focus moves only when the scene opened; a refusal keeps it here, by
    // the error. The move itself waits for the new step to render.
    if (await onRequest((shown) => reopenScene(origin, shown, index))) onReopened();
  }
  return (
    <section className="scene-strip" aria-labelledby="scene-strip-title">
      <div className="strip-head">
        <h2 id="scene-strip-title">Scenes</h2>
        <p className="muted" role="status">
          {summary}
        </p>
      </div>
      <ol ref={list} tabIndex={0} aria-label="Scenes of this job">
        {scenes.map((scene) => {
          const still = outputSrc(origin, scene.stillUrl);
          return (
            <li
              key={scene.index}
              data-scene={scene.index}
              className={`scene-pill ${scene.status}`}
              aria-current={scene.status === "current" ? "step" : undefined}
            >
              {still ? <img src={still} alt="" loading="lazy" /> : <span className="still-gap" aria-hidden="true" />}
              <span className="scene-text">
                <strong className="scene-name">{scene.name}</strong>
                <span className="scene-range">{scene.range}</span>
                <span className="scene-status">
                  <span className="sdot" aria-hidden="true" />
                  {scene.statusText}
                  {scene.previewed && scene.status !== "pending" ? (
                    <span className="previewed"> · preview ready</span>
                  ) : null}
                </span>
                {scene.reopen ? (
                  <button
                    type="button"
                    className="secondary small"
                    disabled={locked}
                    aria-label={scene.reopen.label}
                    onClick={() => void reopen(scene.index)}
                  >
                    {scene.reopen.text}
                  </button>
                ) : null}
              </span>
            </li>
          );
        })}
      </ol>
      {problem ? <p className="muted strip-problem">Part of this job could not be read: {problem}</p> : null}
    </section>
  );
}

function WizardStepView({
  origin,
  view,
  busy,
  message,
  onSend,
  onRequest,
  onCheckAgain,
  onEditScenes,
  onView,
  overallShown,
  notify,
}: {
  origin: string;
  view: WizardView;
  busy: boolean;
  message: string | null;
  onSend: SendAction;
  onRequest: RequestAction;
  onCheckAgain: () => Promise<boolean>;
  onEditScenes: EditScenes;
  onView: (view: WizardView) => void;
  overallShown: MutableRefObject<SteadyOverall | null>;
  // "Notify me when it's done", for the long runs.
  notify?: ReactNode;
}) {
  if (isWelcomeStep(view.step)) {
    return <WelcomeStep view={view} busy={busy} message={message} onSend={onSend} />;
  }

  if (view.step === "pick_file") {
    return (
      <ChooseStep
        origin={origin}
        busy={busy}
        message={message}
        onSend={onSend}
        onView={onView}
        stopped={<StoppedNote view={view} />}
        back={<Back view={view} to={["setup", "segments"]} busy={busy} onSend={onSend} />}
      />
    );
  }

  if (view.interrupted && RUNNERS[view.step]) {
    return (
      <Interrupted
        origin={origin}
        view={view}
        busy={busy}
        message={message}
        onRequest={onRequest}
        onCheckAgain={onCheckAgain}
      />
    );
  }

  if (view.step === "sampling" || view.step === "proposing") {
    return (
      <FindingStep
        view={view}
        message={message}
        stop={
          <StopButton origin={origin} view={view} busy={busy} onRequest={onRequest} idleLabel="Cancel" />
        }
        notify={notify}
      />
    );
  }

  if (view.step === "segments") {
    return (
      <SegmentsStep
        origin={origin}
        list={view.segments ?? { segments: [], error: null, note: "" }}
        busy={busy}
        message={message}
        onSend={onSend}
        onEdit={onEditScenes}
        view={view}
      />
    );
  }

  if (view.step === "analyzing") {
    const count = view.job?.scene_count || view.segments?.segments.length;
    return (
      <Busy
        title={`Scene ${(view.segment_index ?? 0) + 1}${count ? ` of ${count}` : ""}`}
        progress={view.progress ?? null}
        overallKey={overallKey(view)}
        overallShown={overallShown}
        message={message}
      >
        Gemini is choosing clean-up, strength, and finishing for this scene.
        A few seconds of it are previewed on the local GPU, and Gemini checks
        the preview for faces, hands, text, and texture that went wrong.
      </Busy>
    );
  }

  if (view.step === "preview_segment") {
    return (
      <PreviewStep
        origin={origin}
        preview={
          view.segment_preview ?? { index: 0, count: 0, scale: null, error: null, shot: null }
        }
        busy={busy}
        message={message}
        onSend={onSend}
        view={view}
      />
    );
  }

  if (view.step === "ready_to_render") {
    return (
      <ReadyStep
        looks={view.looks ?? { scale: null, error: null, shots: [] }}
        busy={busy}
        message={message}
        onSend={onSend}
        view={view}
      />
    );
  }

  if (view.step === "rendering") {
    return (
      <Busy
        title="Upscaling"
        progress={view.progress ?? null}
        overallKey={overallKey(view)}
        overallShown={overallShown}
        message={message}
        stop={<StopButton origin={origin} view={view} busy={busy} onRequest={onRequest} />}
        notify={notify}
      >
        Restoring and upscaling on the local GPU. FFmpeg only trims, splices,
        and copies audio. This can take several minutes.
      </Busy>
    );
  }

  if (view.step === "result") {
    return <ResultStep origin={origin} view={view} busy={busy} message={message} onSend={onSend} />;
  }

  if (view.step === "done") {
    return (
      <>
        <h1>Done</h1>
        <p className="lede">
          Your enhanced file is in the job directory as <code>output.mp4</code>.
        </p>
        {view.output_ready ? (
          <video
            controls
            src={outputSrc(origin, view.output_url) ?? undefined}
            aria-label="Upscaled result"
          />
        ) : null}
        <Versions origin={origin} view={view} />
        <div className="actions">
          <button
            type="button"
            className="secondary"
            disabled={busy}
            onClick={() => void onSend("new", {})}
          >
            Start a new video
          </button>
        </div>
      </>
    );
  }

  return (
    <>
      <h1>Continue in the classic wizard</h1>
      <p>
        This step ({view.step}) is served by the Python wizard page. Open the
        URL printed by <code>super-processor review</code> to finish it. This
        shell drives every step from introduction to result.
      </p>
      {message ? <Alert text={message} /> : null}
    </>
  );
}

function Busy({
  title,
  progress,
  overallKey: steadyKey,
  overallShown,
  message,
  stop,
  notify,
  children,
}: {
  title: string;
  progress: Progress | null;
  // The run the whole job's percentage is kept steady for.
  overallKey?: string;
  // Where the last shown percentage is kept (the page's, so it outlives Busy).
  overallShown?: MutableRefObject<SteadyOverall | null>;
  message: string | null;
  stop?: ReactNode;
  notify?: ReactNode;
  children: ReactNode;
}) {
  const pct = progress ? Math.max(0, Math.min(100, Math.round(progress.pct))) : null;
  // The whole job's percentage never goes back while one run works, even
  // when a poll arrives late or a phase reports a lower estimate.
  const own = useRef<SteadyOverall | null>(null);
  const shown = overallShown ?? own;
  let overall: number | null = null;
  if (typeof progress?.overall === "number") {
    shown.current = steadyOverall(shown.current, steadyKey ?? "", progress.overall);
    overall = Math.round(shown.current.value);
  }
  // The view arrives every two seconds; count the clock down between them.
  // The tick names the view it was counted from, so a new view starts at
  // zero on its first paint instead of with the old view's seconds.
  const [tick, setTick] = useState<{ of: Progress | null; since: number }>({ of: null, since: 0 });
  useEffect(() => {
    if (!progress) return undefined;
    const arrived = Date.now();
    const timer = window.setInterval(
      () => setTick({ of: progress, since: (Date.now() - arrived) / 1000 }),
      1000,
    );
    return () => window.clearInterval(timer);
  }, [progress]);
  const since = tick.of === progress ? tick.since : 0;
  const timing = progress ? progressTiming(progress, since) : "";
  const step = progress ? `${pct}% · ${progress.message}${timing ? ` · ${timing}` : ""}` : "";
  return (
    <>
      <h1>{title}</h1>
      <p className="lede" role="status">
        {children}
      </p>
      {overall !== null ? (
        <>
          <p className="progress">
            <progress max={100} value={overall} aria-label={`Whole job: ${overall}%`}>
              {overall}%
            </progress>{" "}
            <strong>{overall}% of the job</strong>
          </p>
          <p className="progress muted">This step: {step}</p>
        </>
      ) : pct !== null ? (
        <p className="progress">
          <progress max={100} value={pct} aria-label={`${title}: ${pct}%`}>
            {pct}%
          </progress>{" "}
          {step}
        </p>
      ) : null}
      {/* Always in the page, so a note that appears is read out. */}
      <div role="status">{progress?.note ? <p className="notice">{progress.note}</p> : null}</div>
      {message ? <Alert text={message} /> : null}
      {notify}
      {stop}
    </>
  );
}

function StopButton({
  origin,
  view,
  busy,
  onRequest,
  idleLabel,
}: {
  origin: string;
  view: WizardView;
  busy: boolean;
  onRequest: RequestAction;
  // The mockup's word for the button while nothing is stopping (03: "Cancel").
  idleLabel?: string;
}) {
  const control = stopControl(view, busy);
  const stopping = Boolean(view.cancel_requested) || busy;
  return (
    <>
      {/* Always in the page, so the "Stopping" line is announced when it appears. */}
      <div role="status">{control.status ? <p className="stopped-note">{control.status}</p> : null}</div>
      {control.shown ? (
        <div className="actions">
          <Button
            kind="secondary"
            size="md"
            disabled={control.disabled}
            // The run the page shows, so a late click never stops a newer one.
            onClick={() => void onRequest((shown) => cancelRun(origin, shown))}
          >
            {idleLabel && !stopping ? idleLabel : control.label}
          </Button>
        </div>
      ) : null}
    </>
  );
}

/** After a stop: a plain note on the step the run went back to, not an error. */
function StoppedNote({ view }: { view: WizardView }) {
  const text = stoppedNote(view);
  // The live region is always there, so a note that appears is read out.
  return <div role="status">{text ? <p className="stopped-note">{text}</p> : null}</div>;
}

function Back({
  view,
  to,
  busy,
  onSend,
}: {
  view: WizardView;
  to: (keyof Navigation)[];
  busy: boolean;
  onSend: SendAction;
}) {
  const shown = to.filter((target) => view.navigation?.[target]);
  if (!shown.length) return null;
  return (
    <div className="actions back">
      {shown.map((target) => (
        <button
          key={target}
          type="button"
          className="link"
          disabled={busy}
          onClick={() => void onSend("goto", { to: target })}
        >
          ← {BACK_TEXT[target]}
        </button>
      ))}
    </div>
  );
}

function Alert({ text }: { text: string }) {
  return (
    <p className="error" role="alert">
      {text}
    </p>
  );
}

function ResultStep({
  origin,
  view,
  busy,
  message,
  onSend,
}: {
  origin: string;
  view: WizardView;
  busy: boolean;
  message: string | null;
  onSend: SendAction;
}) {
  const [note, setNote] = useState("");
  const src = outputSrc(origin, view.output_url);

  function submit(event: FormEvent) {
    event.preventDefault();
    void onSend("result", { note });
  }

  return (
    <>
      <h1>{resultTitle(view)}</h1>
      <StoppedNote view={view} />
      {view.scale !== null && view.strength !== null ? (
        <p className="muted">
          Scale {view.scale}× ·{" "}
          {view.strength_summary ?? `strength ${view.strength.toFixed(2)}`}.
          Lower strength means less invented texture.
        </p>
      ) : null}
      {src ? (
        <video controls src={src} aria-label="Upscaled result">
          <p>
            Your browser cannot play this video. The file is served at{" "}
            <a href={src}>output.mp4</a>.
          </p>
        </video>
      ) : (
        <p>The result file is not ready yet.</p>
      )}
      <SeamNote seams={view.seams ?? []} error={view.seam_error ?? null} />
      <Versions origin={origin} view={view} />
      <p className="lede">Are you happy, or do you want to say something?</p>
      {message ? <Alert text={message} /> : null}
      <div className="actions">
        <button
          type="button"
          disabled={busy}
          onClick={() => void onSend("result", { mood: "happy" })}
        >
          I am happy
        </button>
      </div>
      <form onSubmit={submit}>
        <label htmlFor="revise-note">
          Tell me what to change
          <textarea
            id="revise-note"
            name="note"
            rows={3}
            value={note}
            onChange={(event) => setNote(event.target.value)}
            placeholder="e.g. reduce artificial detail from 00:30 to 00:40"
          />
        </label>
        <div className="actions">
          <button type="submit" disabled={busy}>
            Revise this range
          </button>
          <button
            type="button"
            className="secondary"
            disabled={busy}
            onClick={() => void onSend("result", { note, restart: "1" })}
          >
            Start over from the scenes with this note
          </button>
        </div>
      </form>
      <div className="actions">
        <button
          type="button"
          className="secondary"
          disabled={busy}
          onClick={() => void onSend("new", {})}
        >
          Start a new video
        </button>
      </div>
    </>
  );
}

/**
 * Every kept result of the job, newest first, each with a player that opens
 * in place and a download link. The one the job's output is now is marked.
 */
function Versions({ origin, view }: { origin: string; view: WizardView }) {
  const [open, setOpen] = useState<number | null>(null);
  const list = versionsNewestFirst(view.versions);
  if (!list.length) return null;
  const now = Date.now();
  return (
    <section className="versions" aria-labelledby="versions-title">
      <h2 id="versions-title">Versions</h2>
      <p className="muted versions-summary">{versionsSummary(view.versions)}</p>
      <ol className="versions-list">
        {list.map((version) => (
          <VersionRow
            key={version.number}
            origin={origin}
            title={view.job?.title ?? null}
            version={version}
            now={now}
            open={open === version.number}
            onToggle={() => setOpen((shown) => (shown === version.number ? null : version.number))}
          />
        ))}
      </ol>
    </section>
  );
}

function VersionRow({
  origin,
  title,
  version,
  now,
  open,
  onToggle,
}: {
  origin: string;
  title: string | null;
  version: OutputVersion;
  now: number;
  open: boolean;
  onToggle: () => void;
}) {
  const src = outputSrc(origin, version.url) ?? undefined;
  const label = versionLabel(version);
  const note = versionNote(version);
  const playerId = `version-player-${version.number}`;
  return (
    <li className={`version-row${version.current ? " current" : ""}`}>
      <div className="version-text">
        <p className="version-name">
          <strong>{label}</strong>
          {version.current ? <span className="badge ok">Current</span> : null}
        </p>
        <p className="version-meta">{versionMeta(version, now)}</p>
        {note ? <p className="version-note">{note}</p> : null}
      </div>
      <div className="version-actions">
        <button
          type="button"
          className="secondary small"
          aria-expanded={open}
          aria-controls={playerId}
          aria-label={`${open ? "Hide" : "Play"} ${label.toLowerCase()}`}
          onClick={onToggle}
        >
          {open ? "Hide" : "Play"}
        </button>
        <a
          className="button secondary small"
          href={src}
          download={downloadName(title, version)}
          aria-label={`Download ${label.toLowerCase()}`}
        >
          Download
        </a>
      </div>
      <div id={playerId} className="version-player" hidden={!open}>
        {open ? <video controls preload="metadata" src={src} aria-label={label} /> : null}
      </div>
    </li>
  );
}

function words(items: string[]): string {
  return items.map((item) => item.replace(/_/g, " ")).join(", ");
}

function Notice({ text }: { text: string | null }) {
  if (!text) return null;
  return (
    <p className="notice" role="status">
      {text}
    </p>
  );
}

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
              className="cut-block"
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
                {formatTime(segment.end_s)}
              </span>
            </div>
          );
        })}
      </div>
      <div className="cut-scale" aria-hidden="true">
        <span>0:00</span>
        <span>{formatTime(total)}</span>
      </div>
      <p id="cut-help" className="muted cut-help">
        Drag a cut to move it, or use "Move cut…" under a scene to type a time.
        With the keyboard, focus a cut and use the arrow keys (Page Up and Page
        Down move it 10 s), then press Enter to save it or Esc to put it back.
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
      <div className="actions">
        <button type="submit" className="small" disabled={locked}>
          {submitText}
        </button>
        <button type="button" className="secondary small" onClick={onClose}>
          Cancel
        </button>
      </div>
    </form>
  );
}

function SegmentsStep({
  origin,
  list,
  busy,
  message,
  onSend,
  onEdit,
  view,
}: {
  origin: string;
  list: SegmentList;
  busy: boolean;
  message: string | null;
  onSend: SendAction;
  onEdit: EditScenes;
  view: WizardView;
}) {
  const [note, setNote] = useState("");
  // What the last change by hand did, or why it could not be made.
  const [said, setSaid] = useState<{ text: string; refused: boolean } | null>(null);
  // The time form open under a scene: split it, or move the cut after it.
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

  useEffect(() => {
    if (focusScene === null) return;
    sceneList.current?.querySelector<HTMLElement>(`[data-scene-title="${focusScene}"]`)?.focus();
    setFocusScene(null);
  }, [focusScene, list.revision]);

  // A new list (a change, or a reload) closes a time form left open on it.
  useEffect(() => setOpen(null), [list.revision]);

  function propose(event: FormEvent) {
    event.preventDefault();
    void onEdit({ action: "note", note });
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

  return (
    <>
      <h1>Scenes</h1>
      <p className="lede">
        {list.error
          ? "Gemini could not split the video, so it is one scene for now. Approve it to go on, or ask Gemini to try again."
          : "Gemini split the video where the picture needs different settings. Approve the scenes, change the cuts yourself, or say where they should go and Gemini will try again."}
      </p>
      <StoppedNote view={view} />
      {message ? <Alert text={message} /> : null}
      <ScenesProblemCard list={list} locked={locked} onEdit={onEdit} />
      <Notice
        text={
          [
            scenesProblem(list) ? "" : list.error,
            list.video_problem ? `Gemini saw stills instead of the video: ${list.video_problem}.` : "",
          ]
            .filter(Boolean)
            .join(" ") || null
        }
      />
      {scaleChoiceText(list.scale, list.scale_reason) ? (
        <p className="muted">{scaleChoiceText(list.scale, list.scale_reason)}</p>
      ) : null}
      {editable ? (
        <section className="cut-editor" aria-labelledby="cut-editor-title">
          <h2 id="cut-editor-title">Timeline</h2>
          <p className="muted">
            Move, merge or split the cuts yourself. Each scene needs at least{" "}
            {limits.minS} s, and there can be up to {limits.maxScenes} scenes.
          </p>
          <CutTimeline
            origin={origin}
            segments={segments}
            limits={limits}
            locked={locked}
            onRefused={refused}
            onMove={move}
          />
        </section>
      ) : null}
      <p className={said?.refused ? "error edit-status" : "muted edit-status"} role="status">
        {said ? said.text : scenesText(count)}
      </p>
      <ol className="shots" ref={sceneList}>
        {segments.map((segment, index) => (
          <li key={index} className="shot">
            {segment.still_url ? (
              <img
                src={outputSrc(origin, segment.still_url) ?? undefined}
                alt={`Still from scene ${index + 1}`}
              />
            ) : null}
            <div>
              <h2 tabIndex={-1} data-scene-title={index}>
                {index + 1}. {segment.label || "Unlabelled scene"}
              </h2>
              <p className="muted">
                {formatTime(segment.start_s)}–{formatTime(segment.end_s)} ·{" "}
                {(segment.end_s - segment.start_s).toFixed(1)} s
              </p>
              <p>
                {segment.contains.length ? (
                  segment.contains.map((item) => (
                    <span key={item} className="chip">
                      {item.replace(/_/g, " ")}
                    </span>
                  ))
                ) : (
                  <span className="muted">Nothing to protect</span>
                )}
              </p>
              {editable ? (
                <div className="actions scene-edits">
                  {index < count - 1 ? (
                    <button
                      type="button"
                      className="secondary small"
                      disabled={locked}
                      aria-label={`Merge scenes ${index + 1} and ${index + 2}`}
                      onClick={() => merge(index)}
                    >
                      Merge with {index + 2}
                    </button>
                  ) : null}
                  <button
                    type="button"
                    className="secondary small"
                    disabled={locked}
                    aria-expanded={open?.kind === "split" && open.index === index}
                    aria-controls={open?.kind === "split" && open.index === index ? `split-${index}-form` : undefined}
                    aria-label={`Split scene ${index + 1}`}
                    onClick={() => openSplit(index)}
                  >
                    Split…
                  </button>
                  {index < count - 1 ? (
                    <button
                      type="button"
                      className="secondary small"
                      disabled={locked}
                      aria-expanded={open?.kind === "move" && open.index === index}
                      aria-controls={open?.kind === "move" && open.index === index ? `move-${index}-form` : undefined}
                      aria-label={`Move the cut between scenes ${index + 1} and ${index + 2}`}
                      onClick={() => openMove(index)}
                    >
                      Move cut…
                    </button>
                  ) : null}
                </div>
              ) : null}
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
            </div>
          </li>
        ))}
      </ol>
      <div className="actions">
        <button type="button" disabled={locked} onClick={() => void onEdit({ action: "approve" })}>
          Approve {scenesText(count)}
        </button>
      </div>
      <form onSubmit={propose}>
        <label htmlFor="segments-note">
          Not right? Say what to change
          <input
            id="segments-note"
            type="text"
            value={note}
            placeholder="e.g. cut where we walk outside"
            onChange={(event) => setNote(event.target.value)}
          />
        </label>
        <div className="actions">
          <button type="submit" className="secondary" disabled={locked}>
            Propose again
          </button>
        </div>
      </form>
      <Back view={view} to={["file"]} busy={busy} onSend={onSend} />
    </>
  );
}

const NEUTRAL: Record<string, number> = {
  deblock: 0,
  denoise: 0,
  contrast: 1,
  brightness: 0,
  saturation: 1,
  gamma: 1,
  grain: 0,
};

export function settingsSummary(strength: number, look: Record<string, number>): string {
  const parts = [`strength ${strength.toFixed(2)}`];
  for (const [name, neutral] of Object.entries(NEUTRAL)) {
    const value = look[name] ?? neutral;
    if (Math.abs(value - neutral) > 1e-6) parts.push(`${name} ${value.toFixed(2)}`);
  }
  return parts.join(" · ");
}

function Boxes({ regions }: { regions: Region[] }) {
  return (
    <>
      {regions.map((region, i) => (
        <span
          key={`${region.kind}-${i}`}
          className={`box ${region.kind}`}
          title={region.kind}
          style={{
            left: `${region.x0 * 100}%`,
            top: `${region.y0 * 100}%`,
            width: `${(region.x1 - region.x0) * 100}%`,
            height: `${(region.y1 - region.y0) * 100}%`,
          }}
        />
      ))}
    </>
  );
}

function Protected({ regions, contains }: { regions: Region[]; contains: string[] }) {
  const kinds = [...new Set(regions.map((region) => region.kind))].sort();
  if (kinds.length) {
    return <p className="muted">Protected inside the boxes: {kinds.join(", ")}.</p>;
  }
  const flagged = contains.filter((item) => ["faces", "hands", "text"].includes(item));
  if (!flagged.length) return null;
  return (
    <p className="muted">
      Contains {flagged.join(", ")} but no boxes were placed, so the whole shot
      is held to a lower strength.
    </p>
  );
}

function SeamNote({
  seams,
  error,
}: {
  seams: { time_s: number; score: number }[];
  error: string | null;
}) {
  if (error) return <p className="muted">{error}</p>;
  if (!seams.length) return null;
  const listed = seams.map((seam) => formatTime(seam.time_s)).join(", ");
  const first = seams[0].time_s;
  const example = `smooth ${formatTime(Math.max(0, first - 1))} to ${formatTime(first + 1)}`;
  return (
    <p className="muted">
      Possible seam{seams.length > 1 ? "s" : ""} at {listed}: the picture jumps
      there and the source does not. To re-render across one in a single pass,
      write a note like “{example}”.
    </p>
  );
}

function Interrupted({
  origin,
  view,
  busy,
  message,
  onRequest,
  onCheckAgain,
}: {
  origin: string;
  view: WizardView;
  busy: boolean;
  message: string | null;
  onRequest: RequestAction;
  onCheckAgain: () => Promise<boolean>;
}) {
  // What the last "Check again" found, read out.
  const [checked, setChecked] = useState<string | null>(null);
  // Nothing runs until the editor chooses, so reopening a job never spends
  // GPU time or Gemini quota by itself.
  if (view.running_elsewhere) {
    return (
      <>
        <h1>Working in another window</h1>
        <p className="lede">
          Another Super Processor window on this computer is working on this job.
          Wait for it to finish there, then check again.
        </p>
        {message ? <Alert text={message} /> : null}
        <p className="muted" role="status">
          {checked}
        </p>
        <div className="actions">
          <button
            type="button"
            disabled={busy}
            onClick={() => {
              setChecked(null);
              void onCheckAgain().then((ok) => {
                if (ok) setChecked(`Still working there (checked at ${new Date().toLocaleTimeString()}).`);
              });
            }}
          >
            {busy ? "Checking…" : "Check again"}
          </button>
        </div>
      </>
    );
  }
  const choice = interruptedChoice(view);
  return (
    <>
      <h1>{choice.title}</h1>
      <p className="lede">{choice.lede}</p>
      {view.cancellable ? <p className="muted">{choice.stopHint}</p> : null}
      {message ? <Alert text={message} /> : null}
      <div className="actions">
        <button type="button" disabled={busy} onClick={() => void onRequest((shown) => resumeRun(origin, shown))}>
          {choice.resumeLabel}
        </button>
        {view.cancellable ? (
          <button
            type="button"
            className="secondary"
            disabled={busy}
            // No run works on an interrupted step, so no run_id is sent.
            onClick={() => void onRequest((shown) => cancelRun(origin, shown))}
          >
            {choice.stopLabel}
          </button>
        ) : null}
      </div>
    </>
  );
}

function poster(videoUrl: string | null): string | null {
  // Each preview has a still of the same name, saved for the check. The
  // query (the plan's generation) stays on the still's URL too.
  return videoUrl ? videoUrl.replace(/\.mp4(\?|$)/, ".jpg$1") : null;
}

function checkSummary(check: LookShot["check"]): string {
  let note = (check.note || "").trim();
  if (note && !/[.!?]$/.test(note)) note += ".";
  if (check.ok === null) return "Not checked.";
  if (check.ok) return note ? `Check passed. ${note}` : "Check passed.";
  let text = `Check found: ${words(check.problems) || "a problem"}.`;
  if (note) text += ` ${note}`;
  if (check.adjusted) text += " Settings were adjusted and the preview redone.";
  return text;
}

function LookCard({ origin, shot }: { origin: string; shot: LookShot }) {
  const summary = settingsSummary(shot.strength, shot.look);
  return (
    <>
      <h2>
        {shot.index + 1}. {formatTime(shot.start_s)}–{formatTime(shot.end_s)} ·{" "}
        {shot.label || "Unlabelled scene"}
      </h2>
      {shot.after_url ? (
        <div className="pair">
          <figure>
            <figcaption>Before</figcaption>
            <div className="frame">
              <video
                controls
                muted
                loop
                preload="metadata"
                src={outputSrc(origin, shot.before_url) ?? undefined}
                poster={outputSrc(origin, poster(shot.before_url)) ?? undefined}
                aria-label={`Scene ${shot.index + 1} before`}
              />
              <Boxes regions={shot.regions ?? []} />
            </div>
          </figure>
          <figure>
            <figcaption>After</figcaption>
            <div className="frame">
              <video
                controls
                muted
                loop
                preload="metadata"
                src={outputSrc(origin, shot.after_url) ?? undefined}
                poster={outputSrc(origin, poster(shot.after_url)) ?? undefined}
                aria-label={`Scene ${shot.index + 1} after`}
              />
              <Boxes regions={shot.regions ?? []} />
            </div>
          </figure>
        </div>
      ) : null}
      <p>
        <strong>
          {shot.use_upscale === false ? `No upscaling: resized only. ${summary}` : summary}
        </strong>
      </p>
      <Protected regions={shot.regions ?? []} contains={shot.contains} />
      <p>{shot.reason}</p>
      <p className="muted">
        {checkSummary(shot.check)}
        {typeof shot.shimmer === "number"
          ? ` Texture stability: ${shot.shimmer.toFixed(1)}× as jumpy as the source.`
          : ""}
      </p>
    </>
  );
}

function PreviewStep({
  origin,
  preview,
  busy,
  message,
  onSend,
  view,
}: {
  origin: string;
  preview: SegmentPreview;
  busy: boolean;
  message: string | null;
  onSend: SendAction;
  view: WizardView;
}) {
  const [note, setNote] = useState("");
  const [resample, setResample] = useState(false);
  const last = preview.index + 1 >= preview.count;

  function redo(event: FormEvent) {
    event.preventDefault();
    void onSend("preview", { action: "redo", note, resample: resample ? "1" : "" });
  }

  return (
    <>
      <h1>
        Scene {preview.index + 1} of {preview.count}
      </h1>
      <p className="lede">
        A few seconds of this scene, before and after. Approve it, or tell me
        what to change.
      </p>
      {preview.scale ? (
        <p className="muted">{outputSizeLine(preview.scale)}</p>
      ) : null}
      {message ? <Alert text={message} /> : null}
      <Notice text={preview.error} />
      {preview.shot ? <LookCard origin={origin} shot={preview.shot} /> : null}
      {/* Always in the page, so a change is read out; empty without --prefetch. */}
      <p className="muted prefetch-note" role="status">
        {prefetchNote(preview)}
      </p>
      <div className="actions">
        <button
          type="button"
          disabled={busy}
          onClick={() => void onSend("preview", { action: "approve" })}
        >
          {last ? "Approve and finish" : "Approve and go to the next scene"}
        </button>
      </div>
      <form onSubmit={redo}>
        <label htmlFor="preview-note">
          Not right? Tell me what to change in this scene
          <input
            id="preview-note"
            type="text"
            value={note}
            placeholder="e.g. skin looks waxy, keep it softer"
            onChange={(event) => setNote(event.target.value)}
          />
        </label>
        <label htmlFor="preview-resample" className="inline">
          <input
            id="preview-resample"
            type="checkbox"
            checked={resample}
            onChange={(event) => setResample(event.target.checked)}
          />
          Preview a different part
        </label>
        <div className="actions">
          <button type="submit" className="secondary" disabled={busy}>
            Redo this scene
          </button>
        </div>
      </form>
      {preview.shot && preview.scale !== 1 ? (
        <UpscaleToggle shot={preview.shot} busy={busy} onSend={onSend} />
      ) : null}
      <Back view={view} to={["segments"]} busy={busy} onSend={onSend} />
    </>
  );
}

function UpscaleToggle({
  shot,
  busy,
  onSend,
}: {
  shot: LookShot;
  busy: boolean;
  onSend: SendAction;
}) {
  const on = shot.use_upscale !== false;
  const who = typeof shot.upscale_override === "boolean" ? "You" : "Gemini";
  return (
    <div className="actions">
      <p className="muted">
        {who} chose: this scene {on ? "uses upscaling" : "skips upscaling (plain resize)"}.
      </p>
      <button
        type="button"
        className="secondary"
        disabled={busy}
        onClick={() => void onSend("preview", { action: "upscale", on: on ? "0" : "1" })}
      >
        {on ? "Skip upscaling for this scene" : "Use upscaling for this scene"}
      </button>
    </div>
  );
}

function ReadyStep({
  looks,
  busy,
  message,
  onSend,
  view,
}: {
  looks: Looks;
  busy: boolean;
  message: string | null;
  onSend: SendAction;
  view: WizardView;
}) {
  const scales = [...(looks.scales ?? [])].sort((a, b) => b - a);
  const [scale, setScale] = useState<number | null>(
    looks.scale !== null && scales.includes(looks.scale) ? looks.scale : (scales[0] ?? null),
  );
  const [width, height] = looks.source_size ?? [0, 0];
  // After a failed render the error shows, and so does Render, to try again.
  const text = readyText(view);
  return (
    <>
      <h1>{text.title}</h1>
      <p className="lede">{text.lede}</p>
      <StoppedNote view={view} />
      {message ? <Alert text={message} /> : null}
      <ul className="summary">
        {looks.shots.map((shot) => {
          const summary = settingsSummary(shot.strength, shot.look);
          return (
            <li key={shot.index}>
              {formatTime(shot.start_s)}–{formatTime(shot.end_s)} ·{" "}
              {shot.label || "Unlabelled scene"}:{" "}
              {shot.use_upscale === false ? `No upscaling: resized only. ${summary}` : summary}
            </li>
          );
        })}
      </ul>
      {looks.scale_reason ? <p className="muted">{scaleChoiceText(looks.scale, looks.scale_reason)}</p> : null}
      {scales.length ? (
        <fieldset className="options">
          <legend>Upscale</legend>
          {scales.map((option) => {
            const size = width && height ? ` · ${width * option}×${height * option}` : "";
            return (
              <label key={option} className="option">
                <input
                  type="radio"
                  name="scale"
                  value={option}
                  checked={scale === option}
                  onChange={() => setScale(option)}
                />
                <span>
                  {option === 1 ? (
                    <>
                      <strong>Don&apos;t upscale</strong>
                      {size} · clean-up and finishing only, no upscaling
                    </>
                  ) : (
                    <>
                      <strong>Upscale {option}×</strong>
                      {size} · SPAN on the GPU
                    </>
                  )}
                </span>
              </label>
            );
          })}
        </fieldset>
      ) : null}
      {offersRender(view) ? (
        <div className="actions">
          <button
            type="button"
            disabled={busy}
            onClick={() =>
              void onSend("ready", scale === null ? { action: "render" } : { action: "render", scale: String(scale) })
            }
          >
            {text.renderLabel}
          </button>
        </div>
      ) : null}
      <Back view={view} to={["segments"]} busy={busy} onSend={onSend} />
    </>
  );
}
