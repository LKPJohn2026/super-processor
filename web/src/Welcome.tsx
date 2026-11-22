import { useEffect, useState, type FormEvent, type ReactNode } from "react";
import type { WizardView } from "./api";
import { Banner, Button, Card, Field, Icon } from "./ui";
import { NEEDS_KEY_TEXT, keyCheck, keyError, welcomeStart } from "./welcomeView";

const AI_STUDIO = "https://aistudio.google.com/apikey";
const GEMINI_PRICING = "https://ai.google.dev/gemini-api/docs/pricing";
const KEY_FIELD = "gemini-key";

type SendAction = (action: string, fields: Record<string, string>) => Promise<boolean | void>;

/** A link that opens in a new tab and says so to screen readers. */
export function OutLink({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noreferrer">
      {children}
      <span aria-hidden="true"> ↗</span>
      <span className="sr-only"> (opens in a new tab)</span>
    </a>
  );
}

/**
 * The hero of mockup 01: overline, display heading, the promise, then the
 * actions. `aside` sits to its right on a wide screen (the picture).
 */
export function WelcomeHero({
  overline = "Welcome",
  heading = "Make old video sharper and cleaner — on your own computer",
  children,
  actions,
  aside,
}: {
  overline?: string;
  heading?: string;
  children: ReactNode;
  actions: ReactNode;
  aside?: ReactNode;
}) {
  return (
    <section className="welcome-hero" aria-labelledby="welcome-heading">
      <div className="welcome-hero-text">
        <p className="overline">{overline}</p>
        <h1 id="welcome-heading" className="display">
          {heading}
        </h1>
        <div className="muted welcome-lede">{children}</div>
        <div className="welcome-actions">{actions}</div>
      </div>
      {aside}
    </section>
  );
}

/**
 * Where the mockup shows a before/after still. There is no job yet, so no
 * real picture: an empty stage that says what will show there.
 */
export function WelcomeFrame() {
  return (
    <figure className="welcome-frame">
      <div className="welcome-frame-stage" aria-hidden="true">
        <span className="welcome-frame-half before" />
        <span className="welcome-frame-half after" />
        <span className="welcome-frame-line" />
        <span className="tag tag-left">Original</span>
        <span className="tag tag-right">Restored</span>
        <Icon name="film" size={36} color="var(--on-stage)" className="welcome-frame-icon" />
      </div>
      <figcaption className="caption welcome-frame-caption">
        Each scene’s before and after shows here before anything is rendered.
      </figcaption>
    </figure>
  );
}

/** "What leaves your computer" (mockup 01), shared with the static page. */
export function PrivacyCard() {
  return (
    <Card as="section" className="col gap16 welcome-privacy" aria-labelledby="privacy-heading">
      <h2 id="privacy-heading" className="h3">
        What leaves your computer
      </h2>
      <div className="privacy-split">
        <div className="privacy-box stays">
          <p className="row gap6 body-s strong privacy-title">
            <Icon name="lock" size={16} />
            Stays here
          </p>
          <ul className="caption privacy-list">
            <li>The full-quality video</li>
            <li>Its sound</li>
            <li>The restored video</li>
          </ul>
        </div>
        <div className="privacy-box sent">
          <p className="row gap6 body-s strong privacy-title">
            <Icon name="send" size={16} />
            Sent to Google Gemini
          </p>
          <ul className="caption privacy-list">
            <li>A small silent copy (up to 960 px), deleted after about 48 h</li>
            <li>A few stills</li>
            <li>Notes you write</li>
          </ul>
        </div>
      </div>
      <p className="caption muted">
        <strong>Will this cost money?</strong> Usually within Gemini’s free tier.{" "}
        <OutLink href={GEMINI_PRICING}>Check your plan</OutLink>
      </p>
    </Card>
  );
}

/** One line of "What you need": an icon, what it is, and what was found. */
function NeedRow({
  icon,
  tone,
  label,
  detail,
  action,
}: {
  icon: "check-circle" | "x-circle" | "cpu" | "info";
  tone: "ok" | "bad" | "plain";
  label?: string;
  detail: ReactNode;
  action?: ReactNode;
}) {
  const color = tone === "ok" ? "var(--ok)" : tone === "bad" ? "var(--err)" : "var(--ink-subtle)";
  return (
    <div className="need-row">
      <Icon name={icon} size={20} color={color} />
      <div className="col grow">
        {label ? <p className="body-s strong">{label}</p> : null}
        <p className={tone === "bad" ? "caption check-bad" : "caption subtle"}>{detail}</p>
      </div>
      {action}
    </div>
  );
}

/**
 * The Welcome & setup screen (mockup 01) for the intro, LLM choice, local
 * model and Gemini key steps. "Choose a video" goes on to the file pick
 * (saving a typed key first); the key and the LLM choice stay here.
 */
export function WelcomeStep({
  view,
  busy,
  message,
  onSend,
}: {
  view: WizardView;
  busy: boolean;
  message: string | null;
  onSend: SendAction;
}) {
  const hasKey = view.has_gemini_key;
  const refused = keyError(view);
  const [typed, setTyped] = useState("");
  // The key field: always without a key, after "Replace key" with one.
  const [replacing, setReplacing] = useState(Boolean(refused));
  const [needsKey, setNeedsKey] = useState(false);
  useEffect(() => {
    if (refused) setReplacing(true);
  }, [refused]);
  const showField = !hasKey || replacing;
  const check = keyCheck(hasKey);
  const fieldError = needsKey ? NEEDS_KEY_TEXT : refused;

  function start() {
    const next = welcomeStart(hasKey, showField ? typed : "");
    if (next.kind === "needs_key") {
      setNeedsKey(true);
      document.getElementById(KEY_FIELD)?.focus();
      return;
    }
    setNeedsKey(false);
    void onSend(next.action, next.fields);
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    start();
  }

  function cancelReplace() {
    setReplacing(false);
    setTyped("");
    setNeedsKey(false);
    window.requestAnimationFrame(() => document.getElementById("replace-key")?.focus());
  }

  const banner = message && message !== refused ? message : null;

  return (
    <div className="welcome-screen">
      <WelcomeHero
        actions={
          <>
            <Button icon="film" disabled={busy} onClick={start}>
              Choose a video
            </Button>
            <span className="caption subtle">
              {hasKey ? "Your Gemini key is ready." : "Takes about 1 minute to set up"}
            </span>
          </>
        }
        aside={<WelcomeFrame />}
      >
        <p>
          Pick a film or a home video. Gemini suggests settings scene by scene, you check every scene before
          anything is rendered, and the full-quality video never leaves this computer.
        </p>
      </WelcomeHero>
      {banner ? (
        <Banner tone="error" role="alert" title="That did not work">
          {banner}
        </Banner>
      ) : null}
      <div className="welcome-cards">
        <Card as="section" className="col gap16 welcome-need" aria-labelledby="need-heading">
          <h2 id="need-heading" className="h3">
            What you need
          </h2>
          <NeedRow
            icon="cpu"
            tone="plain"
            label="NVIDIA graphics card"
            detail="The previews and the render run on it, on this computer."
          />
          <NeedRow
            icon={check.ok ? "check-circle" : "x-circle"}
            tone={check.ok ? "ok" : "bad"}
            label={check.label}
            detail={check.detail}
            action={
              hasKey && !replacing ? (
                <Button id="replace-key" kind="ghost" size="md" disabled={busy} onClick={() => {
                    setReplacing(true);
                    // The button goes; the field it opened takes focus.
                    window.requestAnimationFrame(() => document.getElementById(KEY_FIELD)?.focus());
                  }}>
                  Replace key
                </Button>
              ) : null
            }
          />
          {showField ? (
            <form className="key-form" onSubmit={submit}>
              <Field
                id={KEY_FIELD}
                name="api_key"
                type="password"
                autoComplete="off"
                spellCheck={false}
                label={hasKey ? "New Gemini API key" : "Gemini API key"}
                value={typed}
                onChange={(event) => {
                  setTyped(event.target.value);
                  if (needsKey) setNeedsKey(false);
                }}
                error={fieldError}
                help="Saved on this computer, readable only by your user."
              />
              <p className="caption subtle">
                No key yet? Open <OutLink href={AI_STUDIO}>Google AI Studio</OutLink>, accept the terms if asked,
                create an API key and paste it above.
              </p>
              <div className="row gap8 wrap">
                <Button type="submit" kind="secondary" size="md" icon="key" disabled={busy}>
                  Save key and choose a video
                </Button>
                {hasKey ? (
                  <Button kind="ghost" size="md" disabled={busy} onClick={cancelReplace}>
                    Keep the current key
                  </Button>
                ) : null}
              </div>
            </form>
          ) : null}
          {view.step === "local_llm_stub" ? (
            <Banner
              tone="info"
              title="Local models are not wired up yet"
              actions={
                <Button size="md" disabled={busy} onClick={() => void onSend("local-llm", { action: "gemini" })}>
                  Use Gemini
                </Button>
              }
            >
              Gemini is the way to finish a job today. A local model you run yourself will come later.
            </Banner>
          ) : (
            <NeedRow
              icon="info"
              tone="plain"
              // Only the words (01): a button here would lead to a step with nothing behind it.
              detail="Gemini watches the video and suggests settings. Local models: coming later."
            />
          )}
        </Card>
        <PrivacyCard />
      </div>
    </div>
  );
}
