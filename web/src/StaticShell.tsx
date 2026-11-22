import { useEffect, useId, useState, type FormEvent } from "react";
import { documentTitle, originHost } from "./shellView";
import { AppShell } from "./shell/AppShell";
import { TopBar } from "./shell/TopBar";
import { Banner, Button, Card, Field, Icon } from "./ui";
import { PrivacyCard, WelcomeHero } from "./Welcome";

const DEFAULT_ADDRESS = "http://127.0.0.1:45143";

// The five steps of a job, in the sidebar's words (mockups 02–09).
const TOUR = [
  {
    id: "video",
    title: "Video",
    body: "Choose a video on the computer that runs the wizard, up to 1080p and 30 minutes. A video dropped on the page is copied into the jobs folder there.",
  },
  {
    id: "scenes",
    title: "Scenes",
    body: "A small copy without its sound goes to Gemini, which splits the video where the light, the place or the picture quality changes. Move, merge or split the cuts, then approve them.",
  },
  {
    id: "previews",
    title: "Previews",
    body: "One scene at a time, Gemini suggests the clean-up and how much detail to add, and a few seconds are shown before and after. Approve each scene, or tell Gemini what to change.",
  },
  {
    id: "render",
    title: "Render",
    body: "The whole video is restored on your graphics card, twice the size or at its own size, with its sound copied across unchanged. You can stop a render and start it again later.",
  },
  {
    id: "result",
    title: "Result",
    body: "Watch the restored video. Keep it, or say which stretch to restore again and what to change; earlier versions are kept.",
  },
] as const;

type StaticShellProps = {
  checking: boolean;
  connectError: string | null;
  // The address last tried ("" for this page's own server).
  tried: string;
  // Opened from "Change address": the address field takes focus.
  focusAddress?: boolean;
  onConnect: (url: string) => void;
};

/**
 * The page with no wizard to talk to (a static build, or a wizard that is
 * not running): what Super Processor does, a way to connect to the wizard,
 * and a walk through the steps. Nothing here starts any work.
 */
export function StaticShell({ checking, connectError, tried, focusAddress, onConnect }: StaticShellProps) {
  const [index, setIndex] = useState(0);
  const [url, setUrl] = useState(() => tried || DEFAULT_ADDRESS);
  const step = TOUR[index];
  const tourId = useId();

  useEffect(() => {
    if (focusAddress) document.getElementById("wizard-url")?.focus();
  }, [focusAddress]);

  function submit(event: FormEvent) {
    event.preventDefault();
    onConnect(url.trim());
  }

  const connect = (
    <Card as="section" className="col gap16 connect-card" aria-labelledby="connect-heading">
      <h2 id="connect-heading" className="h3">
        Connect to your computer
      </h2>
      {checking ? (
        <p role="status" className="row gap8 body-s">
          <Icon name="clock" color="var(--focus)" />
          Checking for a local wizard…
        </p>
      ) : (
        <p role="status" className="row gap8 body-s">
          <Icon name="wifi" color="var(--ink-subtle)" />
          No live wizard is connected.
        </p>
      )}
      <form onSubmit={submit} className="col gap12">
        <Field
          id="wizard-url"
          name="wizard-url"
          label="Local wizard address"
          type="url"
          inputMode="url"
          value={url}
          onChange={(event) => setUrl(event.target.value)}
          autoComplete="off"
          spellCheck={false}
          help="The address super-processor review prints after “open in your browser”, such as http://127.0.0.1:45143/?token=…"
        />
        <div className="row gap12 wrap">
          <Button type="submit" icon="wifi" disabled={checking}>
            Connect to local wizard
          </Button>
        </div>
      </form>
      {connectError ? (
        <Banner tone="error" role="alert" title={`Can’t reach the wizard at ${originHost(tried, DEFAULT_ADDRESS)}`}>
          {connectError}
        </Banner>
      ) : null}
    </Card>
  );

  return (
    <AppShell
      topBar={<TopBar jobTitle={null} connection={connectError ? "err" : null} onAllJobs={null} />}
      title={documentTitle(null)}
      headingKey="offline"
      wide
    >
      <div className="welcome-screen">
        <WelcomeHero
          actions={
            <p className="caption subtle">
              Start it with <code>super-processor review</code> on the computer with the graphics card.
            </p>
          }
          aside={connect}
        >
          <p>
            Pick a film or a home video. Gemini suggests settings scene by scene, you check every scene before
            anything is rendered, and the full-quality video never leaves your computer.
          </p>
          <p>
            This page only shows how it works. The restoring runs in the wizard on your own computer: open the
            address it prints, or connect this page to it.
          </p>
        </WelcomeHero>
        <div className="welcome-cards">
          <Card as="section" className="col gap16 tour-card" aria-labelledby="tour-heading">
            <div className="col gap4">
              <h2 id="tour-heading" className="h3">
                How a job goes
              </h2>
              <p className="caption subtle">Move through the steps with the buttons. Nothing here starts any work.</p>
            </div>
            <ol className="tour-steps" aria-label="Steps of a job">
              {TOUR.map((item, itemIndex) => {
                const state = itemIndex === index ? "current" : itemIndex < index ? "done" : "upcoming";
                return (
                  <li key={item.id}>
                    <button
                      type="button"
                      className={`tour-step ${state}`}
                      aria-current={state === "current" ? "step" : undefined}
                      aria-controls={tourId}
                      onClick={() => setIndex(itemIndex)}
                    >
                      <span className="marker" aria-hidden="true">
                        {itemIndex + 1}
                      </span>
                      {item.title}
                    </button>
                  </li>
                );
              })}
            </ol>
            <div id={tourId} className="col gap8 tour-body" aria-live="polite">
              <h3 className="h3">
                <span className="sr-only">Step {index + 1} of {TOUR.length}: </span>
                {step.title}
              </h3>
              <p className="body-s muted">{step.body}</p>
            </div>
            <div className="row gap12 wrap">
              <Button
                kind="secondary"
                size="md"
                icon="chevron-left"
                onClick={() => setIndex((value) => Math.max(0, value - 1))}
                disabled={index === 0}
              >
                Previous step
              </Button>
              <Button
                kind="secondary"
                size="md"
                onClick={() => setIndex((value) => Math.min(TOUR.length - 1, value + 1))}
                disabled={index === TOUR.length - 1}
              >
                Next step
                <Icon name="chevron-right" size={16} />
              </Button>
            </div>
          </Card>
          <PrivacyCard />
        </div>
      </div>
    </AppShell>
  );
}
