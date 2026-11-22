import { useState, type FormEvent } from "react";
import { documentTitle } from "./shellView";
import { AppShell } from "./shell/AppShell";
import { TopBar } from "./shell/TopBar";
import { Banner, Button, Card, Field, Icon } from "./ui";

const TOUR = [
  {
    id: "intro",
    title: "Introduction",
    body: "Super Processor enhances footage you already shot. SPAN adds fine detail as it upscales 2×, and strength sets how much (0 is a plain upscale). It keeps every frame and its timing and does not generate new scenes.",
  },
  {
    id: "llm",
    title: "LLM choice",
    body: "Choose Gemini with a Google AI Studio key, or the local-LLM placeholder. Media stays on this machine except for Gemini calls you authorize.",
  },
  {
    id: "setup",
    title: "Gemini setup",
    body: "Create a key in Google AI Studio and paste it into the local wizard. The key is stored on your machine, not on this static page.",
  },
  {
    id: "file",
    title: "Pick a file",
    body: "Give the absolute path of a video on the computer running the wizard. A local GPU model restores and upscales it. You set scale and strength.",
  },
  {
    id: "segments",
    title: "Segments",
    body: "A small copy of the video, without its sound, is uploaded to Gemini, which watches it at one frame a second and splits it where the light, place, or picture quality changes. Gemini also chooses whether to upscale. You approve the segments, or say where the cuts should go.",
  },
  {
    id: "previews",
    title: "Previews",
    body: "One segment at a time, Gemini picks clean-up, strength, and finishing, or leaves the upscaler out where it would only harm. A few seconds of the segment are previewed before and after, and Gemini checks for wrong faces, hands, text, and fake texture. Approve it, or say what to change.",
  },
  {
    id: "render",
    title: "Render",
    body: "Choose whether to upscale: 2× with SPAN on the local GPU, or keep the size and apply only the clean-up and finishing. FFmpeg only trims, splices, and copies audio. A long clip is processed in pieces so it never has to fit in memory at once.",
  },
  {
    id: "result",
    title: "Result",
    body: "Watch output.mp4. Say you are happy, or describe a time range to restore again at a different strength. Lower strength means less invented texture.",
  },
] as const;

type StaticShellProps = {
  checking: boolean;
  connectError: string | null;
  onConnect: (url: string) => void;
};

export function StaticShell({ checking, connectError, onConnect }: StaticShellProps) {
  const [index, setIndex] = useState(0);
  const [url, setUrl] = useState("http://127.0.0.1:45143");
  const step = TOUR[index];

  function submit(event: FormEvent) {
    event.preventDefault();
    onConnect(url.trim());
  }

  return (
    <AppShell
      topBar={<TopBar jobTitle={null} connection={connectError ? "err" : null} onAllJobs={null} />}
      title={documentTitle(null)}
      headingKey="offline"
      wide
    >
      <div className="welcome">
        <div className="col gap20">
          <p className="overline">Welcome</p>
          <h1 className="display">Restore the video you already shot</h1>
          <p className="muted">
            This page is a static shell. It explains the wizard and stays usable when no GPU server is
            attached. The live upscale runs only on your machine, after you start{" "}
            <code>super-processor review</code>.
          </p>
          <p className="muted">
            GitHub Pages cannot drive an RTX GPU. Open the localhost URL that command prints, or connect
            this shell to that server from a local preview.
          </p>
        </div>
        <Card className="col gap16">
          <h2 className="h3">Connect to your computer</h2>
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
              label="Local wizard URL"
              type="url"
              inputMode="url"
              value={url}
              onChange={(event) => setUrl(event.target.value)}
              autoComplete="off"
              help="The address super-processor review prints, such as http://127.0.0.1:45143."
            />
            <div className="row gap12 wrap">
              <Button type="submit" icon="wifi">
                Connect to local wizard
              </Button>
            </div>
          </form>
          {connectError ? (
            <Banner tone="error" role="alert" title="Can’t reach the wizard">
              {connectError}
            </Banner>
          ) : null}
        </Card>

        <Card as="section" className="col gap16" aria-labelledby="tour-heading">
          <div className="col gap4">
            <h2 id="tour-heading" className="h2">
              Wizard steps
            </h2>
            <p className="caption subtle" id="tour-help">
              Move through the steps with the buttons. Nothing here starts an upscale.
            </p>
          </div>
          <ol className="tour-steps" aria-label="Wizard steps">
            {TOUR.map((item, itemIndex) => (
              <li key={item.id}>
                <button
                  type="button"
                  className={itemIndex === index ? "chip sel" : "chip"}
                  aria-current={itemIndex === index ? "step" : undefined}
                  onClick={() => setIndex(itemIndex)}
                >
                  {itemIndex === index ? <Icon name="check" size={14} color="var(--leaf)" /> : null}
                  {item.title}
                </button>
              </li>
            ))}
          </ol>
          <div className="col gap8" aria-live="polite">
            <h3 className="h3">{step.title}</h3>
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
              size="md"
              onClick={() => setIndex((value) => Math.min(TOUR.length - 1, value + 1))}
              disabled={index === TOUR.length - 1}
            >
              Next step
            </Button>
          </div>
        </Card>
      </div>
    </AppShell>
  );
}
