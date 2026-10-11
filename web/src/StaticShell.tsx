import { useState, type FormEvent } from "react";

const TOUR = [
  {
    id: "intro",
    title: "Introduction",
    body: "Super Processor enhances footage you already shot. FlashVSR adds fine detail as it upscales, and strength sets how much (0 is a plain upscale). It keeps every frame and its timing and does not generate new scenes.",
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
    id: "shots",
    title: "Shots",
    body: "FFmpeg finds where each shot starts and measures how blocky, noisy, soft, dark, or flat it is. Gemini labels each shot from one still. You merge or split shots, then approve the list.",
  },
  {
    id: "looks",
    title: "Looks",
    body: "Gemini picks clean-up, strength, and finishing for each shot. Each shot gets a short before/after preview, which Gemini checks for wrong faces, hands, text, and fake texture. Approve them all, or tell it what to change in one shot.",
  },
  {
    id: "upscale",
    title: "Upscaling",
    body: "FlashVSR restores and upscales on the local GPU. FFmpeg only trims, splices, and copies audio. A long clip is processed in overlapping pieces so it can fit in GPU memory.",
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
    <>
      <p className="brand">Super Processor</p>
      <div className="panel">
        <h1>Restore the video you already shot</h1>
        <p className="lede">
          This page is a static shell. It explains the wizard and stays usable
          when no GPU server is attached. The live upscale runs only on your
          machine, after you start <code>super-processor review</code>.
        </p>
        <p>
          GitHub Pages cannot drive an RTX GPU. Open the localhost URL that
          command prints, or connect this shell to that server from a local
          preview.
        </p>
        {checking ? (
          <p role="status">Checking for a local wizard…</p>
        ) : (
          <p role="status">No live wizard is connected.</p>
        )}
        <form onSubmit={submit}>
          <label htmlFor="wizard-url">
            Local wizard URL
            <input
              id="wizard-url"
              name="wizard-url"
              type="url"
              inputMode="url"
              value={url}
              onChange={(event) => setUrl(event.target.value)}
              autoComplete="off"
            />
          </label>
          <div className="actions">
            <button type="submit">Connect to local wizard</button>
          </div>
        </form>
        {connectError ? (
          <p className="error" role="alert">
            {connectError}
          </p>
        ) : null}

        <h2 id="tour-heading">Wizard steps</h2>
        <p className="muted" id="tour-help">
          Move through the steps with the buttons. Nothing here starts an
          upscale.
        </p>
        <ol className="steps" aria-label="Wizard steps">
          {TOUR.map((item, itemIndex) => (
            <li key={item.id} className={itemIndex === index ? "active" : undefined}>
              <button
                type="button"
                className="secondary"
                aria-current={itemIndex === index ? "step" : undefined}
                onClick={() => setIndex(itemIndex)}
              >
                {item.title}
              </button>
            </li>
          ))}
        </ol>
        <section aria-labelledby="tour-heading">
          <h2>{step.title}</h2>
          <p>{step.body}</p>
          <div className="actions">
            <button
              type="button"
              className="secondary"
              onClick={() => setIndex((value) => Math.max(0, value - 1))}
              disabled={index === 0}
            >
              Previous step
            </button>
            <button
              type="button"
              onClick={() =>
                setIndex((value) => Math.min(TOUR.length - 1, value + 1))
              }
              disabled={index === TOUR.length - 1}
            >
              Next step
            </button>
          </div>
        </section>
      </div>
    </>
  );
}
