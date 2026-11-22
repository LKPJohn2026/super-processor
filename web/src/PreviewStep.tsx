import { useEffect, useState, type FormEvent, type ReactNode } from "react";
import { outputSizeLine, outputSrc, type SegmentPreview, type WizardView } from "./api";
import { CompareViewer } from "./CompareViewer";
import { stripSummary } from "./jobView";
import {
  FIX_CHIPS,
  RECHECK_NOTE,
  RESAMPLE_NOTE,
  afterTag,
  approveLabel,
  checkState,
  fixNote,
  previewOverline,
  previewRange,
  previewTitle,
  sceneGets,
  settingsSummary,
  strengthWord,
  type FixChipKey,
} from "./previewView";
import { Banner, Button, Chip, Field, Icon, Segmented } from "./ui";
import { prefetchNote } from "./versions";

// Resolves true when the server took the action (false: the error shows).
type SendAction = (action: string, fields: Record<string, string>) => Promise<boolean | void>;

function poster(videoUrl: string | null): string | null {
  // Each preview has a still of the same name, saved for the check. The
  // query (the plan's generation) stays on the still's URL too.
  return videoUrl ? videoUrl.replace(/\.mp4(\?|$)/, ".jpg$1") : null;
}

/**
 * Scene preview (mockup 05; S6 when Gemini could not check it): the scene
 * strip, the comparison viewer, and beside it the check, what the scene
 * gets, AI detail, "Something off?" and Advanced settings.
 */
export function PreviewStep({
  origin,
  preview,
  busy,
  message,
  onSend,
  view,
  strip,
  back,
}: {
  origin: string;
  preview: SegmentPreview;
  busy: boolean;
  message: string | null;
  onSend: SendAction;
  view: WizardView;
  strip: ReactNode;
  back: ReactNode;
}) {
  const shot = preview.shot;
  const [chosen, setChosen] = useState<FixChipKey[]>([]);
  const [note, setNote] = useState("");
  const [noteError, setNoteError] = useState<string | null>(null);
  const [clipS, setClipS] = useState<number | null>(null);
  // A new scene starts with no chips chosen; a typed note is kept until sent.
  useEffect(() => {
    setChosen([]);
    setNoteError(null);
    setClipS(null);
  }, [preview.index]);

  const scenes = (view.job?.scenes ?? []).map((scene) => ({ index: scene.index, label: scene.label }));
  const count = preview.count || view.job?.scene_count || 0;
  const check = shot ? checkState(shot.check, shot.shimmer) : null;
  const unchecked = Boolean(shot?.after_url && check && !check.checked);
  const summary = stripSummary(view);
  const title = previewTitle(shot, preview.index);
  const range = previewRange(shot);
  const locked = busy || Boolean(view.busy);

  function redo(event: FormEvent) {
    event.preventDefault();
    const text = fixNote(chosen, note);
    if (!text) {
      setNoteError("Pick what is off, or say it in words.");
      return;
    }
    setNoteError(null);
    void onSend("preview", { action: "redo", note: text, resample: "" }).then((ok) => {
      if (ok !== false) {
        setNote("");
        setChosen([]);
      }
    });
  }

  function resample() {
    const text = fixNote(chosen, note) || RESAMPLE_NOTE;
    void onSend("preview", { action: "redo", note: text, resample: "1" });
  }

  const before = shot ? outputSrc(origin, shot.before_url) : null;
  const after = shot ? outputSrc(origin, shot.after_url) : null;
  const seconds = clipS && clipS > 0 ? Math.round(clipS) : null;
  const upscaling = preview.scale !== 1;
  const aiOn = shot?.use_upscale !== false;
  const who = typeof shot?.upscale_override === "boolean" ? "You" : "Gemini";
  const word = shot ? strengthWord(shot.strength) : "";
  const onLabel = word === "None" ? "On" : word;
  const gets = shot ? sceneGets(shot, preview.scale) : [];

  return (
    <div className="preview-step col gap20">
      <header className="preview-head">
        <div className="col gap4">
          <p className="overline">{previewOverline(preview.index, count)}</p>
          <h1>
            {title}
            {range ? <span className="h1-caption"> {range}</span> : null}
          </h1>
        </div>
        {summary ? <p className="caption subtle">{summary}</p> : null}
      </header>
      {message ? (
        <Banner tone="error" role="alert" title="Something went wrong">
          {message}
        </Banner>
      ) : null}
      {preview.error ? (
        <Banner tone="warning" title="About these settings">
          {preview.error}
        </Banner>
      ) : null}
      {strip}
      {unchecked && check ? (
        <Banner
          tone="warning"
          title={check.title}
          actions={
            <>
              <Button
                kind="primary"
                size="md"
                icon="rotate"
                disabled={locked}
                onClick={() => void onSend("preview", { action: "redo", note: RECHECK_NOTE, resample: "" })}
              >
                Try Gemini again
              </Button>
              <Button
                kind="secondary"
                size="md"
                disabled={locked}
                onClick={() => void onSend("preview", { action: "approve" })}
              >
                Approve anyway
              </Button>
            </>
          }
        >
          {check.text}
        </Banner>
      ) : null}
      <div className="preview-main">
        <div className="preview-left col gap16">
          {shot && before && after ? (
            <CompareViewer
              beforeSrc={before}
              afterSrc={after}
              beforePoster={outputSrc(origin, poster(shot.before_url)) ?? undefined}
              afterPoster={outputSrc(origin, poster(shot.after_url)) ?? undefined}
              regions={shot.regions ?? []}
              afterLabel={afterTag(preview.scale, shot.use_upscale)}
              sceneName={`scene ${preview.index + 1}, ${title}`}
              onDuration={setClipS}
            />
          ) : (
            <div className="viewer empty">
              <p>There is no preview of this scene yet.</p>
            </div>
          )}
          <div className="row gap12 wrap preview-actions">
            <Button
              kind="primary"
              size="lg"
              icon="check"
              disabled={locked}
              onClick={() => void onSend("preview", { action: "approve" })}
            >
              {approveLabel(preview.index, count, scenes)}
            </Button>
            <Button kind="ghost" size="lg" disabled={locked} onClick={resample}>
              {seconds ? `Preview a different ${seconds} seconds` : "Preview a different part"}
            </Button>
          </div>
          {/* Always in the page, so a change is read out; empty without --prefetch. */}
          <p className="caption subtle prefetch-note" role="status">
            {prefetchNote(preview)}
          </p>
          {preview.scale ? <p className="caption subtle">{outputSizeLine(preview.scale)}</p> : null}
          {back}
        </div>
        <aside className="preview-panel col gap16" aria-label="This scene's settings">
          {check ? (
            <div className={`check-box ${check.tone}`}>
              <p className="check-title">
                <Icon name={check.tone === "success" ? "check-circle" : "alert"} size={18} />
                {check.checked ? check.title : "Not checked by Gemini"}
              </p>
              <p className="body-s">
                {check.checked ? check.text : "Approve it only if it looks right to you."}
              </p>
            </div>
          ) : null}
          {shot ? (
            <section className="col gap12" aria-labelledby="gets-title">
              <h2 id="gets-title" className="overline">
                What this scene gets
              </h2>
              <ul className="gets">
                {gets.map((item) => (
                  <li key={item.text}>
                    <Icon name={item.icon} size={18} color="var(--ink-muted)" />
                    <span className="body-s grow">{item.text}</span>
                    {item.amount ? <span className="caption subtle">{item.amount}</span> : null}
                  </li>
                ))}
                {gets.length ? null : (
                  <li>
                    <span className="body-s grow">No clean-up or colour changes</span>
                  </li>
                )}
              </ul>
            </section>
          ) : null}
          {shot ? (
            <section className="col gap8" aria-labelledby="ai-detail-title">
              <div className="row between gap8">
                <h2 id="ai-detail-title" className="body-s strong">
                  AI detail
                </h2>
                <span className="caption subtle">
                  {upscaling ? `${who} chose ${aiOn ? onLabel : "Off"}` : "Not used"}
                </span>
              </div>
              {upscaling ? (
                <>
                  <Segmented
                    label="AI detail"
                    options={[
                      { value: "off", label: "Off" },
                      { value: "on", label: onLabel },
                    ]}
                    value={aiOn ? "on" : "off"}
                    disabled={locked}
                    onChange={(value) => {
                      if ((value === "on") !== aiOn) void onSend("preview", { action: "upscale", on: value === "on" ? "1" : "0" });
                    }}
                  />
                  <p className="caption subtle">
                    Off just resizes. Pick it if the AI makes this scene look painted or crawly. Changing it
                    redoes this preview.
                  </p>
                </>
              ) : (
                <p className="caption subtle">The video keeps its size, so no AI detail is added.</p>
              )}
            </section>
          ) : null}
          <hr className="divider" />
          <form className="col gap8" onSubmit={redo} aria-labelledby="off-title">
            <h2 id="off-title" className="body-s strong">
              Something off?
            </h2>
            <div className="row gap6 wrap" role="group" aria-label="Tell Gemini it is">
              {FIX_CHIPS.map((chip) => (
                <Chip
                  key={chip.key}
                  selected={chosen.includes(chip.key)}
                  disabled={locked}
                  onToggle={(on) => {
                    setNoteError(null);
                    setChosen((value) => (on ? [...value.filter((key) => key !== chip.key), chip.key] : value.filter((key) => key !== chip.key)));
                  }}
                >
                  {chip.text}
                </Chip>
              ))}
            </div>
            <Field
              id="preview-note"
              className="compact"
              label="Or tell Gemini (optional)"
              type="text"
              autoComplete="off"
              value={note}
              placeholder="e.g. skin looks waxy, keep it softer"
              error={noteError}
              onChange={(event) => {
                setNoteError(null);
                setNote(event.target.value);
              }}
            />
            <Button type="submit" kind="secondary" size="md" icon="rotate" disabled={locked} className="self-start">
              Redo this preview
            </Button>
          </form>
          <div className="advanced-row">
            {shot ? (
              <details className="advanced">
                <summary>
                  <Icon name="chevron-right" size={16} className="advanced-chevron" />
                  Advanced settings
                </summary>
                <dl>
                  <dt>Settings</dt>
                  <dd className="mono">{settingsSummary(shot.strength, shot.look)}</dd>
                  <dt>Upscaling</dt>
                  <dd>
                    {shot.use_upscale === false ? "Off: plain resize" : "On"}
                    {preview.scale ? ` · output ${preview.scale}×` : ""}
                  </dd>
                  {typeof shot.shimmer === "number" ? (
                    <>
                      <dt>Texture stability</dt>
                      <dd>{shot.shimmer.toFixed(1)}× as jumpy as the source</dd>
                    </>
                  ) : null}
                  {shot.reason ? (
                    <>
                      <dt>Gemini’s reason</dt>
                      <dd>{shot.reason}</dd>
                    </>
                  ) : null}
                </dl>
              </details>
            ) : null}
            <p className="caption subtle kbd-hints">
              <span className="sr-only">Viewer keys: Space plays, B flips before and after, the arrows step a frame, Z shows 100%.</span>
              <span aria-hidden="true">
                <span className="kbd">Space</span> <span className="kbd">B</span> <span className="kbd">←</span>
                <span className="kbd">→</span> <span className="kbd">Z</span>
              </span>
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}
