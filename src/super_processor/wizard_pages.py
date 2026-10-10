"""HTML templates for the localhost Gemini wizard website."""

from __future__ import annotations

from html import escape

# Phase keys for the step rail (setup → result).
_PHASE_ORDER = ("setup", "file", "upscale", "result")


def _step_rail(active: str | None) -> str:
    labels = {
        "setup": "Setup",
        "file": "File",
        "upscale": "Upscale",
        "result": "Result",
    }
    active_idx = _PHASE_ORDER.index(active) if active in _PHASE_ORDER else -1
    items: list[str] = []
    for index, key in enumerate(_PHASE_ORDER):
        label = labels[key]
        if index < active_idx:
            cls = "step done"
        elif index == active_idx:
            cls = "step active"
        else:
            cls = "step upcoming"
        items.append(f'<span class="{cls}">{escape(label)}</span>')
    return '<nav class="steps" aria-label="Progress">' + "".join(items) + "</nav>"


def _page(
    title: str,
    body: str,
    *,
    phase: str | None = None,
    hero: bool = False,
) -> str:
    rail = _step_rail(phase) if not hero else ""
    brand = '<p class="brand-mark">Super Processor</p>' if not hero else ""
    main_class = "hero-main" if hero else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)} · Super Processor</title>
<style>
:root {{
  --ink: #1c2418;
  --muted: #5c6654;
  --sand: #e8dfd0;
  --leaf: #2f3d28;
  --leaf-hot: #3d5234;
  --paper: #f3eee4;
  --line: rgba(28, 36, 24, 0.14);
  --err: #8b2e2e;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  min-height: 100vh;
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  color: var(--ink);
  background:
    radial-gradient(
      ellipse 80% 50% at 10% -10%,
      rgba(120, 150, 90, 0.35),
      transparent 55%
    ),
    radial-gradient(
      ellipse 60% 40% at 100% 0%,
      rgba(180, 130, 70, 0.22),
      transparent 50%
    ),
    linear-gradient(165deg, #1a2216 0%, #2a3424 38%, #3a3228 100%);
}}
body::before {{
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  opacity: 0.12;
  background:
    repeating-linear-gradient(
      -12deg,
      transparent,
      transparent 2px,
      rgba(255, 255, 255, 0.02) 2px,
      rgba(255, 255, 255, 0.02) 3px
    );
}}
.shell {{
  position: relative;
  max-width: 42rem;
  margin: 0 auto;
  padding: 1.25rem 1.15rem 3.5rem;
}}
.panel {{
  background: color-mix(in srgb, var(--paper) 92%, white);
  border: 1px solid var(--line);
  border-radius: 2px;
  padding: 1.75rem 1.4rem 2rem;
  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.28);
}}
.hero-main .panel {{
  min-height: min(72vh, 36rem);
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  background:
    linear-gradient(
      180deg,
      transparent 0%,
      color-mix(in srgb, var(--paper) 95%, white) 55%
    ),
    radial-gradient(
      ellipse at 70% 20%,
      rgba(200, 160, 90, 0.35),
      transparent 55%
    ),
    linear-gradient(145deg, #d8e0cc, var(--sand));
}}
.brand-mark {{
  margin: 0 0 0.75rem;
  font-size: 0.8rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 600;
}}
.brand-hero {{
  font-family: "Fraunces", "Palatino Linotype", serif;
  font-weight: 650;
  font-size: clamp(2.4rem, 7vw, 3.4rem);
  line-height: 1.05;
  margin: 0 0 0.85rem;
  letter-spacing: -0.02em;
}}
h1 {{
  font-family: "Fraunces", "Palatino Linotype", serif;
  font-weight: 650;
  font-size: 1.85rem;
  margin: 0 0 0.65rem;
  letter-spacing: -0.02em;
}}
h2 {{
  font-family: "Fraunces", serif;
  font-size: 1.15rem;
  margin: 1.25rem 0 0.5rem;
}}
p, li {{ line-height: 1.55; }}
.lede {{ font-size: 1.05rem; max-width: 34rem; }}
.muted {{ color: var(--muted); font-size: 0.95rem; }}
.error {{ color: var(--err); }}
.steps {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem 0.75rem;
  margin: 0 0 1rem;
  padding: 0;
  font-size: 0.78rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}}
.step {{ color: rgba(243, 238, 228, 0.45); }}
.step.done {{ color: rgba(243, 238, 228, 0.75); }}
.step.active {{
  color: var(--sand);
  font-weight: 600;
  border-bottom: 2px solid color-mix(in srgb, var(--sand) 70%, transparent);
}}
.choice {{
  display: block;
  width: 100%;
  margin: 0.65rem 0;
  padding: 1rem 1.1rem;
  text-align: left;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.45);
  color: var(--ink);
  font: inherit;
  cursor: pointer;
  transition: border-color 0.15s ease, background 0.15s ease;
}}
.choice:hover {{
  border-color: var(--leaf);
  background: rgba(255, 255, 255, 0.75);
}}
.choice strong {{
  display: block;
  font-family: "Fraunces", serif;
  font-size: 1.1rem;
  margin-bottom: 0.25rem;
}}
.card {{
  margin: 0.85rem 0;
  padding: 0.95rem 1rem;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.4);
}}
button, .btn {{
  display: inline-block;
  margin: 0.4rem 0.4rem 0.4rem 0;
  padding: 0.65rem 1.05rem;
  border: 1px solid var(--leaf);
  background: var(--leaf);
  color: var(--paper);
  text-decoration: none;
  cursor: pointer;
  font: inherit;
  font-weight: 500;
}}
button:hover, .btn:hover {{ background: var(--leaf-hot); }}
button.secondary, a.secondary {{
  background: transparent;
  color: var(--leaf);
}}
button.full {{ width: 100%; text-align: center; }}
ol.setup-steps {{
  padding-left: 1.2rem;
  margin: 0.5rem 0 1rem;
}}
ol.setup-steps li {{ margin: 0.35rem 0; }}
input[type=text], input[type=password], textarea {{
  width: 100%;
  padding: 0.6rem 0.7rem;
  margin: 0.35rem 0 0.85rem;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.7);
  font: inherit;
}}
img, video {{
  max-width: 100%;
  display: block;
  margin: 0.7rem 0 1rem;
  background: #1a1a1a;
}}
.seg-progress {{
  font-size: 0.85rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 0 0 0.5rem;
}}
@media (max-width: 520px) {{
  .shell {{ padding: 0.85rem 0.75rem 2.5rem; }}
  .panel {{ padding: 1.35rem 1rem 1.6rem; }}
}}
</style>
</head>
<body>
<div class="shell {main_class}">
{rail}
{brand}
<div class="panel">
{body}
</div>
</div>
</body>
</html>
"""


def render_intro() -> str:
    body = """
<p class="brand-hero">Super Processor</p>
<p class="lede">Enhance the video you already shot. Gemini diagnoses issues;
FFmpeg applies allowlisted repairs — no invented frames, faces, or scenes.</p>
<p class="muted">Localhost website · media stays on your machine</p>
<form method="post" action="/intro">
<button type="submit" class="full">Continue</button>
</form>
"""
    return _page("Introduction", body, hero=True)


def render_llm_choice(*, error: str | None = None) -> str:
    err = f'<p class="error">{escape(error)}</p>' if error else ""
    body = f"""
<h1>What type of LLM do you have now?</h1>
<p class="lede">Pick how diagnosis should run. Media never leaves this machine
except Gemini API calls you authorize.</p>
{err}
<form method="post" action="/llm">
<button class="choice" type="submit" name="choice" value="gemini">
<strong>A. I don’t have a local LLM</strong>
<span class="muted">Use a Google AI Studio (Gemini) API key</span>
</button>
<button class="choice" type="submit" name="choice" value="local">
<strong>B. I have a local LLM</strong>
<span class="muted">Configure a local model (coming later)</span>
</button>
</form>
"""
    return _page("LLM type", body, phase="setup")


def render_local_llm_stub() -> str:
    body = """
<h1>Local LLM setup</h1>
<p class="lede">Local models are not wired yet. This screen holds the place
for the full setup checklist.</p>
<ol class="setup-steps">
<li>Choose a local runtime (Ollama, LM Studio, …)</li>
<li>Pull a multimodal model that accepts image frames</li>
<li>Expose an OpenAI-compatible or native endpoint</li>
<li>Point Super Processor at that endpoint</li>
<li>Set context / vision limits for frame batches</li>
<li>Smoke-test a still diagnosis</li>
<li>Return here to continue the wizard</li>
</ol>
<p class="muted">For now, continue with Gemini so you can finish a job today.</p>
<form method="post" action="/local-llm">
<button type="submit" name="action" value="gemini">Use Gemini instead</button>
</form>
"""
    return _page("Local LLM", body, phase="setup")


def render_setup(*, has_key: bool, error: str | None = None) -> str:
    status = (
        "A Gemini API key is already configured on this machine."
        if has_key
        else "Follow these steps, then paste your key below."
    )
    err = f'<p class="error">{escape(error)}</p>' if error else ""
    body = f"""
<h1>Gemini API setup</h1>
<p>{escape(status)}</p>
{err}
<ol class="setup-steps">
<li>Open <a href="https://aistudio.google.com/apikey" target="_blank"
rel="noreferrer">Google AI Studio</a></li>
<li>Accept the terms if prompted</li>
<li>Click to create an API key</li>
<li>Copy the key</li>
<li>Paste it here</li>
</ol>
<form method="post" action="/setup">
<label>API key
<input name="api_key" type="password" autocomplete="off"
placeholder="AIza…">
</label>
<button type="submit">Save and continue</button>
</form>
"""
    if has_key:
        body += (
            '<form method="post" action="/setup">'
            '<input type="hidden" name="skip" value="1">'
            '<button type="submit" class="secondary">Use existing key</button>'
            "</form>"
        )
    return _page("Gemini setup", body, phase="setup")


def render_pick(*, error: str | None = None) -> str:
    err = f'<p class="error">{escape(error)}</p>' if error else ""
    body = f"""
<h1>Pick a video file</h1>
<p class="lede">Choose a short clip. A local GPU model restores and upscales it.
The first pass usually covers the whole file.</p>
<p class="muted">Up to 1080p and 30 minutes.</p>
{err}
<form method="post" action="/pick">
<label>Absolute path to video
<input name="path" type="text" placeholder="/home/you/clip.mp4">
</label>
<button type="submit" class="full">Pick a video file</button>
</form>
"""
    return _page("Pick a video", body, phase="file")


def render_rendering() -> str:
    body = """
<h1>Upscaling</h1>
<p class="lede">Restoring and upscaling on the local GPU. FFmpeg only trims,
splices, and copies audio.</p>
<p class="muted">A short clip finishes sooner than a longer one. The page
will refresh.</p>
<meta http-equiv="refresh" content="2;url=/render?run=1">
"""
    return _page("Upscaling", body, phase="upscale")


def render_result(
    *,
    output_url: str,
    error: str | None = None,
    scale: int | None = None,
    strength: float | None = None,
) -> str:
    err = f'<p class="error">{escape(error)}</p>' if error else ""
    knobs = ""
    if scale is not None and strength is not None:
        knobs = (
            f'<p class="muted">Scale {scale}× · strength {strength:.2f}. '
            "Lower strength means less invented texture.</p>"
        )
    body = f"""
<h1>Result</h1>
{err}
{knobs}
<video controls src="{escape(output_url)}"></video>
<p class="lede">Are you happy, or do you want to say something?</p>
<form method="post" action="/result">
<button type="submit" name="mood" value="happy">A. I am happy</button>
</form>
<form method="post" action="/result" class="card">
<label><strong>B. Tell me what to change</strong>
<textarea name="note" rows="3"
placeholder="e.g. reduce artificial detail from 00:30 to 00:40"></textarea>
</label>
<button type="submit">Revise this range</button>
</form>
"""
    return _page("Result", body, phase="result")


def render_done() -> str:
    body = """
<h1>Done</h1>
<p class="lede">Your enhanced file is ready in the job directory as
<code>output.mp4</code>.</p>
<p class="muted">You can close this tab or start another job with
<code>super-processor review</code>.</p>
"""
    return _page("Done", body, phase="result")
