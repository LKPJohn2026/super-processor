"""HTML templates for the localhost Gemini wizard."""

from __future__ import annotations

from html import escape

from .gemini import SegmentEnhanceResult, SplitProposal
from .probe import MediaFacts
from .segments import TimelineSegment


def _page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{escape(title)}</title>
<style>
body {{
  margin: 0;
  min-height: 100vh;
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  color: #1a1f16;
  background:
    radial-gradient(ellipse at 20% 0%, #d9e8c8 0%, transparent 55%),
    radial-gradient(ellipse at 90% 20%, #f0d9b5 0%, transparent 45%),
    linear-gradient(160deg, #f7f3ea 0%, #e7efe0 48%, #f3ebe2 100%);
}}
main {{
  max-width: 42rem;
  margin: 0 auto;
  padding: 2.5rem 1.25rem 4rem;
}}
h1 {{
  font-family: "Fraunces", "Palatino Linotype", serif;
  font-weight: 600;
  font-size: 2rem;
  margin: 0 0 0.75rem;
}}
p, li {{ line-height: 1.5; }}
.card {{
  margin: 1rem 0;
  padding: 1rem 1.1rem;
  background: rgba(255,255,255,0.55);
  border: 1px solid rgba(40,50,30,0.12);
}}
button, .btn {{
  display: inline-block;
  margin: 0.35rem 0.35rem 0.35rem 0;
  padding: 0.55rem 0.9rem;
  border: 1px solid #2c3a22;
  background: #2c3a22;
  color: #f7f3ea;
  text-decoration: none;
  cursor: pointer;
  font: inherit;
}}
button.secondary, a.secondary {{
  background: transparent;
  color: #2c3a22;
}}
input[type=text], input[type=password], textarea {{
  width: 100%;
  box-sizing: border-box;
  padding: 0.5rem;
  margin: 0.4rem 0 0.8rem;
  font: inherit;
}}
img, video {{
  max-width: 100%;
  display: block;
  margin: 0.6rem 0 1rem;
  background: #222;
}}
.muted {{ color: #5a6350; font-size: 0.95rem; }}
.error {{ color: #7a1f1f; }}
</style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
"""


def render_intro() -> str:
    body = """
<h1>Super Processor</h1>
<p>Enhance the video you already shot. We do not invent new frames, faces, or
scenes — Gemini diagnoses issues; FFmpeg applies allowlisted repairs.</p>
<p class="muted">Localhost wizard · media stays on your machine</p>
<form method="post" action="/intro"><button type="submit">Continue</button></form>
"""
    return _page("Introduction", body)


def render_setup(*, has_key: bool, error: str | None = None) -> str:
    status = (
        "A Gemini API key is already configured."
        if has_key
        else ("Paste a Google AI Studio API key to continue.")
    )
    err = f'<p class="error">{escape(error)}</p>' if error else ""
    body = f"""
<h1>Gemini setup</h1>
<p>{escape(status)}</p>
{err}
<div class="card">
<ol>
<li>Open <a href="https://aistudio.google.com/apikey"
target="_blank" rel="noreferrer">Google AI Studio</a></li>
<li>Accept the terms if prompted</li>
<li>Create an API key</li>
<li>Copy the key</li>
<li>Paste it below</li>
</ol>
</div>
<p class="muted">Local LLM support comes later. This wizard is Gemini-only.</p>
<form method="post" action="/setup">
<label>API key<input name="api_key" type="password" autocomplete="off"></label>
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
    return _page("Gemini setup", body)


def render_pick(*, error: str | None = None) -> str:
    err = f'<p class="error">{escape(error)}</p>' if error else ""
    body = f"""
<h1>Pick a video file</h1>
<p>Choose a local file to start. Sampling and Gemini analysis usually take
about 1–3 minutes.</p>
{err}
<form method="post" action="/pick">
<label>Absolute path to video
<input name="path" type="text" placeholder="/home/you/clip.mp4">
</label>
<button type="submit">Analyze</button>
</form>
"""
    return _page("Pick a video", body)


def render_analyzing() -> str:
    body = """
<h1>Analyzing</h1>
<p>Sampling frames and asking Gemini for split layouts…</p>
<p class="muted">This usually takes about 1–3 minutes. The page will refresh.</p>
<meta http-equiv="refresh" content="1;url=/analyze?run=1">
"""
    return _page("Analyzing", body)


def render_overview(
    *,
    facts: MediaFacts,
    highlights: list[str],
) -> str:
    bullets = "".join(f"<li>{escape(item)}</li>" for item in highlights) or (
        "<li>No highlight bullets returned</li>"
    )
    audio = "yes" if facts.has_audio else "no"
    video = facts.primary_video()
    if video is not None and video.width and video.height:
        size = f"{video.width}×{video.height}"
        fps = video.avg_frame_rate or "?"
        video_line = f"{size} @ {fps} fps"
    else:
        video_line = "unknown"
    duration = facts.duration_s if facts.duration_s is not None else 0.0
    body = f"""
<h1>Quick overview</h1>
<div class="card">
<p><strong>Duration:</strong> {duration:.1f}s</p>
<p><strong>Video:</strong> {escape(video_line)}</p>
<p><strong>Audio:</strong> {audio}</p>
</div>
<ul>{bullets}</ul>
<form method="post" action="/overview">
<button type="submit">Choose a split</button>
</form>
"""
    return _page("Overview", body)


def render_split_choice(
    *,
    proposal: SplitProposal,
    error: str | None = None,
) -> str:
    err = f'<p class="error">{escape(error)}</p>' if error else ""
    cards: list[str] = []
    for index, layout in enumerate(proposal.layouts):
        segs = "".join(
            f"<li>{escape(f'{seg.start_s:.0f}–{seg.end_s:.0f}s {seg.label}')}</li>"
            for seg in layout.segments
        )
        cards.append(
            f'<div class="card"><form method="post" action="/split">'
            f'<input type="hidden" name="layout" value="{index}">'
            f"<p><strong>Option {index + 1}</strong> — "
            f"{layout.segment_count} segments</p>"
            f"<p>{escape(layout.summary)}</p><ul>{segs}</ul>"
            f'<button type="submit">Use this split</button></form></div>'
        )
    body = f"""
<h1>How should we split this?</h1>
<p>As Gemini sees it, this video can be worked as independent segments.
Pick the layout that best describes the video, or tell us something else.</p>
{err}
{"".join(cards)}
<form method="post" action="/split">
<label>Something else
<textarea name="note" rows="3"
placeholder="e.g. add a segment for the lat pulldown"></textarea>
</label>
<button type="submit">Revise split</button>
</form>
"""
    return _page("Split choice", body)


def render_enhance(
    *,
    segment: TimelineSegment,
    result: SegmentEnhanceResult,
    preview_url: str | None,
    error: str | None = None,
) -> str:
    err = f'<p class="error">{escape(error)}</p>' if error else ""
    still = ""
    if segment.still_path:
        still = f'<img src="/{escape(segment.still_path[:-4])}.bmp" alt="keyframe">'
    preview = ""
    if preview_url:
        preview = f'<video controls src="{escape(preview_url)}"></video>'
    options = []
    for option in result.options:
        options.append(
            f'<div class="card"><form method="post" action="/segment">'
            f'<input type="hidden" name="option" value="{escape(option.id)}">'
            f"<p><strong>{escape(option.id)}</strong> {escape(option.label)}</p>"
            f'<button type="submit">Preview this</button></form></div>'
        )
    accept = ""
    if preview_url:
        accept = (
            '<form method="post" action="/segment">'
            '<input type="hidden" name="accept" value="1">'
            '<button type="submit">Accept and continue</button></form>'
        )
    span = f"{segment.start_s:.0f}–{segment.end_s:.0f}s"
    body = f"""
<h1>Segment {segment.index}: {escape(span)}</h1>
<p>{escape(segment.context)} · {escape(segment.problem)}</p>
{still}
<p>Issues: {escape(", ".join(result.issues) or "none listed")}</p>
{err}
{"".join(options)}
{preview}
{accept}
<form method="post" action="/segment">
<label>Something else
<textarea name="note" rows="3"
placeholder="e.g. warmer white balance, less denoise"></textarea>
</label>
<p class="muted">Describe how to improve these options. Gemini will return a
new set; repeat until you preview and accept one.</p>
<button type="submit">Revise options</button>
</form>
"""
    return _page("Enhance segment", body)


def render_rendering() -> str:
    body = """
<h1>Rendering</h1>
<p>Encoding each segment and concatenating the final file…</p>
<p class="muted">This often takes about 5–10 minutes for a short clip; longer
files take longer. The page will refresh.</p>
<meta http-equiv="refresh" content="1;url=/render?run=1">
"""
    return _page("Rendering", body)


def render_result(*, output_url: str, error: str | None = None) -> str:
    err = f'<p class="error">{escape(error)}</p>' if error else ""
    body = f"""
<h1>Result</h1>
{err}
<video controls src="{escape(output_url)}"></video>
<p>Do you feel happy with the result, or should we revise?</p>
<form method="post" action="/result">
<button type="submit" name="mood" value="happy">I am happy</button>
<button type="submit" name="mood" value="revise" class="secondary">
Something else</button>
</form>
"""
    return _page("Result", body)


def render_done() -> str:
    body = """
<h1>Done</h1>
<p>Your enhanced file is ready in the job directory as <code>output.mp4</code>.</p>
"""
    return _page("Done", body)
