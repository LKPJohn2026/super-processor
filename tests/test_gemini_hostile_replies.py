"""Hostile JSON from Gemini never crashes the wizard or reaches a plan.

Every place that reads a model reply or a Gemini API answer either raises a
clean error (GeminiError, UpscaleError, ...) or keeps only bounded values:
finite numbers in range, capped counts and text, known names, and state that
``json.dumps(..., allow_nan=False)`` accepts. The transport is faked; nothing
here touches the network or needs FFmpeg.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import re
import time
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from super_processor import gemini
from super_processor.gemini import (
    ChatTurn,
    GeminiClient,
    GeminiError,
    GeminiVideo,
    ModelInfo,
    UrllibGeminiTransport,
    _offered_scale,
    _parse_expiry,
    _parse_json_object,
    contents_from_chat,
    load_chat,
    parse_model_info,
    summarize_gemini_error,
)
from super_processor.look import LOOK_BOUNDS, LookError, ShotLook
from super_processor.regions import (
    KINDS,
    MAX_REGIONS,
    Region,
    region_from_model,
    strength_expr,
)
from super_processor.segment_loop import (
    MAX_LOOP_SEGMENTS,
    LoopSegment,
    SegmentList,
    SegmentLoopError,
    _proposed_spans,
    check_coverage,
    load_segment_list,
    repair_coverage,
    save_segment_list,
    whole_video,
)
from super_processor.segments import SEGMENTS_FILE_NAME, STILLS_DIR_NAME
from super_processor.shot_plan import (
    CHECK_PROBLEMS,
    DEFAULT_STRENGTH,
    PROTECTED_STRENGTH,
    ShotCheck,
    ShotPlan,
    ShotRecipe,
    apply_checks,
    apply_proposals,
    apply_regions,
    look_from_model,
    new_plan,
    reply_scale,
    save_plan,
)
from super_processor.shots import (
    CONTENTS,
    ISSUES,
    MAX_SHOTS,
    Shot,
    ShotList,
    apply_labels,
    load_shots,
    save_shots,
)
from super_processor.upscale import (
    UpscaleError,
    UpscalePlan,
    UpscaleSpan,
    load_upscale_plan,
    replace_overlapping,
    save_upscale_plan,
)
from super_processor.wizard import (
    GEMINI_VIDEO_FILE,
    WizardController,
    WizardState,
    WizardStep,
    save_job_wizard_state,
    save_session_state,
)
from super_processor.wizard_pages import _look_card

# Errors that mean our code tripped over a reply instead of refusing it.
CRASHES = (
    KeyError,
    TypeError,
    IndexError,
    AttributeError,
    RecursionError,
    OverflowError,
    ZeroDivisionError,
    MemoryError,
)

# Fake key, joined at run time so no source line looks like a real key.
KEY = "AI" + "za" + "HOSTILE0" * 4 + "abc"
assert len(KEY) == 39

LONG = "x" * 100_000
TRAVERSAL = "../../etc/passwd"
WINDOWS = "C:\\Windows\\system32\\cmd.exe"
FILTER_META = "',;[]:=\\ scale=1:1,eq"
CONTROL = "a\x00b\x07c\u2028d\u2029<script>alert(1)</script>\r\n"
HOSTILE_STRINGS = ("1", "", LONG, TRAVERSAL, WINDOWS, FILTER_META, CONTROL)

# Every field of a reply is replaced by each of these in turn.
HOSTILE_VALUES: tuple[Any, ...] = (
    None,
    True,
    False,
    *HOSTILE_STRINGS,
    {"nested": {"x": 1}},
    [1, 2, 3],
    math.nan,
    math.inf,
    -math.inf,
    1e308,
    -1e308,
    -1,
    0,
    2**63,
    10**30,
    1e-320,
    2.5,
    999,
    -999,
)

# Whole reply texts that are not a usable object.
RAW_TEXTS: tuple[str, ...] = (
    "not json at all",
    '{"shots": [',
    "\ufeff{}",
    '```json\n{"shots": []}\n```',
    "[]",
    '"just a string"',
    "42",
    "null",
    "true",
    "NaN",
    "Infinity",
    "{" + '"a": ' * 2000 + "1" + "}" * 2000,
    "[" * 2000 + "]" * 2000,
    '{"a":' + "[" * 100_000 + "]" * 100_000 + "}",
    '{"a": 1e999, "b": -1e999, "c": NaN, "d": -Infinity}',
    '{"shots": "\x00\u2028"}',
    "",
)

# Shapes of the reply object that each method should refuse or survive.
WRONG_ROOTS: tuple[Any, ...] = ([], "s", 1, None, True, [{"shots": []}])


@pytest.fixture(autouse=True)
def _one_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUPER_PROCESSOR_GEMINI_CANDIDATES", raising=False)
    monkeypatch.delenv("SUPER_PROCESSOR_GEMINI_MODEL", raising=False)


class _Replies:
    """Answers every generate call with ``text`` as the model's reply."""

    def __init__(self, text: str) -> None:
        self.text = text

    def generate(self, **_: Any) -> dict[str, Any]:
        return {"candidates": [{"content": {"parts": [{"text": self.text}]}}]}


def _text(payload: Any) -> str:
    # json.dumps writes NaN and Infinity as bare tokens, as a model might.
    return json.dumps(payload)


def _client(text: str) -> GeminiClient:
    return GeminiClient(api_key=KEY, model="gemini-3.8-flash", transport=_Replies(text))


def _strict(value: Any) -> str:
    """``value`` as strict JSON; fails the test on NaN or Infinity."""
    return json.dumps(value, allow_nan=False)


def _still(tmp_path: Path) -> Path:
    path = tmp_path / "job" / "still.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8\xff\xd9")
    return path


def _bounded_text(value: Any, cap: int) -> None:
    assert isinstance(value, str)
    assert len(value) <= cap


# Most a logged reply may take in the chat replayed with every later call.
MAX_TURN_BYTES = 64_000


def _check_chat(job_dir: Path) -> None:
    """The saved chat is strict JSON, small per turn, and holds no long text."""
    chat = job_dir / gemini.CHAT_FILE_NAME
    if not chat.exists():
        return
    text = chat.read_text(encoding="utf-8")
    assert LONG[:1000] not in text
    for turn in json.loads(text)["turns"]:
        assert len(_strict(turn)) < MAX_TURN_BYTES
    assert load_chat(job_dir) is not None


# Calls ----------------------------------------------------------------------


def _calls(
    still: Path, job_dir: Path | None
) -> dict[str, Callable[[GeminiClient], Any]]:
    """One call per method that reads a model reply."""
    return {
        "probe": lambda c: c.pick_working_model(),
        "label_shots": lambda c: c.label_shots(
            shots=[{"start_s": 0.0, "end_s": 1.0}],
            stills=[still],
            issues=ISSUES,
            contains=CONTENTS,
            job_dir=job_dir,
        ),
        "propose_looks": lambda c: c.propose_looks(
            shots=[{"index": 0, "start_s": 0.0, "end_s": 1.0}],
            stills=[still],
            bounds=LOOK_BOUNDS,
            scales=(2,),
            protected_strength=PROTECTED_STRENGTH,
            job_dir=job_dir,
        ),
        "propose_segments": lambda c: c.propose_segments(
            stills=[still],
            times=[0.0],
            duration_s=10.0,
            contains=CONTENTS,
            scales=(1, 2),
            job_dir=job_dir,
        ),
        "locate_regions": lambda c: c.locate_regions(
            shots=[{"index": 0}],
            stills=[[still]],
            kinds=KINDS,
            max_regions=MAX_REGIONS,
            job_dir=job_dir,
        ),
        "check_previews": lambda c: c.check_previews(
            shots=[{"index": 0}],
            pairs=[(still, still)],
            problems=CHECK_PROBLEMS,
            job_dir=job_dir,
        ),
        "revise": lambda c: c.revise_upscale_params(
            note="less texture",
            prior_scale=2,
            prior_strength=0.5,
            prior_start_s=0.0,
            prior_end_s=5.0,
            duration_s=5.0,
            job_dir=job_dir,
        ),
    }


METHODS = tuple(_calls(Path("x"), None))


def _call(name: str, text: str, tmp_path: Path, *, log: bool = True) -> Any:
    """Run method ``name`` against ``text``; a clean refusal returns None.

    With ``log`` the job's chat is kept and checked after every call.
    """
    still = _still(tmp_path)
    job_dir = still.parent if log else None
    try:
        return _calls(still, job_dir)[name](_client(text))
    except GeminiError:
        return None
    finally:
        if job_dir is not None:
            _check_chat(job_dir)


@pytest.mark.parametrize("name", METHODS)
@pytest.mark.parametrize("text", RAW_TEXTS, ids=range(len(RAW_TEXTS)))
def test_a_raw_reply_is_an_object_or_a_clean_error(
    name: str, text: str, tmp_path: Path
) -> None:
    result = _call(name, text, tmp_path)
    if result is not None:
        _strict(result)
    # The chat replayed next time is still readable and holds no NaN.
    _check_chat(tmp_path / "job")


@pytest.mark.parametrize("name", METHODS)
@pytest.mark.parametrize("root", WRONG_ROOTS, ids=repr)
def test_a_reply_of_the_wrong_type_is_refused(
    name: str, root: Any, tmp_path: Path
) -> None:
    assert _call(name, _text(root), tmp_path) is None


@pytest.mark.parametrize("name", METHODS)
def test_an_object_without_the_expected_list_is_refused_or_empty(
    name: str, tmp_path: Path
) -> None:
    payloads: list[dict[str, Any]] = [
        {},
        {"shots": None},
        {"segments": {}},
        {"shots": "x"},
    ]
    for payload in payloads:
        result = _call(name, _text(payload), tmp_path)
        if name == "probe":
            # The probe only needs an object back.
            assert result == "gemini-3.8-flash"
        elif name == "revise":
            # Revise hands back the object; nothing in it makes a span.
            assert isinstance(result, dict)
            assert reply_scale(result.get("scale")) is None
            asked = {
                k: result[k] for k in ("start_s", "end_s", "strength") if k in result
            }
            with pytest.raises(UpscaleError):
                UpscaleSpan.from_dict({**asked, "scale": 2})
        else:
            assert result is None


def test_nesting_deeper_than_any_schema_is_refused() -> None:
    assert _parse_json_object('{"a": [[[1]]]}') == {"a": [[[1]]]}
    for depth in (40, 2_000, 100_000):
        text = '{"a":' + "[" * depth + "]" * depth + "}"
        with pytest.raises(GeminiError, match="nested too deep"):
            _parse_json_object(text)


def test_nan_tokens_in_raw_text_are_read_as_null() -> None:
    parsed = _parse_json_object('{"a": NaN, "b": [Infinity, -Infinity], "c": 1e999}')
    assert parsed == {"a": None, "b": [None, None], "c": None}
    big = _parse_json_object('{"n": ' + "9" * 30 + "}")
    assert big == {"n": None}


def test_duplicate_keys_keep_one_value() -> None:
    assert _parse_json_object('{"scale": 2, "scale": NaN}') == {"scale": None}


# Field by field ---------------------------------------------------------------


def _mutants(base: dict[str, Any], paths: list[tuple[Any, ...]]) -> Iterator[Any]:
    """``base`` with each field at ``paths`` set to each hostile value,
    removed, or joined by an unknown key."""
    for path in paths:
        for value in (*HOSTILE_VALUES, "__missing__"):
            reply = copy.deepcopy(base)
            parent = reply
            for key in path[:-1]:
                parent = parent[key]
            if value == "__missing__":
                if isinstance(parent, dict):
                    parent.pop(path[-1], None)
                else:
                    parent.pop(path[-1])
            else:
                parent[path[-1]] = value
            yield reply
    extra = copy.deepcopy(base)
    extra["unknown"] = {"anything": [LONG]}
    yield extra


def _fields(base: dict[str, Any]) -> list[tuple[Any, ...]]:
    """Every path in ``base``, objects and list items alike."""
    paths: list[tuple[Any, ...]] = []

    def walk(node: Any, prefix: tuple[Any, ...]) -> None:
        items = (
            node.items()
            if isinstance(node, dict)
            else enumerate(node)
            if isinstance(node, list)
            else ()
        )
        for key, child in items:
            paths.append((*prefix, key))
            walk(child, (*prefix, key))

    walk(base, ())
    return paths


NEUTRAL_LOOK = {name: neutral for name, (_lo, _hi, neutral) in LOOK_BOUNDS.items()}
FILTER_TEXT = re.compile(r"[a-z0-9=:.,_\-]*")
# The blend expression's whole vocabulary: numbers, operators, and the
# fixed names it is built from.
EXPR_TEXT = re.compile(r"(?:[0-9.,()+\-*/XYWH]|min|max|clip)*")


def _shots(contains: tuple[str, ...] = ("faces", "hands", "text")) -> ShotList:
    return ShotList(
        shots=[
            Shot(start_s=0.0, end_s=5.0, label="a", contains=contains, still="s.jpg"),
            Shot(start_s=5.0, end_s=10.0, label="b", contains=contains, still="t.jpg"),
        ]
    )


def _check_look(look: ShotLook) -> None:
    for name, (low, high, _neutral) in LOOK_BOUNDS.items():
        value = getattr(look, name)
        assert isinstance(value, float) and math.isfinite(value)
        assert low <= value <= high
    # Filter text is built from numbers only, never from reply text.
    assert FILTER_TEXT.fullmatch(look.clean_filter())
    assert FILTER_TEXT.fullmatch(look.finish_filter())


def _check_plan(plan: ShotPlan) -> None:
    for recipe in plan.recipes:
        assert math.isfinite(recipe.strength) and 0.0 <= recipe.strength <= 1.0
        _check_look(recipe.look)
        _bounded_text(recipe.reason, 240)
        _bounded_text(recipe.check.note, 240)
        assert set(recipe.check.problems) <= set(CHECK_PROBLEMS)
        assert recipe.check.ok in (True, False, None)
        assert isinstance(recipe.use_upscale, bool)
    _strict(plan.to_dict())
    assert ShotPlan.from_dict(json.loads(_strict(plan.to_dict()))) == plan


LOOKS_BASE: dict[str, Any] = {
    "scale": 2,
    "shots": [
        {
            "index": 0,
            "strength": 0.4,
            "use_upscale": True,
            "look": dict(NEUTRAL_LOOK, deblock=0.5),
            "reason": "blocky sky",
        }
    ],
}


@pytest.mark.parametrize("path", _fields(LOOKS_BASE), ids=repr)
def test_hostile_look_proposals_stay_in_bounds(
    path: tuple[Any, ...], tmp_path: Path
) -> None:
    shots = _shots()
    for reply in _mutants(LOOKS_BASE, [path]):
        result = _call("propose_looks", _text(reply), tmp_path)
        if result is None:
            continue
        plan = apply_proposals(new_plan(shots, 2), shots, list(result["shots"]), [0, 1])
        _check_plan(plan)
        # Protected content without boxes is held down.
        assert plan.recipes[0].strength <= PROTECTED_STRENGTH


CHECK_BASE: dict[str, Any] = {
    "shots": [
        {
            "index": 1,
            "ok": False,
            "problems": ["waxy_skin"],
            "note": "skin looks waxy",
            "strength": 0.3,
            "look": dict(NEUTRAL_LOOK, denoise=0.4),
        }
    ]
}


@pytest.mark.parametrize("path", _fields(CHECK_BASE), ids=repr)
def test_hostile_preview_checks_stay_in_bounds(
    path: tuple[Any, ...], tmp_path: Path
) -> None:
    shots = _shots()
    for reply in _mutants(CHECK_BASE, [path]):
        verdicts = _call("check_previews", _text(reply), tmp_path)
        if verdicts is None:
            continue
        plan, changed = apply_checks(new_plan(shots, 2), shots, verdicts, [0, 1])
        _check_plan(plan)
        assert set(changed) <= {0, 1}


def test_a_check_note_is_one_short_line() -> None:
    shots = _shots()
    plan, _ = apply_checks(
        new_plan(shots, 2), shots, [{"index": 0, "ok": True, "note": LONG}], [0]
    )
    assert plan.recipes[0].check.note == "x" * 240
    plan, _ = apply_checks(
        new_plan(shots, 2), shots, [{"index": 0, "ok": True, "note": CONTROL}], [0]
    )
    assert "\n" not in plan.recipes[0].check.note
    assert "\u2028" not in plan.recipes[0].check.note
    plan, _ = apply_checks(
        new_plan(shots, 2), shots, [{"index": 0, "ok": True, "note": {"x": 1}}], [0]
    )
    assert plan.recipes[0].check.note == ""


REGIONS_BASE: dict[str, Any] = {
    "shots": [
        {
            "index": 0,
            "regions": [
                {"kind": "faces", "box_2d": [100, 200, 500, 600]},
                {"kind": "text", "box_2d": [0, 0, 1000, 1000]},
            ],
        }
    ]
}


def _check_regions(shots: ShotList) -> None:
    for shot in shots.shots:
        assert len(shot.regions) <= MAX_REGIONS
        for region in shot.regions:
            assert region.kind in KINDS and region.kind in shot.contains
            corners = (region.x0, region.y0, region.x1, region.y1)
            assert all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in corners)
            assert region.x0 < region.x1 and region.y0 < region.y1
        assert EXPR_TEXT.fullmatch(strength_expr(0.5, shot.regions))
    _strict(shots.to_dict())


@pytest.mark.parametrize("path", _fields(REGIONS_BASE), ids=repr)
def test_hostile_regions_stay_in_the_frame(
    path: tuple[Any, ...], tmp_path: Path
) -> None:
    shots = _shots()
    for reply in _mutants(REGIONS_BASE, [path]):
        found = _call("locate_regions", _text(reply), tmp_path)
        if found is None:
            continue
        _check_regions(apply_regions(shots, found, [0, 1]))


LABELS_BASE: dict[str, Any] = {
    "shots": [
        {"index": 0, "label": "a kitchen", "issues": ["blocky"], "contains": ["faces"]}
    ]
}


@pytest.mark.parametrize("path", _fields(LABELS_BASE), ids=repr)
def test_hostile_labels_stay_known_and_short(
    path: tuple[Any, ...], tmp_path: Path
) -> None:
    shots = _shots(())
    for reply in _mutants(LABELS_BASE, [path]):
        labels = _call("label_shots", _text(reply), tmp_path)
        if labels is None:
            continue
        labelled = apply_labels(shots, labels)
        assert len(labelled.shots) == len(shots.shots)
        for shot in labelled.shots:
            _bounded_text(shot.label, 80)
            assert "\n" not in shot.label and " " not in shot.label
            assert set(shot.issues) <= set(ISSUES)
            assert set(shot.contains) <= set(CONTENTS)
        _strict(labelled.to_dict())


SEGMENTS_BASE: dict[str, Any] = {
    "segments": [
        {"start_s": 0.0, "end_s": 5.0, "label": "inside", "contains": ["faces"]},
        {"start_s": 5.0, "end_s": 10.0, "label": "outside", "contains": ["text"]},
    ],
    "scale": 2,
    "scale_reason": "the source is small",
}


def _check_segments(segments: list[LoopSegment], duration_s: float) -> None:
    assert 1 <= len(segments) <= MAX_LOOP_SEGMENTS
    check_coverage(segments, duration_s)
    for segment in segments:
        _bounded_text(segment.label, 80)
        assert set(segment.contains) <= set(CONTENTS)
    _strict(SegmentList(segments=segments).to_dict())


@pytest.mark.parametrize("path", _fields(SEGMENTS_BASE), ids=repr)
def test_hostile_segments_still_cover_the_video(
    path: tuple[Any, ...], tmp_path: Path
) -> None:
    for reply in _mutants(SEGMENTS_BASE, [path]):
        result = _call("propose_segments", _text(reply), tmp_path)
        if result is None:
            continue
        assert result["scale"] in (None, 1, 2)
        _bounded_text(result["scale_reason"], 240)
        _check_segments(repair_coverage(result["segments"], 10.0), 10.0)


REVISE_BASE: dict[str, Any] = {
    "start_s": 1.0,
    "end_s": 4.0,
    "scale": 2,
    "strength": 0.3,
}


def _plan_five() -> UpscalePlan:
    return UpscalePlan(spans=(UpscaleSpan(0.0, 5.0, scale=2, strength=0.5),))


@pytest.mark.parametrize("path", _fields(REVISE_BASE), ids=repr)
def test_hostile_revise_replies_make_a_sane_patch_or_none(
    path: tuple[Any, ...], tmp_path: Path
) -> None:
    for reply in _mutants(REVISE_BASE, [path]):
        raw = _call("revise", _text(reply), tmp_path)
        if raw is None:
            continue
        assert reply_scale(raw.get("scale")) in (None, 1, 2)
        asked = {k: raw[k] for k in ("start_s", "end_s", "strength") if k in raw}
        try:
            patch = UpscaleSpan.from_dict({**asked, "scale": 2})
        except UpscaleError:
            continue
        assert math.isfinite(patch.start_s) and math.isfinite(patch.end_s)
        assert 0.0 <= patch.strength <= 1.0
        try:
            plan = replace_overlapping(
                _plan_five(), patch, 5.0, preserve_shot_settings=True
            )
        except UpscaleError:
            continue
        _strict(plan.to_dict())
        assert all(0.0 <= s.start_s < s.end_s <= 5.0 for s in plan.spans)


# Arrays of 50,000 ---------------------------------------------------------------


def test_fifty_thousand_segments_are_cut_down_quickly(tmp_path: Path) -> None:
    many = [
        {"start_s": i * 0.01, "end_s": (i + 1) * 0.01, "label": f"s{i}"}
        for i in range(50_000)
    ]
    started = time.perf_counter()
    result = _call("propose_segments", _text({"segments": many}), tmp_path)
    assert result is not None
    segments = repair_coverage(result["segments"], 500.0)
    assert time.perf_counter() - started < 10
    _check_segments(segments, 500.0)


def test_fifty_thousand_items_everywhere_stay_capped(tmp_path: Path) -> None:
    shots = _shots()
    many_names = ["faces", "nope", "blocky", "waxy_skin"] * 12_500
    many_boxes = [{"kind": "faces", "box_2d": [10, 10, 900, 900]}] * 50_000
    started = time.perf_counter()
    proposals = [
        {"index": i % 3, "strength": 0.9, "look": NEUTRAL_LOOK, "reason": "r"}
        for i in range(50_000)
    ]
    _check_plan(apply_proposals(new_plan(shots, 2), shots, proposals, [0, 1]))
    checks = [
        {"index": i % 2, "ok": False, "problems": many_names, "strength": 0.1}
        for i in range(50)
    ]
    plan, _ = apply_checks(new_plan(shots, 2), shots, checks, [0, 1])
    _check_plan(plan)
    found = [{"index": 0, "regions": many_boxes}, {"index": 1, "regions": []}]
    _check_regions(apply_regions(shots, found, [0, 1]))
    labels = [
        {"index": i % 2, "label": "x", "issues": many_names, "contains": many_names}
        for i in range(20)
    ]
    labelled = apply_labels(shots, labels)
    assert len(labelled.shots) == 2 <= MAX_SHOTS
    assert region_from_model("faces", [1.0] * 50_000) is None
    assert time.perf_counter() - started < 10


# Items for shots nobody asked about ------------------------------------------------

# None of these is shot 0; true and false are not 1 and 0.
OTHER_INDEXES: tuple[Any, ...] = (1, -1, 999, 2**63, True, False)


def test_reply_items_for_other_shots_change_nothing() -> None:
    shots = _shots()
    plan = new_plan(shots, 2)
    loud = dict(NEUTRAL_LOOK, contrast=1.3)
    proposals = [
        {"index": i, "strength": 0.9, "look": loud, "reason": "x", "use_upscale": False}
        for i in OTHER_INDEXES
    ]
    after = apply_proposals(plan, shots, proposals, [0])
    assert after.recipes[1] == plan.recipes[1]
    # Shot 0 got no recipe of its own, so it uses the measurements.
    assert after.recipes[0].look != look_from_model(loud)
    assert after.recipes[0].use_upscale is True
    checks = [
        {"index": i, "ok": False, "strength": 0.05, "look": loud, "note": "n"}
        for i in OTHER_INDEXES
    ]
    checked, changed = apply_checks(plan, shots, checks, [0])
    assert changed == [] and checked.recipes == plan.recipes
    box = {"kind": "faces", "box_2d": [100, 100, 600, 600]}
    found = [{"index": i, "regions": [box]} for i in OTHER_INDEXES]
    boxed = apply_regions(shots, found, [0])
    assert [shot.regions for shot in boxed.shots] == [(), ()]
    labels = [
        {"index": i, "label": "zzz", "issues": ["blocky"], "contains": ["text"]}
        for i in (-1, 999, 2**63, True, False)
    ]
    assert apply_labels(shots, labels).shots == shots.shots


def test_non_finite_spans_are_dropped() -> None:
    raw: list[dict[str, Any]] = [
        {"start_s": math.nan, "end_s": 1.0},
        {"start_s": 2.0, "end_s": math.nan},
        {"start_s": 10**400, "end_s": 1.0},
        {"start_s": -math.inf, "end_s": 2.0},
        {"start_s": 0.0, "end_s": math.inf},
        {"start_s": 1.0, "end_s": 4.0},
    ]
    spans = _proposed_spans(raw, 10.0)
    # -inf and inf are clamped to the clip; NaN and 10**400 are dropped.
    assert [(start, end) for start, end, _l, _c in spans] == [
        (0.0, 2.0),
        (0.0, 10.0),
        (1.0, 4.0),
    ]
    _check_segments(repair_coverage(raw, 10.0), 10.0)


# Small readers of reply values ----------------------------------------------------


@pytest.mark.parametrize("value", HOSTILE_VALUES, ids=repr)
def test_scale_readers_keep_an_allowed_scale(value: Any) -> None:
    assert reply_scale(value) in (None, 1, 2)
    assert _offered_scale(value, (1, 2)) in (None, 1, 2)
    assert _offered_scale(value, ()) is None
    if isinstance(value, bool):
        # true is not 1.
        assert reply_scale(value) is None
        assert _offered_scale(value, (1, 2)) is None


@pytest.mark.parametrize("value", HOSTILE_VALUES, ids=repr)
def test_a_check_verdict_is_a_real_bool_or_none(value: Any) -> None:
    check = ShotCheck.from_dict({"ok": value, "note": value})
    if isinstance(value, bool):
        assert check.ok is value
    else:
        assert check.ok is None
    _bounded_text(check.note, 240)
    assert "\n" not in check.note and "\u2028" not in check.note


@pytest.mark.parametrize("value", HOSTILE_VALUES, ids=repr)
def test_look_readers_clamp_or_refuse(value: Any) -> None:
    for name in LOOK_BOUNDS:
        _check_look(look_from_model({name: value}))
        try:
            look = ShotLook.from_dict({name: value})
        except LookError:
            continue
        _check_look(look)
    _check_look(look_from_model(value))
    with contextlib.suppress(LookError):
        _check_look(ShotLook.from_dict(value))
    if isinstance(value, bool) or (isinstance(value, float) and math.isnan(value)):
        # Neither true nor NaN is a setting: the shot keeps no change, and
        # a stored look with one is refused.
        for name in LOOK_BOUNDS:
            assert look_from_model({name: value}) == ShotLook()
            with pytest.raises(LookError):
                ShotLook.from_dict({name: value})
        shots = _shots(())
        proposal: dict[str, Any] = {"index": 0, "strength": value, "reason": "r"}
        plan = apply_proposals(new_plan(shots, 2), shots, [proposal], [0])
        assert plan.recipes[0].strength == DEFAULT_STRENGTH


def test_a_look_refuses_an_int_too_big_for_a_float() -> None:
    with pytest.raises(LookError):
        ShotLook.from_dict({"contrast": 10**400})
    # Clamped to the end it is past, not an OverflowError.
    assert look_from_model({"contrast": 10**400}).contrast == LOOK_BOUNDS["contrast"][1]
    assert (
        look_from_model({"contrast": -(10**400)}).contrast == LOOK_BOUNDS["contrast"][0]
    )


@pytest.mark.parametrize("value", HOSTILE_VALUES, ids=repr)
def test_region_reader_keeps_boxes_in_the_frame(value: Any) -> None:
    boxes: list[Any] = [
        value,
        [value, 0, 500, 500],
        [0, value, 500, 500],
        [0, 0, value, 500],
        [0, 0, 500, value],
    ]
    for kind in ("faces", value, " TEXT ", TRAVERSAL):
        for box in boxes:
            region = region_from_model(kind, box)
            if region is None:
                continue
            assert isinstance(region, Region) and region.kind in KINDS
            assert 0.0 <= region.x0 < region.x1 <= 1.0
            assert 0.0 <= region.y0 < region.y1 <= 1.0


@pytest.mark.parametrize(
    "box",
    [
        [2000, 2000, 3000, 3000],
        [-3000, -3000, -2000, -2000],
        [10**400, 0, 500, 500],
        [0, 0, 500, math.inf],
        [math.nan, 0, 500, 500],
    ],
    ids=repr,
)
def test_a_box_outside_the_frame_or_not_finite_is_dropped(box: list[Any]) -> None:
    assert region_from_model("faces", box) is None


@pytest.mark.parametrize("value", HOSTILE_VALUES, ids=repr)
def test_a_stored_span_refuses_or_stays_sane(value: Any) -> None:
    for field in ("start_s", "end_s", "scale", "strength", "look", "regions"):
        data = dict(REVISE_BASE, **{field: value})
        try:
            span = UpscaleSpan.from_dict(data)
        except UpscaleError:
            continue
        assert span.scale in (1, 2) and 0.0 <= span.strength <= 1.0
        _strict(span.to_dict())
    with pytest.raises(UpscaleError):
        UpscaleSpan.from_dict(value)
    for field in ("start_s", "scale"):
        with pytest.raises(UpscaleError):
            UpscaleSpan.from_dict(dict(REVISE_BASE, **{field: 10**400}))


@pytest.mark.parametrize("value", HOSTILE_VALUES, ids=repr)
def test_a_stored_recipe_stays_sane(value: Any) -> None:
    for field in ("strength", "look", "reason", "check", "rev", "shimmer"):
        recipe = ShotRecipe.from_dict({field: value})
        _check_look(recipe.look)
        assert 0.0 <= recipe.strength <= 1.0
        _strict(recipe.to_dict())


# Gemini API answers ---------------------------------------------------------------


class _Lister:
    def __init__(self, rows: Any) -> None:
        self.rows = rows

    def generate(self, **_: Any) -> dict[str, Any]:
        return _Replies('{"ok": true}').generate()

    def list_models(self, **_: Any) -> Any:
        return self.rows


MODEL_ROW: dict[str, Any] = {
    "name": "models/gemini-3.8-flash",
    "displayName": "Flash",
    "description": "multimodal video",
    "supportedGenerationMethods": ["generateContent"],
    "rpmLimit": 15,
    "rpmUsed": 1,
    "rateLimits": {"rpm": "1/15"},
}


MODEL_NAME = re.compile(r"[a-z0-9][a-z0-9.-]{0,63}")


def _check_models(client: GeminiClient) -> None:
    try:
        infos = client.list_model_infos()
        names = client.candidate_models(refresh=True)
    except GeminiError:
        return
    for info in infos:
        assert isinstance(info, ModelInfo)
        _strict(info.to_dict())
    assert len(names) <= gemini.MAX_CANDIDATE_MODELS
    # A name goes into a URL and into error text: plain ids only.
    assert all(MODEL_NAME.fullmatch(name) for name in names)


@pytest.mark.parametrize("path", _fields(MODEL_ROW), ids=repr)
def test_hostile_model_rows_are_skipped_or_read(path: tuple[Any, ...]) -> None:
    for row in _mutants(MODEL_ROW, [path]):
        _check_models(GeminiClient(api_key=KEY, transport=_Lister([row])))
        parse_model_info(row)


@pytest.mark.parametrize(
    "rows",
    [None, {"models": []}, "models", 7, [None, 1, "x", []], [MODEL_ROW] * 50_000],
    ids=["none", "object", "string", "number", "junk-items", "fifty-thousand"],
)
def test_a_hostile_model_list_is_refused_or_read(rows: Any) -> None:
    client = GeminiClient(api_key=KEY, transport=_Lister(rows))
    _check_models(client)
    with contextlib.suppress(GeminiError):
        assert client.pick_working_model()


@pytest.mark.parametrize(
    "rpm",
    [
        math.inf,
        -math.inf,
        math.nan,
        10**400,
        "1e999",
        "x/y",
        "9" * 5000,
        {"used": 1e999},
    ],
    ids=repr,
)
def test_rate_limits_that_are_not_counts_are_ignored(rpm: Any) -> None:
    for row in (
        {"name": "gemini-x-flash", "rpmLimit": rpm, "rpmUsed": rpm},
        {"name": "gemini-x-flash", "rateLimits": {"rpm": rpm}},
        {"name": "gemini-x-flash", "rateLimits": {"rpm": {"used": rpm, "limit": rpm}}},
    ):
        info = parse_model_info(row)
        _strict(info.to_dict())


BAD_MODEL_NAMES: tuple[str, ...] = (
    "models/gemini-flash-" + "x" * 100_000,
    "models/gemini flash <script>",
    "models/gemini-\nflash",
    "models/gemini-flash?key=1",
    "models/gemini-flash/../../x",
    "models/Gemini-Flash",
    "models/gemini-\u2028flash",
    "models/gemini-" + "a" * 60 + "-flash",
)


def test_strange_model_names_are_never_candidates() -> None:
    rows = [dict(MODEL_ROW, name=name) for name in BAD_MODEL_NAMES]
    client = GeminiClient(api_key=KEY, transport=_Lister([*rows, MODEL_ROW]))
    assert client.candidate_models(refresh=True) == ["gemini-3.8-flash"]


class _AllFail:
    """Eight models listed; every one answers with a long, broken error."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def list_models(self, **_: Any) -> list[dict[str, Any]]:
        return [dict(MODEL_ROW, name=f"models/gemini-{i}-flash") for i in range(8)]

    def generate(self, *, model: str, **_: Any) -> dict[str, Any]:
        self.calls.append(model)
        raise GeminiError(f"Gemini HTTP 429: {model} {LONG}\n{CONTROL}", status=429)


def test_when_every_model_fails_the_error_is_one_short_line() -> None:
    transport = _AllFail()
    client = GeminiClient(api_key=KEY, model="gemini-0-flash", transport=transport)
    with pytest.raises(GeminiError) as failed:
        client.pick_working_model()
    message = str(failed.value)
    assert len(transport.calls) == 8
    assert len(message) <= 500
    assert "\n" not in message and "\u2028" not in message


# The real transport, over a fake urlopen ------------------------------------------


class _Answer:
    """What urlopen hands back: ``body`` and some headers."""

    def __init__(self, body: bytes, headers: dict[str, str] | None = None) -> None:
        self.body = body
        self.headers = headers or {}

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> _Answer:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def _serve(
    monkeypatch: pytest.MonkeyPatch, answer: Callable[[int], bytes]
) -> list[str]:
    """Answer every request with ``answer(n)``; nothing leaves the machine."""
    urls: list[str] = []

    def urlopen(req: urllib.request.Request, timeout: float = 0) -> _Answer:
        del timeout
        urls.append(req.full_url)
        if req.get_header("X-goog-upload-command") == "start":
            return _Answer(b"{}", {"X-Goog-Upload-URL": "https://upload.test/s?id=1"})
        return _Answer(answer(len(urls)))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return urls


API_BODIES: tuple[bytes, ...] = (
    b"",
    b"not json",
    b"\xff\xfe\x00garbage",
    b"\xef\xbb\xbf{}",
    b"[]",
    b'"x"',
    b"7",
    b"null",
    b'{"a": NaN}',
    b'{"a": [-Infinity]}',
    b'{"a":' + b"[" * 100_000 + b"]" * 100_000 + b"}",
    b"[" * 100_000,
    b'{"truncated": ',
)


@pytest.mark.parametrize("body", API_BODIES, ids=range(len(API_BODIES)))
def test_a_broken_api_answer_is_a_gemini_error(
    body: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve(monkeypatch, lambda _n: body)
    transport = UrllibGeminiTransport()
    upload = tmp_path / "up.mp4"
    upload.write_bytes(b"video")
    calls: list[Callable[[], Any]] = [
        lambda: transport.generate(model="gemini-x", api_key=KEY, body={}),
        lambda: transport.get_file(api_key=KEY, name="files/abc"),
        lambda: transport.list_models(api_key=KEY),
        lambda: transport.upload_file(
            api_key=KEY, path=upload, mime_type="video/mp4", display_name="up.mp4"
        ),
        lambda: GeminiClient(api_key=KEY, transport=transport).pick_working_model(),
    ]
    for call in calls:
        with pytest.raises(GeminiError):
            call()


def test_a_good_api_answer_still_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file = {"name": "files/abc", "uri": "https://x.test/f", "state": "ACTIVE"}
    _serve(monkeypatch, lambda _n: json.dumps({"file": file, **file}).encode())
    transport = UrllibGeminiTransport()
    upload = tmp_path / "up.mp4"
    upload.write_bytes(b"video")
    assert transport.get_file(api_key=KEY, name="files/abc")["state"] == "ACTIVE"
    sent = transport.upload_file(
        api_key=KEY, path=upload, mime_type="video/mp4", display_name="up.mp4"
    )
    assert sent == file


@pytest.mark.parametrize("per_page", [1, 500])
@pytest.mark.parametrize(
    "token",
    [lambda n: "same", lambda n: f"t{n}", lambda n: {"t": n}, lambda n: "x" * 5000],
    ids=["same", "new-each-time", "object", "huge"],
)
def test_endless_model_pages_stop(
    token: Callable[[int], Any], per_page: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    def page(n: int) -> bytes:
        body = {"models": [MODEL_ROW] * per_page, "nextPageToken": token(n)}
        return json.dumps(body).encode()

    urls = _serve(monkeypatch, page)
    started = time.perf_counter()
    rows = UrllibGeminiTransport().list_models(api_key=KEY)
    assert time.perf_counter() - started < 5
    assert len(urls) <= 20
    assert per_page <= len(rows) <= 2000


ERROR_BODIES: tuple[str, ...] = (
    "",
    "not json",
    '{"error": "boom"}',
    '{"error": [1, 2]}',
    '{"error": {"details": 5}}',
    '{"error": {"details": [{"violations": 5}]}}',
    '{"error": {"details": [{"retryDelay": "1e999s"}]}}',
    '{"error": {"details": [{"retryDelay": "infs"}]}}',
    '{"error": {"details": [{"retryDelay": "nans"}]}}',
    '{"error": {"details": [{"retryDelay": "-5s"}]}}',
    '{"error": {"details": [{"retryDelay": "1e300s"}]}}',
    '{"error": {"message": NaN, "status": Infinity}}',
    '{"error": {"message": "' + LONG + '"}}',
    "[" * 100_000,
    '{"error": ' + "[" * 100_000 + "]" * 100_000 + "}",
    "\ufeff{}",
    '[{"error": {}}]',
    "null",
)


@pytest.mark.parametrize("body", ERROR_BODIES, ids=range(len(ERROR_BODIES)))
@pytest.mark.parametrize("status", [400, 404, 429, 500])
def test_an_error_body_becomes_one_short_line(body: str, status: int) -> None:
    line = summarize_gemini_error(status, body)
    assert isinstance(line, str)
    assert len(line) <= 200
    assert "\n" not in line


def test_a_retry_delay_too_long_to_wait_is_not_shown() -> None:
    body = '{"error": {"details": [{"retryDelay": "1e300s"}]}}'
    assert summarize_gemini_error(429, body) == "quota used up"
    body = '{"error": {"details": [{"retryDelay": "90s"}]}}'
    assert (
        summarize_gemini_error(429, body) == "quota used up; try again in about 2 min"
    )


FILE_BASE: dict[str, Any] = {
    "name": "files/abc-123",
    "uri": "https://example.test/files/abc-123",
    "mimeType": "video/mp4",
    "state": "ACTIVE",
    "expirationTime": "2030-01-01T00:00:00.123456789Z",
}


class _Files:
    """File API double: the upload and every later look at it answer ``file``."""

    def __init__(self, file: Any) -> None:
        self.file = file
        self.deleted: list[str] = []

    def generate(self, **_: Any) -> dict[str, Any]:  # pragma: no cover - unused
        raise AssertionError("no generate expected")

    def upload_file(self, **_: Any) -> Any:
        return self.file

    def get_file(self, **_: Any) -> Any:
        return self.file

    def delete_file(self, *, name: str, **_: Any) -> None:
        raise ValueError(f"cannot delete {name}")


def _upload(file: Any, tmp_path: Path) -> GeminiVideo | None:
    clock = iter(range(0, 10_000, 100))
    client = GeminiClient(api_key=KEY, transport=_Files(file))
    source = tmp_path / "copy.mp4"
    source.write_bytes(b"x")
    try:
        return client.upload_video(
            source, clock=lambda: float(next(clock)), sleep=lambda _s: None
        )
    except GeminiError:
        return None


def _check_video(video: GeminiVideo) -> None:
    assert re.fullmatch(r"files/[A-Za-z0-9][A-Za-z0-9_-]{0,127}", video.name)
    assert math.isfinite(video.expires_at)
    # Both are sent back to Gemini with every request that watches it.
    assert video.uri.startswith("https://") and len(video.uri) <= 2048
    assert re.fullmatch(r"[!-~]+", video.uri)
    assert re.fullmatch(r"video/[a-z0-9.+-]+", video.mime_type)
    _strict(video.to_dict())
    again = GeminiVideo.from_dict(json.loads(_strict(video.to_dict())))
    assert again == video


@pytest.mark.parametrize("path", _fields(FILE_BASE), ids=repr)
def test_hostile_upload_answers_give_a_clean_video_or_error(
    path: tuple[Any, ...], tmp_path: Path
) -> None:
    for file in _mutants(FILE_BASE, [path]):
        video = _upload(file, tmp_path)
        if video is not None:
            _check_video(video)
        client = GeminiClient(api_key=KEY, transport=_Files(file))
        alive = client.video_alive(GeminiVideo(name="files/abc", uri="u"))
        assert alive is (file.get("state") == "ACTIVE")
        # A failed delete is logged, never raised.
        assert client.delete_video(GeminiVideo(name="files/abc", uri="u")) is False


@pytest.mark.parametrize("file", WRONG_ROOTS, ids=repr)
def test_an_upload_answer_that_is_not_an_object_is_refused(
    file: Any, tmp_path: Path
) -> None:
    assert _upload(file, tmp_path) is None
    client = GeminiClient(api_key=KEY, transport=_Files(file))
    assert client.video_alive(GeminiVideo(name="files/abc", uri="u")) is False


@pytest.mark.parametrize(
    "stamp",
    [
        "0001-01-01T00:00:00",
        "9999-12-31T23:59:59",
        "9999-12-31T23:59:59-23:59",
        "2030-01-01T00:00:00." + "9" * 5000 + "Z",
        "not a time",
        "",
        LONG,
    ],
    ids=range(7),
)
def test_an_expiry_time_is_finite_or_zero(stamp: str) -> None:
    assert math.isfinite(_parse_expiry(stamp))


@pytest.mark.parametrize("value", HOSTILE_VALUES, ids=repr)
def test_a_saved_video_record_is_finite_or_dropped(value: Any) -> None:
    base = {
        "name": "files/abc",
        "uri": "u",
        "mime_type": "video/mp4",
        "expires_at": 4e9,
    }
    for field in base:
        video = GeminiVideo.from_dict(dict(base, **{field: value}))
        if video is not None:
            assert math.isfinite(video.expires_at)
            _strict(video.to_dict())
    assert GeminiVideo.from_dict(value) is None
    big = GeminiVideo.from_dict(dict(base, expires_at=10**400))
    assert big is not None and big.expires_at == 0.0


# Saved chat --------------------------------------------------------------------


@pytest.mark.parametrize("value", HOSTILE_VALUES, ids=repr)
def test_a_saved_chat_turn_is_read_or_refused(value: Any) -> None:
    base = {"role": "model", "text": "{}", "frame_refs": [], "structured": {}}
    for field in base:
        try:
            turn = ChatTurn.from_dict(dict(base, **{field: value}))
        except GeminiError:
            continue
        assert isinstance(turn.role, str) and isinstance(turn.text, str)
        _strict(contents_from_chat([turn]))
    with pytest.raises(GeminiError):
        ChatTurn.from_dict(value)


CHAT_FILES: tuple[bytes, ...] = (
    b"",
    b"not json",
    b"\xef\xbb\xbf{}",
    b"\xff\xfe\x00garbage",
    b"[]",
    b'{"turns": 5}',
    b'{"turns": [1, "x", null]}',
    b'{"turns": [{"text": "no role"}]}',
    b'{"turns": [{"role": "user", "text": NaN, "structured": {"a": Infinity}}]}',
    b'{"turns": [' + b"[" * 100_000 + b"]" * 100_000 + b"]}",
    json.dumps({"turns": [{"role": "user", "text": LONG + CONTROL}] * 5}).encode(),
)


@pytest.mark.parametrize("raw", CHAT_FILES, ids=range(len(CHAT_FILES)))
def test_a_damaged_chat_file_is_refused_or_replayed(raw: bytes, tmp_path: Path) -> None:
    (tmp_path / gemini.CHAT_FILE_NAME).write_bytes(raw)
    try:
        turns = load_chat(tmp_path)
    except GeminiError:
        turns = None
    if turns is not None:
        _strict(contents_from_chat(turns))
    # A call that replays it fails cleanly or answers.
    still = tmp_path / "still.jpg"
    still.write_bytes(b"\xff\xd8\xff\xd9")
    try:
        _client(_text(REVISE_BASE)).revise_upscale_params(
            note="less",
            prior_scale=2,
            prior_strength=0.5,
            prior_start_s=0.0,
            prior_end_s=5.0,
            duration_s=5.0,
            job_dir=tmp_path,
        )
    except GeminiError:
        assert turns is None


# Through the wizard ----------------------------------------------------------------


class _Video:
    width = 320
    height = 240


class _Facts:
    format_name = "mov,mp4,m4a,3gp,3g2,mj2"
    has_video = True
    duration_s = 10.0

    def primary_video(self) -> _Video:
        return _Video()


def _wizard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    step: WizardStep,
    text: str,
    *,
    upload: bool = False,
) -> tuple[WizardController, Path, WizardState]:
    monkeypatch.setattr("super_processor.wizard.probe_file", lambda _path: _Facts())
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"not-a-real-video")
    controller = WizardController(tmp_path / "jobs", gemini=_client(text))
    if not upload:
        # No upload: Gemini works from the stills.
        monkeypatch.setattr(controller, "_gemini_video", lambda *_a, **_k: None)
    job_id = controller.store.create(source).job_id
    job_dir = controller.store.job_dir(job_id)
    stills = job_dir / STILLS_DIR_NAME
    stills.mkdir(parents=True, exist_ok=True)
    for ms in (0, 5000):
        (stills / f"t_{ms:08d}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    state = WizardState(step=step, job_id=job_id)
    save_job_wizard_state(job_dir, state)
    save_session_state(controller.jobs_dir, state)
    return controller, job_dir, state


# Longest text the wizard keeps or shows from a reply (an error line).
MAX_SHOWN = 500
# File names the wizard makes; reply text never names a file.
FILE_NAME = re.compile(r"[A-Za-z0-9._-]+")


def _check_values(value: Any, where: str) -> None:
    """Every string short and on one line, every float finite."""
    if isinstance(value, str):
        assert len(value) <= MAX_SHOWN, where
        for char in ("\x00", "\u2028", "\u2029", "\n"):
            assert char not in value, where
    elif isinstance(value, float):
        assert math.isfinite(value), where
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_values(key, where)
            _check_values(item, f"{where}.{key}")
    elif isinstance(value, list):
        for item in value:
            _check_values(item, where)


def _check_state(
    controller: WizardController, job_dir: Path, *, skip: tuple[str, ...] = ()
) -> dict[str, Any]:
    """api_view and every JSON file the job saved are strict and bounded.

    ``skip`` names files a test damaged on purpose.
    """
    view = controller.api_view()
    _strict(view)
    _check_values(view, "api_view")
    for path in controller.jobs_dir.rglob("*"):
        assert FILE_NAME.fullmatch(path.name), path.name
        if path.suffix != ".json" or path.name in skip:
            continue
        if path.name == gemini.CHAT_FILE_NAME:
            # Holds our own long prompts; checked on its own terms.
            _check_chat(path.parent)
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        _strict(data)
        _check_values(data, path.name)
    return view


STYLE = re.compile(r'style="([^"]*)"')
STYLE_VALUE = re.compile(r"(?:[a-z-]+:-?\d+(?:\.\d+)?%;?)+")


def _check_html(html: str) -> None:
    """Reply text on a page is escaped, short, and never styles anything."""
    assert "<script>alert" not in html
    assert LONG[:1000] not in html
    for value in STYLE.findall(html):
        assert STYLE_VALUE.fullmatch(value), value


WIZARD_SEGMENT_REPLIES: tuple[str, ...] = (
    _text(SEGMENTS_BASE),
    _text(dict(SEGMENTS_BASE, scale=math.nan, scale_reason=LONG)),
    _text(
        {
            "segments": [
                {"start_s": math.nan, "end_s": math.inf, "label": TRAVERSAL},
                {
                    "start_s": -1e308,
                    "end_s": 1e308,
                    "label": LONG,
                    "contains": [WINDOWS],
                },
                {"start_s": "1", "end_s": True, "label": FILTER_META},
            ],
            "scale": 2**63,
            "scale_reason": CONTROL,
        }
    ),
    _text(
        {
            "segments": [
                {"start_s": i * 0.01, "end_s": (i + 1) * 0.01, "label": TRAVERSAL}
                for i in range(50_000)
            ]
        }
    ),
    "{" + '"a": ' * 2000 + "1" + "}" * 2000,
    "not json",
    _text([]),
    _text(
        {
            "segments": [
                {"start_s": 0.0, "end_s": 5.0, "label": CONTROL, "contains": []},
                {"start_s": 5.0, "end_s": 10.0, "label": LONG, "contains": []},
            ],
            "scale": 2,
            "scale_reason": CONTROL,
        }
    ),
)


@pytest.mark.parametrize(
    "text", WIZARD_SEGMENT_REPLIES, ids=range(len(WIZARD_SEGMENT_REPLIES))
)
def test_hostile_segments_reach_wizard_state_bounded(
    text: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller, job_dir, state = _wizard(
        tmp_path, monkeypatch, WizardStep.PROPOSING, text
    )
    controller._propose_segments(state)
    assert controller.current_state().step is WizardStep.SEGMENTS
    found = load_segment_list(job_dir)
    assert found is not None
    _check_segments(found.segments, 10.0)
    assert found.scale in (None, 1, 2)
    _bounded_text(found.scale_reason, 240)
    view = _check_state(controller, job_dir)
    assert view["segments"] is not None
    assert len(view["segments"]["segments"]) <= MAX_LOOP_SEGMENTS
    html = controller.render()
    _check_html(html)
    if "<script>" in json.dumps(found.to_dict()):
        # Shown, but as text.
        assert "&lt;script&gt;" in html


WIZARD_REVISE_REPLIES: tuple[dict[str, Any] | str, ...] = (
    REVISE_BASE,
    dict(REVISE_BASE, strength=math.nan),
    dict(REVISE_BASE, start_s=-1e308, end_s=1e308),
    dict(REVISE_BASE, scale=2**63),
    dict(REVISE_BASE, start_s=TRAVERSAL),
    # Fields outside the schema are ignored, not applied or refused.
    dict(
        REVISE_BASE,
        source=TRAVERSAL,
        look={"contrast": 1.3},
        use_upscale=False,
        regions=[{"kind": "faces", "x0": 0, "y0": 0, "x1": 1, "y1": 1}],
    ),
    "not json",
    "{" + '"a": ' * 2000 + "1" + "}" * 2000,
)


@pytest.mark.parametrize(
    "reply", WIZARD_REVISE_REPLIES, ids=range(len(WIZARD_REVISE_REPLIES))
)
def test_a_hostile_revise_reply_reaches_wizard_state_bounded(
    reply: dict[str, Any] | str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = reply if isinstance(reply, str) else _text(reply)
    controller, job_dir, state = _wizard(tmp_path, monkeypatch, WizardStep.RESULT, text)
    save_upscale_plan(
        job_dir, UpscalePlan(spans=(UpscaleSpan(0.0, 10.0, scale=2, strength=0.5),))
    )
    controller._result_post(state, {"mood": ["unhappy"], "note": ["less texture"]})
    after = controller.current_state()
    if after.step is WizardStep.RENDERING:
        plan = load_upscale_plan(job_dir)
        assert plan is not None
        for span in (*plan.spans, plan.pending, plan.latest_revision):
            if span is None:
                continue
            assert 0.0 <= span.start_s < span.end_s <= 10.0
            assert 0.0 <= span.strength <= 1.0 and span.scale == 2
            assert span.look == ShotLook() and span.use_upscale and not span.regions
    else:
        assert after.step is WizardStep.RESULT and after.error
        assert len(after.error) <= 500
    _check_state(controller, job_dir)


def test_extra_reply_fields_do_not_block_a_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = _text(dict(REVISE_BASE, source="elsewhere", look={"bogus": 1}))
    controller, job_dir, state = _wizard(tmp_path, monkeypatch, WizardStep.RESULT, text)
    save_upscale_plan(
        job_dir, UpscalePlan(spans=(UpscaleSpan(0.0, 10.0, scale=2, strength=0.5),))
    )
    controller._result_post(state, {"mood": ["unhappy"], "note": ["less texture"]})
    after = controller.current_state()
    assert after.error is None
    assert after.step is WizardStep.RENDERING


def test_hostile_looks_reach_the_ready_view_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller, job_dir, _state = _wizard(
        tmp_path, monkeypatch, WizardStep.READY_TO_RENDER, "{}"
    )
    shots = _shots()
    plan = new_plan(shots, 2)
    proposals = [
        {
            "index": 0,
            "strength": math.nan,
            "look": {"contrast": math.inf, "gamma": -1e308, "grain": TRAVERSAL},
            "reason": LONG,
            "use_upscale": "no",
        },
        {"index": 1, "strength": 1e308, "look": [FILTER_META], "reason": CONTROL},
    ]
    plan = apply_proposals(plan, shots, proposals, [0, 1])
    checks = [{"index": 1, "ok": False, "note": LONG, "problems": [TRAVERSAL]}]
    plan, _ = apply_checks(plan, shots, checks, [0, 1])
    found = [{"index": 0, "regions": [{"kind": "faces", "box_2d": [math.nan] * 4}]}]
    shots = apply_regions(shots, found, [0, 1])
    save_shots(job_dir, shots)
    save_plan(job_dir, plan)
    view = _check_state(controller, job_dir)
    looks = view["looks"]
    assert looks is not None and len(looks["shots"]) == 2
    for item in looks["shots"]:
        _bounded_text(item["reason"], 240)
        _bounded_text(item["check"]["note"], 240)
        assert 0.0 <= item["strength"] <= 1.0
    _check_pages(controller, looks)


def _check_pages(controller: WizardController, looks: dict[str, Any]) -> None:
    """The ready page, and each shot's preview card, show reply text safely."""
    pages = [controller.render()]
    for item in looks["shots"]:
        card = dict(item, before_url="/previews/b.mp4", after_url="/previews/a.mp4")
        pages.append(_look_card(card, ""))
    html = "".join(pages)
    _check_html(html)
    shown = [
        text
        for item in looks["shots"]
        for text in (
            item["label"],
            item["reason"],
            item["check"]["note"] if item["check"]["ok"] is not None else "",
        )
    ]
    if any("<script>" in text for text in shown):
        # Shown, but as text.
        assert "&lt;script&gt;" in html


@pytest.mark.parametrize("scale", [1, 4, 2**63, 1.0, -2, True], ids=repr)
def test_the_reply_never_picks_the_size(
    scale: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The note says nothing about size, so the scale stays 2 whatever the
    # reply names, and the note still renders.
    text = _text(dict(REVISE_BASE, scale=scale))
    controller, job_dir, state = _wizard(tmp_path, monkeypatch, WizardStep.RESULT, text)
    save_upscale_plan(
        job_dir, UpscalePlan(spans=(UpscaleSpan(0.0, 10.0, scale=2, strength=0.5),))
    )
    controller._result_post(state, {"mood": ["unhappy"], "note": ["less texture"]})
    after = controller.current_state()
    assert after.error is None
    assert after.step is WizardStep.RENDERING
    plan = load_upscale_plan(job_dir)
    assert plan is not None
    assert all(span.scale == 2 for span in plan.spans)
    _check_state(controller, job_dir)


# Looks, boxes and checks through the wizard --------------------------------------


def _no_ffmpeg(
    monkeypatch: pytest.MonkeyPatch, controller: WizardController, still: Path
) -> None:
    """Previews and stills come from fakes; only Gemini's replies are real."""
    monkeypatch.setattr(
        "super_processor.wizard.render_previews",
        lambda _source, _job, _shots, plan, _indexes, **_: plan,
    )
    monkeypatch.setattr(
        "super_processor.wizard.measure_shimmer",
        lambda _job, plan, _indexes, **_: plan,
    )
    monkeypatch.setattr(
        "super_processor.wizard.locate_stills",
        lambda _source, _job, _shots, targets, **_: [[still] * 3 for _ in targets],
    )
    monkeypatch.setattr(
        "super_processor.wizard.check_stills",
        lambda _job, _plan, _index: (still, still),
    )
    monkeypatch.setattr(controller, "upscale_engine", lambda: None)


WIZARD_LOOK_REPLIES: tuple[str, ...] = (
    _text(LOOKS_BASE),
    _text(CHECK_BASE),
    _text(REGIONS_BASE),
    _text(
        {
            "scale": 2**63,
            "shots": [
                {
                    "index": 0,
                    "strength": math.nan,
                    "look": {"contrast": math.inf, "gamma": True, "grain": TRAVERSAL},
                    "reason": LONG + CONTROL,
                    "use_upscale": "no",
                    "ok": "no",
                    "problems": [TRAVERSAL, "waxy_skin"] * 25_000,
                    "note": CONTROL + LONG,
                    "regions": [
                        {"kind": "faces", "box_2d": [math.nan] * 4},
                        {"kind": FILTER_META, "box_2d": [0, 0, 1000, 1000]},
                        {"kind": "text", "box_2d": [10, 10, 900, 900]},
                    ]
                    * 10,
                },
                {"index": True, "strength": 1.0, "reason": CONTROL},
                {"index": 2**63, "ok": False, "strength": 0.0},
                {"index": -1, "regions": [{"kind": "faces", "box_2d": [0, 0, 9, 9]}]},
            ],
        }
    ),
    _text({"shots": [{"index": 1, "ok": False, "reason": CONTROL, "note": CONTROL}]}),
    *RAW_TEXTS,
)


@pytest.mark.parametrize(
    "text", WIZARD_LOOK_REPLIES, ids=range(len(WIZARD_LOOK_REPLIES))
)
def test_hostile_looks_through_the_wizard_stay_bounded(
    text: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller, job_dir, _state = _wizard(
        tmp_path, monkeypatch, WizardStep.READY_TO_RENDER, text
    )
    for name in ("s.jpg", "t.jpg"):
        (job_dir / name).write_bytes(b"\xff\xd8\xff\xd9")
    _no_ffmpeg(monkeypatch, controller, job_dir / "s.jpg")
    shots = _shots()
    save_shots(job_dir, shots)
    source = tmp_path / "clip.mp4"
    warnings: list[str] = []
    # Only GeminiError (or ShotError) is caught there; anything else escapes.
    shots = controller._locate(shots, source, job_dir, warnings)
    _check_regions(shots)
    plan = controller._preview_segment(
        job_dir,
        source,
        shots,
        new_plan(shots, 2),
        [0, 1],
        video=None,
        report=lambda _pct, _message: None,
        warnings=warnings,
    )
    save_plan(job_dir, plan)
    _check_plan(plan)
    for warning in warnings:
        assert len(warning) <= MAX_SHOWN and "\n" not in warning
    # The well-formed replies did reach the plan and the boxes.
    if text == _text(LOOKS_BASE):
        assert plan.recipes[0].reason == "blocky sky"
    if text == _text(REGIONS_BASE):
        assert [r.kind for r in shots.shots[0].regions] == ["faces", "text"]
    saved = load_shots(job_dir)
    assert saved is not None
    _check_regions(saved)
    view = _check_state(controller, job_dir)
    assert view["looks"] is not None
    _check_pages(controller, view["looks"])


# Damaged job files ----------------------------------------------------------------

# Files no reader can use: each is refused or ignored, never a crash.
DAMAGED_FILES: tuple[bytes, ...] = (
    b"",
    b"not json",
    b"\xff\xfe\x00garbage",
    b"\xef\xbb\xbf{}",
    b"[]",
    b"null",
    b'"text"',
    b"[" * 100_000 + b"]" * 100_000,
    b'{"a": ' + b"[" * 100_000 + b"]" * 100_000 + b"}",
)


@pytest.mark.parametrize("raw", CHAT_FILES, ids=range(len(CHAT_FILES)))
def test_a_damaged_chat_leaves_the_wizard_a_fallback(
    raw: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller, job_dir, state = _wizard(
        tmp_path, monkeypatch, WizardStep.PROPOSING, _text(SEGMENTS_BASE)
    )
    (job_dir / gemini.CHAT_FILE_NAME).write_bytes(raw)
    try:
        load_chat(job_dir)
    except GeminiError:
        readable = False
    else:
        readable = True
    controller._propose_segments(state)
    assert controller.current_state().step is WizardStep.SEGMENTS
    found = load_segment_list(job_dir)
    assert found is not None
    if readable:
        assert found.error is None and len(found.segments) == 2
    else:
        # The whole video is one segment, and the editor is told why.
        assert found.segments == [whole_video(10.0)]
        assert found.error is not None and gemini.CHAT_FILE_NAME in found.error
        _bounded_text(found.error, MAX_SHOWN)
    _check_state(controller, job_dir, skip=(gemini.CHAT_FILE_NAME,))


VIDEO_FILES: tuple[bytes, ...] = (
    *DAMAGED_FILES,
    b'{"video": 5, "error": 7}',
    b'{"video": {"name": "files/abc", "uri": "u", "expires_at": NaN}}',
    b'{"video": {"name": "../../etc/passwd", "uri": "https://x.test/f"}}',
    json.dumps({"video": None, "error": LONG + CONTROL}).encode(),
    json.dumps({"video": {"name": "files/abc", "uri": LONG}}).encode(),
)


def _broken_copy(*_a: Any, **_k: Any) -> Path:
    raise SegmentLoopError("no FFmpeg in this test")


@pytest.mark.parametrize("raw", VIDEO_FILES, ids=range(len(VIDEO_FILES)))
def test_a_damaged_video_record_is_ignored(
    raw: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller, job_dir, state = _wizard(
        tmp_path, monkeypatch, WizardStep.SEGMENTS, "{}", upload=True
    )
    monkeypatch.setattr("super_processor.wizard.make_gemini_copy", _broken_copy)
    record = job_dir / GEMINI_VIDEO_FILE
    record.write_bytes(raw)
    controller._forget_gemini_video(job_dir)
    record.write_bytes(raw)
    source = tmp_path / "clip.mp4"
    assert controller._gemini_video(state, job_dir, source, "proposing", 25, 75) is None
    problem = controller._video_problem(job_dir)
    _bounded_text(problem, MAX_SHOWN)
    assert "\n" not in problem
    if b"error" not in raw:
        # Nothing usable was on record, so a new copy was tried.
        assert "no FFmpeg in this test" in problem
    save_segment_list(job_dir, SegmentList(segments=[whole_video(10.0)]))
    _check_state(controller, job_dir, skip=(GEMINI_VIDEO_FILE,))
    _check_html(controller.render())


def _segment_file(**changes: Any) -> bytes:
    segments = SegmentList(
        segments=[LoopSegment(0.0, 5.0, label="a"), LoopSegment(5.0, 10.0, label="b")],
        scale=2,
        scale_reason="small source",
    )
    return json.dumps({**segments.to_dict(), **changes}).encode()


def _one_segment(**changes: Any) -> bytes:
    item = {"start_s": 0.0, "end_s": 10.0, "label": "a", "contains": []}
    return _segment_file(segments=[{**item, **changes}])


# Files that must be refused (SegmentLoopError) ...
BROKEN_SEGMENT_FILES: tuple[bytes, ...] = (
    *DAMAGED_FILES,
    _segment_file(kind="other"),
    _segment_file(segments={"0": {}}),
    _segment_file(segments=[1]),
    _one_segment(start_s=None),
    _one_segment(start_s=True),
    _one_segment(end_s=[1]),
    _one_segment(start_s=5.0, end_s=1.0),
    _one_segment(start_s=-1.0),
    _one_segment(start_s=0.5).replace(b"0.5", b"1" + b"0" * 400),
    _one_segment(start_s=0.5).replace(b"0.5", b"1e400"),
    _one_segment(start_s=0.5).replace(b"0.5", b"NaN"),
    _one_segment(end_s=0.5).replace(b"0.5", b"Infinity"),
)
# ... and files that load, with their text cut down.
ODD_SEGMENT_FILES: tuple[bytes, ...] = (
    _segment_file(note=LONG + CONTROL, error=CONTROL + LONG, scale_reason=LONG),
    _segment_file(approved="yes", revision=-5, scale=True, error=["x"]),
    _segment_file(scale=1, scale_reason=CONTROL),
    _one_segment(label=LONG + CONTROL, note=LONG, contains=[TRAVERSAL, "faces"]),
    _one_segment(label={"nested": [LONG]}, contains=["text"] * 50_000),
    _one_segment(start_s="0", end_s="10"),
)


@pytest.mark.parametrize(
    "raw",
    [*BROKEN_SEGMENT_FILES, *ODD_SEGMENT_FILES],
    ids=range(len(BROKEN_SEGMENT_FILES) + len(ODD_SEGMENT_FILES)),
)
@pytest.mark.parametrize(
    "step", [WizardStep.SEGMENTS, WizardStep.READY_TO_RENDER], ids=str
)
def test_a_damaged_segment_list_is_refused_or_cut_down(
    raw: bytes, step: WizardStep, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller, job_dir, _state = _wizard(tmp_path, monkeypatch, step, "{}")
    (job_dir / SEGMENTS_FILE_NAME).write_bytes(raw)
    if raw in BROKEN_SEGMENT_FILES:
        with pytest.raises(SegmentLoopError):
            load_segment_list(job_dir)
    else:
        found = load_segment_list(job_dir)
        assert found is not None
        for segment in found.segments:
            _bounded_text(segment.label, 80)
            _bounded_text(segment.note, 240)
            assert set(segment.contains) <= set(CONTENTS)
        _bounded_text(found.scale_reason, 240)
        _bounded_text(found.note, MAX_SHOWN)
        assert found.error is None or len(found.error) <= MAX_SHOWN
        # Only a real true approves; a bad revision starts again at 0.
        assert found.approved is False and found.revision == 0
        _strict(found.to_dict())
        _check_values(found.to_dict(), SEGMENTS_FILE_NAME)
    # The pages that read it on every poll still answer.
    _check_state(controller, job_dir, skip=(SEGMENTS_FILE_NAME,))
    _check_html(controller.render())
