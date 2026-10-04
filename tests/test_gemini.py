"""Tests for Gemini structured schemas, sampling, and chat persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from super_processor.gemini import (
    FRAME_CAP,
    ChatTurn,
    GeminiClient,
    GeminiError,
    SegmentEnhanceResult,
    SplitProposal,
    StructuredOp,
    append_chat_turn,
    frame_timestamps,
    load_chat,
    sample_fps_for_duration,
    save_chat,
    validate_enhance_result,
    validate_split_proposal,
)


def test_sample_fps_bounds() -> None:
    assert sample_fps_for_duration(30.0) == 10.0
    assert sample_fps_for_duration(600.0) == 1.0
    mid = sample_fps_for_duration(330.0)
    assert 1.0 < mid < 10.0


def test_frame_timestamps_respect_cap() -> None:
    stamps = frame_timestamps(60.0, frame_cap=20)
    assert len(stamps) <= 20
    assert stamps[0] == 0.0


def test_frame_timestamps_long_clip_uses_one_fps() -> None:
    stamps = frame_timestamps(120.0, frame_cap=FRAME_CAP)
    assert len(stamps) <= FRAME_CAP
    assert stamps[0] == 0.0


def test_chat_round_trip(tmp_path: Path) -> None:
    turns = [
        ChatTurn(role="user", text="hello", frame_refs=["a.jpg"]),
        ChatTurn(role="model", text="{}", structured={"ok": True}),
    ]
    save_chat(tmp_path, turns)
    loaded = load_chat(tmp_path)
    assert loaded[0].text == "hello"
    assert loaded[1].structured == {"ok": True}
    append_chat_turn(tmp_path, ChatTurn(role="user", text="more"))
    assert len(load_chat(tmp_path)) == 3


def test_validate_split_rejects_short_segment() -> None:
    proposal = SplitProposal.from_dict(
        {
            "highlights": [],
            "layouts": [
                {
                    "segment_count": 1,
                    "summary": "one",
                    "segments": [
                        {
                            "start_s": 0,
                            "end_s": 3,
                            "label": "short",
                            "issues": ["noisy"],
                        }
                    ],
                }
            ],
        }
    )
    with pytest.raises(GeminiError, match="shorter than"):
        validate_split_proposal(proposal, duration_s=60.0)


def test_validate_enhance_allowlist() -> None:
    result = SegmentEnhanceResult.from_dict(
        {
            "issues": ["dark"],
            "options": [
                {
                    "id": "A",
                    "label": "lift",
                    "ops": [{"op": "contrast", "params": {"brightness": 0.1}}],
                },
                {
                    "id": "B",
                    "label": "warm",
                    "ops": [{"op": "white_balance", "params": {"temperature": 5600}}],
                },
                {
                    "id": "C",
                    "label": "bad",
                    "ops": [{"op": "reframe_vertical", "params": {}}],
                },
            ],
        }
    )
    with pytest.raises(GeminiError, match="not allowlisted"):
        validate_enhance_result(result)


class _FakeTransport:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.last_body: dict[str, Any] | None = None

    def generate(
        self,
        *,
        model: str,
        api_key: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        self.last_body = body
        text = json.dumps(self.payload)
        return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def test_client_propose_splits_uses_schema(tmp_path: Path) -> None:
    payload = {
        "highlights": ["opening"],
        "layouts": [
            {
                "segment_count": 2,
                "summary": "two looks",
                "segments": [
                    {
                        "start_s": 0,
                        "end_s": 30,
                        "label": "dark",
                        "issues": ["low_light"],
                    },
                    {
                        "start_s": 30,
                        "end_s": 60,
                        "label": "bright",
                        "issues": ["low_contrast"],
                    },
                ],
            }
        ],
    }
    transport = _FakeTransport(payload)
    client = GeminiClient(api_key="test-key", transport=transport)
    frame = tmp_path / "f.jpg"
    frame.write_bytes(b"\xff\xd8\xff\xd9")
    proposal = client.propose_splits(
        frame_paths=[frame],
        duration_s=60.0,
        job_dir=tmp_path,
    )
    assert len(proposal.layouts) == 1
    assert transport.last_body is not None
    assert (
        transport.last_body["generationConfig"]["responseMimeType"]
        == "application/json"
    )
    assert len(load_chat(tmp_path)) == 2


def test_client_propose_enhance(tmp_path: Path) -> None:
    payload = {
        "issues": ["low_light"],
        "options": [
            {
                "id": "A",
                "label": "lift",
                "ops": [
                    {
                        "op": "contrast",
                        "params": {"contrast": 1.2, "brightness": 0.1, "gamma": 1.0},
                    }
                ],
            },
            {
                "id": "B",
                "label": "denoise",
                "ops": [{"op": "denoise", "params": {"strength": 0.3}}],
            },
            {
                "id": "C",
                "label": "sharpen",
                "ops": [
                    {"op": "sharpen", "params": {"luma_amount": 0.5, "luma_size": 5}}
                ],
            },
        ],
    }
    client = GeminiClient(api_key="k", transport=_FakeTransport(payload))
    result = client.propose_enhance(
        frame_paths=[],
        segment_label="low_light",
        start_s=0,
        end_s=40,
        job_dir=tmp_path,
    )
    assert len(result.options) == 3
    assert isinstance(result.options[0].ops[0], StructuredOp)


def test_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        "super_processor.gemini.resolve_secret",
        lambda _env: None,
    )
    with pytest.raises(GeminiError, match="GEMINI_API_KEY"):
        GeminiClient(api_key="")


def test_validate_split_overlap() -> None:
    proposal = SplitProposal.from_dict(
        {
            "highlights": [],
            "layouts": [
                {
                    "segment_count": 2,
                    "summary": "bad",
                    "segments": [
                        {"start_s": 0, "end_s": 20, "label": "a", "issues": ["noisy"]},
                        {"start_s": 15, "end_s": 40, "label": "b", "issues": ["noisy"]},
                    ],
                }
            ],
        }
    )
    with pytest.raises(GeminiError, match="overlap"):
        validate_split_proposal(proposal, duration_s=60.0)


def test_validate_enhance_option_count() -> None:
    op = {"op": "denoise", "params": {"strength": 0.2}}
    result = SegmentEnhanceResult.from_dict(
        {
            "issues": [],
            "options": [
                {"id": "A", "label": "a", "ops": [op]},
                {"id": "B", "label": "b", "ops": [op]},
            ],
        }
    )
    with pytest.raises(GeminiError, match="3 to 5"):
        validate_enhance_result(result)


def test_validate_split_empty_layouts() -> None:
    proposal = SplitProposal.from_dict({"highlights": [], "layouts": []})
    with pytest.raises(GeminiError, match="no split layouts"):
        validate_split_proposal(proposal, duration_s=10.0)


def test_frame_timestamps_zero_duration() -> None:
    assert frame_timestamps(0.0) == [0.0]
