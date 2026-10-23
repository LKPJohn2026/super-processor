"""Tests for Gemini structured schemas, sampling, and chat persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from super_processor.gemini import (
    ChatTurn,
    GeminiClient,
    GeminiError,
    ModelInfo,
    append_chat_turn,
    contents_from_chat,
    load_chat,
    model_accepts_text_or_video,
    model_has_usable_rpm,
    parse_model_info,
    save_chat,
    select_candidate_models,
)


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


def test_contents_from_chat_alternates_roles() -> None:
    contents = contents_from_chat(
        [
            ChatTurn(role="user", text="first"),
            ChatTurn(role="user", text="still first"),
            ChatTurn(role="model", text='{"ok": true}'),
            ChatTurn(role="user", text="  "),
        ]
    )
    assert [item["role"] for item in contents] == ["user", "model"]
    assert contents[0]["parts"][0]["text"] == "first\nstill first"
    assert contents[1]["parts"][0]["text"] == '{"ok": true}'


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


def test_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        "super_processor.gemini.resolve_secret",
        lambda _env: None,
    )
    with pytest.raises(GeminiError, match="GEMINI_API_KEY"):
        GeminiClient(api_key="")


def _sample_enhance_payload(*, label: str = "lift") -> dict[str, Any]:
    return {
        "issues": ["low_light"],
        "options": [
            {
                "id": "A",
                "label": label,
                "ops": [{"op": "contrast", "params": {"brightness": 0.1}}],
            },
            {
                "id": "B",
                "label": "denoise",
                "ops": [{"op": "denoise", "params": {"strength": 0.3}}],
            },
            {
                "id": "C",
                "label": "sharpen",
                "ops": [{"op": "sharpen", "params": {"luma_amount": 0.5}}],
            },
        ],
    }


def test_parse_model_info_rpm_fraction() -> None:
    info = parse_model_info(
        {
            "name": "models/gemini-3.8-flash",
            "displayName": "Gemini 3.8 Flash",
            "description": "Multimodal model for text, image, and video",
            "supportedGenerationMethods": ["generateContent"],
            "rateLimits": {"rpm": "0/10"},
        }
    )
    assert info.name == "gemini-3.8-flash"
    assert info.rpm_used == 0
    assert info.rpm_limit == 10
    assert info.rpm_remaining == 10
    assert model_accepts_text_or_video(info)
    assert model_has_usable_rpm(info)


def test_image_generation_models_rejected() -> None:
    info = ModelInfo(
        name="gemini-3.1-flash-image",
        methods=("generateContent",),
        description="image generation",
        rpm_limit=10,
        rpm_used=0,
    )
    assert not model_accepts_text_or_video(info)


def test_model_rpm_zero_limit_rejected() -> None:
    info = ModelInfo(
        name="gemini-2.0-flash",
        methods=("generateContent",),
        description="image video",
        rpm_limit=0,
        rpm_used=0,
    )
    assert not model_has_usable_rpm(info)


def test_model_rpm_exhausted_rejected() -> None:
    info = ModelInfo(
        name="gemini-2.0-flash",
        methods=("generateContent",),
        description="image video",
        rpm_limit=5,
        rpm_used=5,
    )
    assert not model_has_usable_rpm(info)


def test_select_candidate_models_filters_and_orders() -> None:
    models = [
        ModelInfo(
            name="text-embedding-004",
            methods=("embedContent",),
            description="embeddings",
            rpm_limit=100,
            rpm_used=0,
        ),
        ModelInfo(
            name="gemini-2.0-flash",
            methods=("generateContent",),
            description="multimodal image video",
            rpm_limit=0,
            rpm_used=0,
        ),
        ModelInfo(
            name="gemini-3.8-flash",
            methods=("generateContent",),
            description="text image video",
            rpm_limit=10,
            rpm_used=0,
        ),
        ModelInfo(
            name="gemini-2.5-flash",
            methods=("generateContent",),
            description="multimodal",
            rpm_limit=5,
            rpm_used=1,
        ),
    ]
    names = select_candidate_models(models, preferred="gemini-2.5-flash")
    assert names[0] == "gemini-2.5-flash"
    assert "gemini-3.8-flash" in names
    assert "gemini-2.0-flash" not in names
    assert "text-embedding-004" not in names


class _FailoverTransport:
    def __init__(self, payload: dict[str, Any], *, fail_models: set[str]) -> None:
        self.payload = payload
        self.fail_models = fail_models
        self.calls: list[str] = []

    def list_models(self, *, api_key: str) -> list[dict[str, Any]]:
        del api_key
        return [
            {
                "name": "models/gemini-gone",
                "description": "multimodal image video",
                "supportedGenerationMethods": ["generateContent"],
                "rateLimits": {"rpm": "0/5"},
            },
            {
                "name": "models/gemini-3.8-flash",
                "description": "multimodal image video",
                "supportedGenerationMethods": ["generateContent"],
                "rateLimits": {"rpm": "0/10"},
            },
        ]

    def generate(
        self,
        *,
        model: str,
        api_key: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        del api_key, body
        self.calls.append(model)
        if model in self.fail_models:
            raise GeminiError(
                f"Gemini HTTP 404: {model} is no longer available", status=404
            )
        text = json.dumps(self.payload)
        return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def test_model_info_to_dict_and_rpm_shapes() -> None:
    info = parse_model_info(
        {
            "model": "models/gemini-2.5-pro",
            "display_name": "Pro",
            "supported_actions": ["generateContent"],
            "rpmLimit": 15,
            "rpmUsed": "bad",
            "rate_limits": {"rpm": {"used": 2, "limit": 15}},
        }
    )
    assert info.name == "gemini-2.5-pro"
    assert info.rpm_used == 2
    assert info.rpm_limit == 15
    assert info.to_dict()["rpm_remaining"] == 13
    broken = parse_model_info({"name": "models/gemini-x", "rateLimits": {"rpm": "x/y"}})
    assert broken.rpm_limit is None
    assert model_accepts_text_or_video(
        ModelInfo(name="gemini-2.5-flash", methods=("generateContent",))
    )
    assert not model_accepts_text_or_video(ModelInfo(name=""))
    assert not model_accepts_text_or_video(
        ModelInfo(name="gemini-embedding-001", methods=("embedContent",))
    )
    assert not model_accepts_text_or_video(
        ModelInfo(name="other-model", methods=("generateContent",))
    )


def test_client_pick_working_model_all_fail() -> None:
    transport = _FailoverTransport(
        {"ok": True}, fail_models={"gemini-gone", "gemini-3.8-flash"}
    )
    client = GeminiClient(
        api_key="k",
        model="gemini-gone",
        transport=transport,
    )
    with pytest.raises(GeminiError, match="usable quota"):
        client.pick_working_model()


def test_client_pick_working_model_skips_exhausted() -> None:
    payload = {"ok": True}
    transport = _FailoverTransport(payload, fail_models={"gemini-gone"})
    client = GeminiClient(
        api_key="k",
        model="gemini-gone",
        transport=transport,
    )
    picked = client.pick_working_model()
    assert picked == "gemini-3.8-flash"
    assert client.model == "gemini-3.8-flash"
    assert transport.calls[0] == "gemini-gone"


def test_client_fails_over_when_a_model_times_out() -> None:
    class TimeoutTransport(_FailoverTransport):
        def generate(
            self,
            *,
            model: str,
            api_key: str,
            body: dict[str, Any],
        ) -> dict[str, Any]:
            if model == "gemini-gone":
                self.calls.append(model)
                raise GeminiError("Gemini request failed: The read operation timed out")
            return super().generate(model=model, api_key=api_key, body=body)

    transport = TimeoutTransport({"ok": True}, fail_models=set())
    client = GeminiClient(api_key="k", model="gemini-gone", transport=transport)
    assert client.pick_working_model() == "gemini-3.8-flash"
    # One timeout moves on at once: no same-model retries, and the slow model
    # is not marked exhausted for the rest of the session.
    assert transport.calls == ["gemini-gone", "gemini-3.8-flash"]
    assert "gemini-gone" not in client._exhausted_models


_UPSCALE_REPLY = {"start_s": 1, "end_s": 3, "scale": 2, "strength": 0.2}


def _revise(client: GeminiClient, job_dir: Path) -> dict[str, Any]:
    return client.revise_upscale_params(
        note="less artificial detail from 1s to 3s",
        prior_scale=2,
        prior_strength=0.5,
        prior_start_s=0.0,
        prior_end_s=20.0,
        duration_s=20.0,
        job_dir=job_dir,
    )


def test_revise_resends_history_to_each_model(tmp_path: Path) -> None:
    save_chat(
        tmp_path,
        [
            ChatTurn(role="user", text="label the opening shot"),
            ChatTurn(role="model", text='{"shots": []}'),
        ],
    )
    bodies: list[dict[str, Any]] = []
    transport = _FailoverTransport(_UPSCALE_REPLY, fail_models={"gemini-gone"})

    def generate(
        *,
        model: str,
        api_key: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        bodies.append({"model": model, **body})
        return _FailoverTransport.generate(
            transport, model=model, api_key=api_key, body=body
        )

    transport.generate = generate  # type: ignore[method-assign]
    client = GeminiClient(api_key="k", model="gemini-gone", transport=transport)
    _revise(client, tmp_path)
    assert len(bodies) >= 2
    for body in bodies:
        roles = [item["role"] for item in body["contents"]]
        assert roles[:2] == ["user", "model"]
        assert roles[-1] == "user"
        assert body["contents"][0]["parts"][0]["text"] == "label the opening shot"
        assert "less artificial detail" in body["contents"][-1]["parts"][0]["text"]
        assert "systemInstruction" in body
        assert "previous_interaction_id" not in body
        assert "generationConfig" in body
    assert bodies[0]["model"] != bodies[-1]["model"]


def test_client_retries_transient_503(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("super_processor.gemini.time.sleep", sleeps.append)
    payload = _UPSCALE_REPLY

    class FlakyTransport:
        def __init__(self) -> None:
            self.calls = 0

        def list_models(self, *, api_key: str) -> list[dict[str, Any]]:
            del api_key
            return [
                {
                    "name": "models/gemini-3.8-flash",
                    "description": "multimodal image video",
                    "supportedGenerationMethods": ["generateContent"],
                    "rateLimits": {"rpm": "0/10"},
                }
            ]

        def generate(
            self,
            *,
            model: str,
            api_key: str,
            body: dict[str, Any],
        ) -> dict[str, Any]:
            del model, api_key, body
            self.calls += 1
            if self.calls < 3:
                raise GeminiError("Gemini HTTP 503: Service Unavailable", status=503)
            return {
                "candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]
            }

    transport = FlakyTransport()
    client = GeminiClient(
        api_key="k",
        model="gemini-3.8-flash",
        transport=transport,
    )
    result = _revise(client, tmp_path)
    assert result["strength"] == 0.2
    assert transport.calls == 3
    assert sleeps == [1.5, 3.0]


def test_client_failsover_unavailable_model(tmp_path: Path) -> None:
    payload = _UPSCALE_REPLY
    transport = _FailoverTransport(payload, fail_models={"gemini-gone"})
    client = GeminiClient(
        api_key="k",
        model="gemini-gone",
        transport=transport,
    )
    result = _revise(client, tmp_path)
    assert result["strength"] == 0.2
    assert transport.calls[0] == "gemini-gone"
    assert "gemini-3.8-flash" in transport.calls
    assert client.model == "gemini-3.8-flash"


def test_client_generate_all_candidates_fail(tmp_path: Path) -> None:
    transport = _FailoverTransport({}, fail_models={"gemini-gone", "gemini-3.8-flash"})
    client = GeminiClient(
        api_key="k",
        model="gemini-gone",
        transport=transport,
    )
    with pytest.raises(GeminiError, match="All candidate"):
        _revise(client, tmp_path)


def test_client_non_retryable_error_raises(tmp_path: Path) -> None:
    class BoomTransport:
        def list_models(self, *, api_key: str) -> list[dict[str, Any]]:
            del api_key
            return [
                {
                    "name": "models/gemini-3.8-flash",
                    "description": "multimodal image video",
                    "supportedGenerationMethods": ["generateContent"],
                    "rateLimits": {"rpm": "0/10"},
                }
            ]

        def generate(
            self,
            *,
            model: str,
            api_key: str,
            body: dict[str, Any],
        ) -> dict[str, Any]:
            del model, api_key, body
            raise GeminiError("Gemini HTTP 400: bad request", status=400)

    client = GeminiClient(
        api_key="k",
        model="gemini-3.8-flash",
        transport=BoomTransport(),
    )
    with pytest.raises(GeminiError, match="bad request"):
        _revise(client, tmp_path)


def test_candidate_models_honors_env_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "SUPER_PROCESSOR_GEMINI_CANDIDATES",
        "gemini-2.5-flash,models/gemini-2.5-pro",
    )

    class ListTransport:
        def list_models(self, *, api_key: str) -> list[dict[str, Any]]:
            del api_key
            return []

        def generate(
            self,
            *,
            model: str,
            api_key: str,
            body: dict[str, Any],
        ) -> dict[str, Any]:
            del model, api_key, body
            raise GeminiError("unused", status=500)

    client = GeminiClient(
        api_key="k",
        model="gemini-3.8-flash",
        transport=ListTransport(),
    )
    names = client.candidate_models(refresh=True)
    assert names[0] == "gemini-3.8-flash"
    assert "gemini-2.5-flash" in names
    assert "gemini-2.5-pro" in names


def _segment_stills(job_dir: Path, count: int) -> list[Path]:
    folder = job_dir / "segment_stills"
    folder.mkdir(parents=True, exist_ok=True)
    stills = []
    for index in range(count):
        path = folder / f"t_{index * 1000:08d}.jpg"
        path.write_bytes(b"\xff\xd8jpeg")
        stills.append(path)
    return stills


def test_propose_segments_sends_stills_with_times(tmp_path: Path) -> None:
    reply = {"segments": [{"start_s": 0.0, "end_s": 3.0, "label": "kitchen"}]}
    transport = _FakeTransport(reply)
    client = GeminiClient(api_key="k", transport=transport)
    out = client.propose_segments(
        stills=_segment_stills(tmp_path, 3),
        times=[0.0, 1.0, 2.0],
        duration_s=3.0,
        job_dir=tmp_path,
    )
    assert out == reply["segments"]
    assert transport.last_body is not None
    parts = transport.last_body["contents"][-1]["parts"]
    assert [p["text"] for p in parts[1::2]] == ["t=0.000s", "t=1.000s", "t=2.000s"]
    assert all("inlineData" in p for p in parts[2::2])
    assert "did not accept" not in parts[0]["text"]
    turns = load_chat(tmp_path)
    assert [turn.role for turn in turns] == ["user", "model"]


def test_propose_segments_note_replays_the_last_proposal(tmp_path: Path) -> None:
    first = {"segments": [{"start_s": 0.0, "end_s": 3.0, "label": "one take"}]}
    transport = _FakeTransport(first)
    client = GeminiClient(api_key="k", transport=transport)
    stills = _segment_stills(tmp_path, 2)
    client.propose_segments(
        stills=stills, times=[0.0, 2.0], duration_s=3.0, job_dir=tmp_path
    )
    transport.payload = {
        "segments": [
            {"start_s": 0.0, "end_s": 1.5, "label": "indoor"},
            {"start_s": 1.5, "end_s": 3.0, "label": "outdoor"},
        ]
    }
    client.propose_segments(
        stills=stills,
        times=[0.0, 2.0],
        duration_s=3.0,
        note="split   where we walk outside",
        job_dir=tmp_path,
    )
    assert transport.last_body is not None
    contents = transport.last_body["contents"]
    assert [item["role"] for item in contents] == ["user", "model", "user"]
    assert "one take" in contents[1]["parts"][0]["text"]
    assert "'split where we walk outside'" in contents[2]["parts"][0]["text"]


def test_propose_segments_refuses_bad_input_and_replies(tmp_path: Path) -> None:
    client = GeminiClient(api_key="k", transport=_FakeTransport({"segments": []}))
    stills = _segment_stills(tmp_path, 1)
    with pytest.raises(GeminiError, match="one time per still"):
        client.propose_segments(stills=stills, times=[], duration_s=1.0)
    with pytest.raises(GeminiError, match="no segments"):
        client.propose_segments(
            stills=stills, times=[0.0], duration_s=1.0, job_dir=tmp_path
        )
    outside = tmp_path.parent / "elsewhere.jpg"
    outside.write_bytes(b"\xff\xd8")
    with pytest.raises(GeminiError, match="outside the job"):
        client.propose_segments(
            stills=[outside], times=[0.0], duration_s=1.0, job_dir=tmp_path
        )
