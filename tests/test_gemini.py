"""The Gemini client: model choice and failover, replies, and the saved chat."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from super_processor.gemini import (
    MAX_CANDIDATE_MODELS,
    ChatTurn,
    GeminiClient,
    GeminiError,
    ModelInfo,
    append_chat_turn,
    append_chat_turns,
    contents_from_chat,
    load_chat,
    model_accepts_text_or_video,
    model_has_usable_rpm,
    parse_model_info,
    save_chat,
    select_candidate_models,
    summarize_gemini_error,
)


def test_the_chat_survives_a_save_and_load(tmp_path: Path) -> None:
    turns = [
        ChatTurn(role="user", text="hello", frame_refs=["a.jpg"]),
        ChatTurn(role="model", text="{}", structured={"ok": True}),
    ]
    path = save_chat(tmp_path, turns)
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1
    assert load_chat(tmp_path) == turns
    appended = append_chat_turn(tmp_path, ChatTurn(role="user", text="more"))
    assert load_chat(tmp_path) == appended == [*turns, ChatTurn("user", "more")]
    # The write goes through a temporary file that is moved into place.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["gemini_chat.json"]


def test_append_chat_turns_adds_them_in_one_go(tmp_path: Path) -> None:
    append_chat_turn(tmp_path, ChatTurn(role="user", text="first"))
    pair = [ChatTurn("user", "question"), ChatTurn("model", "{}")]
    assert append_chat_turns(tmp_path, pair) == [ChatTurn("user", "first"), *pair]
    assert load_chat(tmp_path) == [ChatTurn("user", "first"), *pair]


def test_two_threads_logging_at_once_keep_every_pair_together(
    tmp_path: Path,
) -> None:
    client = GeminiClient(api_key="k", transport=_FakeTransport({}))
    start = threading.Barrier(2)

    def log_twenty(name: str) -> None:
        start.wait()
        for index in range(20):
            client._log(tmp_path, f"{name} {index}", {"who": name, "n": index})

    threads = [threading.Thread(target=log_twenty, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    turns = load_chat(tmp_path)
    # No append is lost, and no other thread's turn lands between a question
    # and its answer.
    assert len(turns) == 80
    pairs = list(zip(turns[0::2], turns[1::2], strict=True))
    assert all(q.role == "user" and a.role == "model" for q, a in pairs)
    answers = [a.structured for _q, a in pairs if a.structured is not None]
    assert len(answers) == 40
    assert [q.text for q, _a in pairs] == [f"{a['who']} {a['n']}" for a in answers]
    for name in ("a", "b"):
        mine = [a["n"] for a in answers if a["who"] == name]
        assert mine == list(range(20))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["gemini_chat.json"]


def test_turns_out_replays_the_chat_but_does_not_save(tmp_path: Path) -> None:
    earlier = [ChatTurn("user", "earlier ask"), ChatTurn("model", '{"x": 1}')]
    save_chat(tmp_path, earlier)
    reply = {"segments": [{"start_s": 0.0, "end_s": 3.0, "label": "kitchen"}]}
    transport = _FakeTransport(reply)
    client = GeminiClient(api_key="k", transport=transport)
    held: list[ChatTurn] = []
    client.propose_segments(
        stills=_segment_stills(tmp_path, 2),
        times=[0.0, 1.0],
        duration_s=3.0,
        job_dir=tmp_path,
        turns_out=held,
    )
    assert transport.last_body is not None
    sent = transport.last_body["contents"]
    assert sent[0]["parts"][0]["text"] == "earlier ask"
    assert load_chat(tmp_path) == earlier
    assert [turn.role for turn in held] == ["user", "model"]
    assert held[1].structured == reply
    # The caller decides later: appending the held pair matches a normal call.
    append_chat_turns(tmp_path, held)
    assert load_chat(tmp_path) == [*earlier, *held]


def test_turns_out_also_holds_label_and_revise_turns(tmp_path: Path) -> None:
    held: list[ChatTurn] = []
    client = GeminiClient(api_key="k", transport=_FakeTransport(_UPSCALE_REPLY))
    client.revise_upscale_params(
        note="softer",
        prior_scale=2,
        prior_strength=0.5,
        prior_start_s=0.0,
        prior_end_s=20.0,
        duration_s=20.0,
        job_dir=tmp_path,
        turns_out=held,
    )
    still = tmp_path / "s.jpg"
    still.write_bytes(b"\xff\xd8jpeg")
    labels = GeminiClient(api_key="k", transport=_FakeTransport({"shots": []}))
    labels.label_shots(
        shots=[{"start_s": 0.0, "end_s": 1.0}],
        stills=[still],
        issues=("noise",),
        contains=("faces",),
        job_dir=tmp_path,
        turns_out=held,
    )
    assert [turn.role for turn in held] == ["user", "model", "user", "model"]
    assert held[1].structured == _UPSCALE_REPLY
    assert held[3].structured == {"shots": []}
    assert load_chat(tmp_path) == []


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
    names = select_candidate_models(models, preferred="models/gemini-2.5-flash")
    # Preferred first, once; then the rest in listed order. Embeddings and a
    # model with an RPM limit of 0 never become candidates.
    assert names == ["gemini-2.5-flash", "gemini-3.8-flash"]
    assert select_candidate_models(models) == ["gemini-3.8-flash", "gemini-2.5-flash"]


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
    # The configured model, then the extra names in the order given, with
    # their ``models/`` prefix gone.
    assert names == ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-pro"]


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
        contains=("faces", "text"),
        job_dir=tmp_path,
    )
    assert out == {"segments": reply["segments"], "scale": None, "scale_reason": ""}
    assert transport.last_body is not None
    parts = transport.last_body["contents"][-1]["parts"]
    assert [p["text"] for p in parts[1::2]] == ["t=0.000s", "t=1.000s", "t=2.000s"]
    assert all("inlineData" in p for p in parts[2::2])
    assert "did not accept" not in parts[0]["text"]
    assert "faces, text" in parts[0]["text"]
    assert "at most 10; fewer is better" in parts[0]["text"]
    assert "at least 3 seconds long" in parts[0]["text"]
    schema = transport.last_body["generationConfig"]["responseSchema"]
    assert "contains" in schema["properties"]["segments"]["items"]["required"]
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


_QUOTA_BODY = {
    "error": {
        "code": 429,
        "message": "You exceeded your current quota, please check your plan. "
        "For more information, head to: https://ai.google.dev/rate-limits.",
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.Help", "links": []},
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [
                    {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}
                ],
            },
            {
                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                "retryDelay": "57511s",
            },
        ],
    }
}


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (429, _QUOTA_BODY, "daily quota used up; try again in about 16 h"),
        (
            429,
            {
                "error": {
                    "status": "RESOURCE_EXHAUSTED",
                    "details": [{"retryDelay": "90s"}],
                }
            },
            "quota used up; try again in about 2 min",
        ),
        (
            429,
            {"error": {"code": 429, "details": [{"retryDelay": "xs"}, "junk"]}},
            "quota used up",
        ),
        (
            400,
            {
                "error": {
                    "message": "API key not valid. Please pass a valid API key.",
                    "details": [{"reason": "API_KEY_INVALID"}],
                }
            },
            "the API key was rejected; replace it on the setup screen",
        ),
        (
            404,
            {"error": {"message": "Model x is no longer available to new users."}},
            "this model is no longer available",
        ),
        (500, {"error": {"message": "Internal error. Retry later."}}, "Internal error"),
        (503, "Service Unavailable", "Service Unavailable"),
        (400, {"error": {"message": "x" * 300}}, "x" * 160 + "…"),
    ],
)
def test_gemini_errors_are_one_readable_line(
    status: int, body: object, expected: str
) -> None:
    detail = body if isinstance(body, str) else json.dumps(body)
    assert summarize_gemini_error(status, detail) == expected


def test_failover_groups_models_that_failed_the_same_way(tmp_path: Path) -> None:
    transport = _FailoverTransport({}, fail_models={"gemini-gone", "gemini-3.8-flash"})
    transport.list_models = lambda *, api_key: [  # type: ignore[method-assign]
        {
            "name": f"models/{name}",
            "description": "multimodal image video",
            "supportedGenerationMethods": ["generateContent"],
            "rateLimits": {"rpm": "0/10"},
        }
        for name in ("gemini-gone", "gemini-3.8-flash")
    ]
    original = transport.generate

    def same_reason(
        *, model: str, api_key: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            return original(model=model, api_key=api_key, body=body)
        except GeminiError as exc:
            raise GeminiError("Gemini HTTP 404: retired", status=404) from exc

    transport.generate = same_reason  # type: ignore[method-assign]
    client = GeminiClient(api_key="k", model="gemini-gone", transport=transport)
    with pytest.raises(GeminiError) as failed:
        _revise(client, tmp_path)
    assert str(failed.value) == (
        "All candidate Gemini models failed: "
        "gemini-gone, gemini-3.8-flash: Gemini HTTP 404: retired"
    )


def _reply(text: str) -> dict[str, Any]:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


class _Scripted:
    """No ``models.list``; answers each model from ``script`` in turn."""

    def __init__(self, *answers: dict[str, Any] | GeminiError) -> None:
        self.answers = list(answers)
        self.calls: list[str] = []

    def generate(
        self, *, model: str, api_key: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        del api_key, body
        self.calls.append(model)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, GeminiError):
            raise answer
        return answer


def test_an_exhausted_model_is_not_tried_again_this_session(tmp_path: Path) -> None:
    transport = _FailoverTransport(_UPSCALE_REPLY, fail_models={"gemini-gone"})
    client = GeminiClient(api_key="k", model="gemini-gone", transport=transport)
    _revise(client, tmp_path)
    assert transport.calls == ["gemini-gone", "gemini-3.8-flash"]
    transport.calls.clear()
    # The retired model answered 404 once; later calls go straight past it.
    _revise(client, tmp_path)
    assert transport.calls == ["gemini-3.8-flash"]
    assert client.candidate_models() == ["gemini-3.8-flash"]


def test_a_model_that_stays_busy_is_left_after_its_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SUPER_PROCESSOR_GEMINI_CANDIDATES", "gemini-2.5-flash")
    sleeps: list[float] = []
    monkeypatch.setattr("super_processor.gemini.time.sleep", sleeps.append)
    busy = GeminiError("Gemini HTTP 503: high demand", status=503)
    transport = _Scripted(busy, busy, busy, _reply(json.dumps(_UPSCALE_REPLY)))
    client = GeminiClient(api_key="k", model="gemini-3.8-flash", transport=transport)
    assert client._generate_with_failover({"contents": []}) == _reply(
        json.dumps(_UPSCALE_REPLY)
    )
    # Three tries of the busy model (two retries), then the next one.
    assert transport.calls == ["gemini-3.8-flash"] * 3 + ["gemini-2.5-flash"]
    assert sleeps == [1.5, 3.0]
    assert client.model == "gemini-2.5-flash"
    # Busy is not gone: the model stays a candidate for later calls.
    assert "gemini-3.8-flash" in client.candidate_models()


def test_without_models_list_only_the_configured_model_is_tried() -> None:
    transport = _Scripted(GeminiError("Gemini HTTP 429: quota used up", status=429))
    client = GeminiClient(api_key="k", model="gemini-3.8-flash", transport=transport)
    assert client.candidate_models() == ["gemini-3.8-flash"]
    with pytest.raises(GeminiError) as failed:
        client._generate_with_failover({"contents": []})
    assert str(failed.value) == (
        "All candidate Gemini models failed: "
        "gemini-3.8-flash: Gemini HTTP 429: quota used up"
    )


def test_a_model_name_that_cannot_watch_video_leaves_no_candidates() -> None:
    # An image-generation model set in the environment is never called.
    transport = _Scripted(_reply("{}"))
    client = GeminiClient(
        api_key="k", model="gemini-2.5-flash-image", transport=transport
    )
    with pytest.raises(GeminiError, match="models failed: no candidates$"):
        client._generate_with_failover({"contents": []})
    assert transport.calls == []


def test_candidates_are_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    names = [f"gemini-{n}-flash" for n in range(20)]
    monkeypatch.setenv("SUPER_PROCESSOR_GEMINI_CANDIDATES", ",".join(names))
    client = GeminiClient(api_key="k", model=names[0], transport=_Scripted({}))
    assert client.candidate_models() == names[:MAX_CANDIDATE_MODELS]


def test_the_working_model_leads_the_candidates_after_a_probe() -> None:
    transport = _FailoverTransport({"ok": True}, fail_models=set())
    transport.list_models = lambda *, api_key: [  # type: ignore[method-assign]
        {"name": f"models/{name}", "supportedGenerationMethods": ["generateContent"]}
        for name in ("gemini-a-flash", "gemini-b-flash", "gemini-c-flash")
    ]
    original = transport.generate

    def slow_first(*, model: str, api_key: str, body: dict[str, Any]) -> Any:
        if model == "gemini-a-flash":
            transport.calls.append(model)
            raise GeminiError("Gemini request failed: The read operation timed out")
        return original(model=model, api_key=api_key, body=body)

    transport.generate = slow_first  # type: ignore[method-assign]
    client = GeminiClient(api_key="k", model="gemini-a-flash", transport=transport)
    assert client.pick_working_model() == "gemini-b-flash"
    # The slow model is kept, but later calls start with the one that answered.
    assert client.candidate_models() == [
        "gemini-b-flash",
        "gemini-a-flash",
        "gemini-c-flash",
    ]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({"candidates": []}, "unexpected Gemini response shape"),
        ({"candidates": [{"content": {}}]}, "unexpected Gemini response shape"),
        ({"candidates": [{"content": {"parts": [{}]}}]}, "returned no text"),
        (_reply("not json"), "JSON parse failed"),
        (_reply("[1, 2]"), "must be an object"),
    ],
)
def test_an_unreadable_reply_is_an_error_not_a_result(
    tmp_path: Path, response: dict[str, Any], message: str
) -> None:
    client = GeminiClient(api_key="k", transport=_Scripted(response))
    with pytest.raises(GeminiError, match=message):
        _revise(client, tmp_path)
    # Nothing half-read goes into the chat that later calls replay.
    assert load_chat(tmp_path) == []


def test_a_reply_split_over_parts_is_joined() -> None:
    halves = {
        "candidates": [{"content": {"parts": [{"text": '{"ok"'}, {"text": ": 1}"}]}}]
    }
    client = GeminiClient(api_key="k", transport=_Scripted(halves))
    assert client._generate(contents=[], schema={}, system="s") == {"ok": 1}


def test_revise_offers_only_the_scales_there_are(tmp_path: Path) -> None:
    transport = _FakeTransport(_UPSCALE_REPLY)
    client = GeminiClient(api_key="k", transport=transport)
    assert _revise(client, tmp_path) == _UPSCALE_REPLY
    assert transport.last_body is not None
    prompt = transport.last_body["contents"][-1]["parts"][0]["text"]
    assert "scale (only one of: 1, 2;" in prompt
    assert "Keep scale=2 unless" in prompt
    assert "4" not in prompt.split("scale (only one of:")[1].split(";")[0]
    turns = load_chat(tmp_path)
    assert [t.role for t in turns] == ["user", "model"]
    assert turns[1].structured == _UPSCALE_REPLY


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{", "cannot read gemini_chat.json"),
        ('{"turns": {"0": {}}}', "turns must be a list"),
        ('{"turns": [{"role": "user", "frame_refs": "a.jpg"}]}', "frame_refs"),
        ('{"turns": [{"role": "model", "structured": [1]}]}', "structured"),
    ],
)
def test_a_damaged_chat_file_is_reported(
    tmp_path: Path, content: str, message: str
) -> None:
    (tmp_path / "gemini_chat.json").write_text(content, encoding="utf-8")
    with pytest.raises(GeminiError, match=message):
        load_chat(tmp_path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("[1]", "root must be an object"),
        ('"turns"', "root must be an object"),
        ('{"turns": [{"text": "x"}]}', "needs a role"),
    ],
)
def test_a_chat_file_of_the_wrong_shape_is_a_gemini_error(
    tmp_path: Path, content: str, message: str
) -> None:
    (tmp_path / "gemini_chat.json").write_text(content, encoding="utf-8")
    with pytest.raises(GeminiError, match=message):
        load_chat(tmp_path)


def test_a_missing_chat_is_empty(tmp_path: Path) -> None:
    assert load_chat(tmp_path / "new-job") == []


def test_an_unreadable_still_is_an_error(tmp_path: Path) -> None:
    client = GeminiClient(api_key="k", transport=_FakeTransport({"segments": []}))
    with pytest.raises(GeminiError, match="could not read gone.jpg"):
        client.propose_segments(
            stills=[tmp_path / "gone.jpg"],
            times=[0.0],
            duration_s=1.0,
            job_dir=tmp_path,
        )


def test_a_short_retry_hint_reads_in_seconds() -> None:
    body = json.dumps({"error": {"details": [{"retryDelay": "41.4s"}]}})
    assert summarize_gemini_error(429, body) == "quota used up; try again in 41 s"


def test_rpm_text_and_bad_numbers() -> None:
    info = parse_model_info(
        {"name": "gemini-x-flash", "rpm": "15", "rpm_used": None, "RPM": 1}
    )
    assert (info.rpm_limit, info.rpm_used, info.rpm_remaining) == (15, None, None)
    nested = parse_model_info(
        {"name": "gemini-x", "rateLimits": {"RPM": {"used": "x", "limit": 9}}}
    )
    assert (nested.rpm_used, nested.rpm_limit) == (None, None)
    # Unknown RPM is worth a probe; an exhausted one is not.
    assert model_has_usable_rpm(ModelInfo(name="gemini-x-flash"))
    assert not model_accepts_text_or_video(
        ModelInfo(name="gemini-x-flash", methods=("countTokens",))
    )
