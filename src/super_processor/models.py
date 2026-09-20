"""Optional planner LLM / VLM adapters (Recipe JSON only; no shell tools)."""

from __future__ import annotations

import contextlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .plan import apply_instruction_patches
from .recipe import Recipe


@dataclass(slots=True)
class ModelEndpoint:
    """Configured local or remote OpenAI-compatible chat endpoint."""

    name: str
    base_url: str
    model: str
    api_key_env: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
        }


def default_endpoints() -> list[ModelEndpoint]:
    """Return built-in Ollama / LM Studio / BYOK endpoint presets."""
    return [
        ModelEndpoint(
            name="ollama",
            base_url=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
            model=os.environ.get("SUPER_PROCESSOR_OLLAMA_MODEL", "llama3.2"),
            api_key_env=None,
        ),
        ModelEndpoint(
            name="lmstudio",
            base_url=os.environ.get("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
            model=os.environ.get("SUPER_PROCESSOR_LMSTUDIO_MODEL", "local-model"),
            api_key_env=None,
        ),
        ModelEndpoint(
            name="openai",
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.environ.get("SUPER_PROCESSOR_OPENAI_MODEL", "gpt-4o-mini"),
            api_key_env="OPENAI_API_KEY",
        ),
    ]


class PlannerError(RuntimeError):
    """Raised when planner model interaction fails."""


def planner_available(endpoint: ModelEndpoint) -> bool:
    """Return True when the endpoint looks reachable (best-effort)."""
    try:
        url = endpoint.base_url.rstrip("/")
        if endpoint.name == "ollama":
            req = urllib.request.Request(url + "/api/tags", method="GET")
        else:
            req = urllib.request.Request(url + "/models", method="GET")
        if endpoint.api_key_env:
            key = os.environ.get(endpoint.api_key_env)
            if not key:
                return False
            req.add_header("Authorization", f"Bearer {key}")
        with urllib.request.urlopen(req, timeout=1.5) as response:
            return int(getattr(response, "status", 200)) < 500
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def list_models(*, remote_frames_consent: bool = False) -> dict[str, Any]:
    """Describe configured model endpoints and consent state."""
    endpoints = []
    for item in default_endpoints():
        endpoints.append(
            {
                **item.to_dict(),
                "reachable": planner_available(item),
            }
        )
    return {
        "endpoints": endpoints,
        "remote_frames_consent": remote_frames_consent,
        "safe_mode_default": True,
    }


def apply_llm_plan_patch(
    recipe: Recipe,
    instruction: str,
    *,
    safe_mode: bool = True,
    endpoint_name: str = "ollama",
) -> Recipe:
    """Patch a recipe using NL rules; optionally call a local LLM for advice text.

    In safe_mode the LLM is never contacted. Outside safe_mode, a chat completion
    may be requested, but only the deterministic patcher mutates the recipe.
    Models never receive a shell tool and never return executable commands.
    """
    if safe_mode or not instruction.strip():
        return apply_instruction_patches(recipe, instruction)

    endpoint = next(
        (item for item in default_endpoints() if item.name == endpoint_name),
        default_endpoints()[0],
    )
    # Best-effort advisory call; failures fall back to CV/NL patches only.
    with contextlib.suppress(Exception):
        _advisory_chat(endpoint, instruction, recipe)
    return apply_instruction_patches(recipe, instruction)


def _advisory_chat(endpoint: ModelEndpoint, instruction: str, recipe: Recipe) -> str:
    """Request advisory natural-language guidance; ignore structured tool use."""
    payload = {
        "model": endpoint.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You advise on Super Processor recipe tweaks only. "
                    "Never invent shell commands or non-allowlisted ops. "
                    "Reply with short plain advice."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Instruction: {instruction}\n"
                    f"Current recipe ops: {json.dumps(recipe.to_dict()['ops'])}"
                ),
            },
        ],
        "temperature": 0,
    }
    url = endpoint.base_url.rstrip("/")
    if endpoint.name == "ollama":
        url = url + "/v1/chat/completions"
    else:
        url = url + "/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if endpoint.api_key_env:
        key = os.environ.get(endpoint.api_key_env)
        if not key:
            raise PlannerError(f"missing API key env {endpoint.api_key_env}")
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=8) as response:
        body = json.loads(response.read().decode("utf-8"))
    choices = body.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")
