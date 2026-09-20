"""Optional planner LLM / VLM adapters (Recipe JSON only; no shell tools)."""

from __future__ import annotations

import contextlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .plan import apply_instruction_patches
from .recipe import DEFAULT_OP_ORDER, OpName, Recipe
from .validator import PARAM_BOUNDS


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


def resolve_secret(env_var: str) -> str | None:
    """Resolve a secret from the environment, then optional OS keyring."""
    value = os.environ.get(env_var)
    if value:
        return value
    with contextlib.suppress(Exception):
        import keyring

        stored = keyring.get_password("super-processor", env_var)
        if stored:
            return stored
    return None


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
            key = resolve_secret(endpoint.api_key_env)
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


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _clamp_params(op_name: OpName, params: dict[str, Any]) -> dict[str, float]:
    """Keep only allowlisted numeric params inside published bounds."""
    bounds = PARAM_BOUNDS.get(op_name, {})
    cleaned: dict[str, float] = {}
    for key, value in params.items():
        if key not in bounds:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        low, high = bounds[key]
        cleaned[key] = max(low, min(high, number))
    return cleaned


def apply_structured_patches(recipe: Recipe, payload: dict[str, Any]) -> Recipe:
    """Apply a validated structured patch document onto a recipe.

    Expected shape::

        {"patches": [{"op": "denoise", "enabled": true, "params": {...}}]}

    Unknown ops/params are ignored. Models never introduce shell commands.
    """
    patches = payload.get("patches")
    if not isinstance(patches, list):
        return recipe
    ops = {op.op: op for op in recipe.ops}
    for item in patches:
        if not isinstance(item, dict):
            continue
        name = str(item.get("op", ""))
        try:
            op_name = OpName(name)
        except ValueError:
            continue
        if op_name not in ops:
            continue
        current = ops[op_name]
        enabled = item.get("enabled")
        if isinstance(enabled, bool):
            current.enabled = enabled
        raw_params = item.get("params")
        if isinstance(raw_params, dict):
            merged = dict(current.params)
            merged.update(_clamp_params(op_name, raw_params))
            current.params = merged
            if current.params and enabled is None:
                current.enabled = True
        ops[op_name] = current
    recipe.ops = [ops[name] for name in DEFAULT_OP_ORDER]
    return recipe


def parse_structured_patch(text: str) -> dict[str, Any] | None:
    """Extract a JSON patch object from model output, if present."""
    text = text.strip()
    if not text:
        return None
    candidates = [text]
    match = _JSON_BLOCK.search(text)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            data = json.loads(candidate)
            if isinstance(data, dict) and "patches" in data:
                return data
    return None


def apply_llm_plan_patch(
    recipe: Recipe,
    instruction: str,
    *,
    safe_mode: bool = True,
    endpoint_name: str = "ollama",
) -> Recipe:
    """Patch a recipe using NL rules and optional structured LLM patches.

    In safe_mode the LLM is never contacted. Outside safe_mode, a chat completion
    may return a JSON ``patches`` document which is merged through allowlisted
    bounds; deterministic NL patches always run as a final fallback layer.
    Models never receive a shell tool and never return executable commands.
    """
    if not instruction.strip():
        return recipe
    if safe_mode:
        return apply_instruction_patches(recipe, instruction)

    endpoint = next(
        (item for item in default_endpoints() if item.name == endpoint_name),
        default_endpoints()[0],
    )
    advice = ""
    with contextlib.suppress(Exception):
        advice = _advisory_chat(endpoint, instruction, recipe)
    structured = parse_structured_patch(advice)
    if structured is not None:
        recipe = apply_structured_patches(recipe, structured)
    return apply_instruction_patches(recipe, instruction)


def _advisory_chat(endpoint: ModelEndpoint, instruction: str, recipe: Recipe) -> str:
    """Request a structured recipe patch; ignore any non-JSON chatter."""
    allowlisted = [name.value for name in OpName]
    payload = {
        "model": endpoint.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You advise Super Processor recipe tweaks only. "
                    "Reply with ONE JSON object of the form "
                    '{"patches":[{"op":"<name>","enabled":true,"params":{...}}]}. '
                    f"Allowed ops: {', '.join(allowlisted)}. "
                    "Never invent shell commands, filters, or non-allowlisted ops."
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
        key = resolve_secret(endpoint.api_key_env)
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
