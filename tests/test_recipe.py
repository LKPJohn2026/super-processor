"""Tests for the allowlisted Recipe JSON schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from super_processor.recipe import (
    RECIPE_VERSION,
    OpName,
    RecipeError,
    empty_recipe,
    load_recipe,
    parse_recipe_json,
    write_recipe,
)


def test_empty_recipe_contains_allowlisted_ops() -> None:
    recipe = empty_recipe("abcd1234abcd1234", "/tmp/clip.mp4")
    assert recipe.version == RECIPE_VERSION
    assert [op.op for op in recipe.ops] == list(OpName)
    assert all(not op.enabled for op in recipe.ops)


def test_round_trip_write_load(tmp_path: Path) -> None:
    recipe = empty_recipe("abcd1234abcd1234", "/tmp/clip.mp4")
    recipe.ops[0].enabled = True
    recipe.ops[0].params = {"temperature": 5600, "tint": 5}
    write_recipe(tmp_path, recipe)
    loaded = load_recipe(tmp_path)
    assert loaded.to_dict() == recipe.to_dict()


def test_rejects_unknown_op() -> None:
    payload = empty_recipe("abcd1234abcd1234", "/tmp/clip.mp4").to_dict()
    payload["ops"].append({"op": "face_swap", "enabled": True, "params": {}})
    with pytest.raises(RecipeError, match="unsupported op"):
        parse_recipe_json(json.dumps(payload))


def test_rejects_duplicate_ops() -> None:
    payload = empty_recipe("abcd1234abcd1234", "/tmp/clip.mp4").to_dict()
    payload["ops"].append({"op": "contrast", "enabled": True, "params": {}})
    with pytest.raises(RecipeError, match="unique"):
        parse_recipe_json(json.dumps(payload))


def test_rejects_wrong_version() -> None:
    payload = empty_recipe("abcd1234abcd1234", "/tmp/clip.mp4").to_dict()
    payload["version"] = "999"
    with pytest.raises(RecipeError, match="unsupported recipe version"):
        parse_recipe_json(json.dumps(payload))
