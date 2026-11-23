"""Text from a model reply or a saved file, made safe to keep and show."""

from __future__ import annotations

from typing import Any


def one_line(raw: Any, cap: int) -> str:
    """``raw`` as one line of visible characters, at most ``cap`` long.

    Runs of whitespace (line breaks included) become one space, and control
    characters such as NUL are dropped. Anything but text becomes "".
    """
    if not isinstance(raw, str):
        return ""
    words = " ".join(raw.split())
    return "".join(char for char in words if char.isprintable())[:cap]
