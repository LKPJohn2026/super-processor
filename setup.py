"""Build configuration for the Cython extension."""

from __future__ import annotations

import os
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup


def build_extensions() -> list[Extension]:
    """Create extension modules from their canonical ``.pyx`` sources."""
    annotate = os.environ.get("CYTHON_ANNOTATE", "").lower() in {"1", "true", "yes"}
    extensions = [
        Extension(
            name="super_processor._core",
            sources=["src/super_processor/_core.pyx"],
        )
    ]
    return cythonize(
        extensions,
        annotate=annotate,
        compiler_directives={
            "boundscheck": False,
            "cdivision": True,
            "embedsignature": True,
            "initializedcheck": False,
            "language_level": 3,
            "wraparound": False,
        },
    )


# CI and the release set this, so no sdist or wheel ships without the web app
# (`npm run build:package` in web/). Local builds may leave it out.
REQUIRE_WEB_DIST = "SUPER_PROCESSOR_REQUIRE_WEB_DIST"


def check_web_dist() -> None:
    """Stop the build when the web app is required but was not built."""
    if os.environ.get(REQUIRE_WEB_DIST) != "1":
        return
    index = Path(__file__).resolve().parent / "src/super_processor/web_dist/index.html"
    if not index.is_file():
        raise SystemExit(
            f"{REQUIRE_WEB_DIST}=1 but {index} is missing; run "
            "`npm ci && npm run build:package` in web/ first"
        )


check_web_dist()
setup(ext_modules=build_extensions())
