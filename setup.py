"""Build configuration for the Cython extension."""

from __future__ import annotations

import os

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


setup(ext_modules=build_extensions())
