"""Regression checks for the sdist manifest using existing build dependencies."""

from __future__ import annotations

from pathlib import Path

from setuptools._distutils.filelist import FileList

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_keeps_web_html_but_excludes_cython_annotations() -> None:
    # setuptools-scm seeds the manifest with tracked files before MANIFEST.in runs.
    tracked = [
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "setup.py",
        "docs/example.md",
        "web/index.html",
        "web/package.json",
        "web/package-lock.json",
        "web/src/main.tsx",
        "src/super_processor/__init__.py",
        "src/super_processor/_core.pyx",
        "src/super_processor/_core.pyi",
        "src/super_processor/py.typed",
    ]
    generated = [
        "src/super_processor/_core.html",
        "src/super_processor/nested/annotation.html",
        "src/super_processor/_core.c",
        "src/super_processor/_core.so",
        "src/super_processor/_version.py",
    ]
    manifest = FileList()
    manifest.set_allfiles(tracked + generated)
    manifest.files = tracked + generated
    for line in (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines():
        manifest.process_template_line(line)

    assert set(tracked) <= set(manifest.files)
    assert not set(generated) & set(manifest.files)
