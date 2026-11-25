"""Regression checks for the sdist manifest using existing build dependencies."""

from __future__ import annotations

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

# Build tooling is a dev extra, not a dependency of an installed wheel.
filelist = pytest.importorskip("setuptools._distutils.filelist")

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_keeps_web_html_but_excludes_cython_annotations() -> None:
    """The built web app is untracked but kept; Cython annotations are not."""
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
    # Built by `npm run build:package`; not in git, so not seeded above.
    built_app = [
        "src/super_processor/web_dist/index.html",
        "src/super_processor/web_dist/assets/index-abc.js",
        "src/super_processor/web_dist/assets/index-abc.css",
        "src/super_processor/web_dist/assets/font-abc.woff2",
    ]
    generated = [
        "src/super_processor/_core.html",
        "src/super_processor/nested/annotation.html",
        "src/super_processor/_core.c",
        "src/super_processor/_core.so",
        "src/super_processor/_version.py",
    ]
    # FileList matches native separators, just like setuptools' real file list.
    tracked = [str(Path(path)) for path in tracked]
    built_app = [str(Path(path)) for path in built_app]
    generated = [str(Path(path)) for path in generated]
    manifest = filelist.FileList()
    manifest.set_allfiles(tracked + built_app + generated)
    manifest.files = tracked + generated
    for line in (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines():
        # setuptools reads the template with comments and blank lines removed.
        line = line.split("#", 1)[0].strip()
        if line:
            manifest.process_template_line(line)

    assert set(tracked) <= set(manifest.files)
    assert set(built_app) <= set(manifest.files)
    assert not set(generated) & set(manifest.files)


def test_package_data_keeps_the_web_app_html() -> None:
    """`*.html` in exclude-package-data would also drop web_dist/index.html."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"web_dist/*.html"' in text
    assert '"web_dist/assets/*"' in text
    assert 'super_processor = ["*.c", "*.pyx", "_core.html"]' in text


CHECKER = ROOT / ".github" / "scripts" / "check_web_dist.py"


def _checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_web_dist", CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wheel(path: Path, files: list[str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("super_processor/__init__.py", "")
        for name in files:
            archive.writestr(f"super_processor/web_dist/{name}", "x")


def _sdist(path: Path, files: list[str]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for folder in ("pkg-1.0/src/super_processor/web_dist/assets",):
            info = tarfile.TarInfo(folder)
            info.type = tarfile.DIRTYPE
            archive.addfile(info)
        for name in files:
            info = tarfile.TarInfo(f"pkg-1.0/src/super_processor/web_dist/{name}")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))


@pytest.mark.skipif(not CHECKER.is_file(), reason="needs the repository checkout")
def test_release_check_finds_a_missing_or_different_web_app(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checker = _checker()
    app = ["index.html", "assets/index-abc.js", "assets/index-abc.css"]
    _wheel(tmp_path / "a-1.0-cp310-linux.whl", app)
    _wheel(tmp_path / "a-1.0-cp310-win.whl", app)
    _sdist(tmp_path / "a-1.0.tar.gz", app)
    assert checker.main(["check", str(tmp_path)]) == 0
    assert "identical" in capsys.readouterr().out

    _wheel(tmp_path / "a-1.0-cp310-mac.whl", ["assets/index-abc.js"])
    assert checker.main(["check", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "a-1.0-cp310-mac.whl: no web_dist/index.html" in err
    assert "different web_dist files" in err

    empty = tmp_path / "empty"
    empty.mkdir()
    _sdist(empty / "b-1.0.tar.gz", ["index.html"])
    assert checker.problems(empty) == ["b-1.0.tar.gz: no web_dist/assets/*.js"]
    assert checker.problems(tmp_path / "none") == [
        f"no wheel or sdist in {tmp_path / 'none'}"
    ]
    assert checker.main(["check"]) == 2
