"""Check that every wheel and sdist in a folder carries the same web app.

Usage: python .github/scripts/check_web_dist.py DIST_DIR
"""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path

WHEEL_PREFIX = "super_processor/web_dist/"
SDIST_PART = "/src/super_processor/web_dist/"


def bundle(names: list[str], prefix: str) -> list[str]:
    """The web_dist files in an archive, relative to web_dist/."""
    found = sorted(name.split(prefix, 1)[1] for name in names if prefix in name)
    return [name for name in found if name and not name.endswith("/")]


def problems(dist: Path) -> list[str]:
    bad: list[str] = []
    listings: dict[str, list[str]] = {}
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if not wheels and not sdists:
        return [f"no wheel or sdist in {dist}"]
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            names = [n for n in archive.namelist() if n.startswith(WHEEL_PREFIX)]
        listings[wheel.name] = bundle(names, WHEEL_PREFIX)
    for sdist in sdists:
        with tarfile.open(sdist) as archive:
            # Files only: a tar lists its folders too.
            names = [m.name for m in archive.getmembers() if m.isfile()]
        listings[sdist.name] = bundle(names, SDIST_PART)
    for name, files in listings.items():
        if "index.html" not in files:
            bad.append(f"{name}: no web_dist/index.html")
        if not any(f.startswith("assets/") and f.endswith(".js") for f in files):
            bad.append(f"{name}: no web_dist/assets/*.js")
    if len({tuple(files) for files in listings.values()}) > 1:
        bad.append("the distributions carry different web_dist files")
    return bad


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    found = problems(Path(argv[1]))
    for line in found:
        print(f"error: {line}", file=sys.stderr)
    if not found:
        print("web app present and identical in every distribution")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
