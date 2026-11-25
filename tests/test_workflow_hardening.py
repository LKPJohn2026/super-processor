"""Supply-chain rules for the GitHub workflows, Dependabot and the build toolchain.

The release workflow holds a PyPI trusted-publishing token and a token that can
write releases, so everything it runs must be pinned: actions to commit SHAs,
Python build tools to exact versions with hashes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

# A dev extra; cibuildwheel's test environment does not have it.
yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
REQUIREMENTS = ROOT / ".github" / "requirements"

pytestmark = pytest.mark.skipif(
    not WORKFLOWS.is_dir(), reason="needs the repository checkout (.github/)"
)

SHA_PIN = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")
USES_LINE = re.compile(r"^\s*(?:-\s+)?uses:\s*(\S+)(.*)$")
# Values an attacker can choose (PR titles, branch names, commit messages...).
UNTRUSTED_EXPR = re.compile(r"\$\{\{[^}]*\b(github\.event\.|github\.head_ref)")
# Owners trusted to run in a job that holds a write token.
WRITE_JOB_OWNERS = {"actions", "pypa"}


def _load(text: str) -> dict[str, Any]:
    data = yaml.safe_load(text)
    assert isinstance(data, dict)
    # YAML 1.1 reads the bare key `on` as True.
    if True in data:
        data["on"] = data.pop(True)
    return data


def _workflow_files() -> list[Path]:
    files = sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))
    assert files, "no workflow files found"
    return files


def _is_local(ref: str) -> bool:
    return ref.startswith("./") or ref.startswith("docker://")


def _writes(perms: Any) -> list[str]:
    """Scopes a `permissions:` value grants write on."""
    if perms is None:
        return []
    if isinstance(perms, str):
        return ["*"] if perms == "write-all" else []
    assert isinstance(perms, dict)
    return sorted(str(k) for k, v in perms.items() if v == "write")


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    steps = job.get("steps") or []
    assert isinstance(steps, list)
    return steps


def unpinned_uses(text: str) -> list[str]:
    """`uses:` refs that are not a full commit SHA with a version comment."""
    bad = []
    for n, line in enumerate(text.splitlines(), 1):
        m = USES_LINE.match(line)
        if not m:
            continue
        ref, rest = m.group(1).strip("'\""), m.group(2)
        if _is_local(ref):
            continue
        if not SHA_PIN.match(ref):
            bad.append(f"line {n}: {ref} is not pinned to a 40-hex commit SHA")
        elif not re.match(r"^\s+#\s*v\d", rest):
            bad.append(f"line {n}: {ref} has no '# vX.Y.Z' comment")
    return bad


def permission_problems(data: dict[str, Any]) -> list[str]:
    bad = []
    top = _writes(data.get("permissions"))
    if top:
        bad.append(f"workflow-level permissions grant write: {top}")
    if "permissions" not in data:
        bad.append("no workflow-level permissions (defaults may be read/write)")
    for name, job in (data.get("jobs") or {}).items():
        writes = _writes(job.get("permissions"))
        if {"id-token", "pages", "*"} & set(writes) and "environment" not in job:
            bad.append(f"job {name} grants {writes} without an environment")
        if not writes:
            continue
        for step in _steps(job):
            ref = str(step.get("uses", ""))
            if ref and not _is_local(ref) and ref.split("/")[0] not in WRITE_JOB_OWNERS:
                bad.append(f"job {name} holds {writes} and runs third-party {ref}")
    return bad


def untrusted_in_run(data: dict[str, Any]) -> list[str]:
    bad = []
    for name, job in (data.get("jobs") or {}).items():
        for step in _steps(job):
            script = str(step.get("run", ""))
            if UNTRUSTED_EXPR.search(script):
                bad.append(f"job {name}: {step.get('name')!r} expands untrusted input")
    return bad


def unhashed_pip_installs(data: dict[str, Any]) -> list[str]:
    """pip installs that could resolve a package fresh from the index."""
    scripts: list[str] = [str(v) for v in (data.get("env") or {}).values()]
    for job in (data.get("jobs") or {}).values():
        scripts += [str(v) for v in (job.get("env") or {}).values()]
        for step in _steps(job):
            scripts.append(str(step.get("run", "")))
            scripts += [str(v) for v in (step.get("env") or {}).values()]
    bad = []
    for script in scripts:
        for line in script.splitlines():
            if not re.search(r"\bpip\s+install\b", line):
                continue
            hashed = re.search(r"--require-hashes\s+-r\s+\S+", line)
            local = re.search(r"\bpip\s+install\s+(--[\w-]+\s+)*dist/\S+\s*$", line)
            if not (hashed or local):
                bad.append(line.strip())
    return bad


# --- the checkers catch what they are meant to catch -----------------------

BAD_WORKFLOW = """\
name: Bad
on: [pull_request]
permissions:
  contents: write
jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: write
    steps:
      - uses: actions/checkout@v7
      - uses: someone/action@0123456789abcdef0123456789abcdef01234567
      - uses: ./local-action
      - run: echo "${{ github.event.pull_request.title }}"
      - run: python -m pip install build
      - run: python -m pip install --require-hashes -r reqs.txt
      - run: python -m pip install --no-deps dist/*.whl
"""


def test_checkers_flag_a_bad_workflow() -> None:
    data = _load(BAD_WORKFLOW)
    uses = unpinned_uses(BAD_WORKFLOW)
    assert any("actions/checkout@v7" in b for b in uses)
    assert any("no '# vX.Y.Z' comment" in b for b in uses)
    assert not any("local-action" in b for b in uses)
    perms = permission_problems(data)
    assert any("workflow-level" in b for b in perms)
    assert any("without an environment" in b for b in perms)
    assert any("third-party someone/action" in b for b in perms)
    assert untrusted_in_run(data)
    assert unhashed_pip_installs(data) == ["python -m pip install build"]


# --- the real workflows -----------------------------------------------------


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_actions_are_pinned_to_commit_shas(path: Path) -> None:
    assert unpinned_uses(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_write_permissions_are_job_scoped(path: Path) -> None:
    assert permission_problems(_load(path.read_text(encoding="utf-8"))) == []


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_run_scripts_do_not_expand_event_data(path: Path) -> None:
    assert untrusted_in_run(_load(path.read_text(encoding="utf-8"))) == []


def test_release_installs_only_hash_checked_tools() -> None:
    data = _load((WORKFLOWS / "release.yml").read_text(encoding="utf-8"))
    assert unhashed_pip_installs(data) == []
    env = data["env"]
    assert "CIBW_TEST_REQUIRES" not in env
    assert "--no-isolation" in env["CIBW_BUILD_FRONTEND"]
    for key in ("CIBW_BEFORE_BUILD", "CIBW_BEFORE_TEST"):
        assert (
            "--require-hashes -r {project}/.github/requirements/build.txt" in (env[key])
        )
    sdist = "\n".join(str(s.get("run", "")) for s in _steps(data["jobs"]["sdist"]))
    assert "python -m build --sdist --no-isolation" in sdist


def test_ci_package_job_uses_the_release_toolchain() -> None:
    data = _load((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))
    job = data["jobs"]["package"]
    assert unhashed_pip_installs({"jobs": {"package": job}}) == []
    runs = "\n".join(str(s.get("run", "")) for s in _steps(job))
    assert "python -m build --no-isolation" in runs


def _runs(job: dict[str, Any]) -> str:
    return "\n".join(str(s.get("run", "")) for s in _steps(job))


def _downloads_web_dist(job: dict[str, Any]) -> bool:
    return any(
        "download-artifact" in str(s.get("uses", ""))
        and (s.get("with") or {}).get("name") == "web-dist"
        and (s.get("with") or {}).get("path") == "src/super_processor/web_dist"
        for s in _steps(job)
    )


def test_release_builds_the_web_app_once_from_the_lockfile() -> None:
    data = _load((WORKFLOWS / "release.yml").read_text(encoding="utf-8"))
    web = data["jobs"]["web"]
    assert _writes(web.get("permissions")) == []
    node = [s for s in _steps(web) if "setup-node" in str(s.get("uses", ""))]
    assert len(node) == 1
    # A tag build must not restore a cache a branch run could have saved.
    assert "cache" not in (node[0].get("with") or {})
    build = [s for s in _steps(web) if "npm" in str(s.get("run", ""))]
    assert [s.get("working-directory") for s in build] == ["web"]
    lines = str(build[0]["run"]).split()
    assert lines[:2] == ["npm", "ci"]
    assert "install" not in lines
    assert "npm run build:package" in _runs(web)
    assert "npm run check:package" in _runs(web)
    for name in ("sdist", "wheels"):
        job = data["jobs"][name]
        assert "web" in job["needs"], name
        assert _downloads_web_dist(job), name
    assert data["jobs"]["sdist"]["env"]["SUPER_PROCESSOR_REQUIRE_WEB_DIST"] == "1"
    env = data["env"]["CIBW_ENVIRONMENT"]
    assert "SUPER_PROCESSOR_REQUIRE_WEB_DIST=1" in env
    assert "check_web_dist.py dist" in _runs(data["jobs"]["verify"])


def test_ci_puts_the_web_app_in_the_distributions() -> None:
    data = _load((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))
    frontend, package = data["jobs"]["frontend"], data["jobs"]["package"]
    assert "npm ci" in _runs(frontend)
    assert "npm run build:package" in _runs(frontend)
    uploads = [s for s in _steps(frontend) if "upload-artifact" in str(s.get("uses"))]
    assert [(s["with"]["name"], s["with"]["path"]) for s in uploads] == [
        ("web-dist", "src/super_processor/web_dist")
    ]
    assert "frontend" in package["needs"]
    assert _downloads_web_dist(package)
    assert package["env"]["SUPER_PROCESSOR_REQUIRE_WEB_DIST"] == "1"
    assert "check_web_dist.py dist" in _runs(package)


def test_release_publishes_only_from_v_tags() -> None:
    data = _load((WORKFLOWS / "release.yml").read_text(encoding="utf-8"))
    assert data["on"] == {"push": {"tags": ["v*"]}}
    pypi = data["jobs"]["pypi"]
    assert pypi["environment"]["name"] == "pypi"
    assert "startsWith(github.ref, 'refs/tags/v')" in pypi["if"]
    assert _writes(pypi["permissions"]) == ["id-token"]
    release = data["jobs"]["github-release"]
    assert _writes(release["permissions"]) == ["contents"]
    assert all(
        "uses" not in s or s["uses"].startswith("actions/") for s in _steps(release)
    )
    write_jobs = {n for n, j in data["jobs"].items() if _writes(j.get("permissions"))}
    assert write_jobs == {"github-release", "pypi"}


def test_pages_deploys_only_from_main_pushes() -> None:
    data = _load((WORKFLOWS / "pages.yml").read_text(encoding="utf-8"))
    assert data["permissions"] == {"contents": "read"}
    build, deploy = data["jobs"]["build"], data["jobs"]["deploy"]
    assert "permissions" not in build
    assert deploy["permissions"] == {"pages": "write", "id-token": "write"}
    main_push = "github.event_name == 'push' && github.ref == 'refs/heads/main'"
    assert deploy["if"] == main_push
    upload = [s for s in _steps(build) if "upload-pages-artifact" in str(s.get("uses"))]
    assert [s["if"] for s in upload] == [main_push]


# --- build toolchain --------------------------------------------------------


def _build_system_requires() -> list[str]:
    # Read by hand: tomllib is not in Python 3.10, which CI still runs.
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r"^\[build-system\]\n(.*?)(?=^\[)", text, re.S | re.M)
    assert m, "no [build-system] table"
    req = re.search(r"^requires\s*=\s*\[(.*?)\]", m.group(1), re.S | re.M)
    assert req, "no build-system requires"
    return re.findall(r'"([^"]+)"', req.group(1))


def _pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)(?:\[[^\]]*\])?==([^\s;\\]+)", line)
        if m:
            pins[re.sub(r"[-_.]+", "-", m.group(1)).lower()] = m.group(2)
    return pins


def test_build_system_requires_are_exact_pins() -> None:
    requires = _build_system_requires()
    assert requires
    for spec in requires:
        assert re.fullmatch(r"[A-Za-z0-9_.-]+(\[[\w,-]+\])?==[\w.]+", spec), spec
    names = {re.split(r"[\[=]", s)[0].lower() for s in requires}
    assert "wheel" not in names


def test_build_system_matches_hashed_build_requirements() -> None:
    """The release builds with --no-isolation from build.txt, so they must agree."""
    wanted: dict[str, str] = {}
    for spec in _build_system_requires():
        name, version = re.split(r"(?:\[[^\]]*\])?==", spec)
        wanted[re.sub(r"[-_.]+", "-", name).lower()] = version
    built_in = _pins(REQUIREMENTS / "build.in")
    locked = _pins(REQUIREMENTS / "build.txt")
    for name, version in wanted.items():
        assert built_in.get(name) == version, f"build.in: {name}"
        assert locked.get(name) == version, f"build.txt: {name}"


@pytest.mark.parametrize("name", ["build.txt", "tools.txt"])
def test_requirement_locks_are_fully_hashed(name: str) -> None:
    text = (REQUIREMENTS / name).read_text(encoding="utf-8").replace("\\\n", " ")
    entries = [line for line in text.splitlines() if re.match(r"^[A-Za-z0-9]", line)]
    assert entries
    for line in entries:
        assert re.match(r"^[A-Za-z0-9_.-]+==\S+", line), line
        assert re.search(r"--hash=sha256:[0-9a-f]{64}", line), line


def test_tools_lock_agrees_with_build_lock() -> None:
    build, tools = _pins(REQUIREMENTS / "build.txt"), _pins(REQUIREMENTS / "tools.txt")
    assert {"build", "cibuildwheel", "twine"} <= set(tools)
    for name in set(build) & set(tools):
        assert build[name] == tools[name], name


# --- Dependabot -------------------------------------------------------------


def test_dependabot_watches_actions_python_and_web() -> None:
    data = _load((ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8"))
    entries = {(u["package-ecosystem"], u["directory"]) for u in data["updates"]}
    assert ("npm", "/web") in entries
    assert ("github-actions", "/") in entries
    assert ("pip", "/") in entries
    assert ("pip", "/.github/requirements") in entries
