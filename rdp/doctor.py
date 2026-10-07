"""Preflight: report which external tools and settings each stage needs and whether they are present."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from rdp.config import Settings
from rdp.runner.claude_agent import claude_binary


def _version(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return ((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()[0] if (proc.stdout or proc.stderr) else ""


def checks(settings: Settings) -> list[tuple[str, str, bool, str]]:
    """(stage, name, ok, detail)"""
    out: list[tuple[str, str, bool, str]] = []

    def add(stage: str, name: str, ok: bool, detail: str = "") -> None:
        out.append((stage, name, bool(ok), detail))

    add("all", "dataset", settings.dataset.is_file(), str(settings.dataset))
    home = str(settings.home)
    add("all", "runtime home outside iCloud, no spaces",
        " " not in home and "Mobile Documents" not in home and "CloudDocs" not in home, home)
    add("all", "git", shutil.which("git") is not None, _version(["git", "--version"]))

    claude = shutil.which(claude_binary())
    add("run", "claude CLI", claude is not None,
        _version([claude_binary(), "--version"]) if claude else "not on PATH (set RDP_CLAUDE_BIN or install Claude Code)")
    add("run", "GEMINI_API_KEY", bool(os.environ.get("GEMINI_API_KEY")), "in .env or the environment")
    try:
        import google.genai  # noqa: F401

        add("run", "google-genai package", True)
    except Exception as exc:
        add("run", "google-genai package", False, f"pip install google-genai ({exc})")

    measure = settings.measure
    java17 = Path(measure.get("jdk17_home", "")).expanduser() / "bin" / "java"
    add("measure", "JDK 17", java17.is_file(), str(java17))
    add("measure", "sonar-scanner", shutil.which("sonar-scanner") is not None, "brew install sonar-scanner")
    add("measure", "CK jar", (settings.vendor_dir / "ck.jar").is_file(), "python -m rdp setup-metrics")
    add("measure", "SonarQube server",
        (settings.vendor_dir / "sonarqube" / "bin" / "macosx-universal-64" / "sonar.sh").is_file(),
        "python -m rdp setup-metrics")

    add("verify", "Maven", shutil.which("mvn") is not None, "brew install maven")
    for label, path in sorted((settings.verify.get("jdk_homes") or {}).items(), key=lambda kv: int(kv[0])):
        add("verify", f"JDK {label}", (Path(path).expanduser() / "bin" / "java").is_file(), path)
    return out


def report(settings: Settings) -> int:
    rows = checks(settings)
    missing_by_stage: dict[str, int] = {}
    for stage, name, ok, detail in rows:
        mark = "ok     " if ok else "MISSING"
        print(f"  [{stage:7s}] {mark} {name}" + (f"  ({detail})" if detail else ""))
        if not ok:
            missing_by_stage[stage] = missing_by_stage.get(stage, 0) + 1
    if not missing_by_stage:
        print("all checks passed")
        return 0
    print("missing: " + ", ".join(f"{n} for {s}" for s, n in sorted(missing_by_stage.items())))
    return 1
