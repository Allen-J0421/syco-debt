"""Settings: config/run.toml plus .env, resolved to absolute paths."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from .env into os.environ without overriding existing values."""
    path = path or PROJECT_ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]
    home: Path
    dataset: Path
    config_dir: Path = field(default=CONFIG_DIR)

    # ---- sections -------------------------------------------------------------
    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw.get(name, {}))

    @property
    def agent(self) -> dict[str, Any]:
        return self.section("agent")

    @property
    def prompter(self) -> dict[str, Any]:
        return self.section("prompter")

    @property
    def run(self) -> dict[str, Any]:
        return self.section("run")

    @property
    def measure(self) -> dict[str, Any]:
        return self.section("measure")

    @property
    def verify(self) -> dict[str, Any]:
        return self.section("verify")

    # ---- runtime layout under RDP_HOME ---------------------------------------
    @property
    def repos_dir(self) -> Path:
        return self.home / "repos"

    @property
    def worktrees_dir(self) -> Path:
        return self.home / "worktrees"

    @property
    def vendor_dir(self) -> Path:
        return self.home / "vendor"

    @property
    def scratch_dir(self) -> Path:
        return self.home / "scratch"

    def run_dir(self, run: str) -> Path:
        return self.home / "runs" / run


def load_settings(config_path: Path | None = None, *, home: str | None = None) -> Settings:
    load_dotenv()
    path = config_path or CONFIG_DIR / "run.toml"
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    paths = raw.get("paths", {})
    home_raw = home or os.environ.get("RDP_HOME") or paths.get("home", "~/rdp")
    home_path = Path(home_raw).expanduser().resolve()
    dataset = Path(paths.get("dataset", "../SWE-Refactor/pure_refactoring_data.json")).expanduser()
    if not dataset.is_absolute():
        dataset = (PROJECT_ROOT / dataset).resolve()
    return Settings(raw=raw, home=home_path, dataset=dataset, config_dir=path.parent)
