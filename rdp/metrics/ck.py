"""CK 0.7.0 runner (class.csv + method.csv): coupling between objects and per-method cyclomatic."""

from __future__ import annotations

import subprocess
from pathlib import Path


def run_ck(java_bin: Path, ck_jar: Path, src_dir: Path, out_dir: Path, *, heap: str = "-Xmx4g",
           timeout_s: int = 1800) -> bool:
    """Same invocation as the foundational `ck.sh`. True when metrics were extracted."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "ck.log"
    cmd = [str(java_bin), heap, "-jar", str(ck_jar), str(src_dir), "false", "0", "false", f"{out_dir}/"]
    try:
        with log.open("w") as fh:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False
    cls = out_dir / "class.csv"
    return "Metrics extracted" in log.read_text(errors="replace") and cls.is_file() and cls.stat().st_size > 0
