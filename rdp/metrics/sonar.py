"""SonarQube scans. Python port of the foundational `tools/metrics/lib/sonar.sh` + `common.sh`.

Same analysis as the foundational run: SonarQube Community Build 25.1, sonar-scanner CLI,
sources analysed without bytecode (empty `sonar.java.binaries`), SCM disabled. Each scan uses an
ephemeral project that is deleted afterwards. New here: code-smell issues are exported before
the project is deleted.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from rdp.util import eprint, file_lock

SONAR_METRICS = (
    "complexity,cognitive_complexity,code_smells,duplicated_lines_density,duplicated_lines,"
    "duplicated_blocks,sqale_rating,sqale_index,sqale_debt_ratio,ncloc,lines,files,functions,"
    "classes,comment_lines_density"
)
_EFFORT_RE = re.compile(r"(?:(\d+)d)?\s*(?:(\d+)h)?\s*(?:(\d+)min)?")


def effort_minutes(effort: str | None) -> int:
    """'1h30min' -> 90. SonarQube counts a day as 8 hours."""
    if not effort:
        return 0
    m = _EFFORT_RE.fullmatch(effort.strip())
    if not m:
        return 0
    d, h, mins = (int(x) if x else 0 for x in m.groups())
    return d * 8 * 60 + h * 60 + mins


@dataclass
class SonarServer:
    host: str
    sonar_home: Path           # vendor/sonarqube
    jdk17_home: Path
    token_file: Path
    dummy_bin: Path            # empty dir used as sonar.java.binaries
    work_dir: Path
    scanner_heap: str = "-Xmx2g"
    scan_timeout_s: int = 1800
    token: str = ""

    # ---- HTTP -------------------------------------------------------------------
    def _request(self, path: str, *, auth: tuple[str, str] | None = None, data: dict | None = None,
                 timeout: int = 60) -> tuple[int, str]:
        url = self.host.rstrip("/") + path
        body = urllib.parse.urlencode(data).encode() if data is not None else None
        req = urllib.request.Request(url, data=body, method="POST" if data is not None else "GET")
        user, password = auth if auth is not None else (self.token, "")
        if user:
            cred = base64.b64encode(f"{user}:{password}".encode()).decode()
            req.add_header("Authorization", f"Basic {cred}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError) as exc:
            return 0, str(exc)

    def _json(self, path: str, **kw) -> dict:
        _status, text = self._request(path, **kw)
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return obj if isinstance(obj, dict) else {}

    # ---- lifecycle --------------------------------------------------------------
    def status(self) -> str:
        return str(self._json("/api/system/status", auth=("", "")).get("status", ""))

    @property
    def _launcher(self) -> Path:
        return self.sonar_home / "bin" / "macosx-universal-64" / "sonar.sh"

    def start(self, *, wait_s: int = 300) -> None:
        if self.status() == "UP":
            return
        if not self._launcher.is_file():
            raise RuntimeError(f"SonarQube not installed at {self.sonar_home}; run `python -m rdp setup-metrics`")
        for sub in ("logs", "data", "temp"):
            (self.sonar_home / sub).mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, SONAR_JAVA_PATH=str(self.jdk17_home / "bin" / "java"))
        subprocess.run([str(self._launcher), "start"], env=env, capture_output=True)
        deadline = time.time() + wait_s
        while time.time() < deadline:
            if self.status() == "UP":
                return
            time.sleep(5)
        raise RuntimeError("SonarQube did not come up; see vendor/sonarqube/logs/")

    def stop(self) -> None:
        if self._launcher.is_file():
            subprocess.run([str(self._launcher), "stop"], capture_output=True)

    def ensure_token(self) -> str:
        """Reuse the saved token if valid; otherwise set the admin password once and mint one."""
        with file_lock(self.token_file.with_name(self.token_file.name + ".lock")):
            if self.token_file.is_file():
                tok = self.token_file.read_text().strip()
                if tok and self._json("/api/authentication/validate", auth=(tok, "")).get("valid") is True:
                    self.token = tok
                    return tok
            admin_pass = os.environ.get("SONAR_ADMIN_PASS", "Sonar!2026pass")
            self._request("/api/users/change_password", auth=("admin", "admin"),
                          data={"login": "admin", "previousPassword": "admin", "password": admin_pass})
            tok = self._json("/api/user_tokens/generate", auth=("admin", admin_pass),
                             data={"name": f"rdp-{int(time.time())}"}).get("token", "")
            if not tok:
                raise RuntimeError("could not obtain a SonarQube token (set SONAR_ADMIN_PASS if the "
                                   "admin password was changed)")
            self.token_file.parent.mkdir(parents=True, exist_ok=True)
            self.token_file.write_text(tok)
            self.token_file.chmod(0o600)
            self.token = tok
            return tok

    # ---- one scan ---------------------------------------------------------------
    def scan(self, key: str, src_dir: Path) -> tuple[dict, list[dict]]:
        """Scan a directory. Returns (measures JSON as returned by the API, code-smell issues).
        Raises RuntimeError when the scan or the server-side task fails."""
        self.dummy_bin.mkdir(parents=True, exist_ok=True)
        wd = self.work_dir / f"sw_{key}"
        log_path = self.work_dir / f"{key}.scan.log"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "sonar-scanner",
            f"-Dsonar.host.url={self.host}", f"-Dsonar.token={self.token}",
            f"-Dsonar.projectKey={key}", f"-Dsonar.projectName={key}",
            f"-Dsonar.projectBaseDir={src_dir}", "-Dsonar.sources=.",
            f"-Dsonar.java.binaries={self.dummy_bin}",
            "-Dsonar.sourceEncoding=UTF-8", "-Dsonar.scm.disabled=true",
            f"-Dsonar.working.directory={wd}",
        ]
        env = dict(os.environ, SONAR_SCANNER_OPTS=self.scanner_heap)
        try:
            with log_path.open("w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                               timeout=self.scan_timeout_s, cwd=str(src_dir))
            log_text = log_path.read_text(errors="replace")
            if "EXECUTION SUCCESS" not in log_text:
                raise RuntimeError(f"sonar-scanner failed for {key}: {log_text[-400:]}")
            task_id = ""
            report = wd / "report-task.txt"
            if report.is_file():
                for line in report.read_text().splitlines():
                    if line.startswith("ceTaskId="):
                        task_id = line.split("=", 1)[1].strip()
            status = ""
            for _ in range(600):
                status = str(self._json(f"/api/ce/task?id={task_id}").get("task", {}).get("status", ""))
                if status in ("SUCCESS", "FAILED", "CANCELED"):
                    break
                time.sleep(1)
            if status != "SUCCESS":
                raise RuntimeError(f"SonarQube analysis task {status or 'timed out'} for {key}")
            measures = self._json(
                f"/api/measures/component?component={key}&metricKeys={SONAR_METRICS}")
            if "component" not in measures:
                raise RuntimeError(f"no measures returned for {key}")
            return measures, self._issues(key)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"sonar-scanner timed out for {key}") from exc
        finally:
            self._request("/api/projects/delete", data={"project": key})
            import shutil

            shutil.rmtree(wd, ignore_errors=True)
            log_path.unlink(missing_ok=True)

    def _issues(self, key: str) -> list[dict]:
        out: list[dict] = []
        page = 1
        while True:
            data = self._json(
                f"/api/issues/search?components={key}&types=CODE_SMELL&resolved=false&ps=500&p={page}")
            issues = data.get("issues") or []
            for it in issues:
                component = str(it.get("component", ""))
                out.append({
                    "rule": it.get("rule"),
                    "severity": it.get("severity"),
                    "effort_min": effort_minutes(it.get("effort") or it.get("debt")),
                    "file": component.split(":", 1)[1] if ":" in component else component,
                    "line": it.get("line"),
                    "message": it.get("message"),
                })
            total = int((data.get("paging") or {}).get("total", data.get("total", 0)) or 0)
            if not issues or page * 500 >= total or page >= 20:
                break
            page += 1
        return out


def write_measures(path: Path, measures: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(measures))
