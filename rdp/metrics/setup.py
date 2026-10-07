"""Download the pinned analysis tools into RDP_HOME/vendor (CK jar, SonarQube server)."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

from rdp.util import eprint

CK_URL = "https://repo1.maven.org/maven2/com/github/mauricioaniche/ck/0.7.0/ck-0.7.0-jar-with-dependencies.jar"


def sonar_url(version: str) -> str:
    return f"https://binaries.sonarsource.com/Distribution/sonarqube/sonarqube-{version}.zip"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    eprint(f"  downloading {url}")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.rename(dest)


def setup_vendor(vendor: Path, sonar_version: str) -> None:
    ck_jar = vendor / "ck.jar"
    if not (ck_jar.is_file() and ck_jar.stat().st_size > 0):
        _download(CK_URL, ck_jar)
    eprint(f"  ok  CK jar ({ck_jar})")

    sonar_home = vendor / "sonarqube"
    launcher = sonar_home / "bin" / "macosx-universal-64" / "sonar.sh"
    if not launcher.is_file():
        archive = vendor / "sonarqube.zip"
        _download(sonar_url(sonar_version), archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(vendor)
        shutil.rmtree(sonar_home, ignore_errors=True)
        (vendor / f"sonarqube-{sonar_version}").rename(sonar_home)
        archive.unlink()
        # zipfile drops the executable bit
        for path in (sonar_home / "bin").rglob("*"):
            if path.is_file():
                path.chmod(0o755)
        for path in (sonar_home / "elasticsearch" / "bin").glob("*"):
            if path.is_file():
                path.chmod(0o755)
    eprint(f"  ok  SonarQube {sonar_version} ({sonar_home})")
    (vendor / "dummybin").mkdir(parents=True, exist_ok=True)
