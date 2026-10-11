"""Offline tests for metric-tool installation."""

from __future__ import annotations

import urllib.error
import zipfile

from rdp.metrics import setup


def test_sonarqube_download_falls_back_to_same_version_on_403(tmp_path, monkeypatch):
    version = "25.1.0.102122"
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "ck.jar").write_bytes(b"existing CK jar")
    urls = []

    def fake_download(url, dest):
        urls.append(url)
        if url == setup.sonar_url(version):
            raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)
        with zipfile.ZipFile(dest, "w") as archive:
            archive.writestr(f"sonarqube-{version}/bin/macosx-universal-64/sonar.sh", "#!/bin/sh\n")

    monkeypatch.setattr(setup, "_download", fake_download)
    setup.setup_vendor(vendor, version)

    assert urls == [setup.sonar_url(version), setup.sonar_maven_url(version)]
    assert (vendor / "sonarqube/bin/macosx-universal-64/sonar.sh").is_file()
    assert not (vendor / "sonarqube.zip").exists()
