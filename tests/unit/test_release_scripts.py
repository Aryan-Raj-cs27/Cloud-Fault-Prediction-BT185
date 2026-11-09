"""Release archive installer must reject path traversal and checksum mismatches."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from install_artifacts import safe_members


def _zip(tmp_path: Path, names: list[str]) -> zipfile.ZipFile:
    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name in names:
            zf.writestr(name, "x")
    return zipfile.ZipFile(path)


def test_accepts_expected_layout(tmp_path: Path) -> None:
    zf = _zip(
        tmp_path,
        ["models/fusion/w60/fusion.json", "reports/results_summary.json", "app/demo_samples.json"],
    )
    assert len(safe_members(zf)) == 3


@pytest.mark.parametrize(
    "bad", ["../evil.py", "/etc/passwd", "src/faultaware/x.py", "app/api.py", "models/../../x"]
)
def test_rejects_unsafe_paths(tmp_path: Path, bad: str) -> None:
    with pytest.raises(SystemExit):
        safe_members(_zip(tmp_path, [bad]))


def test_checksum_file_uses_lf_line_endings(tmp_path: Path) -> None:
    """`sha256sum -c` on Linux/macOS fails if the sidecar has a Windows CRLF ending."""
    from package_artifacts import write_checksum

    archive = tmp_path / "faultaware-artifacts.zip"
    archive.write_bytes(b"payload")
    digest = write_checksum(archive)
    raw = (tmp_path / "faultaware-artifacts.zip.sha256").read_bytes()
    assert raw == f"{digest}  faultaware-artifacts.zip\n".encode()
