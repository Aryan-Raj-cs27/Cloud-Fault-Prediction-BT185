"""Install a release archive produced by ``scripts/package_artifacts.py`` on a server.

Verifies the SHA-256 checksum and rejects any archive member that would escape the project
directory (zip-slip) or land outside the expected folders.

Usage:
    python scripts/install_artifacts.py path/to/faultaware-artifacts.zip [--sha256 <hex>]
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_TOP = {"models", "reports", "app"}


def safe_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Validate every member path; raise on anything suspicious."""
    members = []
    for info in zf.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise SystemExit(f"unsafe path in archive: {info.filename}")
        if path.parts[0] not in ALLOWED_TOP:
            raise SystemExit(f"unexpected path in archive: {info.filename}")
        if path.parts[0] == "app" and path.as_posix() != "app/demo_samples.json":
            raise SystemExit(f"archive may not overwrite code: {info.filename}")
        members.append(info)
    return members


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--sha256", help="expected checksum (default: read <archive>.sha256)")
    args = parser.parse_args()
    expected = args.sha256
    sidecar = args.archive.with_name(args.archive.name + ".sha256")
    if expected is None and sidecar.is_file():
        expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    if expected is None:
        raise SystemExit("no checksum given and no .sha256 file found; refusing to install")
    if actual != expected.lower():
        raise SystemExit(f"checksum mismatch: expected {expected}, got {actual}")
    with zipfile.ZipFile(args.archive) as zf:
        members = safe_members(zf)
        zf.extractall(ROOT, members=members)
    print(f"installed {len(members)} files into {ROOT}")


if __name__ == "__main__":
    main()
