"""Bundle exactly what the deployed console needs into one zip (+ SHA-256 file).

Model weights are never committed to git (CLAUDE.md §6.4). Instead, attach this archive to a
GitHub Release (or copy it to the server) and run ``scripts/install_artifacts.py`` after cloning.

Contents: feature spec, fused models of every window (seed-0 artefacts only), fusion
calibrators, report summaries and the demo samples.

Usage:
    python scripts/package_artifacts.py            # -> release/faultaware-artifacts.zip
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def serving_files() -> list[Path]:
    """Paths (relative to the project root) the API needs at runtime."""
    models = ROOT / "models"
    files = [models / "serving" / "feature_spec.json"]
    for meta_file in sorted((models / "fusion").glob("*/fusion.json")):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        tag = meta_file.parent.name
        files += [meta_file, meta_file.parent / "fusion.joblib"]
        for name in (meta["supervised_model"], meta["anomaly_model"]):
            files += sorted(
                p for p in (models / "artefacts" / tag / name).rglob("*") if p.is_file()
            )
    files += sorted((models / "fusion").glob("score_distributions*.npz"))
    reports = ROOT / "reports"
    files += [
        reports / n
        for n in ("results_summary.json", "simulation_summary.json", "simulation_replay.json")
    ]
    files.append(ROOT / "app" / "demo_samples.json")
    missing = [str(p.relative_to(ROOT)) for p in files if not p.is_file()]
    if missing:
        raise SystemExit(
            f"missing artefacts (run make train eval simulate demo-samples): {missing}"
        )
    return [p.relative_to(ROOT) for p in files]


def write_checksum(archive: Path) -> str:
    """Write ``<archive>.sha256`` in ``sha256sum`` format and return the hex digest.

    The line ending is forced to LF: on Windows ``write_text`` would emit CRLF, and
    ``sha256sum -c`` on Linux/macOS then looks for a file named ``"<archive>\\r"``.
    """
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    sidecar = archive.with_name(archive.name + ".sha256")
    sidecar.write_text(f"{digest}  {archive.name}\n", encoding="utf-8", newline="\n")
    return digest


def main() -> None:
    """Write the archive and its checksum."""
    out_dir = ROOT / "release"
    out_dir.mkdir(exist_ok=True)
    archive = out_dir / "faultaware-artifacts.zip"
    files = serving_files()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in files:
            zf.write(ROOT / rel, rel.as_posix())
    digest = write_checksum(archive)
    size_mb = archive.stat().st_size / 1e6
    print(f"{archive} ({len(files)} files, {size_mb:.1f} MB)\nsha256 {digest}")


if __name__ == "__main__":
    main()
