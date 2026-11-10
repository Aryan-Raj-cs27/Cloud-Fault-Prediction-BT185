"""Download Loghub HDFS_v1 into data/raw/ if it is not already there.

Source: Loghub (https://github.com/logpai/loghub), HDFS_v1 — the URL lives in
configs/data/hdfs.yaml (``download_url``). Raw files are never modified after download.

Usage:
    python scripts/download_data.py
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

from faultaware.utils.config import load_config, project_path
from faultaware.utils.logging import get_logger

log = get_logger("download_data")


def main() -> None:
    """Download and extract HDFS.log and anomaly_label.csv."""
    cfg = load_config()
    log_path = project_path(cfg.data.raw_log)
    label_path = project_path(cfg.data.raw_labels)
    if log_path.is_file() and label_path.is_file():
        log.info("raw data already present in %s", log_path.parent)
        return
    raw_dir = log_path.parent
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "HDFS_v1.zip"
    log.info("downloading %s (~186 MB) ...", cfg.data.download_url)
    with urllib.request.urlopen(cfg.data.download_url) as resp, archive.open("wb") as out:
        shutil.copyfileobj(resp, out)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            name = Path(member).name
            if name in {"HDFS.log", "anomaly_label.csv"}:
                with zf.open(member) as src, (raw_dir / name).open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    archive.unlink()
    missing = [p for p in (log_path, label_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"archive did not contain {missing}; see data/README.md")
    log.info("raw data ready in %s", raw_dir)


if __name__ == "__main__":
    main()
