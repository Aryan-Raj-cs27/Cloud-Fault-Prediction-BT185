# Data

| Item | Value |
|---|---|
| Dataset | Loghub **HDFS_v1** (Hadoop Distributed File System logs, 203-node cluster, ~38.7 h, Nov 2008) |
| Origin | Xu et al., "Detecting large-scale system problems by mining console logs", SOSP 2009; redistributed by Loghub (https://github.com/logpai/loghub) |
| Files | `raw/HDFS.log` (1,577,982,906 bytes; 11,175,629 lines), `raw/anomaly_label.csv` (575,061 blocks; 16,838 anomalous = 2.93 %) |
| Download | `python scripts/download_data.py` (URL in `configs/data/hdfs.yaml`; verify it on the Loghub page if it changes) |
| Local copy | obtained by the team before 2026-10-29 (Kaggle mirror of Loghub HDFS_v1) |
| Licence | Loghub datasets are released for research purposes; cite Loghub and Xu et al. |

## Layout
- `raw/` — immutable, git-ignored. Never edited.
- `interim/hdfs_events.parquet` — one row per (log line × block): timestamp, block id, template id, level, IP.
- `interim/hdfs_templates.csv` — 54 masked event templates with counts.
- `processed/sessions.parquet` — one row per labelled block with event arrays + `split`.
- `processed/split_summary.json` — split boundaries, sizes and fault rates.
- `processed/features/<window>/<split>.npz` — cached model inputs.
- `legacy/HDFS_Parsed.csv` — the prototype's 50k-line parse (not used).

## Subset used
All 575,061 blocks. After the chronological split with a 30-min gap: train 402,549 (3.26 % anomalous), val 68,954 (3.45 %), test 70,185 (1.31 %); 33,373 blocks fall in the gaps.

Regenerate everything with `make data` (or `.\make.ps1 data`).
