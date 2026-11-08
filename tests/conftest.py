"""Shared tiny synthetic fixtures (tests never touch the real dataset, CLAUDE.md §10.2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from faultaware.data.parse import parse_lines
from faultaware.data.sessions import build_sessions
from faultaware.data.split import split_sessions
from faultaware.features.build import Vocabulary, build_features

FIXTURES = Path(__file__).parent / "fixtures"


def synthetic_log(n_blocks: int = 300, seed: int = 0) -> tuple[list[str], pd.DataFrame]:
    """Generate HDFS-format lines for ``n_blocks`` blocks; ~15% anomalous.

    Anomalous blocks log an exception during the write phase (within the first seconds) and
    a WARN at deletion time, so both early and full-session models can learn them.
    """
    rng = np.random.default_rng(seed)
    lines: list[tuple[int, str]] = []
    labels = []
    base = 1226262918  # 2008-11-09 20:35:18
    for b in range(n_blocks):
        blk = f"blk_{1000 + b}" if b % 2 else f"blk_-{1000 + b}"
        anomalous = rng.random() < 0.15
        start = base + b * 60
        ip = f"10.250.{rng.integers(1, 20)}.{rng.integers(1, 250)}"
        events = [
            (0, f"INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /user/root/f{b} {blk}"),
            (
                1,
                f"INFO dfs.DataNode$DataXceiver: Receiving block {blk} src: /{ip}:5010 "
                f"dest: /{ip}:50010",
            ),
            (
                2,
                f"INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block {blk} terminating",
            ),
            (
                3,
                f"INFO dfs.DataNode$PacketResponder: Received block {blk} of size 67108864 "
                f"from /{ip}",
            ),
        ]
        for k in range(int(rng.integers(0, 4))):  # normal variability: 0-3 block reads
            events.append(
                (600 + 60 * k, f"INFO dfs.DataNode$DataXceiver: /{ip} Served block {blk} to /{ip}")
            )
        if anomalous:
            events.append(
                (
                    2,
                    f"INFO dfs.DataNode$DataXceiver: writeBlock {blk} received "
                    "exception java.io.IOException: Could not read from stream",
                )
            )
            events.append(
                (
                    3600,
                    f"WARN dfs.FSDataset: Unexpected error trying to delete block "
                    f"{blk}. BlockInfo not found in volumeMap.",
                )
            )
        else:
            events.append(
                (
                    3600,
                    f"INFO dfs.FSDataset: Deleting block {blk} file "
                    f"/mnt/hadoop/dfs/data/current/subdir1/{blk}",
                )
            )
        for off, text in events:
            ts = start + off
            d = pd.Timestamp(ts, unit="s")
            stamp = f"{d:%y%m%d} {d:%H%M%S}"
            level, rest = text.split(" ", 1)
            lines.append((ts, f"{stamp} 143 {level} {rest}"))
        labels.append({"BlockId": blk, "Label": "Anomaly" if anomalous else "Normal"})
    lines.sort(key=lambda x: x[0])
    return [line for _, line in lines], pd.DataFrame(labels)


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Parsed events, sessions with split, vocabulary and features for the full window."""
    lines, labels_df = synthetic_log()
    tmp = tmp_path_factory.mktemp("tiny")
    label_path = tmp / "labels.csv"
    labels_df.to_csv(label_path, index=False)
    events, templates = parse_lines(lines, n_workers=1, chunk_lines=500)
    from faultaware.data.parse import load_labels

    sessions = build_sessions(events, load_labels(label_path))
    split, bounds = split_sessions(sessions, (0.6, 0.2, 0.2), gap_seconds=120)
    names = dict(zip(templates.template_id, templates.template, strict=True))
    vocab = Vocabulary.fit(split[split.split == "train"]["templates"], names)
    warn = {i for i, lv in enumerate(events.attrs["levels"]) if lv == "WARN"}
    feats = {
        sp: build_features(split[split.split == sp], vocab, warn, max_len=16)
        for sp in ("train", "val", "test")
    }
    early = {
        sp: build_features(split[split.split == sp], vocab, warn, max_len=16, observe_seconds=10)
        for sp in ("train", "val", "test")
    }
    return {
        "events": events,
        "templates": templates,
        "sessions": sessions,
        "split": split,
        "bounds": bounds,
        "vocab": vocab,
        "warn": warn,
        "features": feats,
        "early": early,
        "lines": lines,
        "label_path": label_path,
    }
