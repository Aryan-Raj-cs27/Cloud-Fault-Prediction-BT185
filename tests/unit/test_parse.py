"""Unit tests for the raw-log parser."""

from __future__ import annotations

import numpy as np
import pytest

from faultaware.data.parse import mask_content, parse_line, parse_lines, parse_timestamp


def test_parse_timestamp_known_value() -> None:
    assert parse_timestamp("081109", "203518") == 1226262918


def test_parse_timestamp_rejects_invalid_time() -> None:
    with pytest.raises(ValueError):
        parse_timestamp("081109", "256000")


def test_mask_content_replaces_variables() -> None:
    text = "Receiving block blk_-160 src: /10.250.19.102:54106 dest: /10.250.19.102:50010"
    assert mask_content(text) == "Receiving block <BLK> src: <IP> dest: <IP>"


def test_mask_content_paths_and_numbers() -> None:
    text = "Deleting block blk_7 file /mnt/hadoop/dfs/data/current/subdir1/blk_7"
    assert mask_content(text) == "Deleting block <BLK> file <PATH><BLK>"
    assert mask_content("PacketResponder 1 for block blk_5 terminating") == (
        "PacketResponder <NUM> for block <BLK> terminating"
    )


def test_parse_line_extracts_fields() -> None:
    line = (
        "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block "
        "blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010"
    )
    parsed = parse_line(line)
    assert parsed is not None
    assert parsed.blocks == [-1608999687919862906]
    assert parsed.level == "INFO"
    assert parsed.ip == "10.250.19.102"
    assert parsed.template.startswith("INFO dfs.DataNode$DataXceiver: Receiving block <BLK>")


def test_parse_line_returns_none_for_garbage() -> None:
    assert parse_line("not a log line") is None


def test_parse_lines_expands_multi_block_lines() -> None:
    lines = [
        "081109 203518 1 INFO dfs.X: move blk_1 to blk_2",
        "081109 203519 1 INFO dfs.X: done blk_1",
        "garbage",
    ]
    events, templates = parse_lines(lines)
    assert len(events) == 3
    assert sorted(events.block_id.tolist()) == [1, 1, 2]
    assert templates["count"].sum() == 3
    assert events.line_no.tolist() == [0, 0, 1]


def test_parse_lines_parallel_matches_serial(tiny_data: dict) -> None:
    lines = tiny_data["lines"][:400]
    serial, t1 = parse_lines(lines, n_workers=1, chunk_lines=50)
    parallel, t2 = parse_lines(lines, n_workers=2, chunk_lines=50)
    assert np.array_equal(serial.to_numpy(), parallel.to_numpy())
    assert t1.template.tolist() == t2.template.tolist()
