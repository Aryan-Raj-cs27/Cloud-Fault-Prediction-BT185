"""Streaming parser for raw HDFS logs.

Each raw line looks like::

    081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-16 src: /10.250...
    <date> <time> <pid> <level> <component>: <content>

Event templates are obtained by *masking* variable fields (block ids, IPs, paths, numbers) —
a deterministic, label-free alternative to Drain that yields a small template vocabulary for
HDFS (see docs/DECISIONS.md D-002). One output row is emitted per (line, block) pair, because a
line can mention more than one block.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import date
from multiprocessing import Pool
from multiprocessing.pool import Pool as PoolType
from pathlib import Path

import numpy as np
import pandas as pd

from faultaware.utils.logging import get_logger

log = get_logger(__name__)

LINE_RE = re.compile(r"^(\d{6}) (\d{6}) (\d+) (\w+) ([^:\s]+): (.*)$")
BLOCK_RE = re.compile(r"blk_-?\d+")
IP_RE = re.compile(r"/?(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(:\d+)?")
PATH_RE = re.compile(r"(?:/[\w.\-$]+)+/?")
NUM_RE = re.compile(r"(?<![\w<])-?\d+(?![\w>])")
EPOCH = date(1970, 1, 1)


@dataclass
class ParsedLine:
    """A parsed log line before block expansion."""

    timestamp: int  # seconds since the Unix epoch (log timezone, assumed UTC)
    level: str
    template: str
    blocks: list[int]
    ip: str


def mask_content(content: str) -> str:
    """Replace variable fields in a log message with placeholders.

    Args:
        content: The message part of a log line (after ``component:``).

    Returns:
        The masked template text, e.g. ``"Receiving block <BLK> src: <IP> dest: <IP>"``.
    """
    text = BLOCK_RE.sub("<BLK>", content)
    text = IP_RE.sub("<IP>", text)
    text = PATH_RE.sub("<PATH>", text)
    text = NUM_RE.sub("<NUM>", text)
    return text.strip()


def parse_timestamp(date_str: str, time_str: str) -> int:
    """Convert ``YYMMDD`` + ``HHMMSS`` into epoch seconds.

    Args:
        date_str: Six-digit date, e.g. ``"081109"`` (2008-11-09).
        time_str: Six-digit time, e.g. ``"203518"``.

    Returns:
        Seconds since 1970-01-01.

    Raises:
        ValueError: If the strings are not valid dates/times.
    """
    day = date(2000 + int(date_str[:2]), int(date_str[2:4]), int(date_str[4:6]))
    hh, mm, ss = int(time_str[:2]), int(time_str[2:4]), int(time_str[4:6])
    if not (0 <= hh < 24 and 0 <= mm < 60 and 0 <= ss < 60):
        raise ValueError(f"invalid time {time_str}")
    return (day - EPOCH).days * 86400 + hh * 3600 + mm * 60 + ss


def parse_line(line: str) -> ParsedLine | None:
    """Parse one raw HDFS log line.

    Args:
        line: A raw log line.

    Returns:
        The parsed line, or ``None`` if the line does not match the HDFS format.
    """
    match = LINE_RE.match(line.rstrip("\r\n"))
    if match is None:
        return None
    date_s, time_s, _pid, level, component, content = match.groups()
    blocks = list(dict.fromkeys(int(b[4:]) for b in BLOCK_RE.findall(content)))
    ip_match = IP_RE.search(content)
    return ParsedLine(
        timestamp=parse_timestamp(date_s, time_s),
        level=level,
        template=f"{level} {component}: {mask_content(content)}",
        blocks=blocks,
        ip=ip_match.group(1) if ip_match else "",
    )


@dataclass
class _Chunk:
    start_line: int
    lines: list[str]


@dataclass
class _ChunkResult:
    line_no: list[int] = field(default_factory=list)
    timestamp: list[int] = field(default_factory=list)
    block: list[int] = field(default_factory=list)
    template: list[str] = field(default_factory=list)
    level: list[str] = field(default_factory=list)
    ip: list[str] = field(default_factory=list)
    n_unparsed: int = 0


def _parse_chunk(chunk: _Chunk) -> _ChunkResult:
    out = _ChunkResult()
    for offset, raw in enumerate(chunk.lines):
        parsed = parse_line(raw)
        if parsed is None:
            out.n_unparsed += 1
            continue
        for blk in parsed.blocks:
            out.line_no.append(chunk.start_line + offset)
            out.timestamp.append(parsed.timestamp)
            out.block.append(blk)
            out.template.append(parsed.template)
            out.level.append(parsed.level)
            out.ip.append(parsed.ip)
    return out


def _chunks(lines: Iterable[str], chunk_lines: int) -> Iterator[_Chunk]:
    buf: list[str] = []
    start = 0
    for i, line in enumerate(lines):
        if not buf:
            start = i
        buf.append(line)
        if len(buf) >= chunk_lines:
            yield _Chunk(start, buf)
            buf = []
    if buf:
        yield _Chunk(start, buf)


class _Interner:
    """Map strings to dense integer codes in first-seen order."""

    def __init__(self) -> None:
        self.codes: dict[str, int] = {}

    def encode(self, values: list[str]) -> np.ndarray:
        codes = self.codes
        return np.fromiter(
            (codes.setdefault(v, len(codes)) for v in values), dtype=np.int32, count=len(values)
        )

    def vocabulary(self) -> list[str]:
        return list(self.codes)


def parse_lines(
    lines: Iterable[str], n_workers: int = 1, chunk_lines: int = 200_000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse an iterable of raw lines into an event table and a template table.

    Args:
        lines: Raw log lines (any iterable, e.g. an open file).
        n_workers: Worker processes (1 = in-process, used by tests).
        chunk_lines: Lines per work unit.

    Returns:
        ``(events, templates)`` where ``events`` has columns ``line_no, timestamp, block_id,
        template_id, level_id, ip_id`` (one row per line × block) and ``templates`` has
        ``template_id, template, count``.
    """
    templates, levels, ips = _Interner(), _Interner(), _Interner()
    parts: list[pd.DataFrame] = []
    n_unparsed = 0
    chunks = _chunks(lines, chunk_lines)
    pool = Pool(n_workers) if n_workers > 1 else None
    try:
        for res in _bounded_map(pool, chunks, batch=max(1, 2 * n_workers)):
            n_unparsed += res.n_unparsed
            parts.append(_to_frame(res, templates, levels, ips))
            if len(parts) % 10 == 0:
                log.info("parsed %d chunks", len(parts))
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    if n_unparsed:
        log.warning("%d lines did not match the HDFS log format and were skipped", n_unparsed)
    events = pd.concat(parts, ignore_index=True) if parts else _empty_events()
    counts = np.bincount(events["template_id"], minlength=len(templates.codes))
    tmpl = pd.DataFrame(
        {
            "template_id": np.arange(len(templates.codes), dtype=np.int32),
            "template": templates.vocabulary(),
            "count": counts.astype(np.int64),
        }
    )
    events.attrs["levels"] = levels.vocabulary()
    return events, tmpl


def _bounded_map(
    pool: PoolType | None, chunks: Iterator[_Chunk], batch: int
) -> Iterator[_ChunkResult]:
    """Map chunks in bounded batches so the whole file is never held in memory."""
    if pool is None:
        yield from (_parse_chunk(c) for c in chunks)
        return
    while True:
        work = list(itertools.islice(chunks, batch))
        if not work:
            return
        yield from pool.map(_parse_chunk, work)


def _to_frame(
    res: _ChunkResult, templates: _Interner, levels: _Interner, ips: _Interner
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "line_no": np.asarray(res.line_no, dtype=np.int64),
            "timestamp": np.asarray(res.timestamp, dtype=np.int64),
            "block_id": np.asarray(res.block, dtype=np.int64),
            "template_id": templates.encode(res.template),
            "level_id": levels.encode(res.level).astype(np.int8),
            "ip_id": ips.encode(res.ip),
        }
    )


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "line_no": pd.Series(dtype=np.int64),
            "timestamp": pd.Series(dtype=np.int64),
            "block_id": pd.Series(dtype=np.int64),
            "template_id": pd.Series(dtype=np.int32),
            "level_id": pd.Series(dtype=np.int8),
            "ip_id": pd.Series(dtype=np.int32),
        }
    )


def parse_log_file(
    log_path: Path, n_workers: int = 8, chunk_lines: int = 200_000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a raw HDFS log file (see :func:`parse_lines`).

    Args:
        log_path: Path to ``HDFS.log``.
        n_workers: Worker processes.
        chunk_lines: Lines per work unit.

    Returns:
        ``(events, templates)``.
    """
    log.info("parsing %s with %d workers", log_path, n_workers)
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        return parse_lines(handle, n_workers=n_workers, chunk_lines=chunk_lines)


def load_labels(label_path: Path) -> pd.DataFrame:
    """Load Loghub block labels.

    Args:
        label_path: CSV with columns ``BlockId`` (``blk_<int>``) and ``Label``
            (``Normal``/``Anomaly``).

    Returns:
        DataFrame with ``block_id`` (int64) and ``label`` (int8, 1 = anomaly).

    Raises:
        ValueError: If unexpected label values are present.
    """
    raw = pd.read_csv(label_path)
    bad = set(raw["Label"].unique()) - {"Normal", "Anomaly"}
    if bad:
        raise ValueError(f"unexpected labels: {bad}")
    return pd.DataFrame(
        {
            "block_id": raw["BlockId"].str.slice(4).astype(np.int64),
            "label": (raw["Label"] == "Anomaly").astype(np.int8),
        }
    )
