"""Table writers (CSV + Markdown) for ``reports/tables``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub Markdown table (no external dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(v) for v in row.tolist()) + " |")
    return "\n".join(lines) + "\n"


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_table(df: pd.DataFrame, out_stem: Path) -> None:
    """Write ``<stem>.csv`` and ``<stem>.md``."""
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_stem.with_suffix(".csv"), index=False)
    out_stem.with_suffix(".md").write_text(to_markdown(df), encoding="utf-8")
