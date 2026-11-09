"""Source code must never be swallowed by a .gitignore rule meant for artefacts.

Regression: an unanchored ``models/`` rule (meant for trained weights at the repo root) also
ignored ``src/faultaware/models/``, so a fresh clone could not import the model package.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = ("src", "app", "scripts", "tests")


@pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout"
)
def test_no_source_file_is_gitignored() -> None:
    sources = [
        p.relative_to(ROOT).as_posix()
        for d in SOURCE_DIRS
        for p in (ROOT / d).rglob("*.py")
        if "__pycache__" not in p.parts
    ]
    assert sources, "no source files found"
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        input="\n".join(sources),
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    ignored = result.stdout.split()
    assert ignored == [], f"source files ignored by .gitignore: {ignored}"
