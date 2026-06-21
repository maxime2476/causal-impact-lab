#!/usr/bin/env python
"""Reject artifacts containing forbidden authorship/attribution references.

Enforces the project's attribution policy. Wired into git via ``pre-commit``
for two stages:

* ``pre-commit`` stage -- receives staged file paths and scans their contents.
* ``commit-msg`` stage -- receives the path to the commit-message file.

Usage
-----
    python scripts/check_provenance.py FILE [FILE ...]

Exit status is non-zero (blocking the commit) if any forbidden pattern is
found. The patterns are assembled from fragments so this file does not itself
contain the literal strings it forbids.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Case-insensitive forbidden patterns, assembled from fragments so the source
# of this checker never matches itself.
_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"cl" + r"aude",
        r"\bA" + r"I\b",
        r"co-authored-by:\s*cl" + r"aude",
        r"generated\s" + r"+with",
        r"language\s" + r"+model",
        r"assist" + r"ant",
        "\U0001f916",  # robot face
    )
)

# Never scan these (binary / lock files).
_SKIP_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".duckdb", ".parquet"}
)


def _iter_offences(path: Path) -> list[tuple[int, str]]:
    """Return ``(line_number, line)`` pairs in *path* that match a pattern."""
    if path.suffix.lower() in _SKIP_SUFFIXES:
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    offences: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if any(pat.search(line) for pat in _PATTERNS):
            offences.append((lineno, line.strip()))
    return offences


def main(argv: list[str]) -> int:
    """Scan *argv* paths; return 1 if any forbidden pattern is found."""
    found = False
    for raw in argv:
        path = Path(raw)
        if not path.is_file():
            continue
        for lineno, line in _iter_offences(path):
            found = True
            print(f"{path}:{lineno}: forbidden attribution pattern -> {line!r}")
    if found:
        print(
            "\nCommit blocked: remove forbidden attribution references.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
