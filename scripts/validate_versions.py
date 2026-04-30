#!/usr/bin/env python3
"""Validate that all 4 pyproject.toml files declare the same version,
and that this version matches the given tag.

Usage:
    python scripts/validate_versions.py v0.2.0
    python scripts/validate_versions.py 0.2.0   # leading 'v' optional

Exits 0 on success, non-zero on mismatch with a clear error message.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PYPROJECTS = [
    "pyproject.toml",
    "packages/ltx-core-mlx/pyproject.toml",
    "packages/ltx-pipelines-mlx/pyproject.toml",
    "packages/ltx-trainer/pyproject.toml",
]

VERSION_LINE_RE = re.compile(r'^\s*version\s*=\s*"([^"]+)"', re.MULTILINE)


def _read_version(path: Path) -> str:
    m = VERSION_LINE_RE.search(path.read_text())
    if not m:
        raise SystemExit(f'error: no `version = "..."` line in {path}')
    return m.group(1)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0] if argv else 'validate_versions.py'} <tag>", file=sys.stderr)
        return 2
    tag = argv[1]
    expected = tag[1:] if tag.startswith("v") else tag

    cwd = Path.cwd()
    versions: dict[str, str] = {}
    for rel in PYPROJECTS:
        versions[rel] = _read_version(cwd / rel)

    mismatched = {rel: v for rel, v in versions.items() if v != expected}
    if mismatched:
        print(f"error: version mismatch with tag {tag!r} (expected {expected!r})", file=sys.stderr)
        for rel, v in versions.items():
            marker = "❌" if rel in mismatched else "✓"
            print(f"  {marker} {rel}: {v}", file=sys.stderr)
        return 1

    print(f"all 4 pyproject.toml files agree: version = {expected}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
