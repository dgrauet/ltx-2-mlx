#!/usr/bin/env python3
"""Bump version in all 4 pyproject.toml files of the ltx-2-mlx workspace.

Usage:
    python scripts/bump_version.py 0.2.0
    python scripts/bump_version.py 0.3.0-rc1

Rewrites the `version = "..."` line in:
- pyproject.toml (root)
- packages/ltx-core-mlx/pyproject.toml
- packages/ltx-pipelines-mlx/pyproject.toml
- packages/ltx-trainer/pyproject.toml

Stdlib only. Refuses invalid SemVer-ish values.
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

VERSION_RE = re.compile(r'^version\s*=\s*"[^"]*"', re.MULTILINE)
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")


def _bump_file(path: Path, new_version: str) -> None:
    text = path.read_text()
    new_text, count = VERSION_RE.subn(f'version = "{new_version}"', text, count=1)
    if count != 1:
        raise SystemExit(f'error: no `version = "..."` line found in {path}')
    path.write_text(new_text)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0] if argv else 'bump_version.py'} <version>", file=sys.stderr)
        return 2
    version = argv[1]
    if not SEMVER_RE.match(version):
        print(f"error: invalid version string: {version!r} (expected MAJOR.MINOR.PATCH[-PRERELEASE])", file=sys.stderr)
        return 2

    cwd = Path.cwd()
    for rel in PYPROJECTS:
        path = cwd / rel
        if not path.exists():
            print(f"error: {rel} not found (cwd={cwd})", file=sys.stderr)
            return 1
        _bump_file(path, version)
        print(f"bumped {rel} -> {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
