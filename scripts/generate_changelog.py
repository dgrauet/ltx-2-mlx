#!/usr/bin/env python3
"""Generate a Markdown changelog from git log, grouped by Conventional Commits prefix.

Usage:
    python scripts/generate_changelog.py             # full history, written to stdout
    python scripts/generate_changelog.py v0.1.0      # since v0.1.0 (exclusive) to HEAD

Sections (in this order, omitted if empty):
    ### Breaking Changes  (any commit subject containing `!:` after the type, or BREAKING in the type itself)
    ### Features          (feat:)
    ### Bug Fixes         (fix:)
    ### Other             (everything else, including docs/chore/refactor/test/style/perf/ci, AND any subject without a recognized prefix)

Stdlib only.
"""

from __future__ import annotations

import re
import subprocess
import sys

# Capture: type, optional bang, optional scope, message
PREFIX_RE = re.compile(r"^(?P<type>[a-zA-Z]+)(?P<bang>!)?(?:\([^)]+\))?:\s*(?P<msg>.+)$")


def _git_log(prev_tag: str | None) -> list[str]:
    range_spec = f"{prev_tag}..HEAD" if prev_tag else "HEAD"
    out = subprocess.check_output(
        ["git", "log", range_spec, "--pretty=format:%s"],
        text=True,
    )
    return [line for line in out.splitlines() if line.strip()]


def _classify(subject: str) -> str:
    """Return one of: 'breaking', 'feat', 'fix', 'other'."""
    m = PREFIX_RE.match(subject)
    if not m:
        return "other"
    type_ = m.group("type").lower()
    if m.group("bang"):
        return "breaking"
    if type_ == "feat":
        return "feat"
    if type_ == "fix":
        return "fix"
    return "other"


def _format_item(subject: str) -> str:
    return f"- {subject}"


def main(argv: list[str]) -> int:
    if len(argv) > 2:
        print(f"usage: {argv[0]} [prev_tag]", file=sys.stderr)
        return 2
    prev_tag = argv[1] if len(argv) == 2 else None

    subjects = _git_log(prev_tag)

    buckets: dict[str, list[str]] = {"breaking": [], "feat": [], "fix": [], "other": []}
    for s in subjects:
        buckets[_classify(s)].append(_format_item(s))

    sections = [
        ("Breaking Changes", buckets["breaking"]),
        ("Features", buckets["feat"]),
        ("Bug Fixes", buckets["fix"]),
        ("Other", buckets["other"]),
    ]

    parts: list[str] = []
    for title, items in sections:
        if items:
            parts.append(f"### {title}\n")
            parts.extend(items)
            parts.append("")

    if not parts:
        parts = ["_No changes._"]

    print("\n".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
