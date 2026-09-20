"""docs/PIPELINES.md must list every CLI flag, and must not list flags the CLI does not have."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from ltx_pipelines_mlx.cli import _build_parser

DOC = Path(__file__).resolve().parents[1] / "docs" / "PIPELINES.md"
SUBCOMMANDS = ("generate", "a2v", "keyframe", "ic-lora", "hdr-ic-lora", "lipdub", "retake", "extend")
#: Flags shared by every subcommand that the guide documents once, under "Common flags".
FLAG_RE = re.compile(r"`(--[a-z0-9][a-z0-9-]*)")


def _subparser(name: str) -> argparse.ArgumentParser:
    parser = _build_parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices[name]
    raise AssertionError("no subparsers")


def parser_flags(subcommand: str) -> set[str]:
    """Long option names of one subcommand, ``--help`` excluded."""
    flags = set()
    for action in _subparser(subcommand)._actions:
        for opt in action.option_strings:
            if opt.startswith("--") and opt != "--help":
                flags.add(opt)
    return flags


def documented_flags(text: str) -> set[str]:
    return set(FLAG_RE.findall(text))


@pytest.fixture(scope="module")
def doc_text() -> str:
    return DOC.read_text(encoding="utf-8")


@pytest.mark.parametrize("subcommand", SUBCOMMANDS)
def test_every_cli_flag_is_documented(subcommand, doc_text):
    missing = parser_flags(subcommand) - documented_flags(doc_text)
    assert not missing, f"{subcommand}: flags missing from docs/PIPELINES.md: {sorted(missing)}"


def test_every_documented_flag_exists(doc_text):
    known = set().union(*(parser_flags(s) for s in SUBCOMMANDS))
    unknown = documented_flags(doc_text) - known
    assert not unknown, f"docs/PIPELINES.md mentions flags the CLI does not have: {sorted(unknown)}"


def test_guide_has_the_required_sections(doc_text):
    for heading in ("## Which pipeline?", "## Pipelines", "## Common flags", "## Flags by pipeline"):
        assert heading in doc_text, heading
