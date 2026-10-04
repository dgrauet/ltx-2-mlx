"""CLI ``--help`` contracts: maturity tier tags on subcommands."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from ltx_pipelines_mlx.cli import _build_parser

MATURITY_DOC = Path(__file__).resolve().parents[1] / "docs" / "PIPELINE_MATURITY.md"


def _subparsers_action(parser: argparse.ArgumentParser) -> argparse._SubParsersAction:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    raise AssertionError("no subparsers")


def _subcommand_help() -> dict[str, str]:
    return {a.dest: a.help or "" for a in _subparsers_action(_build_parser())._choices_actions}


def _tiered_subcommands() -> dict[str, str]:
    """Whole subcommands (not ``generate --flag`` modes) listed under Beta / Experimental."""
    tiers: dict[str, str] = {}
    tier = None
    for line in MATURITY_DOC.read_text().splitlines():
        if line.startswith("### "):
            tier = "beta" if "Beta" in line else "experimental" if "Experimental" in line else None
            continue
        if line.startswith("## "):
            tier = None
        if tier is None or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        m = re.fullmatch(r"`([a-z0-9-]+)`", cells[1]) if len(cells) > 1 else None
        if m:
            tiers[m.group(1)] = tier
    return tiers


def test_maturity_doc_lists_tiered_subcommands():
    tiers = _tiered_subcommands()
    assert tiers.get("hdr-ic-lora") == "experimental"
    assert tiers.get("a2v") == "beta"


@pytest.mark.parametrize("name, tier", sorted(_tiered_subcommands().items()))
def test_non_stable_subcommand_help_carries_tier_tag(name: str, tier: str):
    assert _subcommand_help()[name].startswith(f"[{tier}] ")
