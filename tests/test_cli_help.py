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


# --- examples in cli.py must run as written ---------------------------------------------------

_GENERATE_MODES = ("one_stage", "two_stage", "two_stages_hq", "distilled", "dfr")


def _examples() -> list[str]:
    import ltx_pipelines_mlx.cli as cli

    sources = [("epilog", _build_parser().epilog or ""), ("docstring", cli.__doc__ or "")]
    found = []
    for origin, text in sources:
        for line in text.replace("\\\n", " ").splitlines():  # join shell line continuations
            stripped = " ".join(line.split())
            if stripped.startswith("ltx-2-mlx "):
                found.append(pytest.param(stripped, id=f"{origin}:{stripped[10:50]}"))
    return found


def test_examples_are_found():
    assert len(_examples()) >= 15


@pytest.mark.parametrize("example", _examples())
def test_cli_example_parses(example: str):
    import shlex

    argv = shlex.split(example)[1:]
    args = _build_parser().parse_args(argv)
    if args.command == "generate":
        # generate refuses to run without exactly one mode flag (checked after parsing).
        assert sum(bool(getattr(args, m)) for m in _GENERATE_MODES) == 1, example
