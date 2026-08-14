#!/usr/bin/env python3
"""Opt-in Metal System Trace launcher for qualified LTX workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path


def _build_record_command(
    output: Path,
    target_command: Sequence[str],
    *,
    time_limit: str | None,
    environment: Sequence[str],
) -> list[str]:
    command = [
        "xcrun",
        "xctrace",
        "record",
        "--template",
        "Metal System Trace",
        "--output",
        str(output),
        "--no-prompt",
        "--target-stdout",
        "-",
    ]
    if time_limit is not None:
        command.extend(["--time-limit", time_limit])
    for assignment in environment:
        command.extend(["--env", assignment])
    command.extend(["--launch", "--", *target_command])
    return command


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Launch one command under xctrace's Metal System Trace template. "
            "This is opt-in qualification tooling and is never enabled by normal runs."
        )
    )
    parser.add_argument("--output", required=True, help="Destination .trace bundle")
    parser.add_argument("--manifest", help="JSON manifest path (default: <output>.manifest.json)")
    parser.add_argument("--time-limit", help="Optional xctrace limit such as 5m or 300s")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Environment assignment forwarded to the launched target (repeatable)",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    target = list(args.command)
    if target and target[0] == "--":
        target = target[1:]
    if not target:
        parser.error("provide the target command after --")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else Path(f"{output}.manifest.json")
    toc_path = Path(f"{output}.toc.xml")
    record_command = _build_record_command(
        output,
        target,
        time_limit=args.time_limit,
        environment=args.env,
    )

    started = time.time()
    recorded = subprocess.run(record_command)
    exported_returncode: int | None = None
    if recorded.returncode == 0:
        exported = subprocess.run(
            [
                "xcrun",
                "xctrace",
                "export",
                "--input",
                str(output),
                "--toc",
                "--output",
                str(toc_path),
            ]
        )
        exported_returncode = exported.returncode

    from ltx_pipelines_mlx.utils.perf_profile import runtime_identity

    manifest = {
        "schema": "ltx.metal-system-trace.v1",
        "template": "Metal System Trace",
        "started_unix_seconds": started,
        "ended_unix_seconds": time.time(),
        "target_command": target,
        "forwarded_environment_names": [assignment.partition("=")[0] for assignment in args.env],
        "trace_path": str(output),
        "trace_sha256": _sha256(output),
        "toc_path": str(toc_path),
        "toc_sha256": _sha256(toc_path),
        "record_returncode": recorded.returncode,
        "export_returncode": exported_returncode,
        "runtime": runtime_identity(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return recorded.returncode or (exported_returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
