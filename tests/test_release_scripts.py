"""Unit tests for scripts/{bump_version,validate_versions,generate_changelog}.py.

These are stdlib-only tests with no MLX imports — they can run on any host
including ubuntu-latest in CI.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _make_workspace(tmp_path: Path) -> Path:
    """Create a fake monorepo with 4 pyproject.toml files at version 0.1.0."""
    (tmp_path / "packages/ltx-core-mlx").mkdir(parents=True)
    (tmp_path / "packages/ltx-pipelines-mlx").mkdir(parents=True)
    (tmp_path / "packages/ltx-trainer").mkdir(parents=True)

    root_toml = textwrap.dedent("""\
        [project]
        name = "ltx-2-mlx"
        version = "0.1.0"
        description = "x"
        """)

    def pkg_toml(name: str) -> str:
        return textwrap.dedent(f"""\
            [build-system]
            requires = ["hatchling"]

            [project]
            name = "{name}"
            version = "0.1.0"
            """)

    (tmp_path / "pyproject.toml").write_text(root_toml)
    (tmp_path / "packages/ltx-core-mlx/pyproject.toml").write_text(pkg_toml("ltx-core-mlx"))
    (tmp_path / "packages/ltx-pipelines-mlx/pyproject.toml").write_text(pkg_toml("ltx-pipelines-mlx"))
    (tmp_path / "packages/ltx-trainer/pyproject.toml").write_text(pkg_toml("ltx-trainer-mlx"))
    return tmp_path


def test_bump_version_rewrites_all_four_files(tmp_path):
    workspace = _make_workspace(tmp_path)
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "bump_version.py"), "0.2.0"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    for rel in [
        "pyproject.toml",
        "packages/ltx-core-mlx/pyproject.toml",
        "packages/ltx-pipelines-mlx/pyproject.toml",
        "packages/ltx-trainer/pyproject.toml",
    ]:
        text = (workspace / rel).read_text()
        assert 'version = "0.2.0"' in text, f"{rel} not bumped"
        assert 'version = "0.1.0"' not in text, f"{rel} still has old version"


def test_bump_version_rejects_invalid_semver(tmp_path):
    workspace = _make_workspace(tmp_path)
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "bump_version.py"), "not-a-version"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "version" in result.stderr.lower()


def test_bump_version_accepts_prerelease(tmp_path):
    workspace = _make_workspace(tmp_path)
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "bump_version.py"), "0.3.0-rc1"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert 'version = "0.3.0-rc1"' in (workspace / "pyproject.toml").read_text()


def test_bump_version_handles_indented_version_line(tmp_path):
    """TOML allows indented keys; the bumper should rewrite them and preserve indentation."""
    workspace = _make_workspace(tmp_path)
    indented = textwrap.dedent("""\
        [project]
          name = "ltx-2-mlx"
          version = "0.1.0"
          description = "x"
        """)
    (workspace / "pyproject.toml").write_text(indented)

    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "bump_version.py"), "0.2.0"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    text = (workspace / "pyproject.toml").read_text()
    assert '  version = "0.2.0"' in text, "indentation must be preserved"
    assert 'version = "0.1.0"' not in text
