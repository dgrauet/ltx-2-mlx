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


def test_validate_versions_passes_when_all_match(tmp_path):
    workspace = _make_workspace(tmp_path)
    # bump everything to 0.2.0 first
    subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "bump_version.py"), "0.2.0"],
        cwd=workspace,
        check=True,
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "validate_versions.py"), "v0.2.0"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_validate_versions_fails_on_mismatch(tmp_path):
    workspace = _make_workspace(tmp_path)
    # only bump root, leave packages at 0.1.0
    root = workspace / "pyproject.toml"
    root.write_text(root.read_text().replace('version = "0.1.0"', 'version = "0.2.0"'))

    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "validate_versions.py"), "v0.2.0"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "ltx-core-mlx" in result.stderr or "ltx-core-mlx" in result.stdout


def test_validate_versions_strips_v_prefix(tmp_path):
    workspace = _make_workspace(tmp_path)
    subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "bump_version.py"), "0.2.0"],
        cwd=workspace,
        check=True,
    )
    # accept both with and without v
    for tag in ["v0.2.0", "0.2.0"]:
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "validate_versions.py"), tag],
            cwd=workspace,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"tag={tag}: {result.stderr}"


def test_changelog_groups_commits_by_prefix(tmp_path):
    """Build a tiny git repo with known commits, run the generator, check sections."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

    git("init", "-q", "-b", "main")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    (repo / "README.md").write_text("init\n")
    git("add", ".")
    git("commit", "-qm", "chore: initial")

    for msg in [
        "feat: shiny new pipeline",
        "fix: off-by-one in scheduler",
        "docs: update README",
        "feat!: drop python 3.10",
        "refactor: rename helper",
        "weird message without a prefix",
    ]:
        (repo / "x.txt").write_text(msg)
        git("add", ".")
        git("commit", "-qm", msg)

    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "generate_changelog.py")],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    # Section headers
    assert "### Breaking Changes" in out
    assert "### Features" in out
    assert "### Bug Fixes" in out
    assert "### Other" in out
    # Items in correct sections
    assert "shiny new pipeline" in out
    assert "off-by-one in scheduler" in out
    assert "drop python 3.10" in out  # feat! goes to Breaking
    # Refactor and "weird message" go under Other
    assert "rename helper" in out
    assert "weird message without a prefix" in out


def test_changelog_uses_range_when_prev_tag_given(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

    git("init", "-q", "-b", "main")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    (repo / "f").write_text("a")
    git("add", ".")
    git("commit", "-qm", "feat: alpha")
    git("tag", "v0.1.0")
    (repo / "f").write_text("b")
    git("add", ".")
    git("commit", "-qm", "feat: beta")

    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "generate_changelog.py"), "v0.1.0"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "beta" in result.stdout
    assert "alpha" not in result.stdout  # alpha is before v0.1.0
