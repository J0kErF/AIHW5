"""CLI tests using Typer's runner (offline via the fake backend)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from localforge.cli.app import app

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("version", "baseline", "pull", "run", "compare"):
        assert cmd in result.output


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "localforge" in result.output


def test_run_with_fake_backend() -> None:
    result = runner.invoke(
        app,
        ["run", "--backend", "fake", "--model", "fake/tiny", "--max-new-tokens", "8"],
    )
    assert result.exit_code == 0
    assert "output:" in result.output


def test_compare_fake_suite_writes_report(tmp_path: Path) -> None:
    suite = tmp_path / "fake_suite.yaml"
    suite.write_text(
        "suite_id: faketest\n"
        "models: ['fake/tiny']\n"
        "backends: ['fake']\n"
        "dtypes: ['fp32']\n"
        "prompt: 'hello'\n"
        "max_new_tokens: 8\n",
        encoding="utf-8",
    )
    results_dir = tmp_path / "results"
    result = runner.invoke(
        app,
        ["compare", "--suite", str(suite), "--results-dir", str(results_dir)],
    )
    assert result.exit_code == 0, result.output
    assert (results_dir / "reports" / "faketest" / "report.html").exists()
    assert (results_dir / "reports" / "faketest" / "matrix.md").exists()


def test_compare_rejects_unknown_suite_path() -> None:
    result = runner.invoke(app, ["compare", "--suite", "does_not_exist.yaml"])
    assert result.exit_code != 0
