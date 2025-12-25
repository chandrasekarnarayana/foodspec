import pytest
from typer.testing import CliRunner

try:
    from foodspec.cli import app
except Exception as e:  # pragma: no cover
    pytest.skip(f"Unified CLI not available: {e}")


@pytest.fixture(scope="module")
def runner():
    return CliRunner()


def test_cli_top_level_help_lists_subcommands(runner):
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    # Expect unified subcommands present in help text
    for cmd in ("bench", "predict", "library-search", "fit", "report"):
        assert cmd in result.output


def test_cli_subcommand_help_runs(runner):
    for cmd in ("bench", "predict", "library-search", "fit", "report"):
        res = runner.invoke(app, [cmd, "--help"])
        assert res.exit_code == 0, f"{cmd} help failed: {res.output}"
