from typer.testing import CliRunner

from foodspec.cli.main import app


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_cli_io_help():
    runner = CliRunner()
    result = runner.invoke(app, ["io", "validate", "--help"])
    assert result.exit_code == 0

