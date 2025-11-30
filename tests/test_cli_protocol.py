from typer.testing import CliRunner

from foodspec.cli import app

runner = CliRunner()


def test_cli_protocol_benchmarks(monkeypatch, tmp_path):
    def fake_run(out, random_state=42):
        return {"classification": {"accuracy": 0.9}, "mixture": {"r2": 0.85}}

    monkeypatch.setattr("foodspec.apps.protocol_validation.run_protocol_benchmarks", fake_run)
    result = runner.invoke(app, ["protocol-benchmarks", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "classification" in result.output
