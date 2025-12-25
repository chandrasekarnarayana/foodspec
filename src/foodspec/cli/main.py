"""Command-line interface for foodspec.

This module assembles CLI commands organized into logical groups:
- Data management (csv-to-library, library-search, library-auth, model-info)
- Preprocessing (preprocess)
- Modeling (qc, fit, predict)
- Analysis (oil-auth, heating, domains, mixture, hyperspectral, aging, shelf-life)
- Workflow orchestration (run-exp, protocol-benchmarks, bench)
- Utilities (about, report)
"""

from __future__ import annotations

from typing import Optional
from importlib.metadata import PackageNotFoundError, version as pkg_version

import matplotlib
import typer

matplotlib.use("Agg")

from foodspec.cli.commands.analysis import analysis_app
from foodspec.cli.commands.data import data_app
from foodspec.cli.commands.modeling import modeling_app
from foodspec.cli.commands.preprocess import preprocess_app
from foodspec.cli.commands.utils import utils_app
from foodspec.cli.commands.workflow import workflow_app

app = typer.Typer(help="foodspec command-line interface")

# Global options for root command
@app.callback(invoke_without_command=True)
def root_callback(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        help="Show FoodSpec version and exit",
    )
):
    """Root-level options for the foodspec CLI."""
    if version:
        try:
            v = pkg_version("foodspec")
        except PackageNotFoundError:
            v = "unknown"
        typer.echo(v)
        raise typer.Exit()

# Register command groups
# Data commands
app.command("csv-to-library")(data_app.registered_commands[0].callback)
app.command("library-search")(data_app.registered_commands[1].callback)
app.command("library-auth")(data_app.registered_commands[2].callback)
app.command("model-info")(data_app.registered_commands[3].callback)

# Preprocessing
app.command("preprocess")(preprocess_app.registered_commands[0].callback)

# Modeling
app.command("qc")(modeling_app.registered_commands[0].callback)
app.command("fit")(modeling_app.registered_commands[1].callback)
app.command("predict")(modeling_app.registered_commands[2].callback)

# Analysis
app.command("oil-auth")(analysis_app.registered_commands[0].callback)
app.command("heating")(analysis_app.registered_commands[1].callback)
app.command("domains")(analysis_app.registered_commands[2].callback)
app.command("mixture")(analysis_app.registered_commands[3].callback)
app.command("hyperspectral")(analysis_app.registered_commands[4].callback)
app.command("aging")(analysis_app.registered_commands[5].callback)
app.command("shelf-life")(analysis_app.registered_commands[6].callback)

# Workflow
app.command("run-exp")(workflow_app.registered_commands[0].callback)
app.command("protocol-benchmarks")(workflow_app.registered_commands[1].callback)
app.command("bench")(workflow_app.registered_commands[2].callback)

# Utilities
app.command("about")(utils_app.registered_commands[0].callback)
app.command("report")(utils_app.registered_commands[1].callback)


def main():
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
