"""
FoodSpec v2 Definition of Done:
- Deterministic outputs: seed is explicit; CV splits reproducible.
- No hidden global state.
- Every public API: type hints + docstring + example.
- Errors must be actionable (tell user what to fix).
- Any I/O goes through ArtifactRegistry.
- ProtocolV2 is the source of truth (YAML -> validated model).
- Each module has unit tests.
- Max 500-600 lines per file (human readability).
- All functions and variables: docstrings + comments as necessary.
- Modularity, scalability, flexibility, reproducibility, reliability.
- PEP 8 style, standards, and guidelines enforced.

CLI run command implementation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import typer

from foodspec.core.orchestrator import ExecutionEngine
from foodspec.core.protocol import (
    DataSpec,
    FeatureSpec,
    ModelSpec,
    PreprocessSpec,
    ProtocolV2,
    QCSpec,
    ReportingSpec,
    TaskSpec,
    ValidationSpec,
)


def generate_minimal_protocol(
    input_path: str,
    task_name: str,
    modality: str,
    label: str = "label",
) -> ProtocolV2:
    """Generate a minimal protocol from basic parameters.

    Creates a protocol with sensible defaults for quick experimentation.
    
    Args:
        input_path: Path to input data file (CSV)
        task_name: Task identifier (e.g., "oil_authentication")
        modality: Measurement modality (e.g., "raman", "nir")
        label: Target label column name
        
    Returns:
        ProtocolV2: Validated protocol with defaults applied
        
    Examples:
        >>> protocol = generate_minimal_protocol(
        ...     input_path="data.csv",
        ...     task_name="classify",
        ...     modality="raman"
        ... )
        >>> protocol.data.input
        'data.csv'
    """
    # Create DataSpec with minimal metadata_map
    data_spec = DataSpec(
        input=input_path,
        modality=modality,
        label=label,
        metadata_map={
            "sample_id": "sample_id",
            "modality": modality,
            "label": label,
        },
    )
    
    # Create TaskSpec with basic objective
    task_spec = TaskSpec(
        name=task_name,
        objective="classification",
        constraints={},
    )
    
    # Use default specs for all other components
    protocol = ProtocolV2(
        version="2.0.0",
        data=data_spec,
        task=task_spec,
        preprocess=PreprocessSpec(recipe="basic", steps=[]),
        qc=QCSpec(
            policy="warn",
            thresholds={},
            metrics=[],
        ),
        features=FeatureSpec(strategy="auto", modules=[]),
        model=ModelSpec(family="sklearn", estimator="logreg", params={}),
        validation=ValidationSpec(scheme="train_test_split", metrics=["accuracy"]),
        reporting=ReportingSpec(format="markdown", sections=["summary", "metrics"]),
    )
    
    return protocol.apply_defaults()


def run(
    protocol: Optional[Path] = typer.Option(
        None,
        "--protocol",
        "-p",
        exists=True,
        help="Path to protocol YAML file",
    ),
    outdir: Path = typer.Option(
        "./foodspec_runs/run",
        "--outdir",
        "-o",
        help="Output directory for run artifacts",
    ),
    seed: int = typer.Option(
        0,
        "--seed",
        "-s",
        help="Random seed for reproducibility",
    ),
    # Minimal flags for auto-generation
    input_path: Optional[str] = typer.Option(
        None,
        "--input",
        "-i",
        help="Input data path (auto-generates protocol if no --protocol given)",
    ),
    task: Optional[str] = typer.Option(
        None,
        "--task",
        "-t",
        help="Task name (auto-generates protocol if no --protocol given)",
    ),
    modality: Optional[str] = typer.Option(
        None,
        "--modality",
        "-m",
        help="Modality (e.g., raman, nir) (auto-generates protocol if no --protocol given)",
    ),
    label: str = typer.Option(
        "label",
        "--label",
        "-l",
        help="Label column name (used with auto-generated protocol)",
    ),
) -> None:
    """Run FoodSpec analysis workflow from a protocol file or minimal flags.

    Two modes of operation:
    
    1. Full protocol mode (recommended for production):
        foodspec run --protocol config.yaml --outdir runs/exp1 --seed 42
        
    2. Minimal flags mode (quick experimentation):
        foodspec run --input data.csv --task classify --modality raman --outdir runs/test
        
    Minimal flags mode auto-generates a protocol with sensible defaults. For full
    control over preprocessing, model selection, and validation, use a protocol file.

    Examples:
        # Full protocol mode
        foodspec run --protocol examples/protocol.yaml --outdir runs/exp1 --seed 42
        
        # Minimal flags mode
        foodspec run -i data.csv -t oil_auth -m raman -o runs/quick_test
        
        # Minimal mode with custom label column
        foodspec run -i spectra.csv -t classify -m nir -l class_label
    """
    typer.echo("=" * 70)
    typer.echo("FoodSpec v2.0 - Analysis Run")
    typer.echo("=" * 70)
    
    # Determine mode: protocol file or minimal flags
    if protocol is not None:
        # Mode 1: Use provided protocol file
        typer.echo(f"Protocol: {protocol}")
        protocol_path = protocol
        
    elif input_path and task and modality:
        # Mode 2: Auto-generate protocol from minimal flags
        typer.echo("Mode:     Minimal flags (auto-generating protocol)")
        typer.echo(f"Input:    {input_path}")
        typer.echo(f"Task:     {task}")
        typer.echo(f"Modality: {modality}")
        typer.echo(f"Label:    {label}")
        
        # Generate protocol
        try:
            auto_protocol = generate_minimal_protocol(
                input_path=input_path,
                task_name=task,
                modality=modality,
                label=label,
            )
            
            # Save to temporary file
            temp_protocol = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
                prefix="foodspec_auto_protocol_",
            )
            auto_protocol.dump(temp_protocol.name)
            protocol_path = Path(temp_protocol.name)
            
            typer.echo(f"Generated: {protocol_path}")
            
        except Exception as e:
            typer.secho(f"✗ Error generating protocol: {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
            
    else:
        # Missing required arguments
        typer.secho(
            "✗ Error: Must provide either --protocol OR all of (--input, --task, --modality)",
            fg=typer.colors.RED,
            err=True,
        )
        typer.secho(
            "\nExamples:",
            fg=typer.colors.YELLOW,
            err=True,
        )
        typer.secho(
            "  foodspec run --protocol config.yaml --outdir runs/exp1",
            fg=typer.colors.YELLOW,
            err=True,
        )
        typer.secho(
            "  foodspec run --input data.csv --task classify --modality raman",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=2)
    
    typer.echo(f"Output:   {outdir}")
    typer.echo(f"Seed:     {seed}")
    typer.echo("")

    try:
        # Execute workflow
        engine = ExecutionEngine()
        result = engine.run(
            protocol_or_path=protocol_path,
            outdir=outdir,
            seed=seed,
        )

        # Print summary
        typer.secho("✓ Analysis completed successfully", fg=typer.colors.GREEN)
        typer.echo("")
        typer.echo("Summary:")
        run_id = getattr(result.manifest, "run_id", result.output_dir.name)
        typer.echo(f"  Run ID:       {run_id}")
        typer.echo(f"  Output dir:   {result.output_dir}")
        typer.echo(f"  Manifest:     {result.output_dir / 'manifest.json'}")
        typer.echo(f"  Logs:         {result.output_dir / 'logs.txt'}")
        
        # Print key metrics if available on manifest
        metrics = getattr(result.manifest, "metrics", None)
        if metrics:
            typer.echo("")
            typer.echo("Key Metrics:")
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, float):
                    typer.echo(f"  {metric_name}: {metric_value:.4f}")
                else:
                    typer.echo(f"  {metric_name}: {metric_value}")
        
        # Check if HTML report exists
        report_path = result.output_dir / "artifacts" / "report.html"
        if report_path.exists():
            typer.echo("")
            typer.echo(f"Report:       {report_path}")
            typer.echo(f"              file://{report_path.absolute()}")
        
        typer.echo("")
        typer.echo(f"Results saved to: {result.output_dir}")
        typer.echo("=" * 70)
        
        # Cleanup temporary protocol if auto-generated
        if protocol is None and protocol_path.exists():
            protocol_path.unlink()

    except FileNotFoundError as e:
        typer.secho(f"✗ File not found: {e}", fg=typer.colors.RED, err=True)
        typer.secho(
            "  Check that input files and protocol paths exist.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=1)
    except ValueError as e:
        typer.secho(f"✗ Validation error: {e}", fg=typer.colors.RED, err=True)
        typer.secho(
            "  Fix protocol validation errors and try again.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=1)
    except NotImplementedError as e:
        typer.secho(f"✗ Not implemented: {e}", fg=typer.colors.RED, err=True)
        typer.secho(
            "  This pipeline stage is not yet implemented in this version.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"✗ Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
