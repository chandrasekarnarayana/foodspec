# Workflows API

High-level domain-specific analysis workflows.

The `foodspec.workflows` module provides end-to-end workflows for common food spectroscopy applications.

## Oil Authentication

Complete oil authentication workflows are available through the CLI and protocol system. See the [Oil Authentication Guide](../workflows/authentication/oil_authentication.md) for details.

## Heating Quality

### analyze_heating_trajectory

Analyze thermal degradation patterns over time.

::: foodspec.heating_trajectory.analyze_heating_trajectory
    options:
      show_source: false
      heading_level: 4

## Library Search

### LibrarySearchWorkflow

Search spectral library for matches.

::: foodspec.workflows.library_search.LibrarySearchWorkflow
    options:
      show_source: false
      heading_level: 4

## Shelf Life

### estimate_remaining_shelf_life

Model shelf life from spectral evolution.

::: foodspec.workflows.shelf_life.estimate_remaining_shelf_life
    options:
      show_source: false
      heading_level: 4

## Data Governance

### summarize_dataset

Generate comprehensive dataset quality report.

::: foodspec.core.summary.summarize_dataset
    options:
      show_source: false
      heading_level: 4

### compute_readiness_score

Assess dataset readiness for modeling.

::: foodspec.qc.readiness.compute_readiness_score
    options:
      show_source: false
      heading_level: 4

## See Also

- **[Workflows Guide](../workflows/index.md)** - Workflow methodology
- **[Oil Authentication](../workflows/authentication/oil_authentication.md)** - Authentication workflows
- **[Examples](../examples_gallery.md)** - Complete workflow examples
