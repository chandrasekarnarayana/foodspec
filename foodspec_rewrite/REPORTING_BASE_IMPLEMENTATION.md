# Reporting Base Implementation - Prompt 2 Summary

## ✅ Completed Implementation

### 1. ReportContext (base.py)
**Purpose**: Load and provide access to run artifacts

**Key Features**:
- Loads manifest from `manifest.json`
- Parses protocol snapshot from manifest
- Loads optional artifact tables: metrics.csv, predictions.csv, qc.csv
- Loads trust outputs (uncertainty, abstention, etc.)
- Provides list of available artifacts
- Serializes to dict for template rendering

**Example Usage**:
```python
context = ReportContext.load(Path("/run/dir"))
print(context.manifest.seed)
print(context.metrics)
print(context.available_artifacts)
```

### 2. ReportBuilder (base.py)
**Purpose**: Build HTML reports with artifact validation and navigation

**Key Features**:
- Validates required artifacts based on mode (RESEARCH, REGULATORY, MONITORING)
- Raises actionable errors when artifacts are missing in strict modes
- Renders HTML with sidebar navigation using Jinja2
- Embeds metrics, QC, predictions as tables
- Includes reproducibility metadata
- Supports all report modes with mode-specific content

**Example Usage**:
```python
context = ReportContext.load(Path("/run/dir"))
builder = ReportBuilder(context)
html_path = builder.build_html(
    Path("/run/dir/report.html"),
    mode=ReportMode.REGULATORY
)
```

### 3. collect_figures() Helper
**Purpose**: Index images under artifacts/viz folder

**Key Features**:
- Recursively scans viz directory
- Groups figures by subdirectory (drift, interpretability, uncertainty, pipeline)
- Supports PNG, JPG, JPEG, SVG, GIF
- Returns dict mapping category → list of paths
- Returns empty dict if viz dir doesn't exist

**Example Usage**:
```python
figures = collect_figures(Path("/run/dir"))
print(figures["drift"])  # [Path(...), Path(...)]
```

### 4. Jinja2 Templates (templates/base.html)
**Purpose**: Responsive HTML layout with sidebar navigation

**Key Features**:
- Dark sidebar with navigation links
- Main content area with metadata header
- Section rendering based on mode-enabled sections
- Responsive grid layout for figures
- Info boxes for reproducibility metadata
- Mobile-friendly CSS

**Sections Rendered**:
- Header with metadata (seed, protocol hash, duration)
- Metrics table (if metrics present and section enabled)
- QC results table (if QC present and section enabled)
- Predictions table (if predictions present and section enabled)
- Uncertainty/Trust outputs (if available)
- Visualizations grid (if figures present)
- Reproducibility section (always included)

### 5. Artifact Validation
**Mode-Specific Validation**:
- **RESEARCH**: minimal (manifest, metrics, protocol_snapshot, data_fingerprint)
- **REGULATORY**: strict (requires qc, plus research artifacts)
- **MONITORING**: moderate (like RESEARCH, but flags if missing)

Error handling:
- RESEARCH mode: permissive, missing artifacts silently excluded
- REGULATORY mode: strict errors with list of missing artifacts and available alternatives
- Regulatory mode sets `warnings_as_errors=True` for actionable failure messages

## Tests (32 tests in test_reporting_base.py)

### ReportContext Tests (8 tests)
✓ Loads manifest, metrics, QC data
✓ Handles missing optional artifacts
✓ Raises error for missing manifest
✓ Provides available_artifacts list
✓ Serializes to dict for templating

### collect_figures Tests (4 tests)
✓ Indexes figures from viz directories
✓ Groups by category
✓ Supports multiple image formats
✓ Returns empty dict when no viz dir

### ReportBuilder Validation Tests (4 tests)
✓ Validates research mode minimal artifacts
✓ Validates regulatory mode requires qc
✓ Raises errors for missing regulatory artifacts
✓ Permissive in research mode for missing artifacts

### ReportBuilder HTML Generation Tests (12 tests)
✓ Builds HTML file
✓ Includes title, sidebar, metadata
✓ Renders metrics and QC tables
✓ Embeds figures from viz directories
✓ Creates parent directories
✓ Accepts mode as string or enum
✓ Includes reproducibility section

### ReportBuilder Error Handling Tests (2 tests)
✓ Invalid mode raises error
✓ Error message lists missing artifacts

### Mode-Specific Tests (2 tests)
✓ Research, regulatory, monitoring modes set descriptions

## Acceptance Criteria Met ✅

- ✅ **report.html builds from artifacts alone**: ReportContext loads all required data from run directory
- ✅ **Missing artifacts raise actionable errors**: REGULATORY mode provides clear error messages listing what's missing and what's available
- ✅ **Sidebar navigation**: Template includes left sidebar with section links
- ✅ **Images embedded via relative paths**: collect_figures() returns paths used in template
- ✅ **Jinja2 templates**: Base layout with clean, responsive design
- ✅ **Mode-aware content**: Sections rendered based on mode config

## Files Created

1. **foodspec/reporting/base.py** (450 lines)
   - ReportContext class with load() classmethod
   - ReportBuilder class with build_html() method
   - collect_figures() helper function

2. **foodspec/reporting/templates/base.html** (300+ lines)
   - Complete HTML5 layout
   - Sidebar navigation
   - Responsive CSS grid
   - Jinja2 template syntax

3. **tests/test_reporting_base.py** (450+ lines)
   - 32 comprehensive tests
   - Fixture-based setup
   - Mode-aware validation tests
   - Error handling tests

## Integration with Existing Code

- Imports ReportMode, get_mode_config, validate_artifacts from modes.py
- Imports ArtifactRegistry, RunManifest from core modules
- Exports ReportBuilder, ReportContext, collect_figures from __init__.py
- Works with existing artifact layout and manifest structure
- Respects mode-driven configuration from modes.py

## Next Steps (Future Prompts)

1. Create additional template variants (compact, detailed, print-friendly)
2. Add PDF generation (WeasyPrint or similar)
3. Implement partial report generation (single section)
4. Add custom CSS/branding options
5. Integrate with CLI/orchestrator for end-to-end workflow
