# FoodSpec Visualization & Reporting - Developer Integration Guide

## Architecture Overview

```
foodspec run / orchestration
    ↓
ProtocolRunner.run() → save_outputs()
    ↓
[BUILD REPORT] ← NEW: Automatic after successful run
    ↓
ReportContext.load()  ← Loads all artifacts
    ↓
ExperimentCard  ← Builds summary with risk assessment
ReportBuilder   ← Generates HTML report
    ↓
Output:
  - report.html
  - card.json
  - card.md
```

---

## Key Classes & Functions

### 1. ReportContext (`src/foodspec/reporting/base.py`)

```python
class ReportContext:
    """Load and provide access to run artifacts"""
    
    @classmethod
    def load(cls, run_dir: Path) -> ReportContext:
        """Load from run directory"""
        # Scans for:
        # - manifest.json (required)
        # - metrics.csv (optional)
        # - predictions.csv (optional)
        # - trust/* (optional)
        # - drift/* (optional)
        # - qc/* (optional)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to template-renderable dict"""
    
    @property
    def available_artifacts(self) -> List[str]:
        """List of available artifact types"""
```

**Key Methods**:
- `_load_trust_outputs()` - Scans trust/, drift/, qc/ subdirectories
- `_load_csv()` - Parses metrics/predictions/qc tables
- `collect_figures()` - Indexes all PNG/SVG/JPG files by category

**Usage**:
```python
context = ReportContext.load(Path("runs/exp1"))
print(context.available_artifacts)  # ['manifest', 'metrics', 'trust', ...]
print(context.trust_outputs.keys())  # ['calibration', 'conformal', ...]
```

### 2. ExperimentCard (`src/foodspec/reporting/cards.py`)

```python
@dataclass
class ExperimentCard:
    """Concise summary of a model run"""
    
    # Identity
    run_id: str
    timestamp: str
    task: str
    modality: str
    model: str
    validation_scheme: str
    
    # Metrics (8 fields)
    macro_f1: Optional[float]
    auroc: Optional[float]
    ece: Optional[float]
    coverage: Optional[float]
    abstain_rate: Optional[float]
    mean_set_size: Optional[float]        # NEW
    drift_score: Optional[float]           # NEW
    qc_pass_rate: Optional[float]          # NEW
    
    # Assessment
    confidence_level: ConfidenceLevel      # LOW/MEDIUM/HIGH
    deployment_readiness: DeploymentReadiness  # NOT_READY/PILOT/READY
    key_risks: List[str]
    auto_summary: str
    
    def to_json(self, out_path: Path) -> Path:
    def to_markdown(self, out_path: Path) -> Path:

def build_experiment_card(
    context: ReportContext,
    mode: ReportMode = RESEARCH,
) -> ExperimentCard:
    """Build card from context"""
```

**Key Functions**:
- `_extract_metrics()` - Parses all 8 metrics from tables + trust outputs
- `_assess_confidence()` - Scores confidence based on metrics + validation
- `_assess_deployment_readiness()` - Determines deployment status
- `_identify_risks()` - Identifies key risk factors

**Usage**:
```python
context = ReportContext.load("runs/exp1")
card = build_experiment_card(context, mode=ReportMode.RESEARCH)

# Export
card.to_json(Path("card.json"))
card.to_markdown(Path("card.md"))

# Access fields
print(f"Ready: {card.deployment_readiness.value}")
print(f"F1: {card.macro_f1}")
print(f"Coverage: {card.coverage:.1%}")
```

### 3. ReportBuilder (`src/foodspec/reporting/html.py`)

```python
class ReportBuilder:
    """Build HTML reports from context"""
    
    def __init__(self, context: ReportContext):
        self.context = context
        self._setup_jinja2()  # Initialize template environment
    
    def build_html(
        self,
        out_path: Path,
        mode: ReportMode = RESEARCH,
        title: str = "FoodSpec Report",
    ) -> Path:
        """Build and write HTML report"""
        # 1. Validate artifacts for mode
        # 2. Extract context dict
        # 3. Render template
        # 4. Write to disk
```

**Integration Points**:
- Jinja2 templates in `src/foodspec/reporting/templates/`
- PackageLoader with FileSystemLoader fallback
- Responsive HTML layout with sidebar navigation

**Usage**:
```python
context = ReportContext.load("runs/exp1")
builder = ReportBuilder(context)
builder.build_html(
    Path("report.html"),
    mode=ReportMode.REGULATORY,
    title="GxP Audit Report",
)
```

### 4. PaperFigureExporter (`src/foodspec/reporting/export.py`)

```python
class PaperFigureExporter:
    """Export matplotlib figures in publication presets"""
    
    def __init__(self, preset: str = "joss"):
        # Presets: joss, ieee, elsevier, nature
    
    def export_figure(
        self,
        fig: plt.Figure,
        out_dir: Path,
        name: str,
        formats: tuple = ("png", "svg"),
        dpi_png: int = 300,
    ) -> Dict[str, Path]:
        """Export single figure"""
    
    def create_figure_bundle(
        self,
        figures: Dict[str, plt.Figure],
        out_dir: Path,
        formats: tuple = ("png", "svg"),
    ) -> Path:
        """Export multiple figures with consistent styling"""

def export_paper_figures(
    run_dir: Path,
    out_dir: Optional[Path] = None,
    preset: str = "joss",
    formats: tuple = ("png", "svg"),
) -> Path:
    """High-level API: export all figures from run"""
```

**Usage**:
```python
from foodspec.reporting.export import export_paper_figures

figures_dir = export_paper_figures(
    run_dir="runs/exp1",
    preset="joss",
    formats=("png", "svg"),
)
# Output: runs/exp1/figures_export/
```

### 5. Visualization Functions (`src/foodspec/viz/comprehensive.py`)

```python
def plot_raw_vs_processed_overlay(
    wavenumbers: np.ndarray,
    X_raw: np.ndarray,
    X_processed: np.ndarray,
    n_samples: int = 5,
    seed: int = 0,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Overlay comparison of preprocessing"""

def plot_pca_umap(
    X_embedded: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = "PCA",
    seed: int = 0,
) -> plt.Figure:
    """2D embedding with optional coloring"""

def plot_coverage_efficiency_curve(
    alpha_values: np.ndarray,
    coverage: np.ndarray,
    efficiency: np.ndarray,
    target_alpha: float = 0.1,
    seed: int = 0,
) -> plt.Figure:
    """Conformal prediction trade-off"""

def plot_conformal_set_sizes(
    set_sizes: np.ndarray,
    labels: Optional[np.ndarray] = None,
    seed: int = 0,
) -> plt.Figure:
    """Distribution of prediction set sizes"""

def plot_abstention_distribution(
    abstain_flags: np.ndarray,
    labels: Optional[np.ndarray] = None,
    batch_ids: Optional[np.ndarray] = None,
    seed: int = 0,
) -> plt.Figure:
    """Abstention rates by class/batch"""
```

**All functions**:
- Return matplotlib Figure objects
- Support `seed` parameter (deterministic)
- Support `save_path` for automatic export
- Have `dpi` parameter for resolution control

---

## Data Flow: From Run to Report

### Step 1: Run Completes

```
ProtocolRunner.run(inputs)
  ↓
runner.save_outputs(result, target_dir)
  ├─ Creates: metrics.csv, predictions.csv
  ├─ Creates: qc_report.json
  └─ Creates: manifest.json
```

### Step 2: Automatic Report Generation (NEW)

```python
# In foodspec/cli/main.py, after successful run:

if report:  # Flag controlled by --report/--no-report
    from foodspec.reporting.api import build_report_from_run
    
    artifacts = build_report_from_run(
        run_dir=target,
        out_dir=target,
        mode=mode or "research",
        pdf=False,
    )
    # Returns: {
    #   "report_html": "...",
    #   "card_json": "...",
    #   "card_markdown": "...",
    # }
```

### Step 3: build_report_from_run() Orchestration

```python
def build_report_from_run(
    run_dir: Path,
    out_dir: Path,
    mode: str,
    pdf: bool,
    title: str,
) -> Dict[str, str]:
    """Orchestrate full report generation"""
    
    # 1. Load context
    context = ReportContext.load(run_dir)
    
    # 2. Build experiment card
    card = build_experiment_card(context, mode=mode)
    
    # 3. Generate HTML report
    builder = ReportBuilder(context)
    html_path = builder.build_html(
        out_dir / "report.html",
        mode=mode,
        title=title,
    )
    
    # 4. Export card formats
    card.to_json(out_dir / "card.json")
    card.to_markdown(out_dir / "card.md")
    
    # 5. Optional PDF
    if pdf:
        # PDF export via external tool
        pass
    
    return artifacts
```

---

## Extending the Reporting System

### Add a New Metric to ExperimentCard

1. **Add field to dataclass**:
   ```python
   @dataclass
   class ExperimentCard:
       # ... existing fields ...
       my_metric: Optional[float] = None
   ```

2. **Update _extract_metrics()**:
   ```python
   def _extract_metrics(context):
       metrics = {
           # ... existing metrics ...
           "my_metric": None,
       }
       # Add extraction logic:
       metrics["my_metric"] = _get_float(
           context.trust_outputs,
           ["my_metric", "alternate_name"],
       )
       return metrics
   ```

3. **Update to_markdown()**:
   ```python
   if self.my_metric is not None:
       metrics_items.append(f"- **My Metric**: {self.my_metric:.3f}")
   ```

4. **Update build_experiment_card()**:
   ```python
   return ExperimentCard(
       # ... existing fields ...
       my_metric=metrics["my_metric"],
   )
   ```

### Add a New Visualization Function

1. **Create function in viz/comprehensive.py**:
   ```python
   def plot_my_visualization(
       data: np.ndarray,
       seed: int = 0,
       save_path: Optional[Path] = None,
       dpi: int = 300,
   ) -> plt.Figure:
       """My visualization"""
       apply_style()
       
       fig, ax = plt.subplots(figsize=(6, 4))
       # Plot logic here
       
       if save_path:
           fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
       
       return fig
   ```

2. **Export in viz/__init__.py**:
   ```python
   from .comprehensive import plot_my_visualization
   ```

3. **Use in report**:
   ```python
   fig = plot_my_visualization(data, seed=42)
   exporter.export_figure(fig, out_dir, "my_plot")
   ```

### Add a New Report Mode

1. **Add to ReportMode enum** (`src/foodspec/reporting/modes.py`):
   ```python
   class ReportMode(str, Enum):
       RESEARCH = "research"
       REGULATORY = "regulatory"
       MONITORING = "monitoring"
       MY_MODE = "my_mode"  # NEW
   ```

2. **Add config** (`src/foodspec/reporting/modes.py`):
   ```python
   _MODE_CONFIGS[ReportMode.MY_MODE] = ModeConfig(
       mode=ReportMode.MY_MODE,
       enabled_sections=["summary", "metrics", "custom"],
       required_artifacts=["manifest", "metrics"],
       default_plots=["confusion_matrix", "my_plot"],
       strictness_level=2,
       warnings_as_errors=False,
       description="My custom mode",
   )
   ```

3. **Add template section** (`src/foodspec/reporting/templates/base.html`):
   ```html
   {% if "custom" in enabled_sections %}
     <section id="custom">
       <!-- Custom section content -->
     </section>
   {% endif %}
   ```

---

## Testing Integration

### Golden Run Fixture (`tests/fixtures/run_minimal/`)

Use this minimal run directory for testing:

```python
@pytest.fixture
def minimal_run_dir():
    return Path(__file__).parent.parent / "fixtures" / "run_minimal"

def test_my_report_feature(minimal_run_dir):
    context = ReportContext.load(minimal_run_dir)
    # Test logic
```

### Adding Tests

1. **Create test file** `tests/reporting/test_my_feature.py`:
   ```python
   import pytest
   from foodspec.reporting.base import ReportContext
   
   def test_my_feature(minimal_run_dir):
       context = ReportContext.load(minimal_run_dir)
       assert context.manifest is not None
   ```

2. **Run tests**:
   ```bash
   pytest tests/reporting/test_my_feature.py -v
   ```

---

## Performance Considerations

### ReportContext.load()

- **Time**: ~100-500ms for typical run (50+ artifacts)
- **Optimization**: Artifacts loaded in parallel where possible
- **Memory**: ~50-200 MB for loaded context

### Report Generation

- **HTML**: ~100 KB (lightweight)
- **Card generation**: ~50-100 ms
- **Figure export**: Depends on figure count (10-50 ms per figure)

### Best Practices

1. **Lazy loading**: Only load artifacts when needed
2. **Batch operations**: Export multiple figures in one pass
3. **Caching**: Cache parsed contexts if generating multiple reports

---

## Dependency Management

### Required
- `numpy` - Data handling
- `matplotlib` - Visualization
- `jinja2` - HTML templating

### Optional
- `reportlab` or `weasyprint` - PDF export
- `plotly` - Interactive plots (future)
- `pandas` - DataFrames (already required)

---

## Error Handling

### Graceful Degradation

```python
# Missing artifacts don't fail report generation
context = ReportContext.load(run_dir)
# Returns empty dicts for missing trust/drift/qc

# Report validation is mode-aware
try:
    builder.build_html(path, mode=REGULATORY)
except ValueError as e:
    # Required artifacts missing in regulatory mode
    logger.error(f"Regulatory report requires: {e}")
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)

logger.warning("Missing trust outputs; skipping calibration section")
logger.info(f"Generated report: {html_path}")
logger.debug(f"Loaded {len(context.metrics)} metric rows")
```

---

## Backward Compatibility

**No breaking changes**:
- ✅ Existing `build_report_from_run()` API preserved
- ✅ Existing visualization functions unchanged
- ✅ New functions added (not replaced)
- ✅ Optional report generation (can be disabled)
- ✅ Existing artifact formats unchanged

---

## References

- **User Guide**: [VISUALIZATION_REPORTING_QUICKSTART.md](VISUALIZATION_REPORTING_QUICKSTART.md)
- **Implementation**: [VISUALIZATION_REPORTING_SUMMARY.md](VISUALIZATION_REPORTING_SUMMARY.md)
- **Code**: `src/foodspec/reporting/`, `src/foodspec/viz/comprehensive.py`
- **Tests**: `tests/reporting/test_comprehensive_reports.py`

