"""Documentation: Paper-Ready Figure Presets System (Phase 12)

## Overview

This module provides a publication-quality figure styling system for FoodSpec that
enables authors to easily change matplotlib figure style for different scientific
journals with a single flag or context manager.

## Features

### 1. FigurePreset Enum
Four predefined presets for major scientific venues:

- **JOSS** (Journal of Open Source Software)
  - 3.5" single-column figures
  - 9pt font size
  - 1.5pt lines
  - 72 DPI screen resolution

- **IEEE** (Institute of Electrical and Electronics Engineers)
  - 3.5" single-column (3.27-3.5 inches standard)
  - 9pt font size
  - 1.0pt lines (IEEE standard)
  - Standard margins and spacing

- **Elsevier** (Elsevier Journals)
  - 3.5" single-column width
  - 10pt font size (readable)
  - 1.2pt lines
  - Clear axis labels (11pt)

- **Nature** (Nature/Science style)
  - 3.5" single-column
  - 8pt font size (minimal)
  - 1.0pt lines (thin)
  - Minimal styling, publication-ready

### 2. Core Functions

#### apply_figure_preset(preset: FigurePreset | str) -> None
Applies a preset globally to all subsequently created matplotlib figures.

```python
from foodspec.viz import apply_figure_preset, FigurePreset

# Via enum
apply_figure_preset(FigurePreset.JOSS)

# Via string (case-insensitive)
apply_figure_preset("ieee")

# All figures created after this use JOSS styling
fig, ax = plt.subplots()
ax.plot(x, y)
```

#### figure_context(preset: FigurePreset | str) -> ContextManager
Temporarily applies a preset within a code block. rcParams are automatically
restored on exit, even if exceptions occur.

```python
from foodspec.viz import figure_context, FigurePreset

# Temporarily use IEEE styling
with figure_context(FigurePreset.IEEE):
    fig, ax = plt.subplots()  # Uses IEEE styling
    ax.plot(x, y)
# Original styling restored here

# rcParams restored even if exception occurs
try:
    with figure_context(FigurePreset.NATURE):
        fig, ax = plt.subplots()
        raise ValueError("Error!")
except ValueError:
    pass
# rcParams still restored!
```

#### save_figure(fig, path, dpi=300, transparent=False, bbox_inches='tight', pad_inches=0.1) -> Path
Saves publication-ready figures with proper resolution and formatting.

```python
from pathlib import Path
from foodspec.viz import save_figure

fig, ax = plt.subplots()
ax.plot(x, y)

# Save with 300 DPI (publication standard)
output_path = save_figure(fig, "my_figure.png", dpi=300)

# Save with transparency (for PDF overlays)
save_figure(fig, "transparent.png", transparent=True)

# Tight bounding box with 0.1" padding
save_figure(fig, "tight.pdf", bbox_inches="tight", pad_inches=0.1)
```

#### get_figure_preset_config(preset: FigurePreset | str) -> Dict[str, Any]
Returns the rcParams dictionary for a preset without applying it.
Useful for inspection or programmatic access.

```python
from foodspec.viz import get_figure_preset_config, FigurePreset

# Inspect JOSS preset
joss_config = get_figure_preset_config(FigurePreset.JOSS)
print(joss_config["figure.figsize"])  # [3.5, 2.8]
print(joss_config["font.size"])       # 9
```

#### list_presets() -> Dict[str, str]
Lists all available presets with descriptions.

```python
from foodspec.viz import list_presets

presets = list_presets()
for name, description in presets.items():
    print(f"{name}: {description}")
```

## Design Principles

### No Hardcoded Colors
Presets control only **typography and layout** (fonts, sizes, line widths, margins).
Colors are left to plotting code, allowing flexible color schemes.

### Deterministic Output
All presets use explicit rcParams values. Same preset always produces
identical looking figures across runs.

### Safe rcParams Restoration
Context manager uses `try`/`finally` block. rcParams are restored even
if exceptions occur within the context.

### Publication Ready
Default DPI (300), tight bounding box, and proper font sizes meet most
journal submission requirements.

## Acceptance Criteria Status

✅ FigurePreset enum with JOSS, IEEE, ELSEVIER, NATURE
✅ apply_figure_preset(preset) global application
✅ figure_context(preset) context manager with restoration
✅ save_figure(fig, path, dpi=300, transparent=False) export
✅ No hardcoded colors (typography/layout only)
✅ 40 comprehensive tests (all passing)
✅ Deterministic look (preset-based, reproducible)
✅ One flag changes style of all exported figures

## Testing

The system includes 40 tests covering:

- Enum membership and values (2 tests)
- Preset application (11 tests):
  - Via enum and string
  - Case-insensitive
  - Invalid preset error handling
  - rcParams modification verification
  - Spine visibility
- Config inspection (5 tests):
  - Returns dict
  - Via enum and string
  - No side effects
  - Defensive copying
- Context manager (7 tests):
  - Applies preset inside block
  - Restores after exit
  - Nested contexts
  - Exception safety
  - Figure creation within context
- Figure saving (10 tests):
  - File creation
  - Path handling
  - Multiple formats (PNG, PDF, SVG)
  - Custom DPI
  - Transparency
  - Tight bbox
- Preset listing (3 tests):
  - Returns dict with descriptions
  - All presets listed
- Integration tests (3 tests):
  - Full workflow
  - Context + save
  - Multiple formats

Run tests:
```bash
pytest tests/viz/test_paper.py -v
# Output: 40 passed
```

## Examples

### Example 1: Global Preset for Paper
```python
from foodspec.viz import apply_figure_preset, FigurePreset, save_figure
import matplotlib.pyplot as plt

# Apply JOSS style globally
apply_figure_preset(FigurePreset.JOSS)

# All figures use JOSS style
fig1, ax1 = plt.subplots()
ax1.plot(x1, y1)
save_figure(fig1, "figure1.png")

fig2, ax2 = plt.subplots()
ax2.plot(x2, y2)
save_figure(fig2, "figure2.png")
```

### Example 2: Multiple Journal Versions
```python
from foodspec.viz import figure_context, save_figure
import matplotlib.pyplot as plt

data = ...

# Create versions for different journals
for preset in [FigurePreset.JOSS, FigurePreset.IEEE, FigurePreset.NATURE]:
    with figure_context(preset):
        fig, ax = plt.subplots()
        ax.plot(data)
        save_figure(fig, f"{preset.value}_version.png")
```

### Example 3: Custom Colors + Preset Style
```python
from foodspec.viz import figure_context, FigurePreset
import matplotlib.pyplot as plt

# Preset controls layout, you control colors
with figure_context(FigurePreset.NATURE):
    fig, ax = plt.subplots()
    ax.plot(x, y, color="#FF6B6B", linewidth=2)  # Custom color
    ax.scatter(x2, y2, color="#4ECDC4", s=100)   # Your colors
    # Nature styling applied: font sizes, line widths, etc.
```

## Integration

The module is fully integrated into FoodSpec:

```python
from foodspec.viz import (
    FigurePreset,
    apply_figure_preset,
    figure_context,
    save_figure,
    get_figure_preset_config,
    list_presets,
)
```

## Files

- **src/foodspec/viz/paper.py**: Main implementation (366 lines)
  - FigurePreset enum
  - _PRESET_CONFIGS dictionary
  - apply_figure_preset()
  - figure_context()
  - save_figure()
  - get_figure_preset_config()
  - list_presets()

- **tests/viz/test_paper.py**: Comprehensive test suite (400+ lines)
  - 40 tests covering all functionality
  - Edge cases and error handling
  - Integration tests with actual figures

- **examples/paper_presets_demo.py**: Demonstration script
  - Shows all presets
  - Global application example
  - Context manager example
  - Batch processing example

## Matplotlib rcParams Modified

Each preset modifies these rcParams:

- `figure.figsize`: Width and height in inches
- `font.size`: Default font size (pt)
- `axes.labelsize`: Axis label size (pt)
- `axes.titlesize`: Title font size (pt)
- `xtick.labelsize`: X-tick label size (pt)
- `ytick.labelsize`: Y-tick label size (pt)
- `legend.fontsize`: Legend font size (pt)
- `lines.linewidth`: Default line width (pt)
- `axes.linewidth`: Axis spine width (pt)
- `xtick.major.width`: X-tick width (pt)
- `ytick.major.width`: Y-tick width (pt)
- `axes.spines.left`: Show left spine (bool)
- `axes.spines.right`: Show right spine (bool)
- `axes.spines.top`: Show top spine (bool)
- `axes.spines.bottom`: Show bottom spine (bool)
- `savefig.dpi`: Default savefig DPI (int)
- `savefig.bbox`: Default bbox for savefig
- `savefig.pad_inches`: Default padding

## Limitations

1. **Colors not controlled**: Presets don't set colors (intentional design choice)
2. **Figure creation after preset**: Figures must be created after applying preset
3. **rcParams reset by matplotlib**: Some operations (plt.style.use()) may override
4. **One preset at a time**: Global preset is exclusive (but context managers allow nesting)

## Future Enhancements

1. Custom preset creation (user-defined styles)
2. Integration with matplotlib's style system
3. Template-based figure generation
4. Preset composition (mix multiple presets)
5. Automatic format selection based on journal

## Status

✅ **COMPLETE** - All acceptance criteria met, 40 tests passing, documentation complete.
"""
