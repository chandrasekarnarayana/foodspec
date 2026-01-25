# Metadata Schema Enforcement

FoodSpec v2 enforces mandatory metadata schemas to ensure that validation strategies (LOBO, LOSO, Group K-Fold) and QC grouping have the required metadata columns present in the dataset.

## Overview

When you configure a protocol with:
- **Grouped validation** (e.g., `group_kfold`, `leave_one_group_out`) requiring `validation.group_key`
- **QC grouping** requiring `qc.group_by`

FoodSpec now validates that the required metadata columns are present in your CSV data file **before** attempting to execute the pipeline. This provides early, actionable error messages instead of cryptic failures deep in the execution flow.

## Key Components

### 1. `DataSpec.required_metadata_keys`

Explicitly declare which metadata columns your protocol requires:

```python
from foodspec.core.protocol import DataSpec

spec = DataSpec(
    input="data.csv",
    modality="raman",
    label="target",
    required_metadata_keys=["batch", "instrument"],  # Explicitly required
)
```

**Default:** Empty list `[]` (no validation)

### 2. `QCSpec.group_by`

Specify metadata column for QC grouping (e.g., by instrument, batch, operator):

```python
from foodspec.core.protocol import QCSpec

qc = QCSpec(
    group_by="instrument",  # QC metrics computed per instrument
)
```

This automatically infers `"instrument"` as a required metadata key.

### 3. `ProtocolV2.infer_required_metadata_keys()`

Helper method that automatically infers required metadata keys from your protocol configuration:

```python
protocol = ProtocolV2(
    data=DataSpec(input="data.csv", modality="raman", label="target"),
    task=TaskSpec(name="lobo", objective="classification"),
    validation=ValidationSpec(scheme="group_kfold", group_key="batch"),
    qc=QCSpec(group_by="instrument"),
)

required = protocol.infer_required_metadata_keys()
# Returns: ["batch", "instrument"]

# Apply inferred keys
protocol.data.required_metadata_keys = required
```

**Inference rules:**
- `validation.group_key` → required metadata key
- `qc.group_by` → required metadata key
- Duplicates are automatically removed

## Validation Points

### At Load Time (`load_csv_spectra`)

When you load a CSV file, FoodSpec validates required metadata columns:

```python
from foodspec.io.readers import load_csv_spectra

spec = DataSpec(
    input="data.csv",
    modality="raman",
    label="target",
    required_metadata_keys=["batch", "instrument"],
)

# Raises ValueError if "batch" or "instrument" missing
ds = load_csv_spectra("data.csv", spec)
```

**Error format (actionable):**
```
ValueError: Missing required metadata columns: instrument.
Required for validation/QC configuration.
Available columns: batch, replicate, sample_id, target, 400, 500, 600
```

### At Runtime (`SpectraSet.validate_required_metadata`)

You can also manually validate a `SpectraSet`:

```python
from foodspec.core.data import SpectraSet

ds = SpectraSet(X=..., x=..., metadata=...)
ds.validate_required_metadata(["batch", "instrument"])
```

**Error format:**
```
ValueError: Missing required metadata keys: instrument.
Available: batch, replicate, sample_id
```

## Usage Examples

### Example 1: Leave-One-Batch-Out (LOBO)

```python
from foodspec.core.protocol import ProtocolV2, DataSpec, TaskSpec, ValidationSpec

protocol = ProtocolV2(
    data=DataSpec(
        input="oils.csv",
        modality="raman",
        label="class",
        required_metadata_keys=["batch"],  # Explicitly required
    ),
    task=TaskSpec(name="lobo", objective="classification"),
    validation=ValidationSpec(
        scheme="group_kfold",
        group_key="batch",  # Must be present in metadata
    ),
)
```

**If `oils.csv` is missing the `batch` column:**
```
ValueError: Missing required metadata columns: batch.
Required for validation/QC configuration.
Available columns: class, instrument, sample_id, 400, 500, 600
```

### Example 2: Automatic Inference

```python
# Create protocol without explicitly specifying required keys
protocol = ProtocolV2(
    data=DataSpec(input="data.csv", modality="raman", label="target"),
    task=TaskSpec(name="test", objective="classification"),
    validation=ValidationSpec(scheme="group_kfold", group_key="batch"),
    qc=QCSpec(group_by="instrument"),
)

# Infer required keys from validation + QC config
required = protocol.infer_required_metadata_keys()
print(required)  # ["batch", "instrument"]

# Apply to DataSpec
protocol.data.required_metadata_keys = required

# Now load with validation
ds = load_csv_spectra("data.csv", protocol.data)
```

### Example 3: Metadata Mapping

If your CSV columns have different names, use `metadata_map` to rename them:

```python
spec = DataSpec(
    input="data.csv",
    modality="raman",
    label="exp_target",
    metadata_map={
        "batch": "exp_batch",        # Rename exp_batch → batch
        "instrument": "exp_instrument",  # Rename exp_instrument → instrument
        "label": "exp_target",
    },
    required_metadata_keys=["batch", "instrument"],  # Logical names after mapping
)

# Validation checks for "exp_batch" and "exp_instrument" in CSV
# After loading, metadata columns are "batch" and "instrument"
ds = load_csv_spectra("data.csv", spec)
```

## Common Scenarios

### LOBO Validation
**Requires:** `batch` column
```python
validation = ValidationSpec(scheme="group_kfold", group_key="batch")
```

### LOSO Validation
**Requires:** `sample` or `subject` column
```python
validation = ValidationSpec(scheme="leave_one_group_out", group_key="sample")
```

### QC per Instrument
**Requires:** `instrument` column
```python
qc = QCSpec(group_by="instrument")
```

### Combined Requirements
**Requires:** Both `batch` (for validation) and `instrument` (for QC)
```python
protocol = ProtocolV2(
    data=DataSpec(input="data.csv", modality="raman", label="target"),
    validation=ValidationSpec(scheme="group_kfold", group_key="batch"),
    qc=QCSpec(group_by="instrument"),
)
required = protocol.infer_required_metadata_keys()  # ["batch", "instrument"]
```

## Error Message Format

All validation errors follow this actionable format:

```
ValueError: Missing required metadata columns: <missing_keys>.
Required for validation/QC configuration.
Available columns: <actual_columns>
```

This helps you quickly identify:
1. **What's missing:** The exact column names required
2. **What's available:** What FoodSpec found in your CSV
3. **How to fix:** Rename or add the missing columns

## Implementation Details

### Code Locations

- **Schema definition:** `foodspec/core/protocol.py`
  - `DataSpec.required_metadata_keys: List[str]`
  - `QCSpec.group_by: Optional[str]`
  - `ProtocolV2.infer_required_metadata_keys() -> List[str]`

- **Reader enforcement:** `foodspec/io/readers.py`
  - `load_csv_spectra()` validates required keys before parsing

- **Runtime validation:** `foodspec/core/data.py`
  - `SpectraSet.validate_required_metadata(required_keys: List[str])`

### Test Coverage

Comprehensive tests in `tests/test_metadata_schema.py`:
- DataSpec field validation (2 tests)
- Protocol inference logic (5 tests)
- Reader enforcement with actionable errors (4 tests)
- SpectraSet validation (4 tests)
- Workflow integration scenarios (3 tests)

**Total:** 18 tests covering all validation paths

## Design Rationale

### Why enforce metadata schemas?

1. **Early failure:** Catch missing columns at load time, not during CV splits
2. **Actionable errors:** Show what's missing and what's available
3. **Explicit requirements:** Make protocol dependencies clear
4. **Prevent silent failures:** Grouped validation with missing group keys would fail cryptically

### Why inference helper?

- Reduces boilerplate: no need to manually list required keys
- Single source of truth: validation config determines requirements
- Extensible: future protocol features automatically infer requirements

### Why validate at multiple points?

- **Load time** (`load_csv_spectra`): Immediate feedback when reading files
- **Runtime** (`SpectraSet.validate_required_metadata`): Catch issues in programmatic data construction

## Future Enhancements

Potential extensions (not yet implemented):

1. **Type validation:** Ensure metadata columns have expected dtypes (e.g., `batch` is categorical)
2. **Cardinality checks:** Warn if too few unique values (e.g., only 2 batches for LOBO)
3. **Missing value detection:** Fail if required columns contain NaNs
4. **Schema versioning:** Track metadata schema changes across protocol versions

## See Also

- [Core Concepts: Data Structures](../concepts/data_structures.md)
- [Validation Strategies](../methods/validation.md)
- [Quality Control](../methods/qc.md)
- [Protocol V2 Schema](../reference/protocol_schema.md)
