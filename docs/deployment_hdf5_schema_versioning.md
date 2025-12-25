# HDF5 Schema Versioning

## Overview
FoodSpec HDF5 files (`SpectralDataset.to_hdf5()`) embed a schema version to track compatibility and prevent silent data corruption across FoodSpec releases.

## Schema Version
Each HDF5 file stores:
- **`foodspec_hdf5_schema_version`**: e.g., `"1.1"` (major.minor format)

Current supported version: **1.1**

## Compatibility Checks

### Version Validation on Load
```python
from foodspec import SpectralDataset

# Automatic version check; raises ValueError if incompatible
ds = SpectralDataset.from_hdf5("dataset.h5")
```

### Major Version Mismatch
If the file's major version differs from the current schema major version (e.g., file built with `2.0`, loading with `1.1`), `from_hdf5()` raises `ValueError`.

**Solution:** Re-save the dataset with the current FoodSpec version.

### Minor Version Mismatch
If the file's minor version is newer than the current schema (e.g., file built with `1.2`, loading with `1.1`), `from_hdf5()` raises `ValueError` unless `allow_future=True`.

**Solution:** Either re-save with current FoodSpec, or use `allow_future=True` for testing.

## Override Checks (Emergency Only)

```python
# Bypass all version validation (NOT recommended for production)
ds = SpectralDataset.from_hdf5("future_dataset.h5", allow_future=True)
```

## Backward Compatibility Strategy

- **Legacy files** (v1.0) are auto-detected and loaded into the v1.1 structure with minimal translation.
- **Migration path:** Save legacy files with current FoodSpec to update schema version.

## Migration Example

```python
from foodspec import SpectralDataset

# Load old file
ds_old = SpectralDataset.from_hdf5("legacy_dataset.h5", allow_future=True)

# Re-save with current schema version
ds_old.save_hdf5("legacy_dataset_v1_1.h5")
```

## Best Practices

1. **Regular Export:** If upgrading FoodSpec major versions, re-export HDF5 files.
   ```bash
   python -c "
   from foodspec import SpectralDataset
   ds = SpectralDataset.from_hdf5('data.h5', allow_future=True)
   ds.save_hdf5('data_updated.h5')
   "
   ```

2. **Version Control:** Track the schema version in your metadata.
   ```python
   import h5py
   with h5py.File("dataset.h5", "r") as f:
       version = f.attrs.get("foodspec_hdf5_schema_version")
       print(f"Schema version: {version}")
   ```

3. **Automated Pipelines:** Test backward-compatible reads during CI.
   ```bash
   pytest -k "test_hdf5" --tb=short
   ```

## See Also
- [SpectralDataset & HDF5 I/O](../vendor_io.md)
- [Harmonization & Multi-Instrument Workflows](../hsi_and_harmonization.md)
