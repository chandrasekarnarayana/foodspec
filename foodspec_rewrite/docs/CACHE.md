# Hash-Based Caching for Expensive Stages

P0-B implementation: Hash-based caching for preprocessing and feature extraction.

## Overview

The cache system automatically stores and retrieves expensive stage outputs (preprocessing, features) based on stable cache keys computed from:
- Data fingerprint (SHA256 of input data)
- Stage specification (preprocessing recipe/steps, feature strategy)
- Stage name (preprocess, features)
- Library version (for cache invalidation on upgrades)

## Architecture

- **CacheManager**: Core caching logic (get/put/clear)
- **Storage**: Compressed numpy arrays (.npz) + JSON metadata sidecar
- **Integration**: Wired into ExecutionEngine.run() before expensive stages
- **Manifest tracking**: Cache hits/misses recorded in RunManifest

## Usage

### Basic Example

```python
from foodspec.core import CacheManager, ExecutionEngine
from pathlib import Path

# Enable caching (default behavior)
engine = ExecutionEngine(cache_dir=Path(".cache"), enable_cache=True)

# First run: cache miss, computation happens
result1 = engine.run("protocol.yaml", outdir="runs/exp1", seed=0)
print(f"Cache misses: {result1.manifest.cache_misses}")  # ['preprocess', 'features']

# Second run with same inputs: cache hit, reuses stored arrays
result2 = engine.run("protocol.yaml", outdir="runs/exp2", seed=0)
print(f"Cache hits: {result2.manifest.cache_hits}")  # ['preprocess', 'features']
```

### Direct Cache Access

```python
from foodspec.core import CacheManager
import numpy as np

cache = CacheManager(cache_dir=Path(".cache"))

# Compute stable cache key
key = cache.compute_key(
    data_fingerprint="abc123...",  # From RunManifest.compute_data_fingerprint()
    stage_spec={"recipe": "raman_default"},
    stage_name="preprocess",
    library_version="2.0.0"
)

# Check cache
cached = cache.get(key)
if cached:
    spectra = cached.arrays["spectra"]
    print(f"Cache hit: {spectra.shape}")
else:
    # Expensive computation
    spectra = preprocess_spectra(...)
    
    # Store in cache
    cache.put(
        key=key,
        stage_name="preprocess",
        arrays={"spectra": spectra, "wavenumbers": wn},
        metadata={"n_samples": len(spectra)}
    )
```

### Disable Caching

```python
# Disable for benchmarking or debugging
engine = ExecutionEngine(enable_cache=False)
result = engine.run("protocol.yaml", outdir="runs/exp", seed=0)
```

### Clear Cache

```python
cache = CacheManager(cache_dir=Path(".cache"))
n_removed = cache.clear()
print(f"Cleared {n_removed} cache entries")

# Check cache size
size_mb = cache.size_bytes() / (1024 ** 2)
print(f"Cache size: {size_mb:.2f} MB")
```

## Cache Key Stability

Cache keys are deterministic for identical inputs:

```python
cache = CacheManager(Path(".cache"))

key1 = cache.compute_key("abc", {"recipe": "raman"}, "preprocess", "2.0.0")
key2 = cache.compute_key("abc", {"recipe": "raman"}, "preprocess", "2.0.0")
assert key1 == key2  # Stable

# Different inputs produce different keys (cache invalidation)
key3 = cache.compute_key("def", {"recipe": "raman"}, "preprocess", "2.0.0")
assert key3 != key1  # Different data

key4 = cache.compute_key("abc", {"recipe": "ir"}, "preprocess", "2.0.0")
assert key4 != key1  # Different recipe

key5 = cache.compute_key("abc", {"recipe": "raman"}, "preprocess", "2.1.0")
assert key5 != key1  # Different version
```

## Manifest Tracking

Cache hits/misses are recorded in the manifest:

```python
result = engine.run("protocol.yaml", outdir="runs/exp", seed=0)

# Inspect manifest
manifest = result.manifest
print(f"Cache hits: {manifest.cache_hits}")      # List of stages with cache hit
print(f"Cache misses: {manifest.cache_misses}")  # List of stages with cache miss

# Manifest persisted to manifest.json
import json
manifest_json = json.load(open("runs/exp/manifest.json"))
print(manifest_json["cache_hits"])
print(manifest_json["cache_misses"])
```

## Testing

Comprehensive test coverage in `tests/test_cache.py`:

- ✅ Cache key stability (deterministic hashing)
- ✅ Cache hit returns identical arrays (bit-for-bit)
- ✅ Cache miss returns None
- ✅ Cache invalidation on spec changes
- ✅ Manifest tracking of hits/misses
- ✅ Disabled cache behaves as no-op
- ✅ Corrupted cache entries degrade gracefully

Run tests:
```bash
pytest tests/test_cache.py -v
```

## Implementation Details

### Storage Format

- **Arrays**: `.npz` compressed numpy archives (savez_compressed)
- **Metadata**: `.json` sidecar files with stage info
- **Naming**: `{cache_key}.npz` and `{cache_key}.json`

### Cache Key Computation

```python
payload = {
    "data_fingerprint": "<SHA256 of input data>",
    "stage_spec": {"recipe": "...", "steps": [...]},
    "stage_name": "preprocess",
    "library_version": "2.0.0"
}
cache_key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
```

### Orchestrator Integration

Cache checks happen **before** stage execution:

1. Compute cache key from data fingerprint + stage spec
2. Check `cache.get(key)`
3. If hit: reuse cached arrays, log cache hit
4. If miss: execute stage, store with `cache.put(key, ...)`, log cache miss
5. Record hits/misses in RunManifest

## Future Enhancements

- [ ] TTL-based cache expiration
- [ ] LRU eviction policy when cache exceeds size limit
- [ ] Distributed cache support (Redis/S3)
- [ ] Cache warming from prior runs
- [ ] Partial cache hits (e.g., reuse preprocess but recompute features)
- [ ] Cache statistics dashboard

## Files Modified

- `foodspec/core/cache.py` (new): CacheManager implementation
- `foodspec/core/manifest.py`: Added cache_hits/cache_misses fields
- `foodspec/core/orchestrator.py`: Integrated cache checks before stages
- `foodspec/core/__init__.py`: Exported CacheManager, CacheEntry
- `tests/test_cache.py` (new): Comprehensive test suite

## Verification

All tests pass (16/16):
```bash
$ pytest tests/test_cache.py -v
======================= 16 passed in 0.69s ========================
```
