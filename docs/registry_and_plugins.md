## Registry and Plugins

### Registry
- The `FeatureModelRegistry` (`src/foodspec/registry.py`) records runs and models:
  - dataset hash or input files
  - protocol name/version
  - preprocessing config
  - validation strategy
  - features used
  - model path/type and metrics
  - provenance (timestamp/user/tool version)
- CLI:
  - `foodspec-registry --registry registry.json --query-protocol EdibleOil_Classification_v1`
  - `foodspec-registry --registry registry.json --query-feature I_1742`
- Python:
```python
from foodspec.registry import FeatureModelRegistry
reg = FeatureModelRegistry(Path("~/.foodspec_registry.json").expanduser())
entries = reg.query_by_protocol("EdibleOil_Classification_v1")
print(entries)

# Query models by protocol and feature
models = [e for e in entries if e.model_id]
feat_hits = reg.query_by_feature("I_1742")
print(models, feat_hits)
```
- Auto-registration: set `FOODSPEC_REGISTRY` when saving models to log them automatically.

### Plugins
- Discovery via entry points (`foodspec.plugins`). Plugin manager loads protocol templates, vendor loaders, harmonization strategies.
- CLI:
  - `foodspec-plugin list`
  - `foodspec-plugin install plugin_example_protocol`
- Example plugins:
  - `examples/plugins/plugin_example_protocol`: registers a simple protocol template.
  - `examples/plugins/plugin_example_vendor`: adds a dummy vendor loader (`*.toy`).
- Programmatic discovery:
```python
from foodspec.plugin import PluginManager
pm = PluginManager(); pm.discover()
print(pm.protocols, pm.vendor_loaders)
```
