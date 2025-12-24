# User Guide – Registry & Plugins

Why it matters: provenance (registry) makes analyses auditable; plugins let you extend protocols/vendor loaders without touching core code.

## Registry overview
- The registry (SQLite/JSON index) stores: run_id, protocol name/version, dataset hash/file list, preprocessing summary, validation strategy, model paths/types/metrics, provenance (timestamp, tool version).
- Registry entries are created during runs (if registry is enabled) and when models are frozen.
- Location: defaults to a user-level path (e.g., `~/.foodspect_registry.db`), configurable via env/CLI.

## Registry CLI usage
```bash
foodspec-registry list
foodspec-registry query --protocol oil_basic
foodspec-registry query --model-path path/to/frozen_model.pkl
```
Outputs include model paths and feature definitions where available. See `index.json`/`metadata.json` in run folders for cross-reference. Registry entries are written automatically at run time when enabled.

## Plugin discovery
- FoodSpec discovers plugins via entry points (`foodspect.plugins`).
- Plugins can add: protocol templates, vendor loaders, harmonization strategies, custom steps.
- Example plugins live under `examples/plugins/`.

## Plugin CLI usage
```bash
foodspec-plugin list
foodspec-plugin install my-plugin   # if distributed
foodspec-plugin remove my-plugin
```
After installation, plugin protocols/loaders appear in CLI listings and can be referenced by name.

## How plugin protocols show up
- In the `foodspec-run-protocol` CLI, discovered plugin protocols are listed by name. Provide descriptive names/versions in your plugin metadata.

## Writing your own plugin
- See Developer Guide → `06-developer-guide/writing_plugins.md` and the example packages under `examples/plugins/`.
- Minimal steps: create a package, register entry points, implement `register(registry)` to add protocols/loaders/steps.

## Mini-workflow
1) Run a protocol; confirm `metadata.json` and `index.json` were written, and registry (if enabled) captured the run/model.  
2) Query with `foodspec-registry list` to see the new entry.  
3) Install a plugin example:  
   ```bash
   pip install -e examples/plugins/plugin_example_protocol
   foodspec-plugin list
   ```  
   The plugin protocol should appear in CLI listings; run it like any other protocol.

Cross-links: [cookbook_registry_reporting.md](../03-cookbook/cookbook_registry_reporting.md), [writing_plugins.md](../06-developer-guide/writing_plugins.md).
