# Developer Guide – Extending Protocols & Steps

FoodSpec protocols are YAML/JSON recipes interpreted by the `ProtocolRunner`. You can add new steps or extend existing ones without modifying core logic.

## Step registry overview
- Step types are registered in `src/foodspec/protocol_engine.py`.
- Built-ins include: `preprocess`, `harmonize`, `qc_checks`, `hsi_segment`, `hsi_roi_to_1d`, `rq_analysis`, `output`.
- Each step receives a context dict (dataframes/datasets, config, registry handle, cancel flag) and returns updated context.

## Adding a new step type
1. Implement the handler in a new module (e.g., `src/foodspec/steps/your_step.py`). It should accept `(context, params)` and return the updated context; check the cancel flag if long-running.
2. Register it in the step registry in `protocol_engine.py`.
3. Add schema/validation entries so YAML protocols can declare it (type name + expected params).
4. Update docs and tests with a minimal protocol exercising the new step.

## Making it visible in CLI/GUI
- Protocols that include your step will work automatically in CLI.
- In GUI, ensure the protocol is discoverable (built-in or via plugin). If your step needs UI controls, add them to the cockpit/wizard panels; otherwise, defaults from the protocol are used.

## Extending protocols
- Protocol YAMLs live under `examples/protocols/`.
- Add new fields for your step under `steps:` with `type: your_step` and any params.
- Include `expected_columns` or `expected_metadata` if your step needs them.

## Plugins
- Steps and protocols can also be shipped as plugins (see `writing_plugins.md`). Plugins register entry points so they are discovered without editing core code.

## Tests and docs
- Add a small synthetic test under `tests/` that executes your step via `ProtocolRunner`.
- Document the new step in `docs/04-user-guide/protocols_and_yaml.md` and, if advanced, in `05-advanced-topics/architecture.md`.
