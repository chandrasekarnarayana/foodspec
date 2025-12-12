# Developer Guide – Writing Plugins

FoodSpec supports plugins for protocols, vendor IO, harmonization strategies, and custom steps via Python entry points (`foodspect.plugins`).

## Quick start
1. Create a new package (see `examples/plugins/plugin_example_protocol` and `plugin_example_vendor`).
2. In `pyproject.toml`, add:
   ```toml
   [project.entry-points."foodspect.plugins"]
   your-plugin-name = "your_package.plugin:register"
   ```
3. Implement `register(registry)` to call helpers such as:
   - `registry.register_protocol(name, path_or_dict)`
   - `registry.register_vendor_loader(format_name, loader_fn)`
   - `registry.register_step(step_name, handler_fn)`
4. Install in editable mode:
   ```bash
   pip install -e .
   ```
5. Verify discovery:
   ```bash
   foodspec-plugin list
   ```

## What can a plugin add?
- Protocol templates (YAML/JSON descriptors).
- Vendor loaders for new instruments/formats.
- Harmonization or preprocessing strategies.
- Custom protocol steps.

## Testing plugins
- Use example plugins under `examples/plugins/` as templates.
- Add unit tests that ensure your plugin is discovered and assets load.
- For CLI verification, run `foodspec-plugin list` in a test env.

## Documentation
- Document purpose, expected inputs, and limitations in your plugin README.
- Cross-link from `docs/04-user-guide/registry_and_plugins.md` so users can find it.

## Uninstalling / disabling
- `pip uninstall your-plugin-name`.
- Re-run `foodspec-plugin list` to confirm it is no longer detected.
