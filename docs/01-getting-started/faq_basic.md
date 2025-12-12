# FAQ (Basic)

- **Columns required?** See quickstart_protocol.md and protocol YAML `expected_columns`.
- **Missing dependency?** Run `--check-env` on CLIs to verify.
- **Which preset to use?** Start with “Typical edible oil discrimination” in the GUI.
# FAQ (Basic)

**Do I need ML experience to use FoodSpec?**  
No. Protocols are predefined recipes. You pick one (e.g., oil discrimination), map your columns, and run. Defaults include sensible validation and minimal panels. See [first-steps_gui.md](first-steps_gui.md) or [first-steps_cli.md](first-steps_cli.md).

**What if my data is from a different Raman/FTIR instrument?**  
FoodSpec ingests CSV/HDF5 and has vendor loader stubs. If binary parsing is incomplete, export to CSV or HDF5. Plugins can add loaders; see [registry_and_plugins.md](../04-user-guide/registry_and_plugins.md).

**What is a protocol, in simple terms?**  
A YAML/JSON recipe defining preprocessing, harmonization, QC, HSI (optional), RQ analysis, outputs, and validation strategy. It makes runs repeatable. See [protocols_and_yaml.md](../04-user-guide/protocols_and_yaml.md).

**Should I start with GUI or CLI?**  
GUI gives visual feedback and easier mapping; CLI is best for automation. Start with GUI ([first-steps_gui.md](first-steps_gui.md)), then mirror via CLI ([first-steps_cli.md](first-steps_cli.md)).

**Can I use FoodSpec for matrices beyond oils/chips?**  
Yes. Protocols focus on oils/chips, but any Raman/FTIR data with appropriate peaks/ratios can be processed. Adjust expected columns/peak definitions as needed. See [oil_vs_chips_matrix_effects.md](../02-tutorials/oil_vs_chips_matrix_effects.md) for multi-matrix ideas.

**Where do my results go?**  
Each run creates a timestamped folder with `report.txt/html`, `figures/`, `tables/`, `metadata.json`, `index.json`, and optionally `models/`. GUI shows a link; CLI prints the path. See [first-steps_cli.md](first-steps_cli.md).

**How do I check my installation?**  
Run `foodspec-run-protocol --check-env`. Install extras (`foodspec[gui]`, `foodspec[web]`) if you need those modes. See [installation.md](installation.md).

**Can I extend FoodSpec?**  
Yes. Add protocols, vendor loaders, or harmonization strategies via plugins. See [writing_plugins.md](../06-developer-guide/writing_plugins.md).
