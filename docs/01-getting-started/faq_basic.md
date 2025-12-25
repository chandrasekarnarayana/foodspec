# FAQ (Basic)

<!-- CONTEXT BLOCK (mandatory) -->
**Who needs this?** Beginners with common questions; users encountering issues during first use.
**What problem does this solve?** Quick answers to frequent questions without reading full documentation.
**When to use this?** When stuck on installation, data format, or basic workflow questions.
**Why it matters?** Saves time by addressing common confusion points upfront.
**Time to complete:** 5-10 minutes.
**Prerequisites:** None

---

## Common Questions

**Do I need ML experience to use FoodSpec?**  
No. Protocols are predefined recipes. You pick one (e.g., oil discrimination), map your columns, and run. Defaults include sensible validation and minimal panels. See [first-steps_cli.md](first-steps_cli.md).

**What if my data is from a different Raman/FTIR instrument?**  
FoodSpec ingests CSV/HDF5 and has vendor loader stubs. If binary parsing is incomplete, export to CSV or HDF5. Plugins can add loaders; see [registry_and_plugins.md](../04-user-guide/registry_and_plugins.md).

**What is a protocol, in simple terms?**  
A YAML/JSON recipe defining preprocessing, harmonization, QC, HSI (optional), RQ analysis, outputs, and validation strategy. It makes runs repeatable. See [protocols_and_yaml.md](../04-user-guide/protocols_and_yaml.md).

**Should I start with CLI?**  
Yes. Use the CLI for both first runs and automation. Start with [first-steps_cli.md](first-steps_cli.md).

**Can I use FoodSpec for matrices beyond oils/chips?**  
Yes. Protocols focus on oils/chips, but any Raman/FTIR data with appropriate peaks/ratios can be processed. Adjust expected columns/peak definitions as needed. See [oil_vs_chips_matrix_effects.md](../02-tutorials/oil_vs_chips_matrix_effects.md) for multi-matrix ideas.

**Where do my results go?**  
Each run creates a timestamped folder with `report.txt/html`, `figures/`, `tables/`, `metadata.json`, `index.json`, and optionally `models/`. The CLI prints the path. See [first-steps_cli.md](first-steps_cli.md).

**How do I check my installation?**  
Run `foodspec-run-protocol --check-env`. See [installation.md](installation.md).

**Can I extend FoodSpec?**  
Yes. Add protocols, vendor loaders, or harmonization strategies via plugins. See [writing_plugins.md](../06-developer-guide/writing_plugins.md).
