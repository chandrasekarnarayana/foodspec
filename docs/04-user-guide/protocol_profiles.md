# User Guide – Protocol Profiles

Protocol profiles bundle sensible defaults (preprocessing, validation, figure selection) for common scenarios. Use them as presets in GUI and CLI.

## Profiles (examples)
- **oil_basic**: edible oil discrimination; reference-peak normalization; batch-aware validation; balanced accuracy + confusion matrix + top ratios + minimal panel.
- **oil_heating**: thermal stability; trend analysis (slopes/Spearman with FDR); stability ranking.
- **oil_vs_chips**: matrix effects; divergence metrics/effect sizes; oil-vs-chips plots.
- **hsi_segment_roi**: HSI segmentation (k-means) → ROI averaging → RQ; includes label maps and ROI spectra.

## CLI usage with profiles
Use `--protocol` to point to a profile YAML or a named protocol installed via plugins.
```bash
# Oil discrimination profile
foodspec-run-protocol --input examples/data/oils.csv --protocol examples/protocols/oil_basic.yaml --auto --report-level standard

# Oil vs chips profile (harmonized multi-input)
foodspec-run-protocol \
  --input examples/data/oils.csv \
  --input examples/data/chips.csv \
  --protocol examples/protocols/oil_vs_chips.yaml \
  --auto --report-level full
```
- Figure selection profiles for publish: use `foodspec-publish --profile {quicklook,qa,standard,publication}`; quicklook = fewer figures, publication = richer set (or use `--include-all-figures`).

## GUI usage
- In the protocol selector, choose the profile matching your scenario (oil discrimination, thermal stability, oil vs chips, HSI).
- Click **Validate** then **Run**; the GUI executes the full pipeline and shows a summary with links to the run/publish folders.

## Report levels (CLI `--report-level`)
- `summary`: fewer figures (e.g., fig_limit ≈ 4)
- `standard` (default): typical figure set (e.g., fig_limit ≈ 8)
- `full`: include all figures (`--include-all-figures`)

See also: [automation.md](automation.md), [cli_guide.md](cli_guide.md), and [gui_cockpit_guide.md](gui_cockpit_guide.md).
