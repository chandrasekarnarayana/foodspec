# Quickstart: Run a FoodSpec Protocol in 10 Minutes

## CLI
```bash
foodspec-run-protocol \
  --input data/oils.csv \
  --protocol examples/protocols/EdibleOil_Classification_v1.yml \
  --output-dir runs
```
Outputs: `runs/<protocol>_<input>/` containing `report.txt`, figures, tables, metadata.

## GUI (PyQt cockpit)
```bash
python scripts/foodspec_protocol_cockpit.py
```
1. Select a protocol.
2. Select a CSV/HDF5 (or load a project with multiple datasets).
3. Validate → Run. View status/progress, history, HSI tabs, and run folder link.

## Notes
- Protocols are YAML/JSON in `examples/protocols/`.
- Overrides (CLI): `--seed`, `--cv-folds`, `--normalization-mode`, `--baseline-method`.
- HDF5 stores instrument metadata, preprocessing history, provenance.
