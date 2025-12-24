# Quickstart: Run a FoodSpec Protocol in 10 Minutes

## CLI
```bash
foodspec-run-protocol \
  --input data/oils.csv \
  --protocol examples/protocols/EdibleOil_Classification_v1.yml \
  --output-dir runs
```
Outputs: `runs/<protocol>_<input>/` containing `report.txt`, figures, tables, metadata.

## Tips
- Validate before running; fix blocking errors first.
- Keep protocols versioned; record seeds and CV design in outputs.

## Notes
- Protocols are YAML/JSON in `examples/protocols/`.
- Common overrides: `--seed`, `--cv-folds`, `--normalization-mode`, `--baseline-method`.
- HDF5 stores instrument metadata, preprocessing history, provenance.
