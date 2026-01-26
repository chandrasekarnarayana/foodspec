# Compliance Suite

FoodSpec includes basic compliance checks for ISO 17025 and FDA 21 CFR Part 11.
These checks validate the presence of required run artifacts and audit trails.

Example

```python
from pathlib import Path
from foodspec.compliance import check_iso_17025, check_cfr_part11

run_dir = Path("runs/my_run")
iso = check_iso_17025(run_dir)
cfr = check_cfr_part11(run_dir)
print(iso.score, cfr.score)
```

Scope

- ISO 17025: traceability of inputs, metrics, QC artifacts
- 21 CFR Part 11: electronic records, audit trail, model and dataset cards
