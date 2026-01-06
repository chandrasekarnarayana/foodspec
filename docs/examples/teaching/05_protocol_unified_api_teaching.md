# End-to-End Protocol (Teaching Walkthrough)

**Focus**: Chainable API • QC → Preprocess → Train → Export

## What you will learn
- Apply a full FoodSpec workflow with the unified (Phase 1) API
- Perform QC, preprocessing, model training, diagnostics, and export
- Keep an audit trail (parameters, metrics, diagnostics)

## Prerequisites
- Python 3.10+
- numpy, pandas, scikit-learn, foodspec

## Minimal runnable code
```python
import pandas as pd
from foodspec.core import FoodSpec
from foodspec.datasets import load_example_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

X, y = load_example_data("oil_classification")
fs = FoodSpec(task="classification", name="oil_auth_production")

# QC
qc = fs.quality_check(X=X, y=y, check_type="complete")
assert qc["health"] > 0.5

# Preprocess
fs.add_preprocessing_step("baseline_removal", method="polyfit", order=5)
fs.add_preprocessing_step("normalization", method="vector")
Xp = fs.preprocess(X, fit=True)

# Train + CV
model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
cv = cross_validate(model, Xp, y, cv=5, scoring="accuracy", return_train_score=True)
print({"cv_acc": cv["test_score"].mean(), "train_acc": cv["train_score"].mean()})
```

## Explain the outputs
- `health` > 0.5 ⇒ data quality acceptable
- `cv_acc` vs `train_acc` ⇒ generalization gap check
- The same preprocessing must be reused for new samples (fit=False when deploying)

## Full resources
- Full script: https://github.com/chandrasekarnarayana/foodspec/blob/main/examples/phase1_quickstart.py
- Teaching notebook (download/run): https://github.com/chandrasekarnarayana/foodspec/blob/main/examples/tutorials/05_protocol_unified_api_teaching.ipynb

## Run it yourself
```bash
python examples/phase1_quickstart.py
jupyter notebook examples/tutorials/05_protocol_unified_api_teaching.ipynb
```

## Related docs
- Protocols & YAML → ../user-guide/protocols_and_yaml.md
- End-to-end pipeline → ../workflows/end_to_end_pipeline.md
