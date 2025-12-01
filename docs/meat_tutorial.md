# Meat quality workflow (template)

This template mirrors the oil authentication workflow but targets meat quality/freshness labels.

Steps:
1. Prepare an HDF5 library with spectra and metadata column (e.g., `meat_label` such as fresh/spoiled).
2. Run via CLI:
```bash
foodspec domains \
  libraries/meat_demo.h5 \
  --type meat \
  --label-column meat_label \
  --classifier-name rf \
  --output-dir runs/meat_demo
```
3. Outputs: CV metrics CSV, confusion_matrix.png, summary/report files.

Python API:
```python
from foodspec.apps.meat import run_meat_authentication_workflow
result = run_meat_authentication_workflow(fs, label_column="meat_label")
```
Use this as a starting point and adapt features/labels to your meat quality study.
