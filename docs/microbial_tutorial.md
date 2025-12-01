# Microbial contamination workflow (template)

This workflow reuses the oil-style template for microbial detection (e.g., positive/negative contamination labels).

CLI example:
```bash
foodspec domains \
  libraries/microbial_demo.h5 \
  --type microbial \
  --label-column status \
  --classifier-name svm_rbf \
  --output-dir runs/microbial_demo
```

Python API:
```python
from foodspec.apps.microbial import run_microbial_detection_workflow
result = run_microbial_detection_workflow(fs, label_column="status")
```

Outputs include CV metrics, confusion matrix, and summary reports. Adjust labels and preprocessing as needed for your organism/assay.
