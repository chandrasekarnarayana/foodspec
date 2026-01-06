# Oil Authentication (Teaching Walkthrough)

**Focus**: Supervised classification • Confusion matrix • PCA

## What you will learn
- Load Raman spectra and labels
- Train and cross-validate a classifier
- Read a confusion matrix and PCA scatter plot

## Prerequisites
- Python 3.10+
- numpy, pandas, matplotlib, scikit-learn (installed with FoodSpec)

## Minimal runnable code
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

# Load data
url = "https://github.com/chandrasekarnarayana/foodspec/raw/main/examples/data/oil_synthetic.csv"
df = pd.read_csv(url, index_col=0)
X = df.drop("OilType", axis=1).values
y = df["OilType"].values

# Train + CV metrics
clf = RandomForestClassifier(n_estimators=50, random_state=42)
scores = cross_validate(
    clf, X, y, cv=5,
    scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
)
print({k: scores[f"test_{k}"].mean() for k in ["accuracy","precision_macro","recall_macro","f1_macro"]})
```

## Explain the outputs
- `accuracy/precision/recall/f1` close to 1.0 ⇒ excellent separation
- Use the confusion matrix to see which oils get confused
- PCA plot shows spectral clusters by oil type

## Full resources
- Full script: https://github.com/chandrasekarnarayana/foodspec/blob/main/examples/oil_authentication_quickstart.py
- Teaching notebook (download/run): https://github.com/chandrasekarnarayana/foodspec/blob/main/examples/tutorials/01_oil_authentication_teaching.ipynb
- Example figures: https://github.com/chandrasekarnarayana/foodspec/raw/main/outputs/oil_auth_confusion.png

## Run it yourself
```bash
python examples/oil_authentication_quickstart.py
jupyter notebook examples/tutorials/01_oil_authentication_teaching.ipynb
```

## Related docs
- Methods: classification basics → ../methods/chemometrics/classification_regression.md
- Workflow: oil authentication → ../workflows/authentication/oil_authentication.md
