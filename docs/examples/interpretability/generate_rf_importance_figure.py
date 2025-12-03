import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

rng = np.random.default_rng(0)
n_samples, n_features = 60, 10
X = rng.normal(0, 1, size=(n_samples, n_features))
coefs = np.array([2.0, 1.5, 1.0] + [0.1] * (n_features - 3))
logits = X @ coefs + rng.normal(0, 0.5, size=n_samples)
y = (logits > np.median(logits)).astype(int)

rf = RandomForestClassifier(n_estimators=200, random_state=0)
rf.fit(X, y)
importances = rf.feature_importances_
idx = np.argsort(importances)[::-1]

plt.figure(figsize=(5, 3))
plt.bar(range(5), importances[idx[:5]])
plt.xticks(range(5), [f"f{idx[i]}" for i in range(5)])
plt.ylabel("Importance")
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("docs/assets/rf_feature_importance.png", dpi=200)
