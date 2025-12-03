import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from foodspec.chemometrics.models import make_mlp_regressor
from foodspec.metrics import compute_regression_metrics

rng = np.random.default_rng(0)
n_samples, n_features = 150, 20
wn = np.linspace(600, 1800, n_features)
true_w = np.exp(-0.5 * ((wn - 1500) / 40) ** 2)
X = rng.normal(0, 0.1, size=(n_samples, n_features)) + rng.normal(1.0, 0.2, size=(n_samples, 1)) * true_w
y = X @ true_w + rng.normal(0, 0.05, size=n_samples)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

mlp = make_mlp_regressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=0)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
metrics = compute_regression_metrics(y_test, y_pred)
print(metrics)

plt.figure(figsize=(4, 4))
plt.scatter(y_test, y_pred, alpha=0.7)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, "k--")
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("DL MLP Regression Calibration")
plt.tight_layout()
plt.savefig("docs/assets/dl_mlp_regression_calibration.png", dpi=200)
