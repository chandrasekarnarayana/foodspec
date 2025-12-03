import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression

rng = np.random.default_rng(0)
n_samples, n_features = 80, 50
wn = np.linspace(600, 1800, n_features)
true_loading = np.exp(-0.5 * ((wn - 1650) / 30) ** 2) + 0.5 * np.exp(-0.5 * ((wn - 1200) / 40) ** 2)
X = rng.normal(0, 0.05, size=(n_samples, n_features)) + rng.normal(1.0, 0.2, size=(n_samples, 1)) * true_loading
y = X @ true_loading + rng.normal(0, 0.1, size=n_samples)

pls = PLSRegression(n_components=2)
pls.fit(X, y)
loadings = pls.x_loadings_[:, 0]

plt.figure(figsize=(6, 3))
plt.plot(wn, loadings, label="PLS Loading PC1")
plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Loading")
plt.title("PLS Loadings Highlighting Influential Bands")
plt.legend()
plt.tight_layout()
plt.savefig("docs/assets/pls_loadings.png", dpi=200)
