import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from foodspec.viz.heating import plot_ratio_vs_time


def test_plot_ratio_vs_time():
    time = np.array([0, 1, 2, 3])
    ratio = np.array([1.0, 1.1, 1.2, 1.3])
    model = LinearRegression().fit(time.reshape(-1, 1), ratio)
    ax = plot_ratio_vs_time(time, ratio, model=model)
    assert hasattr(ax, "plot")
    plt.close(ax.figure)

