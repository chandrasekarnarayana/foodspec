"""Generate a small confusion matrix example figure without extra deps.
Saves docs/assets/confusion_matrix_example.png.
"""
import matplotlib.pyplot as plt
import numpy as np

cm = np.array([[18, 2, 0], [1, 14, 3], [0, 2, 20]])
labels = ["Olive", "Sunflower", "Canola"]

fig, ax = plt.subplots(figsize=(4, 3))
im = ax.imshow(cm, cmap="Blues")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

ax.set_xticks(range(len(labels)), labels=labels, rotation=30, ha="right")
ax.set_yticks(range(len(labels)), labels=labels)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion matrix (synthetic example)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig("docs/assets/confusion_matrix_example.png", dpi=150)
plt.close(fig)
