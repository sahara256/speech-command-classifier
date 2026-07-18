import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# -----------------------------
# Load Embeddings
# -----------------------------
embeddings = np.load("embeddings/train_embeddings.npy")

labels = np.load("embeddings/train_labels.npy")
# -----------------------------
# Load Label Encoder
# -----------------------------
label_encoder = joblib.load("models/label_encoder.pkl")

label_names = label_encoder.inverse_transform(labels)
# -----------------------------
# Reduce Dimensions
# -----------------------------
pca = PCA(n_components=2)

embeddings_2d = pca.fit_transform(embeddings)
# -----------------------------
# Plot Embeddings
# -----------------------------
plt.figure(figsize=(10, 8))

unique_labels = np.unique(label_names)

for label in unique_labels:

    indices = label_names == label

    plt.scatter(
        embeddings_2d[indices, 0],
        embeddings_2d[indices, 1],
        label=label,
        alpha=0.7
    )

plt.title("Sentence Embeddings Visualization (PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()