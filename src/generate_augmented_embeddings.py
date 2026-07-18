import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


# ------------------------------------
# Load Sentence Transformer
# ------------------------------------
print("Loading Sentence Transformer...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# ------------------------------------
# Load augmented dataset
# ------------------------------------
train_df = pd.read_csv("data/processed/train_augmented.csv")

texts = train_df["text"].tolist()
labels = train_df["label"].tolist()


# ------------------------------------
# Generate embeddings
# ------------------------------------
print("Generating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True
)


# ------------------------------------
# Save embeddings
# ------------------------------------
np.save(
    "data/processed/train_augmented_embeddings.npy",
    embeddings
)

np.save(
    "data/processed/train_augmented_labels.npy",
    np.array(labels)
)


print("\nEmbeddings generated successfully!")

print(f"Number of samples : {len(texts)}")
print(f"Embedding shape   : {embeddings.shape}")