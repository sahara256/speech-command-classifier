import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# -------------------------------------------------
# Create embeddings folder if it doesn't exist
# -------------------------------------------------

os.makedirs("embeddings", exist_ok=True)

# -------------------------------------------------
# Load Sentence Transformer Model
# -------------------------------------------------

print("Loading Sentence Transformer model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------------------------
# Function to Generate Embeddings
# -------------------------------------------------

def generate_embeddings(csv_path, embedding_path, label_path):
    print(f"\nProcessing: {csv_path}")

    df = pd.read_csv(csv_path)

    texts = df["text"].tolist()
    labels = df["label"].values

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    np.save(embedding_path, embeddings)
    np.save(label_path, labels)

    print(f"Saved embeddings to: {embedding_path}")
    print(f"Saved labels to: {label_path}")
    print(f"Embedding shape: {embeddings.shape}")

# -------------------------------------------------
# Generate Embeddings
# -------------------------------------------------

generate_embeddings(
    "data/processed/extended_train.csv",
    "embeddings/extended_train_embeddings.npy",
    "embeddings/extended_train_labels.npy"
)

generate_embeddings(
    "data/processed/extended_validation.csv",
    "embeddings/extended_validation_embeddings.npy",
    "embeddings/extended_validation_labels.npy"
)

generate_embeddings(
    "data/processed/extended_test.csv",
    "embeddings/extended_test_embeddings.npy",
    "embeddings/extended_test_labels.npy"
)

print("\n" + "=" * 60)
print("Extended Embeddings Generated Successfully!")
print("=" * 60)