import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from preprocessing import clean_text
# -----------------------------
# Load the datasets
# -----------------------------
train = pd.read_csv("data/processed/train.csv")
validation = pd.read_csv("data/processed/validation.csv")
test = pd.read_csv("data/processed/test.csv")
# -----------------------------
# Clean the text
# -----------------------------
train["text"] = train["text"].apply(clean_text)
validation["text"] = validation["text"].apply(clean_text)
test["text"] = test["text"].apply(clean_text)
# -----------------------------
# Load the Label Encoder
# -----------------------------
label_encoder = joblib.load("models/label_encoder.pkl")
# -----------------------------
# Encode Labels
# -----------------------------
train_labels = label_encoder.transform(train["label"])
validation_labels = label_encoder.transform(validation["label"])
test_labels = label_encoder.transform(test["label"])
# -----------------------------
# Load Sentence Transformer
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
# -----------------------------
# Generate Embeddings
# -----------------------------
train_embeddings = model.encode(
    train["text"].tolist(),
    show_progress_bar=True
)

validation_embeddings = model.encode(
    validation["text"].tolist(),
    show_progress_bar=True
)

test_embeddings = model.encode(
    test["text"].tolist(),
    show_progress_bar=True
)
# -----------------------------
# Save Embeddings
# -----------------------------
np.save(
    "embeddings/train_embeddings.npy",
    train_embeddings
)

np.save(
    "embeddings/validation_embeddings.npy",
    validation_embeddings
)

np.save(
    "embeddings/test_embeddings.npy",
    test_embeddings
)
# ----------------------------
# Save Labels
# -----------------------------
np.save(
    "embeddings/train_labels.npy",
    train_labels
)

np.save(
    "embeddings/validation_labels.npy",
    validation_labels
)

np.save(
    "embeddings/test_labels.npy",
    test_labels
)
# -----------------------------
# Display Information
# -----------------------------
print("=" * 60)
print("Embedding Generation Completed Successfully!")
print("=" * 60)

print(f"\nTraining Embeddings Shape     : {train_embeddings.shape}")
print(f"Validation Embeddings Shape  : {validation_embeddings.shape}")
print(f"Testing Embeddings Shape     : {test_embeddings.shape}")

print("\nExample Sentence:")
print(train['text'].iloc[0])

print("\nFirst 10 Embedding Values:")
print(train_embeddings[0][:10])

print(f"\nEmbedding Dimension : {train_embeddings.shape[1]}")