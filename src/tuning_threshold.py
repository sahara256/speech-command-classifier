import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from preprocessing import clean_text


# ---------------------------------
# Load Saved Model
# ---------------------------------
model = joblib.load("models/classifier.pkl")

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ---------------------------------
# Load Validation Dataset
# ---------------------------------
validation = pd.read_csv(
    "data/processed/validation.csv"
)


# ---------------------------------
# Threshold Values
# ---------------------------------
thresholds = np.arange(0.30, 0.95, 0.05)


print("=" * 70)
print("Threshold Tuning")
print("=" * 70)


for threshold in thresholds:

    correct = 0

    total = len(validation)

    for _, row in validation.iterrows():

        sentence = clean_text(row["text"])

        true_label = row["label"]

        embedding = embedding_model.encode([sentence])

        probabilities = model.predict_proba(
            embedding
        )[0]

        max_probability = np.max(probabilities)

        predicted_index = np.argmax(probabilities)

        predicted_label = label_encoder.inverse_transform(
            [predicted_index]
        )[0]

        # --------------------------
        # Apply Threshold
        # --------------------------

        if predicted_label != "OOS" and max_probability < threshold:
            predicted_label = "OOS"

        if predicted_label == true_label:
            correct += 1

    accuracy = correct / total

    print(
        f"Threshold: {threshold:.2f}   Accuracy: {accuracy:.4f}"
    )