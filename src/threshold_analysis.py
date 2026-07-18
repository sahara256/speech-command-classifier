import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from preprocessing import clean_text


# ----------------------------------------------------
# Load Models
# ----------------------------------------------------
model = joblib.load("models/classifier.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------------------------------
# Load Validation Data
# ----------------------------------------------------
validation = pd.read_csv("data/processed/validation.csv")

thresholds = np.arange(0.30, 0.95, 0.05)

print("=" * 100)
print(
    f"{'Threshold':<12}"
    f"{'Known Acc':<15}"
    f"{'OOS Acc':<15}"
    f"{'Overall Acc':<15}"
    f"{'False Reject':<15}"
)
print("=" * 100)

for threshold in thresholds:

    known_total = 0
    known_correct = 0

    oos_total = 0
    oos_correct = 0

    false_reject = 0

    for _, row in validation.iterrows():

        sentence = clean_text(row["text"])
        true_label = row["label"]

        embedding = embedding_model.encode([sentence])

        probabilities = model.predict_proba(embedding)[0]

        max_probability = np.max(probabilities)

        predicted_index = np.argmax(probabilities)

        predicted_label = label_encoder.inverse_transform(
            [predicted_index]
        )[0]

        # Apply threshold only for known predictions
        if predicted_label != "OOS" and max_probability < threshold:
            predicted_label = "OOS"

        # -----------------------------
        # Known Intent Statistics
        # -----------------------------
        if true_label != "OOS":

            known_total += 1

            if predicted_label == true_label:
                known_correct += 1

            if predicted_label == "OOS":
                false_reject += 1

        # -----------------------------
        # OOS Statistics
        # -----------------------------
        else:

            oos_total += 1

            if predicted_label == "OOS":
                oos_correct += 1

    known_accuracy = known_correct / known_total
    oos_accuracy = oos_correct / oos_total

    overall_accuracy = (
        known_correct + oos_correct
    ) / len(validation)

    false_reject_rate = false_reject / known_total

    print(
        f"{threshold:<12.2f}"
        f"{known_accuracy:<15.4f}"
        f"{oos_accuracy:<15.4f}"
        f"{overall_accuracy:<15.4f}"
        f"{false_reject_rate:<15.4f}"
    )