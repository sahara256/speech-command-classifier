import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

# =====================================================
# Load Models
# =====================================================

print("=" * 60)
print("Lightweight Speech Command Classifier")
print("=" * 60)

print("\nLoading MiniLM embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading trained classifier...")
classifier = joblib.load("models/extended_classifier.pkl")

print("\nSystem Ready!")
print("Type 'exit' to quit.\n")

# =====================================================
# Configuration
# =====================================================

CONFIDENCE_THRESHOLD = 0.30

# =====================================================
# Prediction Loop
# =====================================================

while True:

    command = input("Enter Command: ").strip()

    # Exit
    if command.lower() == "exit":
        print("\nThank you for using the Speech Command Classifier!")
        break

    # Empty input
    if command == "":
        print("\nPlease enter a valid command.\n")
        continue

    # -------------------------------------------------
    # Generate Sentence Embedding
    # -------------------------------------------------

    embedding = embedding_model.encode([command])

    # -------------------------------------------------
    # Predict Probabilities
    # -------------------------------------------------

    probabilities = classifier.predict_proba(embedding)[0]

    # -------------------------------------------------
    # Best Prediction
    # -------------------------------------------------

    best_index = np.argmax(probabilities)

    predicted_intent = classifier.classes_[best_index]
    confidence = probabilities[best_index]

    # -------------------------------------------------
    # OOS Detection
    # -------------------------------------------------

    if predicted_intent == "OOS":
        final_prediction = "OOS"
        status = "Unknown Command"

    elif confidence < CONFIDENCE_THRESHOLD:
        final_prediction = "OOS"
        status = "Unknown Command"

    else:
        final_prediction = predicted_intent
        status = "Known Command"

    # -------------------------------------------------
    # Display Prediction
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("Prediction Result")
    print("=" * 60)

    print(f"Input              : {command}")
    print(f"Predicted Intent   : {final_prediction}")
    print(f"Confidence         : {confidence:.2%}")
    print(f"Status             : {status}")

    # -------------------------------------------------
    # Top 3 Predictions
    # -------------------------------------------------

    top_indices = np.argsort(probabilities)[::-1][:3]

    print("\nTop 3 Predictions")
    print("-" * 60)

    for rank, index in enumerate(top_indices, start=1):
        print(
            f"{rank}. "
            f"{classifier.classes_[index]:30}"
            f"{probabilities[index]:.2%}"
        )

    print("=" * 60)
    print()