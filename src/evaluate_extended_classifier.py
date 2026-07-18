import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------------------------
# Load Test Embeddings
# -------------------------------------------------

X_test = np.load("embeddings/extended_test_embeddings.npy")

# Labels are stored as strings
y_test = np.load(
    "embeddings/extended_test_labels.npy",
    allow_pickle=True
)

# -------------------------------------------------
# Load Trained Model
# -------------------------------------------------

classifier = joblib.load(
    "models/extended_classifier.pkl"
)

# -------------------------------------------------
# Predict
# -------------------------------------------------

predictions = classifier.predict(X_test)

# -------------------------------------------------
# Accuracy
# -------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

print("=" * 60)
print("Extended Classifier Test Results")
print("=" * 60)

print(f"\nTest Accuracy : {accuracy:.4f}")

# -------------------------------------------------
# Classification Report
# -------------------------------------------------

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        predictions
    )
)

# -------------------------------------------------
# Confusion Matrix
# -------------------------------------------------

print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)