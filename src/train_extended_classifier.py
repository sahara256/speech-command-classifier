import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------------------------
# Load Training Data
# -------------------------------------------------

X_train = np.load("embeddings/extended_train_embeddings.npy")
y_train = np.load(
    "embeddings/extended_train_labels.npy",
    allow_pickle=True
)

# -------------------------------------------------
# Load Validation Data
# -------------------------------------------------

X_val = np.load("embeddings/extended_validation_embeddings.npy")
y_val = np.load(
    "embeddings/extended_validation_labels.npy",
    allow_pickle=True
)

# -------------------------------------------------
# Train Logistic Regression Classifier
# -------------------------------------------------

print("=" * 60)
print("Training Extended Classifier...")
print("=" * 60)

classifier = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    random_state=42
)

classifier.fit(X_train, y_train)

# -------------------------------------------------
# Training Accuracy
# -------------------------------------------------

train_predictions = classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)

# -------------------------------------------------
# Validation Accuracy
# -------------------------------------------------

val_predictions = classifier.predict(X_val)
validation_accuracy = accuracy_score(y_val, val_predictions)

# -------------------------------------------------
# Print Results
# -------------------------------------------------

print(f"\nTraining Accuracy   : {train_accuracy:.4f}")
print(f"Validation Accuracy : {validation_accuracy:.4f}")

print("\nClassification Report\n")
print(classification_report(y_val, val_predictions))

# -------------------------------------------------
# Save Model
# -------------------------------------------------

joblib.dump(
    classifier,
    "models/extended_classifier.pkl"
)

print("\nExtended classifier saved successfully!")
print("Location: models/extended_classifier.pkl")