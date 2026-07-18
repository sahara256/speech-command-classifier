import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ----------------------------------------
# Load augmented training data
# ----------------------------------------
X_train = np.load("data/processed/train_augmented_embeddings.npy")
y_train = np.load("data/processed/train_augmented_labels.npy")


# ----------------------------------------
# Load validation data
# ----------------------------------------
import pandas as pd

X_val = np.load("embeddings/validation_embeddings.npy")

validation_df = pd.read_csv("data/processed/validation.csv")
y_val = validation_df["label"].values

# ----------------------------------------
# Train Logistic Regression
# ----------------------------------------
print("Training classifier...")

classifier = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver="lbfgs"
)

classifier.fit(X_train, y_train)


# ----------------------------------------
# Training Accuracy
# ----------------------------------------
train_predictions = classifier.predict(X_train)

train_accuracy = accuracy_score(
    y_train,
    train_predictions
)


# ----------------------------------------
# Validation Accuracy
# ----------------------------------------
val_predictions = classifier.predict(X_val)

validation_accuracy = accuracy_score(
    y_val,
    val_predictions
)


# ----------------------------------------
# Save classifier
# ----------------------------------------
joblib.dump(
    classifier,
    "models/augmented_classifier.pkl"
)


# ----------------------------------------
# Results
# ----------------------------------------
print("\n" + "=" * 60)
print("Augmented Classifier Results")
print("=" * 60)

print(f"Training Accuracy   : {train_accuracy:.4f}")
print(f"Validation Accuracy : {validation_accuracy:.4f}")

print("\nClassifier saved as:")
print("models/augmented_classifier.pkl")