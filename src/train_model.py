import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

# ---------------------------------
# Load Training Data
# ---------------------------------
X_train = np.load("embeddings/train_embeddings.npy")
y_train = np.load("embeddings/train_labels.npy")

X_validation = np.load("embeddings/validation_embeddings.npy")
y_validation = np.load("embeddings/validation_labels.npy")

# ---------------------------------
# Create Logistic Regression Model
# ---------------------------------
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver="lbfgs"
)

# ---------------------------------
# Train Model
# ---------------------------------
model.fit(X_train, y_train)
train_pred = model.predict(X_train)

print("Training Accuracy :", accuracy_score(y_train, train_pred))

# ---------------------------------
# Save Model
# ---------------------------------
joblib.dump(model, "models/classifier.pkl")

# ---------------------------------
# Validation Prediction
# ---------------------------------
y_pred = model.predict(X_validation)

# ---------------------------------
# Validation Accuracy
# ---------------------------------
accuracy = accuracy_score(y_validation, y_pred)

print("=" * 50)
print("Validation Accuracy")
print("=" * 50)
print(f"Accuracy : {accuracy:.4f}")


print("\nClassification Report\n")

print(classification_report(
    y_validation,
    y_pred
))