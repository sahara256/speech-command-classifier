import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# -------------------------------------------------
# Load trained classifier
# -------------------------------------------------

classifier = joblib.load("models/augmented_classifier.pkl")

print("Classifier loaded successfully.")

# -------------------------------------------------
# Define input shape
# -------------------------------------------------

initial_type = [
    ("float_input", FloatTensorType([None, 384]))
]

# -------------------------------------------------
# Convert to ONNX
# -------------------------------------------------

onnx_model = convert_sklearn(
    classifier,
    initial_types=initial_type
)

# -------------------------------------------------
# Save ONNX model
# -------------------------------------------------

with open("models/augmented_classifier.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("\nModel exported successfully!")

print("Saved as:")
print("models/augmented_classifier.onnx")