import joblib
import numpy as np
import onnxruntime as ort

# -------------------------------------------------
# Load Scikit-learn classifier
# -------------------------------------------------

classifier = joblib.load("models/augmented_classifier.pkl")

print("Scikit-learn model loaded.")

# -------------------------------------------------
# Load ONNX model
# -------------------------------------------------

session = ort.InferenceSession("models/augmented_classifier.onnx")

print("ONNX model loaded.")

# -------------------------------------------------
# Load validation embeddings
# -------------------------------------------------

X_val = np.load("embeddings/validation_embeddings.npy")

# -------------------------------------------------
# Select a few samples
# -------------------------------------------------

samples = X_val[:10].astype(np.float32)

# -------------------------------------------------
# Predictions from Scikit-learn
# -------------------------------------------------

sklearn_predictions = classifier.predict(samples)

# -------------------------------------------------
# Predictions from ONNX
# -------------------------------------------------

input_name = session.get_inputs()[0].name

onnx_outputs = session.run(
    None,
    {input_name: samples}
)

onnx_predictions = onnx_outputs[0]

# -------------------------------------------------
# Compare predictions
# -------------------------------------------------

print("\nComparison")
print("=" * 60)

for i in range(len(samples)):
    print(f"Sample {i+1}")
    print(f"Scikit-learn : {sklearn_predictions[i]}")
    print(f"ONNX         : {onnx_predictions[i]}")
    print("-" * 60)

# -------------------------------------------------
# Final verification
# -------------------------------------------------

if np.array_equal(sklearn_predictions, onnx_predictions):
    print("\n✅ Verification Successful!")
    print("The ONNX model produces the same predictions as the Scikit-learn model.")
else:
    print("\n❌ Verification Failed!")
    print("The predictions do not match.")