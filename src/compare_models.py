import os
import time
import joblib
import numpy as np
import onnxruntime as ort

# -------------------------------------------------
# Load Validation Embeddings
# -------------------------------------------------

X = np.load("embeddings/validation_embeddings.npy").astype(np.float32)

# -------------------------------------------------
# Load Scikit-learn Model
# -------------------------------------------------

sk_model = joblib.load("models/augmented_classifier.pkl")

# -------------------------------------------------
# Load ONNX Model
# -------------------------------------------------

onnx_session = ort.InferenceSession(
    "models/augmented_classifier.onnx"
)

onnx_input = onnx_session.get_inputs()[0].name

# -------------------------------------------------
# Load Quantized ONNX Model
# -------------------------------------------------

quant_session = ort.InferenceSession(
    "models/augmented_classifier_quantized.onnx"
)

quant_input = quant_session.get_inputs()[0].name

# -------------------------------------------------
# Measure Model Sizes
# -------------------------------------------------

pkl_size = os.path.getsize(
    "models/augmented_classifier.pkl"
) / 1024

onnx_size = os.path.getsize(
    "models/augmented_classifier.onnx"
) / 1024

quant_size = os.path.getsize(
    "models/augmented_classifier_quantized.onnx"
) / 1024

# -------------------------------------------------
# Measure Scikit-learn Inference Time
# -------------------------------------------------

start = time.perf_counter()

for _ in range(1000):
    sk_model.predict(X)

sk_time = (time.perf_counter() - start) / 1000

# -------------------------------------------------
# Measure ONNX Inference Time
# -------------------------------------------------

start = time.perf_counter()

for _ in range(1000):
    onnx_session.run(
        None,
        {onnx_input: X}
    )

onnx_time = (time.perf_counter() - start) / 1000

# -------------------------------------------------
# Measure Quantized ONNX Inference Time
# -------------------------------------------------

start = time.perf_counter()

for _ in range(1000):
    quant_session.run(
        None,
        {quant_input: X}
    )

quant_time = (time.perf_counter() - start) / 1000

# -------------------------------------------------
# Print Results
# -------------------------------------------------

print("=" * 65)
print("Model Comparison")
print("=" * 65)

print(f"{'Model':30}{'Size (KB)':15}{'Inference (ms)'}")

print("-" * 65)

print(f"{'Scikit-learn (.pkl)':30}{pkl_size:<15.2f}{sk_time*1000:.4f}")

print(f"{'ONNX':30}{onnx_size:<15.2f}{onnx_time*1000:.4f}")

print(f"{'Quantized ONNX':30}{quant_size:<15.2f}{quant_time*1000:.4f}")

print("=" * 65)