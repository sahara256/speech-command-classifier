from onnxruntime.quantization import quantize_dynamic, QuantType

# -------------------------------------------------
# Input and output model paths
# -------------------------------------------------

input_model = "models/augmented_classifier.onnx"

output_model = "models/augmented_classifier_quantized.onnx"

# -------------------------------------------------
# Apply Dynamic Quantization
# -------------------------------------------------

quantize_dynamic(
    model_input=input_model,
    model_output=output_model,
    weight_type=QuantType.QInt8
)

print("=" * 60)
print("Model Quantization Completed Successfully!")
print("=" * 60)

print(f"Original Model : {input_model}")
print(f"Quantized Model: {output_model}")