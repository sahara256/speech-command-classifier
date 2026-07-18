import joblib
import numpy as np

from sentence_transformers import SentenceTransformer
from preprocessing import clean_text


# ---------------------------------
# Load Saved Model and Label Encoder
# ---------------------------------
model = joblib.load("models/classifier.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# ---------------------------------
# Load Sentence Transformer
# ---------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------
# Confidence Threshold
# ---------------------------------
THRESHOLD = 0.30


# ---------------------------------
# Prediction Function
# ---------------------------------
def predict_command(text):

    # Step 1: Clean input text
    text = clean_text(text)

    # Step 2: Generate embedding
    embedding = embedding_model.encode([text])

    # Step 3: Predict probabilities
    probabilities = model.predict_proba(embedding)[0]

    print("\n" + "=" * 60)
    print(f"Input Sentence : {text}")
    print("=" * 60)

    print("\nClass Probabilities")
    print("-" * 60)

    for label, prob in zip(label_encoder.classes_, probabilities):
        print(f"{label:30s} {prob:.4f}")

    # Step 4: Highest probability
    max_probability = np.max(probabilities)

    # Step 5: Index of highest probability
    predicted_index = np.argmax(probabilities)

    # Step 6: Convert numerical label to text
    predicted_label = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    # -------------------------------------------------
    # OOS Detection Logic
    # -------------------------------------------------

    # Case 1: Model itself predicts OOS
    if predicted_label == "OOS":
        final_prediction = "OOS"

    # Case 2: Confidence is too low
    elif max_probability < THRESHOLD:
        final_prediction = "OOS"

    # Case 3: Accept prediction
    else:
        final_prediction = predicted_label

    return final_prediction, predicted_label, max_probability


# ---------------------------------
# Test Sentences
# ---------------------------------
# ---------------------------------
# Test Sentences
# ---------------------------------
examples = [

    # -------------------------------
    # Unknown Commands (OOS)
    # -------------------------------
    "Book a hotel",
    "Order pizza",
    "What's the stock price today?",
    "Turn on the TV",
    "Open YouTube",
    "Set an alarm for 7 AM",
    "Call an Uber",
    "Send an email",

    # -------------------------------
    # Known Commands
    # -------------------------------
    "Play the music",
    "Increase the volume",
    "Pause the song",
    "Play the next song",
    "Play the previous song",
    "Decrease the volume",
    "Pick up the call",
    "Decline the call",
    "Activate do not disturb",
    "Deactivate do not disturb",

    # -------------------------------
    # Borderline / Similar Commands
    # -------------------------------
    "Play songs",
    "Turn the volume up",
    "Lower the volume",
    "Answer the phone",
    "Reject the phone call",
]

print("=" * 70)
print("OOS Detection Demo")
print("=" * 70)

for sentence in examples:

    final_prediction, predicted_label, confidence = predict_command(sentence)

    print("\nResult")
    print("-" * 60)
    print(f"Sentence            : {sentence}")
    print(f"Predicted Class     : {predicted_label}")
    print(f"Confidence Score    : {confidence:.4f}")
    print(f"Final Prediction    : {final_prediction}")