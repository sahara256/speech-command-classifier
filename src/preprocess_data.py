import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from preprocessing import clean_text
# -----------------------------
# Load the datasets
# -----------------------------
train = pd.read_csv("data/processed/train.csv")
validation = pd.read_csv("data/processed/validation.csv")
test = pd.read_csv("data/processed/test.csv")
# -----------------------------
# Apply Text Cleaning
# -----------------------------
train["text"] = train["text"].apply(clean_text)
validation["text"] = validation["text"].apply(clean_text)
test["text"] = test["text"].apply(clean_text)
# -----------------------------
# Label Encoding
# -----------------------------
label_encoder = LabelEncoder()

train["label_encoded"] = label_encoder.fit_transform(train["label"])

validation["label_encoded"] = label_encoder.transform(
    validation["label"]
)

test["label_encoded"] = label_encoder.transform(
    test["label"]
)
# -----------------------------
# Save Label Encoder
# -----------------------------
joblib.dump(
    label_encoder,
    "models/label_encoder.pkl"
)
# -----------------------------
# Display Dataset Information
# -----------------------------
print("=" * 50)
print("Preprocessing Completed Successfully!")
print("=" * 50)

print("\nTraining Dataset")
print(train.head())

print("\nValidation Dataset")
print(validation.head())

print("\nTesting Dataset")
print(test.head())

print("\nLabel Mapping")

for index, label in enumerate(label_encoder.classes_):
    print(f"{index} --> {label}")