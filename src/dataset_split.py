import pandas as pd
from sklearn.model_selection import train_test_split

# Load the full dataset
dataset = pd.read_csv("data/processed/full_dataset.csv")

# -----------------------------
# First Split
# 70% Train
# 30% Temporary
# -----------------------------
train, temp = train_test_split(
    dataset,
    test_size=0.30,
    random_state=42,
    stratify=dataset["label"]
)

# -----------------------------
# Second Split
# Split the remaining 30%
# into Validation (15%)
# and Test (15%)
# -----------------------------
validation, test = train_test_split(
    temp,
    test_size=0.50,
    random_state=42,
    stratify=temp["label"]
)

# -----------------------------
# Save the datasets
# -----------------------------
train.to_csv("data/processed/train.csv", index=False)
validation.to_csv("data/processed/validation.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

# -----------------------------
# Print Summary
# -----------------------------
print("=" * 50)
print("Dataset Split Completed Successfully!")
print("=" * 50)

print(f"Training Samples   : {len(train)}")
print(f"Validation Samples : {len(validation)}")
print(f"Testing Samples    : {len(test)}")

print("\nTraining Label Distribution")
print(train["label"].value_counts())

print("\nValidation Label Distribution")
print(validation["label"].value_counts())

print("\nTesting Label Distribution")
print(test["label"].value_counts())

