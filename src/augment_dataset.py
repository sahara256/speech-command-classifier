import random
import pandas as pd

from noise_augmentation import add_asr_noise


# -----------------------------
# Set random seed
# -----------------------------
random.seed(42)


# -----------------------------
# Load training dataset
# -----------------------------
train_df = pd.read_csv("data/processed/train.csv")

print(train_df.columns)
print(train_df.head())
# -----------------------------
# Create augmented samples
# -----------------------------
augmented_rows = []

for _, row in train_df.iterrows():

    original_text = row["text"]
    label = row["label"]

    noisy_text = add_asr_noise(original_text)

    augmented_rows.append({
    "text": noisy_text,
    "label": label

    })


# -----------------------------
# Convert to DataFrame
# -----------------------------
augmented_df = pd.DataFrame(augmented_rows)


# -----------------------------
# Combine original + augmented
# -----------------------------
combined_df = pd.concat(
    [train_df, augmented_df],
    ignore_index=True
)


# -----------------------------
# Shuffle dataset
# -----------------------------
combined_df = combined_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# -----------------------------
# Save
# -----------------------------
combined_df.to_csv(
    "data/processed/train_augmented.csv",
    index=False
)


# -----------------------------
# Print summary
# -----------------------------
print("=" * 60)
print("Training Data Augmentation")
print("=" * 60)

print(f"Original samples  : {len(train_df)}")
print(f"Augmented samples : {len(augmented_df)}")
print(f"Total samples     : {len(combined_df)}")

print("\nFirst 10 rows:\n")

print(combined_df.head(10))