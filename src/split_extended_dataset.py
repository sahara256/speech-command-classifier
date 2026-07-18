import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# Load Extended Dataset
# -------------------------------------------------

df = pd.read_csv("data/processed/extended_dataset.csv")

# -------------------------------------------------
# First Split (70% Train, 30% Temporary)
# -------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

# -------------------------------------------------
# Second Split (15% Validation, 15% Test)
# -------------------------------------------------

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

# -------------------------------------------------
# Save Files
# -------------------------------------------------

train_df.to_csv(
    "data/processed/extended_train.csv",
    index=False
)

validation_df.to_csv(
    "data/processed/extended_validation.csv",
    index=False
)

test_df.to_csv(
    "data/processed/extended_test.csv",
    index=False
)

# -------------------------------------------------
# Print Summary
# -------------------------------------------------

print("=" * 60)
print("Extended Dataset Split Completed!")
print("=" * 60)

print(f"Training Samples   : {len(train_df)}")
print(f"Validation Samples : {len(validation_df)}")
print(f"Test Samples       : {len(test_df)}")

print("\nTraining Distribution:\n")
print(train_df["label"].value_counts())

print("\nValidation Distribution:\n")
print(validation_df["label"].value_counts())

print("\nTest Distribution:\n")
print(test_df["label"].value_counts())