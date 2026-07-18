import pandas as pd

# -------------------------------------------------
# Load Original Dataset
# -------------------------------------------------

original_df = pd.read_csv(
    "data/processed/full_dataset.csv"
)

# -------------------------------------------------
# Load Extension Dataset
# -------------------------------------------------

extension_df = pd.read_csv(
    "data/processed/extension_dataset.csv"
)

# -------------------------------------------------
# Merge Both Datasets
# -------------------------------------------------

merged_df = pd.concat(
    [original_df, extension_df],
    ignore_index=True
)

# -------------------------------------------------
# Shuffle Dataset
# -------------------------------------------------

merged_df = merged_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# -------------------------------------------------
# Save
# -------------------------------------------------

output_path = "data/processed/extended_dataset.csv"

merged_df.to_csv(
    output_path,
    index=False
)

# -------------------------------------------------
# Print Summary
# -------------------------------------------------

print("=" * 60)
print("Extension Dataset Merged Successfully!")
print("=" * 60)

print(f"Original Samples : {len(original_df)}")
print(f"Extension Samples: {len(extension_df)}")
print(f"Total Samples    : {len(merged_df)}")

print("\nIntent Distribution:\n")
print(merged_df["label"].value_counts())

print(f"\nSaved to:\n{output_path}")
