import pandas as pd

# Read the core commands dataset
commands = pd.read_csv("data/raw/core_commands.csv")

# Read the Out-of-Scope (OOS) dataset
oos = pd.read_csv("data/raw/oos.csv")

# Merge both datasets
dataset = pd.concat([commands, oos], ignore_index=True)

# Shuffle the dataset
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the final dataset
dataset.to_csv("data/processed/full_dataset.csv", index=False)

# Display dataset information
print("=" * 50)
print("Dataset Created Successfully!")
print("=" * 50)

print(f"Core Command Samples : {len(commands)}")
print(f"OOS Samples          : {len(oos)}")
print(f"Total Samples        : {len(dataset)}")

print("\nClass Distribution:")
print(dataset["label"].value_counts())

print("\nFirst 5 Rows:")
print(dataset.head())
