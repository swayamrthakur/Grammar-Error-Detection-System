import pandas as pd

# Load datasets
df1 = pd.read_csv("Data/dataset.csv")             # CoNLL
df2 = pd.read_csv("Data/generated_dataset.csv")   # Synthetic
df3 = pd.read_csv("Data/wi_abc_dataset.csv")      # W&I ABC (already contains A+B+C)
df4 = pd.read_csv("Data/cola_dataset.csv")





print(f"CoNLL dataset     : {len(df1)} rows")
print(f"Synthetic dataset : {len(df2)} rows")
print(f"W&I ABC dataset   : {len(df3)} rows")
print(f"CoLA dataset      : {len(df4)} rows")

# Combine
df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# Clean
df = df.dropna(subset=["sentence"])
df = df[df["sentence"].astype(str).str.strip() != ""]
df["label"] = df["label"].astype(int)

# Remove conflicted sentences
label_counts = df.groupby("sentence")["label"].nunique()
conflicted   = label_counts[label_counts > 1].index
if len(conflicted) > 0:
    print(f"\n⚠️  Found {len(conflicted)} conflicted sentences — removing them.")
    df = df[~df["sentence"].isin(conflicted)]
else:
    print("\n✅ No conflicted sentences found.")

# Remove duplicates
df = df.drop_duplicates(subset=["sentence"])

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
df.to_csv("Data/final_dataset.csv", index=False)

print("\nLabel distribution in final dataset:")
print(df["label"].value_counts())
print(f"\nTotal samples: {len(df)}")
print("✅ Saved to Data/final_dataset.csv")