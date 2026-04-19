from datasets import load_dataset
import pandas as pd

print("Downloading CoLA dataset...")
dataset = load_dataset("nyu-mll/glue", "cola")

rows = []
for split in ["train", "validation"]:
    for item in dataset[split]:
        rows.append({
            "sentence": item["sentence"],
            "label":    item["label"]  # 1=acceptable, 0=unacceptable
        })

df = pd.DataFrame(rows)
df = df.drop_duplicates(subset=["sentence"])
df = df.dropna()

print("Label distribution:")
print(df["label"].value_counts())
print(f"Total: {len(df)}")

df.to_csv("Data/cola_dataset.csv", index=False)
print("Saved to Data/cola_dataset.csv")