import pandas as pd

print("=" * 50)
print("DATASET QUALITY REPORT")
print("=" * 50)

files = {
    "CoNLL (dataset.csv)":          "Data/dataset.csv",
    "Synthetic (generated.csv)":    "Data/generated_dataset.csv",
    "W&I A (wi_dataset.csv)":       "Data/wi_dataset.csv",
    "W&I B (wi_b_dataset.csv)":     "Data/wi_b_dataset.csv",
    "W&I C (wi_c_dataset.csv)":     "Data/wi_c_dataset.csv",
    "W&I ABC (wi_abc_dataset.csv)": "Data/wi_abc_dataset.csv",
    "FINAL (final_dataset.csv)":    "Data/final_dataset.csv",
}

for name, path in files.items():
    try:
        df = pd.read_csv(path)
        counts = df["label"].value_counts().sort_index()
        total = len(df)
        label0 = counts.get(0, 0)
        label1 = counts.get(1, 0)
        ratio = label1 / total * 100 if total > 0 else 0
        print(f"\n{name}")
        print(f"  Total     : {total}")
        print(f"  Label 0   : {label0}  ({label0/total*100:.1f}%)")
        print(f"  Label 1   : {label1}  ({label1/total*100:.1f}%)")
        print(f"  Imbalance : {'⚠️  BAD' if ratio > 65 or ratio < 35 else '✅ OK'}")

        # Check for conflicted sentences
        dupes = df.groupby("sentence")["label"].nunique()
        conflicts = (dupes > 1).sum()
        print(f"  Conflicts : {conflicts} {'⚠️  BAD' if conflicts > 0 else '✅ OK'}")

        # Sample sentences
        print(f"  Sample label 0: {df[df['label']==0]['sentence'].iloc[0][:80]}")
        print(f"  Sample label 1: {df[df['label']==1]['sentence'].iloc[0][:80]}")

    except FileNotFoundError:
        print(f"\n{name}: FILE NOT FOUND — skipping")