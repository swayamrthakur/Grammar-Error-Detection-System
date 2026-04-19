"""
convert_m2.py — Fixed M2 parser for Grammar Error Detection
Fixes:
  1. Correct edit offset adjustment after each applied edit
  2. Include correct sentences (label=1) from the M2 file directly
  3. Balanced dataset: correct + incorrect sentences
"""

import pandas as pd
import re


def apply_edits(words, edits):
    """
    Apply edits to a token list with proper offset correction.
    edits: list of (start, end, correction_string)
    Returns corrected sentence string.
    """
    corrected = words[:]
    offset = 0  # tracks index shift after each edit

    for start, end, correction in sorted(edits, key=lambda x: x[0]):
        adj_start = start + offset
        adj_end = end + offset

        if correction == "":
            # Deletion
            del corrected[adj_start:adj_end]
            offset -= (end - start)
        else:
            replacement = correction.split()
            corrected[adj_start:adj_end] = replacement
            offset += len(replacement) - (end - start)

    return " ".join(corrected)


def parse_m2(file_path):
    """
    Parse M2 file and return:
      - (original_sentence, 0)  for sentences WITH errors
      - (corrected_sentence, 1) for grammatically CORRECT sentences
      - Also includes original correct sentences (no edits → label 1)
    """
    data = []

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into sentence blocks
    blocks = content.strip().split("\n\n")

    for block in blocks:
        lines = block.strip().split("\n")
        if not lines or not lines[0].startswith("S "):
            continue

        original = lines[0][2:].strip()
        words = original.split()

        # Collect edits (ignore noop edits marked with -NONE-)
        edits = []
        for line in lines[1:]:
            if not line.startswith("A "):
                continue
            parts = line.split("|||")
            if len(parts) < 5:
                continue

            span_part = parts[0].split()
            try:
                start = int(span_part[1])
                end = int(span_part[2])
            except (IndexError, ValueError):
                continue

            correction = parts[2].strip()
            error_type = parts[1].strip()

            # Skip no-op and UNK edits
            if correction == "-NONE-" or error_type == "noop":
                continue

            edits.append((start, end, correction))

        if edits:
            # Has errors → original is incorrect (label 0)
            if len(words) > 3:
                data.append((original, 0))

            # Apply edits to get corrected version (label 1)
            corrected = apply_edits(words, edits)
            corrected_words = corrected.split()
            if len(corrected_words) > 3 and corrected != original:
                data.append((corrected, 1))
        else:
            # No edits → already correct (label 1)
            if len(words) > 3:
                data.append((original, 1))

    return data


if __name__ == "__main__":
    import sys

    m2_path = sys.argv[1] if len(sys.argv) > 1 else "Data/official-2014.combined.m2"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "Data/dataset.csv"

    print(f"Parsing: {m2_path}")
    data = parse_m2(m2_path)

    df = pd.DataFrame(data, columns=["sentence", "label"])

    # Clean
    df = df.dropna()
    df = df.drop_duplicates(subset=["sentence"])
    df = df[df["sentence"].str.strip().str.len() > 10]
    df = df[df["sentence"].str.count(" ") >= 2]
    df["label"] = df["label"].astype(int)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nLabel distribution:")
    print(df["label"].value_counts())
    print(f"\nTotal samples: {len(df)}")

    df.to_csv(out_path, index=False)
    print(f"\n✅ Dataset saved to {out_path}")
