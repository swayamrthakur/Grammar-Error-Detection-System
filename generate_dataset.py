"""
generate_dataset.py — Fixed synthetic grammar data generator
Fixes:
  1. No duplicate sentences with contradictory labels
  2. All verb corrections are actually correct
  3. Covers more error types beyond just subject-verb agreement
"""

import pandas as pd
import random

random.seed(42)
data = []

# ─────────────────────────────────────────────
# 1. Subject-Verb Agreement (fixed)
# ─────────────────────────────────────────────
# He/She need 3rd-person singular; others don't

third_person = ["He", "She"]
other_subjects = ["They", "We", "I", "The students", "The teacher"]

verb_pairs = [
    # (base_form, 3rd_person_singular)
    ("go",    "goes"),
    ("eat",   "eats"),
    ("play",  "plays"),    # FIXED: was "play" before
    ("have",  "has"),      # FIXED: was "have" before
    ("watch", "watches"),  # FIXED: was "watch" before
    ("run",   "runs"),
    ("make",  "makes"),
    ("work",  "works"),
    ("come",  "comes"),
    ("study", "studies"),
]

places = [
    "to school every day",
    "at the market on weekends",
    "in the park after lunch",
    "to the office early",
    "together on Sundays",
    "hard for the exam",
    "late at night",
    "outside during recess",
]

for sub in third_person:
    for base, third in verb_pairs:
        for place in places:
            wrong   = f"{sub} {base} {place}."    # e.g. "He go to school every day."
            correct = f"{sub} {third} {place}."   # e.g. "He goes to school every day."
            if wrong != correct:                   # safety check — never add contradictory labels
                data.append((wrong, 0))
                data.append((correct, 1))

# Other subjects use base form — only add as correct
for sub in other_subjects:
    for base, _ in verb_pairs:
        for place in places[:4]:  # fewer combos to keep balance
            correct = f"{sub} {base} {place}."
            data.append((correct, 1))

# ─────────────────────────────────────────────
# 2. Article Errors (a/an)
# ─────────────────────────────────────────────
article_pairs = [
    ("a apple",     "an apple"),
    ("a elephant",  "an elephant"),
    ("a hour",      "an hour"),
    ("a umbrella",  "an umbrella"),
    ("an book",     "a book"),
    ("an car",      "a car"),
    ("an pen",      "a pen"),
    ("an table",    "a table"),
    ("an dog",      "a dog"),
    ("an bus",      "a bus"),
]

templates = [
    "She bought {} from the store.",
    "He found {} on the table.",
    "I need {} for my homework.",
    "We saw {} in the garden.",
    "They gave me {}.",
]

for wrong_article, correct_article in article_pairs:
    for tmpl in templates:
        wrong   = tmpl.format(wrong_article)
        correct = tmpl.format(correct_article)
        data.append((wrong, 0))
        data.append((correct, 1))

# ─────────────────────────────────────────────
# 3. Missing Article
# ─────────────────────────────────────────────
missing_article_pairs = [
    ("She is student.",         "She is a student."),
    ("He is doctor.",           "He is a doctor."),
    ("I want to buy car.",      "I want to buy a car."),
    ("They live in house.",     "They live in a house."),
    ("We visited museum.",      "We visited a museum."),
    ("She opened door.",        "She opened the door."),
    ("He closed window.",       "He closed the window."),
    ("I finished homework.",    "I finished the homework."),
    ("They won match.",         "They won the match."),
    ("We missed bus.",          "We missed the bus."),
]

for wrong, correct in missing_article_pairs:
    data.append((wrong, 0))
    data.append((correct, 1))

# ─────────────────────────────────────────────
# 4. Wrong Tense
# ─────────────────────────────────────────────
tense_pairs = [
    ("Yesterday she go to school.",          "Yesterday she went to school."),
    ("Last year he win the competition.",    "Last year he won the competition."),
    ("They come here two days ago.",         "They came here two days ago."),
    ("I see him last Monday.",               "I saw him last Monday."),
    ("She buyed a new phone.",               "She bought a new phone."),
    ("He goed to the market.",               "He went to the market."),
    ("We taked the exam yesterday.",         "We took the exam yesterday."),
    ("They eated dinner early.",             "They ate dinner early."),
    ("I writed a letter to her.",            "I wrote a letter to her."),
    ("She speaked too fast.",                "She spoke too fast."),
]

for wrong, correct in tense_pairs:
    data.append((wrong, 0))
    data.append((correct, 1))

# ─────────────────────────────────────────────
# 5. Clearly Correct Sentences (no errors)
# ─────────────────────────────────────────────
correct_sentences = [
    "She goes to school every morning.",
    "He eats breakfast before leaving.",
    "They play football on weekends.",
    "We have completed the assignment.",
    "I watch television in the evening.",
    "The students study hard for their exams.",
    "She bought an apple from the market.",
    "He is a good doctor.",
    "Yesterday they went to the park.",
    "I saw a beautiful bird in the garden.",
    "We took the bus to the station.",
    "She spoke clearly during the presentation.",
    "The teacher opened the door for the students.",
    "He won the competition last year.",
    "They came home late last night.",
    "I wrote a letter to my friend.",
    "She finished the homework on time.",
    "We missed the last train.",
    "He closed the window before leaving.",
    "They ate dinner together as a family.",
]

for s in correct_sentences:
    data.append((s, 1))

# ─────────────────────────────────────────────
# Build DataFrame and clean
# ─────────────────────────────────────────────
df = pd.DataFrame(data, columns=["sentence", "label"])

# CRITICAL: remove any row where same sentence has both labels
dup_sentences = df.groupby("sentence")["label"].nunique()
conflicted = dup_sentences[dup_sentences > 1].index
df = df[~df["sentence"].isin(conflicted)]

df = df.drop_duplicates(subset=["sentence"])
df = df.dropna()
df = df[df["sentence"].str.strip().str.len() > 5]
df["label"] = df["label"].astype(int)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("✅ Synthetic dataset generated!")
print("Label distribution:")
print(df["label"].value_counts())
print(f"Total samples: {len(df)}")

df.to_csv("Data/generated_dataset.csv", index=False)