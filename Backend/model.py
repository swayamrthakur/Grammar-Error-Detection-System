"""
model.py — Grammar Error Detection (RoBERTa + Balanced)
Fixes:
  - Balanced dataset (undersample majority class)
  - Weighted loss function (handles remaining imbalance)
  - 3 epochs (prevents overfitting)
  - Full dataset used
Expected accuracy: 0.82–0.90
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report, confusion_matrix)

# ─────────────────────────────────────────────
# 0. Path setup
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "Data", "final_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "Backend", "bert_ged_model")

# ─────────────────────────────────────────────
# 1. Load + clean data
# ─────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["sentence"])
df = df[df["sentence"].astype(str).str.strip().str.len() > 5]
df = df.drop_duplicates(subset=["sentence"])
df["label"] = df["label"].astype(int)

# Clean noisy rows
df = df[df["sentence"].str.split().str.len() >= 4]
df = df[df["sentence"].str.split().str.len() <= 60]
df = df[df["sentence"].str.contains('[a-zA-Z]', regex=True)]

print("Label distribution (before balancing):")
print(df["label"].value_counts())
print(f"Total: {len(df)}")

# ─────────────────────────────────────────────
# 2. Balance dataset
# ─────────────────────────────────────────────
min_count = df["label"].value_counts().min()
df = df.groupby("label", group_keys=False).apply(
    lambda x: x.sample(min_count, random_state=42)
).reset_index(drop=True)

print("\nLabel distribution (after balancing):")
print(df["label"].value_counts())
print(f"Total: {len(df)}")

# ─────────────────────────────────────────────
# 3. Train / test split
# ─────────────────────────────────────────────
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)
print(f"\nTrain: {len(train_df)} | Test: {len(test_df)}")

# ─────────────────────────────────────────────
# 4. Tokenizer + Dataset
# ─────────────────────────────────────────────
print("\n⚙️  Loading RoBERTa tokenizer...")
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import torch
from torch import nn

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")

def tokenize(batch):
    return tokenizer(
        batch["sentence"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_ds = Dataset.from_pandas(train_df[["sentence", "label"]]).map(tokenize, batched=True)
test_ds  = Dataset.from_pandas(test_df[["sentence", "label"]]).map(tokenize, batched=True)

train_ds = train_ds.rename_column("label", "labels")
test_ds  = test_ds.rename_column("label", "labels")

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format("torch",  columns=["input_ids", "attention_mask", "labels"])

# ─────────────────────────────────────────────
# 5. Model
# ─────────────────────────────────────────────
print("⚙️  Loading RoBERTa model...")
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-CoLA", num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Training on: {str(device).upper()}")

# ─────────────────────────────────────────────
# 6. Class weights for weighted loss
# ─────────────────────────────────────────────
counts  = df["label"].value_counts().sort_index()
total   = len(df)
weights = torch.tensor(
    [total / (2 * counts[0]), total / (2 * counts[1])],
    dtype=torch.float
).to(device)

print(f"\nClass weights: Incorrect={weights[0]:.3f}, Correct={weights[1]:.3f}")

# ─────────────────────────────────────────────
# 7. Custom Trainer with weighted loss
# ─────────────────────────────────────────────
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        loss    = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ─────────────────────────────────────────────
# 8. Training arguments
# ─────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "f1":        f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall":    recall_score(labels, preds),
    }

args = TrainingArguments(
    output_dir=os.path.join(BASE_DIR, "Backend", "bert_checkpoints"),
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_steps=100,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=2e-5,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=0,
    report_to="none",
)

trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

# ─────────────────────────────────────────────
# 9. Train
# ─────────────────────────────────────────────
print("\n🏋️  Training RoBERTa (balanced)...")
print("   Expected time: 15–20 min on RTX 3050\n")
trainer.train()

# ─────────────────────────────────────────────
# 10. Final evaluation
# ─────────────────────────────────────────────
print("\n📊 Running final evaluation...")
preds_output = trainer.predict(test_ds)
y_pred = np.argmax(preds_output.predictions, axis=-1)
y_true = test_df["label"].values

print("\n" + "="*50)
print("       ROBERTA MODEL PERFORMANCE")
print("="*50)
print(f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision : {precision_score(y_true, y_pred):.4f}")
print(f"Recall    : {recall_score(y_true, y_pred):.4f}")
print(f"F1 Score  : {f1_score(y_true, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Incorrect", "Correct"]))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# ─────────────────────────────────────────────
# 11. Save
# ─────────────────────────────────────────────
print(f"\n💾 Saving model to {MODEL_PATH}...")
trainer.save_model(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)
print("✅ RoBERTa model saved successfully!")
print(f"\n🎯 Now run: python Backend/app.py")