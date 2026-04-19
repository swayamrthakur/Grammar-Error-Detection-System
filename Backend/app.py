"""
app.py — Grammar Error Detection API (Final + Extended Correction)
Uses:
  - RoBERTa fine-tuned model for detection
  - LanguageTool + tense fixer + intervening NP fixer for correction
Endpoints:
  POST /predict       — single sentence grammar check
  POST /predict_batch — multiple sentences
  POST /correct       — grammar correction (LanguageTool + rules)
  GET  /health        — status check
  GET  /              — API info
"""

import os
import re
import torch
import language_tool_python
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# 0. Path setup
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Backend", "bert_ged_model")

# ─────────────────────────────────────────────
# 1. Load RoBERTa model
# ─────────────────────────────────────────────
print("Loading RoBERTa model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"✅ RoBERTa loaded on {str(device).upper()}")

# ─────────────────────────────────────────────
# 2. Load LanguageTool
# ─────────────────────────────────────────────
print("Loading LanguageTool (first run downloads ~200MB)...")
lt = language_tool_python.LanguageTool('en-US')
print("✅ LanguageTool ready")

# ─────────────────────────────────────────────
# 3. Rule-based correction helpers
# ─────────────────────────────────────────────

PAST_MARKERS = [
    'yesterday', 'last year', 'last week', 'last month',
    'last night', 'last monday', 'last tuesday', 'last wednesday',
    'last thursday', 'last friday', 'last saturday', 'last sunday',
    ' ago', 'previously', 'formerly', 'in 2019', 'in 2020',
    'in 2021', 'in 2022', 'in 2023', 'back then', 'at that time',
    'earlier today', 'this morning', 'this afternoon'
]

PRESENT_TO_PAST = [
    (r'\bare\b',   'were'),
    (r'\bis\b',    'was'),
    (r'\bdo\b',    'did'),
    (r'\bdoes\b',  'did'),
    (r'\bhave\b',  'had'),
    (r'\bhas\b',   'had'),
    (r'\bgo\b',    'went'),
    (r'\bgoes\b',  'went'),
    (r'\bcome\b',  'came'),
    (r'\bcomes\b', 'came'),
    (r'\brun\b',   'ran'),
    (r'\bruns\b',  'ran'),
    (r'\beat\b',   'ate'),
    (r'\beats\b',  'ate'),
    (r'\bsee\b',   'saw'),
    (r'\bsees\b',  'saw'),
    (r'\bwake\b',  'woke'),
    (r'\bwakes\b', 'woke'),
    (r'\btake\b',  'took'),
    (r'\btakes\b', 'took'),
    (r'\bmake\b',  'made'),
    (r'\bmakes\b', 'made'),
]

# Nouns that are grammatically singular even when followed by "of + plural"
COLLECTIVE_SINGULAR = [
    'list', 'group', 'set', 'number', 'series', 'type',
    'kind', 'sort', 'variety', 'range', 'collection',
    'batch', 'bunch', 'pair', 'team', 'committee',
    'class', 'category', 'array', 'row', 'box',
    'pile', 'stack', 'bundle', 'cluster', 'network',
    'system', 'body', 'board', 'council', 'panel',
    'majority', 'minority', 'total', 'sum', 'amount',
    'portion', 'percentage', 'fraction', 'quarter',
    'half', 'sample', 'selection', 'sequence', 'string',
]

PLURAL_TO_SINGULAR_VERB = {
    'are':   'is',
    'were':  'was',
    'have':  'has',
    'do':    'does',
    "don't": "doesn't",
    'were':  'was',
}


def fix_tense_context(original, corrected):
    """
    If sentence has past-time markers, convert present verbs
    LanguageTool introduced back to past tense.
    """
    orig_lower = original.lower()
    has_past_marker = any(marker in orig_lower for marker in PAST_MARKERS)
    if not has_past_marker:
        return corrected

    result = corrected
    for pattern, replacement in PRESENT_TO_PAST:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def fix_intervening_np(text, corrected):
    """
    Fix subject-verb agreement when a prepositional phrase
    intervenes between a singular noun head and the verb.
    e.g. 'The list of items are on the table'
      -> 'The list of items is on the table'
    """
    result = corrected
    for noun in COLLECTIVE_SINGULAR:
        for plural_verb, singular_verb in PLURAL_TO_SINGULAR_VERB.items():
            # Pattern: {noun} of [some words] {plural_verb}
            pattern = (
                rf'\b({re.escape(noun)}'
                rf'\s+of\s+\w+(?:\s+\w+){{0,4}}\s+)'
                rf'({re.escape(plural_verb)})\b'
            )
            result = re.sub(
                pattern,
                lambda m: m.group(1) + singular_verb,
                result,
                flags=re.IGNORECASE
            )
    return result


def fix_double_negative(corrected):
    """
    Fix common double negation patterns.
    e.g. 'I don't know nothing' -> 'I don't know anything'
    """
    replacements = [
        (r"\bdon't\s+know\s+nothing\b",   "don't know anything"),
        (r"\bcan't\s+do\s+nothing\b",     "can't do anything"),
        (r"\bdidn't\s+see\s+nobody\b",    "didn't see anybody"),
        (r"\bdidn't\s+go\s+nowhere\b",    "didn't go anywhere"),
        (r"\bnever\s+did\s+nothing\b",    "never did anything"),
        (r"\bno\s+one\s+never\b",         "no one ever"),
        (r"\bnobody\s+never\b",           "nobody ever"),
    ]
    result = corrected
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def fix_article_errors(corrected):
    """
    Fix 'a' before vowel sounds and 'an' before consonant sounds.
    """
    result = corrected
    # 'a' before vowel sound → 'an'
    result = re.sub(
        r'\ba\s+([aeiouAEIOU]\w+)',
        lambda m: 'an ' + m.group(1),
        result
    )
    # 'an' before consonant sound → 'a'
    result = re.sub(
        r'\ban\s+([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]\w+)',
        lambda m: 'a ' + m.group(1),
        result
    )
    return result


def apply_all_fixes(original, lt_corrected):
    """
    Apply all rule-based fixes in sequence after LanguageTool.
    Order matters — tense fix before intervening NP fix.
    """
    result = lt_corrected
    result = fix_tense_context(original, result)
    result = fix_intervening_np(original, result)
    result = fix_double_negative(result)
    result = fix_article_errors(result)
    return result


# ─────────────────────────────────────────────
# 4. Prediction helper
# ─────────────────────────────────────────────
def predict_texts(texts):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    preds  = torch.argmax(logits, dim=1).cpu().tolist()
    probas = torch.softmax(logits, dim=1).cpu().tolist()
    return preds, probas


# ─────────────────────────────────────────────
# 5. Routes
# ─────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No input text provided"}), 400

    preds, probas = predict_texts([text])
    pred  = preds[0]
    proba = probas[0]

    return jsonify({
        "prediction": "Correct" if pred == 1 else "Incorrect",
        "confidence": round(float(max(proba)), 4),
        "label":      int(pred)
    })


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data = request.json
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400
    texts = data.get("texts", [])
    if not texts:
        return jsonify({"error": "No texts provided"}), 400

    preds, probas = predict_texts(texts)
    results = []
    for text, pred, proba in zip(texts, preds, probas):
        results.append({
            "text":       text,
            "prediction": "Correct" if pred == 1 else "Incorrect",
            "confidence": round(float(max(proba)), 4),
            "label":      int(pred)
        })
    return jsonify({"results": results})


@app.route("/correct", methods=["POST"])
def correct():
    data = request.json
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No input text provided"}), 400

    try:
        # Step 1 — LanguageTool
        matches   = lt.check(text)
        corrected = language_tool_python.utils.correct(text, matches)

        # Step 2 — Apply all rule-based fixes
        corrected = apply_all_fixes(text, corrected)

        # Step 3 — No change detected
        if corrected.strip() == text.strip():
            return jsonify({
                "correction": text,
                "note": "No automatic correction available for this error type"
            })

        return jsonify({"correction": corrected})

    except Exception as e:
        return jsonify({"error": f"Correction failed: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":     "ok",
        "model":      "RoBERTa fine-tuned",
        "correction": "LanguageTool + rule-based fixes (free)",
        "device":     str(device),
        "fixes":      ["tense_context", "intervening_np", "double_negative", "article_errors"]
    })


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message":   "Grammar Error Detection API (Free)",
        "endpoints": ["/predict", "/predict_batch", "/correct", "/health"]
    })


# ─────────────────────────────────────────────
# 6. Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 Starting Grammar Error Detection API...")
    print("   Local URL : http://127.0.0.1:5000")
    print("   Health    : http://127.0.0.1:5000/health\n")
    app.run(debug=False, host="0.0.0.0", port=5000)