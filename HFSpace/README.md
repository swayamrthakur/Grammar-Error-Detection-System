---
title: Grammar Error Detection Api
emoji: 📝
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Grammar Error Detection API

A RoBERTa-based grammar error detection API with LanguageTool correction.

## Endpoints

- `POST /predict` — single sentence grammar check
- `POST /predict_batch` — multiple sentences
- `POST /correct` — grammar correction
- `GET /health` — status check
