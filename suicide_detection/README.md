# MindScan AI — Suicide Ideation Detection System
## Deep Learning-Based Detection of Suicidal Tendencies From Social Media Content

**Project-I Report | B.Tech CSE | CGC College of Engineering, Landran**
**Rahul Kumar Rana | Roll No: 2320128 | Supervisor: Ms. Seema Rani**

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Flask app
python app.py

# 3. Open browser at:
http://localhost:5000
```

---

## Project Structure

```
suicide_detection/
├── app.py                  # Flask application (main entry point)
├── requirements.txt
├── models/                 # Saved trained models (auto-created)
│   ├── lr_model.pkl
│   └── tfidf_vectorizer.pkl
├── utils/
│   ├── preprocessor.py     # NLP text preprocessing pipeline
│   └── model_utils.py      # Model training, loading, prediction
└── templates/
    ├── base.html           # Base layout with navbar
    ├── index.html          # Home page with quick demo
    ├── analyze.html        # Full text analysis interface
    ├── dashboard.html      # Charts and model metrics
    ├── methodology.html    # Project methodology (Chapter 7)
    └── about.html          # Project info and team
```

---

## Features

- **Text Analysis**: Classify any social media post as Suicidal / Non-Suicidal
- **Risk Levels**: HIGH / MODERATE / LOW with confidence scores
- **Linguistic Features**: Hopelessness score, negative emotion score, self-referential language
- **Dashboard**: Model performance charts, confusion matrix, analysis history
- **Methodology**: Full pipeline documentation from the project report
- **Retrain**: Retrain the model via the dashboard

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Home page |
| GET | `/analyze` | Text analysis interface |
| POST | `/api/predict` | JSON prediction API |
| GET | `/dashboard` | Model dashboard |
| POST | `/api/retrain` | Retrain model |
| GET | `/methodology` | Methodology page |
| GET | `/about` | About page |

### Prediction API

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel so hopeless and don't want to live anymore"}'
```

Response:
```json
{
  "prediction": 1,
  "label": "Suicidal",
  "risk_level": "HIGH",
  "suicidal_probability": 89.3,
  "non_suicidal_probability": 10.7,
  "confidence": 89.3,
  "linguistic_features": { ... }
}
```

---

## Model Performance (Expected — Chapter 9)

| Model | Accuracy | F1-Score | Method |
|-------|----------|----------|--------|
| Logistic Regression | 82–85% | 0.80–0.83 | TF-IDF |
| LSTM (Deep Learning) | 90–93% | 0.88–0.92 | GloVe |

---

## ⚠️ Disclaimer

This is an **academic research prototype only**. It is NOT a clinical diagnostic tool
and must NOT replace professional psychological assessment.

**Crisis support (India):**
- iCall: 9152987821
- Vandrevala Foundation: 1860-2662-345 (24/7)
