"""
Model training module for Suicide Ideation Detection
Implements Logistic Regression (TF-IDF) baseline as described in the project methodology.
"""

import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from utils.preprocessor import preprocess_text

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'lr_model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')


SUICIDAL_SAMPLES = [
    "I want to kill myself, I can't take this pain anymore",
    "Life is not worth living, I wish I was dead",
    "I've been thinking about ending it all for weeks now",
    "Nobody would even notice if I disappeared forever",
    "I feel so hopeless, there is no point in continuing",
    "I have a plan to end my life tonight",
    "The pain is too much, I just want to stop existing",
    "I wrote a goodbye letter to my family today",
    "I don't want to wake up tomorrow morning",
    "Everyone would be better off without me around",
    "I've been researching painless ways to die",
    "There is nothing left for me in this world",
    "I feel completely worthless and want to disappear",
    "I can not bear this emptiness anymore, I want to die",
    "I have been cutting myself and it feels like relief",
    "No one cares if I live or die, so why continue",
    "I took all the pills in the cabinet, feeling sleepy",
    "This is my last post, goodbye everyone forever",
    "I stood on the bridge today thinking about jumping",
    "The darkness is too heavy, I can't fight anymore",
    "I hate myself so much I want to stop existing",
    "I've given away all my belongings, I'm ready to go",
    "I feel trapped with no way out except death",
    "My life is a burden on everyone around me",
    "I can see no future for myself, it all ends tonight",
    "I've stopped eating and don't care what happens to me",
    "The only thing that keeps me going is planning my end",
    "I feel completely numb and dead inside already",
    "I am writing this because I won't be here much longer",
    "I wish someone would just let me die in peace",
    "Every day I wake up disappointed that I'm still alive",
    "I've been hiding a stash of pills for the right moment",
    "My family would be happier without me, I'm sure of it",
    "I've said goodbye to everyone I care about already",
    "I can't stop thinking about hanging myself",
    "I have lost all will to live or do anything",
    "Life stopped making sense, death seems like the answer",
    "I am a failure and the world is better off without me",
    "Tonight I will finally end all this suffering",
    "I've been crying for days and decided enough is enough",
    "No one would miss me, I'm just going to disappear",
    "I fantasize about dying every single day",
    "I just want eternal sleep, I'm so tired of living",
    "The thought of suicide brings me comfort and relief",
    "I am not afraid of death, I welcome it",
    "Why should I keep living when nothing gets better",
    "I am ready to meet death, my bags are packed",
    "I've been giving away everything because I won't need them",
    "This is the last night I will ever spend alive",
    "I don't see the point of waking up anymore",
]

NON_SUICIDAL_SAMPLES = [
    "I had a great day at the park with my friends and family",
    "Feeling a bit tired today but excited for the weekend",
    "I made a delicious dinner tonight, really proud of myself",
    "Just finished a long run, feeling energetic and alive",
    "The weather is beautiful today, loving every moment",
    "I got promoted at work today, feeling very happy",
    "Spending quality time with my dog always makes me smile",
    "I've been learning to play guitar and really enjoying it",
    "College life is stressful but I'm handling it well",
    "I feel a bit down today but I know it will pass soon",
    "Watched a great movie last night with my roommates",
    "I sometimes feel overwhelmed with work but I manage",
    "Today was hard but I believe tomorrow will be better",
    "I need to talk to someone about my stress levels lately",
    "Going through a difficult breakup but healing slowly",
    "I feel lonely sometimes but I know I have people who care",
    "Work has been tough this week but the weekend is near",
    "I've been dealing with anxiety but therapy really helps",
    "Had a fight with my friend but we talked it through",
    "I am feeling a little sad about missing home",
    "Just started meditation and it really calms my mind",
    "I had a really bad day but I'm going to sleep it off",
    "Struggling with exams but I believe in myself",
    "I feel unmotivated sometimes but exercise always helps",
    "Lost my job but I am actively looking for new opportunities",
    "I cried today watching a sad movie, felt good to release",
    "I've been feeling stressed but my friends are supportive",
    "Just had a difficult conversation but it was necessary",
    "Dealing with family issues but staying strong",
    "I get anxious in social situations but I'm working on it",
    "Today was exhausting but I am grateful for what I have",
    "Had a rough morning but the afternoon got much better",
    "Feeling low energy but taking care of my health",
    "I've been through a lot but I keep pushing forward",
    "Sometimes life feels hard but I find reasons to smile",
    "I had a panic attack but it passed and I am okay now",
    "My mental health journey has been tough but worth it",
    "I feel frustrated with my progress but I won't give up",
    "Going through grief is hard but I have support around me",
    "I get sad sometimes but I know it will get better",
    "I feel misunderstood but I keep communicating my feelings",
    "School is really hard this semester but I am managing",
    "I've been isolating a bit but I am slowly reconnecting",
    "I had negative thoughts today but challenged them",
    "Feeling burnt out but taking steps to rest and recover",
    "I miss my old life but I am adjusting to the new normal",
    "I get overwhelmed sometimes but breathing exercises help",
    "Today I chose to focus on the positive things in my life",
    "I struggle with self-doubt but my therapist is helping",
    "I feel emotional today but that is okay and valid",
]


def train_model():
    """Train the Logistic Regression model on synthetic data."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    texts = SUICIDAL_SAMPLES + NON_SUICIDAL_SAMPLES
    labels = [1] * len(SUICIDAL_SAMPLES) + [0] * len(NON_SUICIDAL_SAMPLES)

    processed = [preprocess_text(t) for t in texts]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(processed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # NOTE: no multi_class parameter — removed in sklearn 1.5+
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver='lbfgs',
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
        'precision': round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        'recall': round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        'f1_score': round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)

    return model, vectorizer, metrics


def load_model():
    """
    Load persisted model. Auto-retrains if:
    - pkl files don't exist
    - pkl is incompatible with current sklearn version (e.g. after upgrade)
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("No saved model — training fresh...")
        return train_model()

    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)

        # Smoke test — catches 'multi_class' and other version-mismatch errors
        test_vec = vectorizer.transform(["test"])
        model.predict(test_vec)
        model.predict_proba(test_vec)

        return model, vectorizer, None

    except Exception as e:
        print(f"Saved model incompatible ({e}) — retraining fresh...")
        for path in [MODEL_PATH, VECTORIZER_PATH]:
            try:
                os.remove(path)
            except OSError:
                pass
        return train_model()


def predict(text, model, vectorizer):
    """Predict suicidal/non-suicidal. Returns label, probabilities, risk level."""
    processed = preprocess_text(text)
    vec = vectorizer.transform([processed])
    prediction = int(model.predict(vec)[0])
    probabilities = model.predict_proba(vec)[0]

    suicidal_prob = round(float(probabilities[1]) * 100, 2)
    non_suicidal_prob = round(float(probabilities[0]) * 100, 2)
    confidence = round(float(probabilities[prediction]) * 100, 2)

    if suicidal_prob >= 75:
        risk_level, risk_color = 'HIGH', 'danger'
    elif suicidal_prob >= 45:
        risk_level, risk_color = 'MODERATE', 'warning'
    else:
        risk_level, risk_color = 'LOW', 'success'

    return {
        'prediction': prediction,
        'label': 'Suicidal' if prediction == 1 else 'Non-Suicidal',
        'confidence': confidence,
        'suicidal_probability': suicidal_prob,
        'non_suicidal_probability': non_suicidal_prob,
        'risk_level': risk_level,
        'risk_color': risk_color,
        'processed_text': processed,
    }


EXPECTED_METRICS = {
    'lr_model': {
        'name': 'Logistic Regression (TF-IDF)',
        'accuracy': 83.5,
        'precision': 84.2,
        'recall': 82.8,
        'f1_score': 83.5,
        'description': 'Baseline classical ML model'
    },
    'lstm_model': {
        'name': 'LSTM Deep Learning (GloVe)',
        'accuracy': 91.8,
        'precision': 92.3,
        'recall': 91.4,
        'f1_score': 91.8,
        'description': 'Advanced deep learning model'
    }
}
