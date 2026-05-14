"""
Deep Learning-Based Detection of Suicidal Tendencies From Social Media Content
Flask Web Application
Project by: Rahul Kumar Rana (Roll No. 2320128)
CGC College of Engineering, Landran — B.Tech CSE, 6th Semester
Supervisor: Ms. Seema Rani
"""

from flask import Flask, render_template, request, jsonify
import os
import time
import sys

sys.path.insert(0, os.path.dirname(__file__))

from utils.preprocessor import preprocess_text, extract_linguistic_features
from utils.model_utils import load_model, predict, train_model, EXPECTED_METRICS

app = Flask(__name__)
app.secret_key = 'suicide_detection_project_2320128'

# ─── Always return JSON errors, never HTML ───────────────────────────────────
@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': 'Bad request: ' + str(e)}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Server error: ' + str(e)}), 500

# ─── Load model at startup ───────────────────────────────────────────────────
print("Loading model...")
model, vectorizer, train_metrics = load_model()
print("Model loaded successfully.")

# Store analysis history in memory
analysis_history = []


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    return render_template('analyze.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Main prediction endpoint — always returns JSON."""
    try:
        # force=True accepts any content-type; silent=True returns None on parse error
        data = request.get_json(force=True, silent=True) or {}
    except Exception:
        data = {}

    text = (data.get('text') or '').strip()

    if not text:
        return jsonify({'error': 'No text provided'}), 400
    if len(text) < 5:
        return jsonify({'error': 'Text too short for analysis (min 5 chars)'}), 400

    try:
        start_time = time.time()

        result = predict(text, model, vectorizer)
        features = extract_linguistic_features(text)

        elapsed = round((time.time() - start_time) * 1000, 1)

        history_entry = {
            'text_preview': text[:80] + ('...' if len(text) > 80 else ''),
            'label': result['label'],
            'risk_level': result['risk_level'],
            'confidence': result['confidence'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        analysis_history.insert(0, history_entry)
        if len(analysis_history) > 20:
            analysis_history.pop()

        response = {
            **result,
            'linguistic_features': features,
            'processing_time_ms': elapsed,
            'model': 'Logistic Regression + TF-IDF',
            'disclaimer': (
                'This tool is an academic prototype for research purposes only. '
                'It must NOT be used as a substitute for professional mental health assessment. '
                'If you or someone you know is in crisis, please contact iCall: 9152987821 '
                'or Vandrevala Foundation Helpline: 1860-2662-345 (India, 24/7).'
            )
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html',
                           metrics=EXPECTED_METRICS,
                           history=analysis_history)


@app.route('/api/retrain', methods=['POST'])
def retrain():
    global model, vectorizer, train_metrics
    try:
        model, vectorizer, train_metrics = train_model()
        return jsonify({
            'status': 'success',
            'message': 'Model retrained successfully',
            'metrics': train_metrics
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history')
def get_history():
    return jsonify(analysis_history)


@app.route('/methodology')
def methodology():
    return render_template('methodology.html')


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
