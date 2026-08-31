import os
import joblib
import numpy as np
from data_utils import clean_text, normalize_input
from urgency import classify_urgency

_pipeline = None
_metadata = None

# Absolute path to project root — works locally and in Vercel serverless
_ROOT = os.path.dirname(os.path.abspath(__file__))

def _load():
    global _pipeline, _metadata
    if _pipeline is None:
        _pipeline = joblib.load(os.path.join(_ROOT, 'models', 'best_model.pkl'))
        _metadata = joblib.load(os.path.join(_ROOT, 'models', 'metadata.pkl'))
        print(f"[predict] Model loaded: {_metadata['best_model_name']}")
        print(f"[predict] Categories: {_metadata['labels']}")

def predict_ticket(raw_text: str) -> dict:
    _load()
    cleaned = clean_text(normalize_input(raw_text))

    category = _pipeline.predict([cleaned])[0]

    # All our models now support predict_proba (LinearSVC is wrapped in
    # CalibratedClassifierCV so it also has predict_proba)
    try:
        proba = _pipeline.predict_proba([cleaned])[0]
        classes = _pipeline.classes_
    except AttributeError:
        # Fallback: softmax over decision function
        dec = _pipeline.decision_function([cleaned])[0]
        classes = _pipeline.classes_
        e = np.exp(dec - np.max(dec))
        proba = e / e.sum()

    all_scores = {
        c: round(float(p), 4)
        for c, p in sorted(zip(classes, proba), key=lambda x: -x[1])
    }
    confidence = round(float(max(proba)), 4)
    urgency, reason = classify_urgency(raw_text)

    return {
        'category': category,
        'confidence': confidence,
        'all_scores': all_scores,
        'urgency': urgency,
        'urgency_reason': reason,
        'model_name': _metadata['best_model_name'],
        'accuracy': round(_metadata.get('accuracy', 0), 4),
        'f1_weighted': round(_metadata.get('f1_weighted', 0), 4),
    }
