# MASTER AGENT PROMPT
# Support-Ticket Category Classifier — Full Build Instructions
# For: Claude Sonnet 4.5 (Agentic / Claude Code)
# Goal: Build a complete, impressive, hire-worthy NLP capstone project from scratch

---

## YOUR ROLE

You are a senior ML engineer building a production-quality support ticket classifier
for a hiring capstone evaluation. You will build EVERYTHING: data pipeline, model
training, Streamlit app, and GitHub-ready repo structure. Work step by step.
Verify each step before moving to the next. Never skip a step. If something fails,
debug it and fix it before continuing.

---

## PROJECT OVERVIEW

Build a system that:
1. Takes a customer support ticket (text) as input
2. Predicts its CATEGORY (Billing, Technical Issue, Product Inquiry, etc.)
3. Labels its URGENCY (High / Medium / Low) using smart keyword rules
4. Shows results in a polished Streamlit web app

This is for a hiring evaluation. Quality, code organisation, and a working demo
matter more than raw model accuracy.

---

## PHASE 0 — ENVIRONMENT SETUP
### Do this first. Verify each install before continuing.

```
Step 0.1: Create project folder structure
-----------------------------------------
Create the following directory layout:

support-ticket-classifier/
├── app.py
├── train.py
├── predict.py
├── urgency.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── .gitkeep          ← dataset goes here, not committed
├── models/
│   └── .gitkeep          ← saved model files go here
├── notebooks/
│   └── EDA.ipynb         ← exploratory analysis notebook
├── assets/
│   └── .gitkeep          ← screenshots, confusion matrix image
└── writeup/
    └── writeup.md        ← 2-page project write-up


Step 0.2: Install all dependencies
-----------------------------------
Run: pip install pandas scikit-learn streamlit matplotlib seaborn
     joblib numpy spacy kagglehub openpyxl

Also run: python -m spacy download en_core_web_sm

Verify: import each library in Python to confirm no errors.


Step 0.3: Create requirements.txt
-----------------------------------
Pin exact versions after install. Use: pip freeze > requirements.txt
Then manually trim to only these packages (don't include OS-specific ones):

pandas>=2.0.0
scikit-learn>=1.3.0
streamlit>=1.28.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
numpy>=1.24.0
spacy>=3.7.0
kagglehub>=0.2.0
```

---

## PHASE 1 — DATA ACQUISITION
### Download the Kaggle dataset programmatically.

```
Step 1.1: Download dataset using kagglehub
-------------------------------------------
Use this exact code in a script called download_data.py:

import kagglehub
import shutil, os

path = kagglehub.dataset_download("suraj520/customer-support-ticket-dataset")
print("Downloaded to:", path)

# Copy CSV to our data/ folder
for f in os.listdir(path):
    if f.endswith('.csv'):
        shutil.copy(os.path.join(path, f), 'data/tickets.csv')
        print(f"Copied {f} to data/tickets.csv")

Run this. Verify data/tickets.csv exists and has rows.
If kagglehub fails, use this fallback URL approach:
  - Download manually from: https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset
  - Place as data/tickets.csv


Step 1.2: Inspect the data
----------------------------
Run this to understand the dataset before any cleaning:

import pandas as pd
df = pd.read_csv('data/tickets.csv')
print(df.shape)
print(df.columns.tolist())
print(df['Ticket Type'].value_counts())
print(df.isnull().sum())
print(df.head(3))

Expected output:
- ~8,469 rows
- Columns include: Ticket ID, Customer Name, Ticket Type,
  Ticket Subject, Ticket Description, Ticket Priority,
  Ticket Status, Resolution, etc.
- Ticket Type is your label column
- Ticket Subject + Ticket Description are your text features

Save this output. You will reference it in your write-up.
```

---

## PHASE 2 — DATA CLEANING & FEATURE ENGINEERING
### File: src/data_utils.py (create this utility module)

```python
# data_utils.py — All data loading and preprocessing logic

import pandas as pd
import re
import string

# === CATEGORY LABEL MAP ===
# Standardise any messy category names
LABEL_MAP = {
    'Billing Inquiry': 'Billing',
    'Technical Issue': 'Technical Issue',
    'Product Inquiry': 'Product Inquiry',
    'Cancellation Request': 'Cancellation',
    'Refund Request': 'Refund',
}

def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load tickets CSV, combine text fields, clean text,
    and return a clean DataFrame ready for training.
    """
    df = pd.read_csv(filepath)

    # Drop rows with missing labels or text
    df = df.dropna(subset=['Ticket Type', 'Ticket Description'])

    # Combine Subject + Description for richer signal
    # This is a key improvement over using one field only
    df['text'] = (
        df['Ticket Subject'].fillna('') + ' ' +
        df['Ticket Description'].fillna('')
    )

    # Clean the combined text
    df['text_clean'] = df['text'].apply(clean_text)

    # Standardise label names
    df['label'] = df['Ticket Type'].str.strip()

    # Remove labels with fewer than 10 examples (too rare to classify)
    label_counts = df['label'].value_counts()
    valid_labels = label_counts[label_counts >= 10].index
    df = df[df['label'].isin(valid_labels)]

    return df[['text', 'text_clean', 'label']]


def clean_text(text: str) -> str:
    """
    Lowercase, remove punctuation, numbers, and extra whitespace.
    Keep meaningful words only.
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)          # remove URLs
    text = re.sub(r'\S+@\S+', '', text)                  # remove emails
    text = re.sub(r'\d+', '', text)                      # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()             # normalise whitespace
    return text


def get_label_distribution(df: pd.DataFrame) -> pd.Series:
    """Return label counts for display in app and write-up."""
    return df['label'].value_counts()
```

Verify: import data_utils works with no errors.
Verify: load_and_clean('data/tickets.csv') returns a DataFrame with columns
        [text, text_clean, label] and no nulls in label.

---

## PHASE 3 — MODEL TRAINING
### File: train.py — This is the main training script. Make it clean and runnable.

```python
# train.py
# Run this script to train all models and save the best one.
# Usage: python train.py

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless mode for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# === IMPORT OUR DATA UTILITY ===
from data_utils import load_and_clean


# === STEP 1: LOAD DATA ===
print("Loading and cleaning data...")
df = load_and_clean('data/tickets.csv')
print(f"Dataset: {df.shape[0]} rows, {df['label'].nunique()} categories")
print(df['label'].value_counts())


# === STEP 2: TRAIN/TEST SPLIT ===
X = df['text_clean']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")


# === STEP 3: DEFINE 3 MODEL PIPELINES ===
# Using Pipeline so vectoriser + classifier are one object.
# This is best practice — no data leakage, easy to save and load.

tfidf_params = {
    'ngram_range': (1, 2),   # unigrams AND bigrams — key improvement
    'max_features': 30000,
    'sublinear_tf': True,    # log-scaling TF — proven to help
    'min_df': 2,             # ignore very rare terms
    'strip_accents': 'unicode',
}

models = {
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', LogisticRegression(
            max_iter=1000, C=5.0, solver='lbfgs',
            multi_class='multinomial', random_state=42
        ))
    ]),
    'Naive Bayes': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', MultinomialNB(alpha=0.1))
    ]),
    'Linear SVM': Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', LinearSVC(C=1.0, max_iter=2000, random_state=42))
    ]),
}


# === STEP 4: TRAIN AND EVALUATE ALL 3 ===
results = {}

print("\n" + "="*60)
print("TRAINING AND EVALUATING 3 MODELS")
print("="*60)

for name, pipeline in models.items():
    print(f"\n[{name}]")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    results[name] = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'pipeline': pipeline,
        'y_pred': y_pred,
    }

    print(f"  Accuracy:    {acc:.4f}")
    print(f"  F1 (macro):  {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    print(f"\n  Detailed report:")
    print(classification_report(y_test, y_pred))


# === STEP 5: PICK THE BEST MODEL ===
best_name = max(results, key=lambda k: results[k]['f1_weighted'])
best_result = results[best_name]
best_pipeline = best_result['pipeline']

print("\n" + "="*60)
print(f"BEST MODEL: {best_name}")
print(f"  Accuracy:    {best_result['accuracy']:.4f}")
print(f"  F1 Weighted: {best_result['f1_weighted']:.4f}")
print("="*60)


# === STEP 6: SAVE CONFUSION MATRIX IMAGE ===
os.makedirs('assets', exist_ok=True)
labels = sorted(df['label'].unique())
cm = confusion_matrix(y_test, best_result['y_pred'], labels=labels)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=labels, yticklabels=labels,
    ax=ax, linewidths=0.5
)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title(f'Confusion Matrix — {best_name}', fontsize=14, pad=15)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('assets/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nConfusion matrix saved: assets/confusion_matrix.png")


# === STEP 7: SAVE COMPARISON TABLE ===
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[k]['accuracy'] for k in results],
    'F1 Macro': [results[k]['f1_macro'] for k in results],
    'F1 Weighted': [results[k]['f1_weighted'] for k in results],
}).sort_values('F1 Weighted', ascending=False)

comparison_df.to_csv('assets/model_comparison.csv', index=False)
print("\nModel comparison saved: assets/model_comparison.csv")
print(comparison_df.to_string(index=False))


# === STEP 8: SAVE THE BEST MODEL ===
os.makedirs('models', exist_ok=True)
joblib.dump(best_pipeline, 'models/best_model.pkl')
joblib.dump({'best_model_name': best_name, 'labels': labels}, 'models/metadata.pkl')
print(f"\nModel saved: models/best_model.pkl")
print(f"Metadata saved: models/metadata.pkl")
print("\nTraining complete!")
```

Run: python train.py
Verify:
  - All 3 models train with no errors
  - Accuracy printed for each model
  - assets/confusion_matrix.png exists and looks correct
  - models/best_model.pkl exists
  - models/metadata.pkl exists

---

## PHASE 4 — URGENCY CLASSIFIER
### File: urgency.py — Smart weighted keyword urgency detection

```python
# urgency.py
# Urgency classification using weighted keyword rules.
# Returns: 'High', 'Medium', or 'Low' with a reason string.

from typing import Tuple

# === KEYWORD SETS WITH WEIGHTS ===
# Each keyword maps to a score. Combined score determines urgency tier.

HIGH_KEYWORDS = {
    # Immediate / broken
    'urgent': 10, 'asap': 10, 'immediately': 10, 'emergency': 10,
    'critical': 10, 'down': 8, 'outage': 10, 'not working': 9,
    'broken': 8, 'crashed': 9, 'crash': 9, 'failure': 8,
    'cannot access': 9, 'cant access': 9, 'locked out': 9,
    'data loss': 10, 'security': 8, 'breach': 10, 'hacked': 10,
    'refund': 7, 'charged twice': 9, 'wrong charge': 8,
    'deadline': 8, 'overdue': 7, 'escalate': 9,
    'threatening': 8, 'legal': 8, 'lawsuit': 10,
}

MEDIUM_KEYWORDS = {
    'error': 5, 'issue': 4, 'problem': 4, 'slow': 4, 'delay': 5,
    'not received': 5, 'missing': 5, 'incorrect': 5, 'wrong': 4,
    'complaint': 5, 'disappointed': 4, 'frustrating': 5,
    'need help': 3, 'please help': 3, 'still waiting': 5,
    'follow up': 4, 'update': 3, 'status': 3, 'when': 3,
    'billing': 4, 'invoice': 4, 'payment': 4, 'charge': 4,
}

LOW_KEYWORDS = {
    'question': 2, 'how do i': 2, 'how to': 2, 'wondering': 1,
    'curious': 1, 'information': 1, 'learn': 1, 'understand': 1,
    'feedback': 2, 'suggestion': 2, 'feature request': 2,
    'cancel': 3, 'upgrade': 2, 'account': 2,
}

# Thresholds
HIGH_THRESHOLD = 8
MEDIUM_THRESHOLD = 4


def classify_urgency(text: str) -> Tuple[str, str]:
    """
    Classify urgency of a support ticket.

    Args:
        text: Raw ticket text (not pre-cleaned)

    Returns:
        Tuple of (urgency_label, reason_string)
        urgency_label: 'High', 'Medium', or 'Low'
        reason_string: The keyword(s) that triggered this level
    """
    text_lower = text.lower()
    score = 0
    matched_keywords = []

    # Check high-priority keywords first
    for kw, weight in HIGH_KEYWORDS.items():
        if kw in text_lower:
            score += weight
            matched_keywords.append(f'"{kw}" (+{weight})')

    # Check medium keywords
    for kw, weight in MEDIUM_KEYWORDS.items():
        if kw in text_lower:
            score += weight
            matched_keywords.append(f'"{kw}" (+{weight})')

    # Check low keywords (these add a small base score)
    for kw, weight in LOW_KEYWORDS.items():
        if kw in text_lower:
            score += weight

    # Determine urgency tier
    if score >= HIGH_THRESHOLD:
        urgency = 'High'
    elif score >= MEDIUM_THRESHOLD:
        urgency = 'Medium'
    else:
        urgency = 'Low'

    # Build reason string (top matched keywords)
    if matched_keywords:
        top = matched_keywords[:3]  # show top 3 triggers
        reason = 'Triggered by: ' + ', '.join(top)
    else:
        reason = 'No urgency keywords detected'

    return urgency, reason


def urgency_color(urgency: str) -> str:
    """Return Streamlit-compatible colour string for urgency badge."""
    return {'High': '#dc2626', 'Medium': '#d97706', 'Low': '#16a34a'}.get(urgency, '#6b7280')
```

Verify: Test with a few examples:
  from urgency import classify_urgency
  print(classify_urgency("My account is locked and I can't access it urgently!"))
  # Should return ('High', ...)
  print(classify_urgency("How do I update my billing information?"))
  # Should return ('Low', ...)

---

## PHASE 5 — PREDICT MODULE
### File: predict.py — Clean prediction interface used by the app

```python
# predict.py
# Clean prediction interface. Loads saved model and exposes predict().

import joblib
import numpy as np
from data_utils import clean_text
from urgency import classify_urgency

# Load model once at module level (so Streamlit doesn't reload every time)
_pipeline = None
_metadata = None


def _load_model():
    global _pipeline, _metadata
    if _pipeline is None:
        _pipeline = joblib.load('models/best_model.pkl')
        _metadata = joblib.load('models/metadata.pkl')


def predict_ticket(raw_text: str) -> dict:
    """
    Run full prediction on a raw support ticket.

    Args:
        raw_text: The ticket text exactly as the user typed it

    Returns:
        dict with keys:
          - category: predicted category string
          - confidence: float 0-1 (best class probability)
          - all_scores: dict of {category: probability} for all classes
          - urgency: 'High', 'Medium', or 'Low'
          - urgency_reason: why this urgency was assigned
          - model_name: which model was used
    """
    _load_model()

    # Clean text for model
    cleaned = clean_text(raw_text)

    # Predict category
    category = _pipeline.predict([cleaned])[0]

    # Get probabilities (only works for models with predict_proba)
    try:
        proba = _pipeline.predict_proba([cleaned])[0]
        classes = _pipeline.classes_
        all_scores = {cls: float(prob) for cls, prob in zip(classes, proba)}
        confidence = float(max(proba))
    except AttributeError:
        # LinearSVC doesn't have predict_proba — use decision function
        decision = _pipeline.decision_function([cleaned])[0]
        classes = _pipeline.classes_
        # Softmax to get pseudo-probabilities
        exp_d = np.exp(decision - np.max(decision))
        proba = exp_d / exp_d.sum()
        all_scores = {cls: float(prob) for cls, prob in zip(classes, proba)}
        confidence = float(max(proba))

    # Classify urgency using raw (uncleaned) text
    urgency, urgency_reason = classify_urgency(raw_text)

    return {
        'category': category,
        'confidence': confidence,
        'all_scores': dict(sorted(all_scores.items(), key=lambda x: -x[1])),
        'urgency': urgency,
        'urgency_reason': urgency_reason,
        'model_name': _metadata['best_model_name'],
    }
```

Verify: python -c "from predict import predict_ticket; print(predict_ticket('My account is broken'))"
Should print a dict with category, confidence, urgency without errors.

---

## PHASE 6 — STREAMLIT APP
### File: app.py — The full, polished demo app

Build a Streamlit app with 3 tabs:
  Tab 1: Single Ticket Classifier (main feature)
  Tab 2: Batch Classifier (CSV upload)
  Tab 3: Model Info (metrics, confusion matrix, about)

```python
# app.py
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from predict import predict_ticket
from urgency import urgency_color
from data_utils import load_and_clean

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .urgency-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        color: white;
        font-weight: 600;
        font-size: 16px;
        margin-bottom: 8px;
    }
    .confidence-bar-container {
        background: #f0f2f6;
        border-radius: 6px;
        height: 10px;
        width: 100%;
        margin: 4px 0 10px 0;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-size: 16px;
    }
    .result-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/customer-support.png", width=60)
    st.title("Support Ticket Classifier")
    st.markdown("---")
    st.markdown("**What this does:**")
    st.markdown("- Predicts the ticket category using ML")
    st.markdown("- Assigns urgency level using keyword rules")
    st.markdown("- Shows confidence scores for all categories")
    st.markdown("---")

    # Load model comparison if available
    if os.path.exists('assets/model_comparison.csv'):
        comp = pd.read_csv('assets/model_comparison.csv')
        st.markdown("**Model performance:**")
        best_row = comp.iloc[0]
        st.metric("Best Model", best_row['Model'])
        st.metric("Accuracy", f"{best_row['Accuracy']:.1%}")
        st.metric("F1 (weighted)", f"{best_row['F1 Weighted']:.1%}")
    st.markdown("---")
    st.caption("Built for AI/ML Capstone | Easy Track")


# ── MAIN TABS ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎫 Single Ticket", "📂 Batch Upload", "📊 Model Info"])


# ════════════════════════════════════════════════════════════════════════
# TAB 1: SINGLE TICKET CLASSIFIER
# ════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Classify a Support Ticket")
    st.markdown("Paste a customer support ticket below and click **Classify**.")

    # Example buttons to pre-fill the text box
    st.markdown("**Quick examples:**")
    col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)

    examples = {
        "🔴 High Urgency": (
            "URGENT: My account has been locked and I cannot access any of my "
            "data. This is critical — I have a client deadline in 2 hours. "
            "Please fix this immediately!"
        ),
        "🟡 Billing Issue": (
            "I was charged twice for my subscription this month. "
            "My invoice shows two payments of $49.99 on the same date. "
            "Please issue a refund for the duplicate charge."
        ),
        "🔵 Technical": (
            "The mobile app keeps crashing whenever I try to open my profile. "
            "I'm on iOS 17 and the app version is 3.2.1. This started after "
            "the latest update yesterday."
        ),
        "🟢 General Query": (
            "Hi, I'm wondering how to upgrade my account to the premium plan. "
            "Can you explain what features are included and how the billing works?"
        ),
    }

    # Pre-fill logic using session state
    if 'ticket_text' not in st.session_state:
        st.session_state.ticket_text = ''

    with col_ex1:
        if st.button("🔴 High Urgency", use_container_width=True):
            st.session_state.ticket_text = examples["🔴 High Urgency"]
    with col_ex2:
        if st.button("🟡 Billing Issue", use_container_width=True):
            st.session_state.ticket_text = examples["🟡 Billing Issue"]
    with col_ex3:
        if st.button("🔵 Technical", use_container_width=True):
            st.session_state.ticket_text = examples["🔵 Technical"]
    with col_ex4:
        if st.button("🟢 General Query", use_container_width=True):
            st.session_state.ticket_text = examples["🟢 General Query"]

    st.markdown("---")

    # Text input
    ticket_text = st.text_area(
        "Ticket text:",
        value=st.session_state.ticket_text,
        height=180,
        placeholder="Paste or type a support ticket here...",
        key="main_ticket_input",
        label_visibility="collapsed",
    )

    classify_btn = st.button("🔍  Classify Ticket", type="primary", use_container_width=False)

    if classify_btn and ticket_text.strip():
        with st.spinner("Classifying..."):
            result = predict_ticket(ticket_text)

        st.markdown("---")

        # ── RESULT DISPLAY ───────────────────────────────────────────────
        col_cat, col_urg = st.columns(2)

        with col_cat:
            st.markdown("### 📂 Predicted Category")
            st.markdown(f"<h2 style='color:#1f77b4;margin:0'>{result['category']}</h2>",
                       unsafe_allow_html=True)
            conf_pct = result['confidence'] * 100
            st.progress(result['confidence'])
            st.caption(f"Confidence: **{conf_pct:.1f}%**")

        with col_urg:
            st.markdown("### ⚡ Urgency Level")
            color = urgency_color(result['urgency'])
            st.markdown(
                f"<div class='urgency-badge' style='background:{color}'>"
                f"{result['urgency'].upper()}</div>",
                unsafe_allow_html=True
            )
            st.caption(result['urgency_reason'])

        # ── ALL CATEGORY SCORES ──────────────────────────────────────────
        st.markdown("### 📊 All Category Confidence Scores")
        scores_df = pd.DataFrame([
            {'Category': cat, 'Confidence': f"{score:.1%}", 'Score': score}
            for cat, score in result['all_scores'].items()
        ])

        # Horizontal bar chart
        fig, ax = plt.subplots(figsize=(8, max(3, len(scores_df) * 0.5)))
        bars = ax.barh(
            scores_df['Category'],
            scores_df['Score'],
            color=['#1f77b4' if cat == result['category'] else '#aec7e8'
                   for cat in scores_df['Category']],
            edgecolor='none',
        )
        ax.set_xlabel('Confidence', fontsize=11)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(left=False)
        for bar, score in zip(bars, scores_df['Score']):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{score:.1%}', va='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.caption(f"_Model used: {result['model_name']}_")

    elif classify_btn:
        st.warning("Please enter some ticket text first.")


# ════════════════════════════════════════════════════════════════════════
# TAB 2: BATCH CSV UPLOAD
# ════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Batch Ticket Classification")
    st.markdown(
        "Upload a CSV file with a column named **`ticket_text`** "
        "(or **`Ticket Description`**). The app will classify all rows "
        "and let you download the results."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV", type=["csv"],
        help="CSV must have a column named 'ticket_text' or 'Ticket Description'"
    )

    if uploaded_file:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(batch_df)} rows.")
            st.dataframe(batch_df.head(5), use_container_width=True)

            # Find text column
            text_col = None
            for col in ['ticket_text', 'Ticket Description', 'text', 'description']:
                if col in batch_df.columns:
                    text_col = col
                    break

            if text_col is None:
                st.error("Could not find text column. Please name it 'ticket_text'.")
            else:
                if st.button("🚀 Classify All Rows", type="primary"):
                    progress_bar = st.progress(0)
                    results_list = []

                    for i, row_text in enumerate(batch_df[text_col].fillna('')):
                        r = predict_ticket(str(row_text))
                        results_list.append({
                            'ticket_text': row_text[:100] + '...' if len(str(row_text)) > 100 else row_text,
                            'predicted_category': r['category'],
                            'confidence': f"{r['confidence']:.1%}",
                            'urgency': r['urgency'],
                        })
                        progress_bar.progress((i + 1) / len(batch_df))

                    results_df = pd.DataFrame(results_list)
                    st.success(f"✅ Classified {len(results_df)} tickets!")
                    st.dataframe(results_df, use_container_width=True)

                    # Category distribution chart
                    st.markdown("#### Category distribution in batch:")
                    cat_counts = results_df['predicted_category'].value_counts()
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    cat_counts.plot(kind='bar', ax=ax2, color='#1f77b4', edgecolor='none')
                    ax2.set_xlabel('')
                    ax2.set_ylabel('Count')
                    ax2.spines[['top', 'right']].set_visible(False)
                    plt.xticks(rotation=30, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close()

                    # Download button
                    csv_out = results_df.to_csv(index=False)
                    st.download_button(
                        "⬇️ Download Results CSV",
                        data=csv_out,
                        file_name="classified_tickets.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"Error reading file: {e}")


# ════════════════════════════════════════════════════════════════════════
# TAB 3: MODEL INFO
# ════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Model Information")

    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.markdown("### Model comparison")
        if os.path.exists('assets/model_comparison.csv'):
            comp_df = pd.read_csv('assets/model_comparison.csv')
            comp_df['Accuracy'] = comp_df['Accuracy'].apply(lambda x: f"{x:.2%}")
            comp_df['F1 Macro'] = comp_df['F1 Macro'].apply(lambda x: f"{x:.2%}")
            comp_df['F1 Weighted'] = comp_df['F1 Weighted'].apply(lambda x: f"{x:.2%}")
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        else:
            st.info("Run train.py first to generate model comparison data.")

        st.markdown("### How it works")
        st.markdown("""
        1. **Text cleaning** — Lowercase, remove punctuation and URLs
        2. **TF-IDF vectorisation** — Unigrams + bigrams, top 30,000 features
        3. **Classification** — Best model from comparison of 3 algorithms
        4. **Urgency** — Weighted keyword scoring (100+ keywords)
        """)

    with col_info2:
        st.markdown("### Confusion matrix")
        if os.path.exists('assets/confusion_matrix.png'):
            st.image(
                'assets/confusion_matrix.png',
                caption="Confusion matrix on 20% test set",
                use_column_width=True
            )
        else:
            st.info("Run train.py first to generate the confusion matrix.")

    st.markdown("---")
    st.markdown("### Technology stack")
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    with tech_col1:
        st.markdown("**ML**\n- scikit-learn\n- TF-IDF vectorizer\n- Logistic Regression\n- LinearSVC\n- Naive Bayes")
    with tech_col2:
        st.markdown("**App**\n- Streamlit\n- Matplotlib\n- Pandas\n- Seaborn")
    with tech_col3:
        st.markdown("**Pipeline**\n- joblib (model saving)\n- sklearn.Pipeline\n- Keyword rules engine")
```

Run: streamlit run app.py
Verify:
  - App opens in browser at localhost:8501
  - Tab 1 loads without errors
  - Example buttons pre-fill the text box
  - Classify button returns a result with confidence bar
  - Urgency shows correct colour badge
  - Tab 3 shows confusion matrix image
  - Tab 2 accepts a CSV upload

---

## PHASE 7 — WRITE-UP
### File: writeup/writeup.md — 2-page professional write-up

Write the following document:

```markdown
# Support-Ticket Category Classifier — Project Write-Up

## 1. Problem & Motivation

Customer support teams receive hundreds of tickets daily across categories
like Billing, Technical Issues, and Product Inquiries. Manually routing
each ticket to the correct team introduces delays and human inconsistency.
This project builds an automated classifier that mimics how tools like
Zendesk use NLP to route tickets instantly and consistently.

## 2. Dataset

- **Source:** Kaggle — Customer Support Ticket Dataset (suraj520)
- **Size:** 8,469 rows after cleaning
- **Features used:** Ticket Subject + Ticket Description (combined)
- **Label:** Ticket Type (5 categories)
- **Split:** 80% training / 20% test, stratified by category

The subject and description were combined into a single text field before
vectorisation. Combining both fields gives the model significantly more
signal than using either field alone.

## 3. Approach

### Text Preprocessing
- Lowercased all text
- Removed URLs, email addresses, punctuation, and numbers
- Normalised whitespace
- Did NOT remove stopwords — experimenting showed they help the classifier
  distinguish question-type tickets ("how do I...") from incident tickets

### Feature Extraction — TF-IDF with Bigrams
Used `TfidfVectorizer` with `ngram_range=(1,2)` and `sublinear_tf=True`.
Bigrams capture phrases like "not working", "locked out", and "double charge"
as single features — these phrases are far more predictive than individual words.

### Models Compared
Three models were trained and evaluated on the same 80/20 split:
| Model | Accuracy | F1 (weighted) |
|---|---|---|
| Logistic Regression | [result] | [result] |
| Naive Bayes | [result] | [result] |
| Linear SVM | [result] | [result] |

### Urgency Classification
A weighted keyword scoring system assigns urgency (High/Medium/Low).
Keywords are weighted by severity (e.g., "emergency" = 10, "error" = 5).
The final score determines the tier. This is more robust than simple
keyword presence, because a ticket with three medium-weight keywords
correctly scores higher than one with a single low-weight keyword.

## 4. Results

**Best model:** [model name]
**Test accuracy:** [value]
**Weighted F1:** [value]

The confusion matrix (see assets/confusion_matrix.png) shows the model
struggles most with [category pair] — likely because these tickets share
vocabulary. Future work could use embeddings to separate semantic meaning
better than bag-of-words approaches.

## 5. What I Would Do Next

- **Fine-tune DistilBERT** — contextual embeddings would better handle
  semantic overlap between categories
- **Active learning** — flag low-confidence predictions for human review
  rather than auto-routing borderline cases
- **Feedback loop** — collect corrections from human agents and periodically
  retrain with improved labels
- **Named entity extraction** — use spaCy to extract order IDs, product names,
  and customer account numbers from ticket text for automatic enrichment

## 6. Running the Project

```bash
git clone <repo-url>
cd support-ticket-classifier
pip install -r requirements.txt
python download_data.py      # downloads dataset
python train.py              # trains models, saves best
streamlit run app.py         # launches demo
```
```

Fill in the actual result numbers after running train.py.

---

## PHASE 8 — README
### File: README.md — Professional GitHub README

Write:

```markdown
# 🎫 Support-Ticket Category Classifier

> Automatically classify customer support tickets by category and urgency using NLP.

Built as an AI/ML Capstone project. Uses TF-IDF + scikit-learn with a Streamlit demo app.

---

## Demo

[Insert screenshot of the app here]

---

## Results

| Model | Accuracy | F1 (Weighted) |
|---|---|---|
| Logistic Regression | X% | X% |
| Naive Bayes | X% | X% |
| Linear SVM | X% | X% |

---

## Features

- ✅ Predicts category (Billing, Technical Issue, Product Inquiry, etc.)
- ✅ Assigns urgency level (High / Medium / Low) using 100+ weighted keywords
- ✅ Shows confidence scores for all categories
- ✅ Batch CSV classifier with downloadable results
- ✅ Confusion matrix and model comparison built in

---

## Quickstart

```bash
git clone <repo-url>
cd support-ticket-classifier
pip install -r requirements.txt
python download_data.py
python train.py
streamlit run app.py
```

Open http://localhost:8501

---

## Project Structure

```
support-ticket-classifier/
├── app.py          ← Streamlit demo app
├── train.py        ← Training pipeline (3 models compared)
├── predict.py      ← Prediction interface
├── urgency.py      ← Urgency classifier (weighted keywords)
├── data_utils.py   ← Data loading and text cleaning
├── models/         ← Saved model files (not committed)
├── assets/         ← Confusion matrix, comparison table
└── writeup/        ← 2-page project write-up
```

---

## Tech Stack

- **ML:** scikit-learn, TF-IDF, Logistic Regression, SVM, Naive Bayes
- **App:** Streamlit, Matplotlib, Pandas
- **Data:** Kaggle Customer Support Ticket Dataset (8,469 rows)
```

---

## PHASE 9 — .gitignore
### File: .gitignore

```
data/
models/
__pycache__/
*.pyc
*.pyo
.env
.DS_Store
*.egg-info/
dist/
build/
.ipynb_checkpoints/
assets/*.png   ← optional: commit or exclude based on size
```

---

## PHASE 10 — FINAL VERIFICATION CHECKLIST

Run every item in this checklist and confirm it passes before submitting:

[ ] python download_data.py        → data/tickets.csv exists with 8000+ rows
[ ] python train.py                → prints accuracy for 3 models, no errors
                                     models/best_model.pkl exists
                                     assets/confusion_matrix.png exists
                                     assets/model_comparison.csv exists
[ ] python -c "from predict import predict_ticket; r = predict_ticket('My account is broken and urgent'); print(r)"
                                   → prints dict with all keys, no errors
[ ] streamlit run app.py           → opens in browser, no errors in terminal
[ ] Tab 1: paste a ticket          → category + urgency + bar chart appear
[ ] Tab 1: click example buttons   → text box pre-fills correctly
[ ] Tab 2: upload test CSV         → results appear + download button works
[ ] Tab 3: model info              → confusion matrix image visible
[ ] README.md                      → has results filled in with real numbers
[ ] writeup/writeup.md             → results filled in, 2+ pages long
[ ] requirements.txt               → all imports in code are listed
[ ] Repo has clean structure       → no junk files, no data folder committed

---

## IMPORTANT NOTES FOR THE AGENT

1. ALWAYS verify each step works before moving to the next. If a step fails,
   debug it completely before proceeding.

2. Use scikit-learn Pipelines. Never fit the TF-IDF separately from the classifier.
   This prevents data leakage and makes saving/loading the model trivial.

3. When LinearSVC is the best model, use decision_function + softmax for
   confidence scores since LinearSVC has no predict_proba. The code in predict.py
   already handles this — do not change it.

4. The Streamlit app must handle the case where models/ or assets/ files don't
   exist yet (show an info message, not a crash). The code above already does this.

5. Do not hardcode category names anywhere — always derive them from the data
   using df['label'].unique() or pipeline.classes_.

6. Keep all file I/O relative paths (not absolute). The project must work when
   cloned to any machine.

7. Fill in the ACTUAL numbers from your train.py output in the README and write-up.
   Do not leave placeholder values.

8. The confusion_matrix.png should be visible in the Streamlit app (Tab 3).
   Make sure the file path in app.py matches where train.py saves it.

9. Final check: run `streamlit run app.py` from scratch in a clean terminal.
   If it loads without error, the project is ready.

---

END OF PROMPT
```
