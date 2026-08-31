import os, joblib, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
from data_utils import load_and_clean, get_natural_examples_df

print("="*60)
print("SUPPORT TICKET CLASSIFIER — TRAINING PIPELINE")
print("="*60)

# === LOAD DATA ===
print("\n[1/6] Loading and cleaning data...")
df = load_and_clean('data/tickets.csv')

# Mix in casually-typed natural-language examples so the model has actually
# seen informal phrasing, not just the formal/lemmatized corpus text — this
# is what closes the real-world generalization gap (see generalization test
# at the end of this script).
natural_df = get_natural_examples_df()
print(f"Adding {len(natural_df)} natural-language augmentation examples...")
df = pd.concat([df, natural_df], ignore_index=True)
print(f"Combined data: {df.shape[0]} rows")

X = df['text_clean']
y = df['label']
labels = sorted(df['label'].unique())
print(f"Categories: {labels}")

# === SPLIT ===
print("\n[2/6] Splitting data (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

# === TFIDF PARAMS (tuned for generalization, not just held-out accuracy) ===
# Lower ngram ceiling + min_df=1 keeps rare casual words (from the natural
# augmentation examples) in the vocabulary instead of pruning them as noise.
tfidf = {
    'ngram_range': (1, 2),
    'max_features': 40000,
    'sublinear_tf': True,
    'min_df': 1,
    'max_df': 0.90,
    'strip_accents': 'unicode',
    'analyzer': 'word',
    'token_pattern': r'(?u)\b[a-z][a-z]+\b',
}

# === DEFINE MODELS ===
print("\n[3/6] Defining 3 model pipelines...")
models = {}

# Logistic Regression — usually best on text
models['Logistic Regression'] = Pipeline([
    ('tfidf', TfidfVectorizer(**tfidf)),
    ('clf', LogisticRegression(
        C=3.0,
        max_iter=2000,
        solver='lbfgs',
        multi_class='multinomial',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    ))
])

# Naive Bayes — fast baseline
models['Naive Bayes'] = Pipeline([
    ('tfidf', TfidfVectorizer(**tfidf)),
    ('clf', MultinomialNB(alpha=0.05))
])

# LinearSVC wrapped in CalibratedClassifierCV to get predict_proba
models['Linear SVM'] = Pipeline([
    ('tfidf', TfidfVectorizer(**tfidf)),
    ('clf', CalibratedClassifierCV(
        LinearSVC(C=1.5, max_iter=3000, random_state=42),
        cv=3
    ))
])

# === TRAIN AND EVALUATE ===
print("\n[4/6] Training all 3 models...\n")
results = {}

for name, pipeline in models.items():
    print(f"  Training: {name}...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    f1m  = f1_score(y_test, y_pred, average='macro',    zero_division=0)
    f1w  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    results[name] = dict(
        accuracy=acc, f1_macro=f1m, f1_weighted=f1w,
        pipeline=pipeline, y_pred=y_pred
    )
    print(f"    Accuracy:    {acc:.4f} ({acc*100:.2f}%)")
    print(f"    F1 Macro:    {f1m:.4f}")
    print(f"    F1 Weighted: {f1w:.4f}")
    print(f"    Per-class report:")
    print(classification_report(y_test, y_pred,
          target_names=labels, zero_division=0))

# === PICK BEST ===
print("\n[5/6] Selecting best model...")
best_name = max(results, key=lambda k: results[k]['f1_weighted'])
best = results[best_name]
print(f"\n{'='*60}")
print(f"BEST MODEL: {best_name}")
print(f"  Accuracy:    {best['accuracy']:.4f} ({best['accuracy']*100:.2f}%)")
print(f"  F1 Macro:    {best['f1_macro']:.4f}")
print(f"  F1 Weighted: {best['f1_weighted']:.4f}")
print(f"{'='*60}")

# === SAVE CONFUSION MATRIX ===
print("\n[6/6] Saving artefacts...")
os.makedirs('assets', exist_ok=True)
cm = confusion_matrix(y_test, best['y_pred'], labels=labels)

fig, ax = plt.subplots(figsize=(max(8, len(labels)*1.8), max(6, len(labels)*1.5)))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=labels, yticklabels=labels,
    ax=ax, linewidths=0.5, linecolor='white',
    cbar_kws={'shrink': 0.8}
)
ax.set_xlabel('Predicted Label', fontsize=12, labelpad=10)
ax.set_ylabel('True Label', fontsize=12, labelpad=10)
ax.set_title(f'Confusion Matrix — {best_name}\nAccuracy: {best["accuracy"]*100:.2f}%',
             fontsize=14, pad=15)
plt.xticks(rotation=40, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('assets/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: assets/confusion_matrix.png")

# === SAVE COMPARISON TABLE ===
comp = pd.DataFrame([
    {
        'Model': k,
        'Accuracy': v['accuracy'],
        'Accuracy_pct': f"{v['accuracy']*100:.2f}%",
        'F1_Macro': v['f1_macro'],
        'F1_Weighted': v['f1_weighted'],
    }
    for k, v in results.items()
]).sort_values('F1_Weighted', ascending=False)

comp.to_csv('assets/model_comparison.csv', index=False)
print("  Saved: assets/model_comparison.csv")
print(f"\n  Model comparison:")
for _, row in comp.iterrows():
    print(f"    {row['Model']:25s} | "
          f"Acc: {row['Accuracy']*100:.2f}% | "
          f"F1w: {row['F1_Weighted']*100:.2f}%")

# === SAVE MODEL ===
os.makedirs('models', exist_ok=True)
joblib.dump(best['pipeline'], 'models/best_model.pkl')
joblib.dump({
    'best_model_name': best_name,
    'labels': labels,
    'accuracy': best['accuracy'],
    'f1_macro': best['f1_macro'],
    'f1_weighted': best['f1_weighted'],
    'n_train': len(X_train),
    'n_test': len(X_test),
    'categories': labels,
}, 'models/metadata.pkl')
print("  Saved: models/best_model.pkl")
print("  Saved: models/metadata.pkl")
print("\nTRAINING COMPLETE!")

# === GENERALIZATION TEST ===
# Casual/natural-language phrasing mapped to this project's real 8 categories
# (Access, Hardware, HR Support, Storage, Purchase, Administrative rights,
# Internal Project, Miscellaneous) — not held-out rows from the same corpus,
# so this measures real-world usability, not just benchmark accuracy.
print("\n" + "="*60)
print("GENERALIZATION TEST - Casual/Natural Language")
print("="*60)

from predict import predict_ticket

casual_tests = [
    ("Access",                 "pls help my acc is locked cant get in"),
    ("Access",                 "forgot my pwd need it reset asap"),
    ("Access",                 "keep getting invalid login error wtf"),
    ("Hardware",                "laptop screen is busted pls send new one"),
    ("Hardware",                "printer jammed again ugh"),
    ("Hardware",                "my pc wont boot up help"),
    ("HR Support",              "need to request time off next week"),
    ("HR Support",              "want to check my remaining pto days"),
    ("Storage",                 "mailbox full cant get new emails ugh"),
    ("Storage",                 "need more storage on shared drive asap"),
    ("Purchase",                "need to order new laptops for new hires"),
    ("Purchase",                "can i get a quote for a monitor"),
    ("Administrative rights",   "need admin rights on my laptop pls"),
    ("Internal Project",        "pls update the project tracker asap"),
    ("Miscellaneous",           "got a sketchy email looks like phishing"),
]

correct = 0
print(f"\n{'Input':<45} {'Expected':<22} {'Got':<22} {'OK?'}")
print("-"*100)
for expected_cat, text in casual_tests:
    r = predict_ticket(text)
    got = r['category']
    ok = 'PASS' if got == expected_cat else 'FAIL'
    if got == expected_cat:
        correct += 1
    print(f"{text[:44]:<45} {expected_cat:<22} {got:<22} {ok}")

gen_acc = correct / len(casual_tests)
print(f"\nGeneralization accuracy: {correct}/{len(casual_tests)} = {gen_acc*100:.1f}%")
print(f"Training accuracy:        {best['accuracy']*100:.2f}%")
print(f"Gap: {(best['accuracy'] - gen_acc)*100:.1f} percentage points")
if gen_acc >= 0.80:
    print("Generalization looks solid")
elif gen_acc >= 0.65:
    print("Acceptable but could be improved")
else:
    print("Generalization still poor - check training data mix")
print("="*60)
