"""
SafeBrief AI — Model Training Script
=====================================
Run this script ONCE to train 3 ML models on OSHA data
and save them as .pkl files for the app to load.

Usage:
    python train_models.py
"""

import os
import json
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ─── CONFIG ──────────────────────────────────────────────
OSHA_FILE = "OSHA_DATA.csv"
MODELS_DIR = "models"

# ─── CREATE MODELS DIRECTORY ─────────────────────────────
os.makedirs(MODELS_DIR, exist_ok=True)


def load_training_data():
    """Load OSHA dataset and prepare text + labels."""

    if not os.path.exists(OSHA_FILE):
        print(f"[ERROR] {OSHA_FILE} not found!")
        print("Please place OSHA_DATA.csv in the project root directory.")
        return None, None

    print(f"[INFO] Loading {OSHA_FILE}...")
    df = pd.read_csv(OSHA_FILE)
    print(f"[INFO] Dataset shape: {df.shape}")

    # ── BUILD TEXT COLUMN ────────────────────────────────
    df["text"] = (
        df["Abstract Text"].fillna("") + " " +
        df["Event Description"].fillna("") + " " +
        df["Event Keywords"].fillna("") + " " +
        df["Nature of Injury"].fillna("") + " " +
        df["Part of Body"].fillna("") + " " +
        df["Event type"].fillna("")
    )

    # ── MAP RISK LABELS ──────────────────────────────────
    def map_risk(value):
        value = str(value).strip().lower()
        if value == "fatal":
            return "HIGH"
        elif value == "nonfatal":
            return "MEDIUM"
        else:
            return "LOW"

    df["label"] = df["Degree of Injury"].apply(map_risk)

    # Drop empty text rows
    df = df[df["text"].str.strip() != ""]

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    print(f"[INFO] Total records: {len(texts)}")
    print(f"[INFO] Label distribution: {pd.Series(labels).value_counts().to_dict()}")

    return texts, labels


def train_and_save():
    """Train 3 ML models, evaluate, and save as .pkl files."""

    # ── LOAD DATA ────────────────────────────────────────
    texts, labels = load_training_data()

    if texts is None:
        return

    # ── TF-IDF VECTORIZATION ─────────────────────────────
    print("\n[INFO] Building TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )
    X = vectorizer.fit_transform(texts)
    print(f"[INFO] Feature matrix shape: {X.shape}")

    # ── TRAIN/TEST SPLIT ─────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"[INFO] Training samples: {X_train.shape[0]}")
    print(f"[INFO] Testing samples:  {X_test.shape[0]}")

    # ── DEFINE MODELS ────────────────────────────────────
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42)
    }

    accuracy_scores = {}

    # ── TRAIN & EVALUATE EACH MODEL ──────────────────────
    for name, model in models.items():

        display_name = name.replace("_", " ").title()
        print(f"\n{'='*60}")
        print(f"  Training: {display_name}")
        print(f"{'='*60}")

        # Train on training set
        model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracy_scores[name] = round(acc * 100, 2)

        print(f"\n  Accuracy: {acc*100:.2f}%\n")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Re-train on FULL dataset for production use
        model.fit(X, labels)

        # Save model
        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f"  [SAVED] {model_path}")

    # ── SAVE VECTORIZER ──────────────────────────────────
    vectorizer_path = os.path.join(MODELS_DIR, "vectorizer.pkl")
    joblib.dump(vectorizer, vectorizer_path)
    print(f"\n  [SAVED] {vectorizer_path}")

    # ── SAVE ACCURACY SCORES ─────────────────────────────
    accuracy_path = os.path.join(MODELS_DIR, "accuracy.json")
    with open(accuracy_path, "w") as f:
        json.dump(accuracy_scores, f, indent=4)
    print(f"  [SAVED] {accuracy_path}")

    # ── SUMMARY ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Models saved in '{MODELS_DIR}/' directory:\n")
    for file in os.listdir(MODELS_DIR):
        size = os.path.getsize(os.path.join(MODELS_DIR, file))
        print(f"    - {file} ({size/1024:.1f} KB)")

    print(f"\n  Accuracy Summary:")
    for name, acc in accuracy_scores.items():
        display_name = name.replace("_", " ").title()
        print(f"    * {display_name}: {acc}%")

    print(f"\n  You can now run: streamlit run app.py\n")


if __name__ == "__main__":
    train_and_save()
