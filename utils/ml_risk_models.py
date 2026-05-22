import os
import json
import joblib

# ─── PATHS ───────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")
LR_PATH = os.path.join(MODELS_DIR, "logistic_regression.pkl")
RF_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")
DT_PATH = os.path.join(MODELS_DIR, "decision_tree.pkl")
ACCURACY_PATH = os.path.join(MODELS_DIR, "accuracy.json")


# ─── LOAD MODELS ─────────────────────────────────────────
def _load_models():
    """Load pre-trained models from .pkl files."""

    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            "Models not found! Please run 'python train_models.py' first "
            "to train and save the models."
        )

    vectorizer = joblib.load(VECTORIZER_PATH)
    lr_model = joblib.load(LR_PATH)
    rf_model = joblib.load(RF_PATH)
    dt_model = joblib.load(DT_PATH)

    return vectorizer, lr_model, rf_model, dt_model


vectorizer, lr_model, rf_model, dt_model = _load_models()


# ─── GET ACCURACY SCORES ────────────────────────────────
def get_accuracy():
    """Return accuracy scores from training evaluation."""

    if not os.path.exists(ACCURACY_PATH):
        return {
            "logistic_regression": "N/A",
            "random_forest": "N/A",
            "decision_tree": "N/A"
        }

    with open(ACCURACY_PATH, "r") as f:
        return json.load(f)


# ─── PREDICT RISK ────────────────────────────────────────
def predict_risk_all(text):
    """Predict risk level using all 3 models with majority vote."""

    input_data = vectorizer.transform([text])

    lr_pred = lr_model.predict(input_data)[0]
    rf_pred = rf_model.predict(input_data)[0]
    dt_pred = dt_model.predict(input_data)[0]

    predictions = [lr_pred, rf_pred, dt_pred]

    final_prediction = max(
        set(predictions),
        key=predictions.count
    )

    return {
        "Logistic Regression": lr_pred,
        "Random Forest": rf_pred,
        "Decision Tree": dt_pred,
        "Final Risk": final_prediction
    }