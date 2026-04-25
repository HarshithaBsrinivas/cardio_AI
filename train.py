"""
Cardio AI Intelligence Dashboard
=================================
train.py — Data generation, model training, evaluation & saving.
Run this first before launching app.py.
 
    python train.py
"""
 
import os
import pickle
import numpy as np
import pandas as pd
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
 
# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
SEED = 42
MODEL_DIR  = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
 
FEATURES = [
    "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
    "fasting_blood_sugar", "rest_ecg", "max_heart_rate",
    "exercise_angina", "st_depression", "st_slope", "num_vessels", "thal",
]
 
FEATURE_LABELS = {
    "age":                "Age (years)",
    "sex":                "Sex",
    "chest_pain_type":    "Chest Pain Type",
    "resting_bp":         "Resting Blood Pressure",
    "cholesterol":        "Serum Cholesterol",
    "fasting_blood_sugar":"Fasting Blood Sugar",
    "rest_ecg":           "Resting ECG",
    "max_heart_rate":     "Max Heart Rate",
    "exercise_angina":    "Exercise Angina",
    "st_depression":      "ST Depression",
    "st_slope":           "ST Slope",
    "num_vessels":        "No. of Vessels",
    "thal":               "Thalassemia",
}
 
 
# ─────────────────────────────────────────────
#  DATA GENERATION
# ─────────────────────────────────────────────
def generate_data(n: int = 1200, seed: int = SEED) -> pd.DataFrame:
    """
    Generate a realistic synthetic heart-disease dataset.
 
    Uses a logistic (sigmoid) model with clinically motivated coefficients
    plus Gaussian noise and a stochastic binomial flip so the target is
    probabilistic — not a hard deterministic threshold.
    """
    rng = np.random.default_rng(seed)
 
    age     = rng.integers(20,  90,  n)
    sex     = rng.integers(0,   2,   n)
    cp      = rng.integers(0,   4,   n)
    bp      = rng.integers(80,  200, n)
    chol    = rng.integers(120, 600, n)
    fbs     = rng.integers(0,   2,   n)
    ecg     = rng.integers(0,   3,   n)
    hr      = rng.integers(70,  210, n)
    ex      = rng.integers(0,   2,   n)
    oldpeak = rng.uniform(0, 6.2, n).round(1)
    slope   = rng.integers(0, 3, n)
    ca      = rng.integers(0, 4, n)
    thal    = rng.integers(0, 3, n)
 
    logit = (
        -5.5
        + 0.045 * (age  - 55)
        + 0.55  *  sex
        + 0.70  * (cp == 3)
        - 0.40  * (cp == 0)
        + 0.012 * (bp   - 130)
        + 0.003 * (chol - 245)
        + 0.35  *  fbs
        + 0.20  *  ecg
        - 0.030 * (hr  - 150)
        + 0.75  *  ex
        + 0.50  *  oldpeak
        - 0.55  * (slope == 0)
        + 0.70  * (slope == 2)
        + 0.62  *  ca
        + 0.45  *  thal
        + rng.normal(0, 0.6, n)
    )
    prob   = 1 / (1 + np.exp(-logit))
    target = rng.binomial(1, prob)
 
    return pd.DataFrame({
        "age": age,            "sex": sex,
        "chest_pain_type": cp, "resting_bp": bp,
        "cholesterol": chol,   "fasting_blood_sugar": fbs,
        "rest_ecg": ecg,       "max_heart_rate": hr,
        "exercise_angina": ex, "st_depression": oldpeak,
        "st_slope": slope,     "num_vessels": ca,
        "thal": thal,          "target": target,
    })
 
 
# ─────────────────────────────────────────────
#  PIPELINE
# ─────────────────────────────────────────────
def build_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   RandomForestClassifier(
            n_estimators      = 300,
            max_depth         = 12,
            min_samples_split = 5,
            min_samples_leaf  = 2,
            max_features      = "sqrt",
            class_weight      = "balanced",
            random_state      = SEED,
            n_jobs            = -1,
        )),
    ])
 
 
# ─────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────
def _sep(title: str = "") -> None:
    print(f"\n{'─'*54}")
    if title:
        print(f"  {title}")
        print(f"{'─'*54}")
 
 
def evaluate(pipeline: Pipeline,
             X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
 
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm  = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
 
    _sep("Hold-out Test Set Metrics")
    print(f"  Accuracy            : {acc:.4f}")
    print(f"  ROC-AUC             : {auc:.4f}")
    print(f"  Sensitivity (Recall): {sensitivity:.4f}")
    print(f"  Specificity         : {specificity:.4f}")
 
    _sep("Confusion Matrix   [Pred → No Disease | Disease]")
    print(f"  True No Disease  :  TN = {tn:>4}   FP = {fp:>4}")
    print(f"  True Disease     :  FN = {fn:>4}   TP = {tp:>4}")
 
    _sep("Classification Report")
    print(classification_report(y_test, y_pred,
                                target_names=["No Disease", "Disease"]))
 
    return {
        "accuracy": acc, "roc_auc": auc,
        "sensitivity": sensitivity, "specificity": specificity,
        "confusion_matrix": cm,
        "y_test": y_test, "y_prob": y_prob,
        "y_pred": y_pred,
    }
 
 
def cross_validate(pipeline: Pipeline,
                   X: pd.DataFrame, y: pd.Series) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    acc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    auc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
 
    _sep("5-Fold Stratified Cross Validation")
    print(f"  Accuracy  folds : {np.round(acc_scores, 4)}")
    print(f"  Accuracy  mean  : {acc_scores.mean():.4f}  ± {acc_scores.std():.4f}")
    print(f"  ROC-AUC   folds : {np.round(auc_scores, 4)}")
    print(f"  ROC-AUC   mean  : {auc_scores.mean():.4f}  ± {auc_scores.std():.4f}")
 
    return {
        "cv_acc":  acc_scores,
        "cv_auc":  auc_scores,
    }
 
 
def print_feature_importance(pipeline: Pipeline,
                              feature_names: list) -> None:
    importances = pipeline.named_steps["model"].feature_importances_
    ranked = sorted(zip(feature_names, importances),
                    key=lambda x: x[1], reverse=True)
    _sep("Feature Importances")
    for rank, (name, imp) in enumerate(ranked, 1):
        bar   = "█" * int(imp * 220)
        label = FEATURE_LABELS.get(name, name)
        print(f"  {rank:>2}. {label:<26} {imp:.4f}  {bar}")
 
 
# ─────────────────────────────────────────────
#  SAVE / LOAD
# ─────────────────────────────────────────────
def save_model(pipeline: Pipeline, feature_names: list,
               metrics: dict, cv_metrics: dict) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    artefact = {
        "pipeline":      pipeline,
        "feature_names": feature_names,
        "metrics":       metrics,
        "cv_metrics":    cv_metrics,
        "seed":          SEED,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artefact, f, protocol=pickle.HIGHEST_PROTOCOL)
 
    size_kb = os.path.getsize(MODEL_PATH) / 1024
    _sep("Model Saved")
    print(f"  Path : {os.path.abspath(MODEL_PATH)}")
    print(f"  Size : {size_kb:.1f} KB")
 
 
def load_model(path: str = MODEL_PATH) -> dict:
    """Load artefact saved by train.py. Returns full dict."""
    with open(path, "rb") as f:
        return pickle.load(f)
 
 
# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 54)
    print("   Cardio AI — Model Training Pipeline")
    print("=" * 54)
 
    # 1 — Data
    print("\n[1/4]  Generating synthetic dataset …")
    df   = generate_data(n=1200)
    X, y = df[FEATURES], df["target"]
    print(f"       Samples : {len(df)}  |  "
          f"Disease prevalence : {y.mean():.1%}")
 
    # 2 — Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    print(f"       Train : {len(X_train)}  |  Test : {len(X_test)}")
 
    # 3 — Train
    print("\n[2/4]  Training Random Forest pipeline …")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("       Done.")
 
    # 4 — Evaluate
    print("\n[3/4]  Evaluating …")
    metrics    = evaluate(pipeline, X_test, y_test)
    cv_metrics = cross_validate(pipeline, X, y)
    print_feature_importance(pipeline, FEATURES)
 
    # 5 — Save
    print("\n[4/4]  Saving artefact …")
    save_model(pipeline, FEATURES, metrics, cv_metrics)
 
    print("\n  ✅  Training complete. Run  streamlit run app.py  next.\n")
 
