"""
Cardio AI Intelligence Dashboard
=================================
app.py — Streamlit front-end.

Run after training:
    python train.py
    streamlit run app.py
"""

import io
import os
import pickle
import warnings
from datetime import datetime

import matplotlib.patchesxzc as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import roc_curve

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Cardio AI Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CONSTANTS  (mirror train.py)
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join("model", "model.pkl")
SEED       = 42

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
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');

* { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background: linear-gradient(135deg,#0a0e1a 0%,#0d1528 60%,#0a1020 100%);
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0f1729 0%,#111d35 100%);
    border-right: 1px solid rgba(59,130,246,.15);
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* Hero header */
.hero {
    background: linear-gradient(135deg,rgba(29,78,216,.15),rgba(124,58,237,.08));
    border: 1px solid rgba(59,130,246,.2);
    border-radius: 20px;
    padding: 32px 40px;
    margin-bottom: 24px;
    text-align: center;
}
.hero-title    { font-size:2.4rem; font-weight:700; color:#f1f5f9; margin:0; }
.hero-subtitle { color:#64748b; font-size:1rem; margin-top:6px; }

/* Metric card */
.mcard {
    background: linear-gradient(135deg,rgba(15,23,42,.9),rgba(30,41,59,.6));
    border: 1px solid rgba(59,130,246,.25);
    border-radius: 14px;
    padding: 18px;
    text-align: center;
}
.mcard-val   { font-size:2rem; font-weight:700; color:#60a5fa; }
.mcard-label { font-size:.75rem; color:#64748b; letter-spacing:1px;
               text-transform:uppercase; margin-top:4px; }

/* Risk badges */
.low    { background:linear-gradient(135deg,#064e3b,#065f46); border:1px solid #10b981;
          color:#6ee7b7; padding:7px 18px; border-radius:20px; font-weight:600; display:inline-block; }
.medium { background:linear-gradient(135deg,#451a03,#78350f); border:1px solid #f59e0b;
          color:#fcd34d; padding:7px 18px; border-radius:20px; font-weight:600; display:inline-block; }
.high   { background:linear-gradient(135deg,#450a0a,#7f1d1d); border:1px solid #ef4444;
          color:#fca5a5; padding:7px 18px; border-radius:20px; font-weight:600; display:inline-block; }

/* Section title */
.stitle {
    font-size:1.05rem; font-weight:600; color:#93c5fd;
    border-left:3px solid #3b82f6; padding-left:10px; margin-bottom:12px;
}

/* Card wrapper */
.card {
    background:rgba(15,23,42,.8);
    border:1px solid rgba(59,130,246,.18);
    border-radius:16px; padding:22px; margin:8px 0;
}

/* Insight row */
.insight {
    background:rgba(15,23,42,.7);
    border:1px solid rgba(59,130,246,.14);
    border-radius:10px; padding:9px 14px;
    margin:5px 0; font-size:.87rem; color:#cbd5e1;
}

/* Buttons */
div[data-testid="stButton"] > button {
    background:linear-gradient(135deg,#1d4ed8,#2563eb) !important;
    color:#fff !important; border:none !important;
    border-radius:10px !important; padding:10px 28px !important;
    font-weight:600 !important;
}
div[data-testid="stButton"] > button:hover {
    background:linear-gradient(135deg,#2563eb,#3b82f6) !important;
    box-shadow:0 4px 20px rgba(59,130,246,.4) !important;
}

hr { border-color:rgba(59,130,246,.1); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DARK MATPLOTLIB STYLE
# ─────────────────────────────────────────────
def dark_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "#0a0e1a",
        "axes.facecolor":   "#0f172a",
        "axes.edgecolor":   "#1e3a5f",
        "axes.labelcolor":  "#94a3b8",
        "xtick.color":      "#64748b",
        "ytick.color":      "#64748b",
        "text.color":       "#e2e8f0",
        "grid.color":       "#1e3a5f",
        "grid.linestyle":   "--",
        "grid.alpha":       0.4,
    })


# ─────────────────────────────────────────────
#  LOAD MODEL  (trains inline if pkl missing)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model() -> dict:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    # ── Fallback: train on the fly (mirrors train.py logic) ──────────
    import numpy as _np
    from sklearn.ensemble import RandomForestClassifier as _RF
    from sklearn.impute import SimpleImputer as _SI
    from sklearn.metrics import (accuracy_score as _acc,
                                 roc_auc_score  as _auc,
                                 confusion_matrix as _cm)
    from sklearn.model_selection import (train_test_split as _tts,
                                         cross_val_score   as _cvs,
                                         StratifiedKFold   as _SKF)
    from sklearn.pipeline import Pipeline as _PL
    from sklearn.preprocessing import StandardScaler as _SS

    rng = _np.random.default_rng(SEED)
    n   = 1200

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
        -5.5 + 0.045*(age-55) + 0.55*sex + 0.70*(cp==3) - 0.40*(cp==0)
        + 0.012*(bp-130) + 0.003*(chol-245) + 0.35*fbs + 0.20*ecg
        - 0.030*(hr-150) + 0.75*ex + 0.50*oldpeak
        - 0.55*(slope==0) + 0.70*(slope==2) + 0.62*ca + 0.45*thal
        + rng.normal(0, 0.6, n)
    )
    prob   = 1 / (1 + _np.exp(-logit))
    target = rng.binomial(1, prob)

    df = pd.DataFrame({
        "age":age,"sex":sex,"chest_pain_type":cp,"resting_bp":bp,
        "cholesterol":chol,"fasting_blood_sugar":fbs,"rest_ecg":ecg,
        "max_heart_rate":hr,"exercise_angina":ex,"st_depression":oldpeak,
        "st_slope":slope,"num_vessels":ca,"thal":thal,"target":target,
    })
    X, y = df[FEATURES], df["target"]
    Xtr, Xte, ytr, yte = _tts(X, y, test_size=.2, random_state=SEED, stratify=y)

    pipe = _PL([
        ("imputer", _SI(strategy="median")),
        ("scaler",  _SS()),
        ("model",   _RF(n_estimators=300, max_depth=12, min_samples_split=5,
                        min_samples_leaf=2, max_features="sqrt",
                        class_weight="balanced", random_state=SEED, n_jobs=-1)),
    ])
    pipe.fit(Xtr, ytr)

    yp  = pipe.predict(Xte)
    ypr = pipe.predict_proba(Xte)[:, 1]
    cm  = _cm(yte, yp)
    cv  = _SKF(n_splits=5, shuffle=True, random_state=SEED)
    cva = _cvs(pipe, X, y, cv=cv, scoring="accuracy")
    fpr, tpr, _ = roc_curve(yte, ypr)

    return {
        "pipeline":      pipe,
        "feature_names": FEATURES,
        "metrics": {
            "accuracy": _acc(yte, yp), "roc_auc": _auc(yte, ypr),
            "confusion_matrix": cm, "y_test": yte,
            "y_prob": ypr, "fpr": fpr, "tpr": tpr,
        },
        "cv_metrics": {"cv_acc": cva},
    }


# ─────────────────────────────────────────────
#  RISK UTILITIES
# ─────────────────────────────────────────────
def classify_risk(prob: float) -> tuple:
    if prob < 0.35:
        return "LOW RISK",      "low",    "✅"
    elif prob < 0.65:
        return "MODERATE RISK", "medium", "⚠️"
    else:
        return "HIGH RISK",     "high",   "🚨"


def get_insights(row: dict, prob: float) -> list:
    ins = []
    if row["age"] > 60:
        ins.append("🔸 Age above 60 — a well-established cardiovascular risk factor.")
    if row["cholesterol"] > 240:
        ins.append(f"🔸 Cholesterol {row['cholesterol']} mg/dL is elevated (>240).")
    if row["resting_bp"] > 140:
        ins.append(f"🔸 Resting BP {row['resting_bp']} mmHg — Stage II hypertension.")
    if row["max_heart_rate"] < 120:
        ins.append("🔸 Low max heart rate — may indicate reduced cardiac reserve.")
    if row["exercise_angina"] == 1:
        ins.append("🔸 Exercise-induced angina — strong predictor of coronary disease.")
    if row["st_depression"] > 2:
        ins.append(f"🔸 ST depression {row['st_depression']} — suggests myocardial ischaemia.")
    if row["num_vessels"] >= 2:
        ins.append(f"🔸 {int(row['num_vessels'])} major vessels involved.")
    if row["chest_pain_type"] == 3:
        ins.append("🔸 Asymptomatic chest pain — often linked to silent ischaemia.")
    if not ins:
        ins.append("✅ No major individual risk flags detected.")
    if prob >= 0.65:
        ins.append("🚨 Urgent referral to a cardiologist is strongly recommended.")
    elif prob >= 0.35:
        ins.append("⚠️ Cardiac follow-up and lifestyle review advised.")
    else:
        ins.append("✅ Maintain regular check-ups and a heart-healthy lifestyle.")
    return ins


# ─────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────
with st.spinner("🔬 Loading Cardio AI model …"):
    artefact = get_model()

model      = artefact["pipeline"]
metrics    = artefact["metrics"]
cv_metrics = artefact["cv_metrics"]


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 10px'>
        <span style='font-size:2.8rem'>🫀</span>
        <div style='font-size:1.1rem;font-weight:700;color:#93c5fd;margin-top:8px'>Cardio AI</div>
        <div style='font-size:.72rem;color:#475569;letter-spacing:1px'>AI Health Dashboard</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    mode = st.radio("Select Mode", [
        "🏥 Patient Mode",
        "👨‍⚕️ Doctor Dashboard",
    ], index=0)
    st.markdown("---")

    st.markdown(f"""
    <div style='font-size:.82rem;color:#475569;line-height:2'>
        <b style='color:#64748b'>⚠️ Important Note</b><br>Predict. Prevent. Protect.<br>
        Smart Insights for Cardiac Health.
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════
#  PAGE: PATIENT MODE
# ═════════════════════════════════════════════
if mode == "🏥 Patient Mode":

    st.markdown("""
    <div class='hero'>
        <div class='hero-title'>🫀 Heart Disease Risk Predictor</div>
        <div class='hero-subtitle'>
            Enter your clinical parameters to receive a personalised cardiovascular risk assessment
        </div>
    </div>""", unsafe_allow_html=True)

    with st.form("patient_form"):

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("<div class='stitle'>👤 Demographics</div>",
                        unsafe_allow_html=True)
            age = st.number_input("Age (years)", 20, 90, 55)
            sex = st.selectbox("Sex", ["Female (0)", "Male (1)"])
            sex = 1 if "Male" in sex else 0
            cp  = st.selectbox("Chest Pain Type", [
                "0 – Typical Angina", "1 – Atypical Angina",
                "2 – Non-Anginal Pain", "3 – Asymptomatic"])
            cp  = int(cp[0])

        with c2:
            st.markdown("<div class='stitle'>🩸 Clinical Parameters</div>",
                        unsafe_allow_html=True)
            resting_bp  = st.number_input("Resting BP (mmHg)",    80,  200, 130)
            cholesterol = st.number_input("Cholesterol (mg/dL)",  100, 600, 245)
            fbs         = st.selectbox("Fasting Blood Sugar > 120 mg/dL",
                                       ["No (0)", "Yes (1)"])
            fbs         = 1 if "Yes" in fbs else 0
            rest_ecg    = st.selectbox("Resting ECG", [
                "0 – Normal", "1 – ST-T Abnormality", "2 – LV Hypertrophy"])
            rest_ecg    = int(rest_ecg[0])

        with c3:
            st.markdown("<div class='stitle'>🏃 Exercise & Stress Data</div>",
                        unsafe_allow_html=True)
            max_hr          = st.number_input("Max Heart Rate (bpm)", 60, 210, 150)
            exercise_angina = st.selectbox("Exercise-Induced Angina",
                                           ["No (0)", "Yes (1)"])
            exercise_angina = 1 if "Yes" in exercise_angina else 0
            st_dep          = st.number_input("ST Depression (Oldpeak)",
                                              0.0, 6.5, 1.0, step=0.1)
            st_slope        = st.selectbox("ST Slope", [
                "0 – Upsloping", "1 – Flat", "2 – Downsloping"])
            st_slope        = int(st_slope[0])

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("<div class='stitle'>🔬 Fluoroscopy</div>",
                        unsafe_allow_html=True)
            num_vessels = st.slider("Number of Major Vessels", 0, 3, 0)
        with d2:
            st.markdown("<div class='stitle'>🧬 Thalassemia</div>",
                        unsafe_allow_html=True)
            thal = st.selectbox("Thalassemia Type", [
                "0 – Normal", "1 – Fixed Defect", "2 – Reversible Defect"])
            thal = int(thal[0])

        submitted = st.form_submit_button("🔮  Predict My Risk",
                                          use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            "age": age, "sex": sex, "chest_pain_type": cp,
            "resting_bp": resting_bp, "cholesterol": cholesterol,
            "fasting_blood_sugar": fbs, "rest_ecg": rest_ecg,
            "max_heart_rate": max_hr, "exercise_angina": exercise_angina,
            "st_depression": st_dep, "st_slope": st_slope,
            "num_vessels": num_vessels, "thal": thal,
        }])

        prob  = model.predict_proba(input_df)[0][1]
        label, badge, icon = classify_risk(prob)
        insights = get_insights(input_df.iloc[0].to_dict(), prob)

        st.markdown("---")

        # ── Metric row ────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        for col, val, lbl in [
            (m1, f"{prob:.1%}",      "Risk Probability"),
            (m2, icon,               "Risk Category"),
            (m3, f"{1-prob:.1%}",    "No Disease Prob."),
            (m4, f"{max(prob,1-prob):.1%}", "Model Confidence"),
        ]:
            col.markdown(f"""
            <div class='mcard'>
                <div class='mcard-val'>{val}</div>
                <div class='mcard-label'>{lbl}</div>
                {'<div style="margin-top:6px"><span class="'+badge+'">'+label+'</span></div>' if lbl=="Risk Category" else ''}
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Gauge + Insights ──────────────────────────────────────────
        gc, ic = st.columns(2)

        with gc:
            st.markdown("<div class='stitle'>Risk Gauge</div>",
                        unsafe_allow_html=True)
            dark_style()
            fig, ax = plt.subplots(figsize=(5, 3.2),
                                   subplot_kw=dict(polar=True))
            fig.patch.set_facecolor("#0a0e1a")
            ax.set_facecolor("#0a0e1a")

            theta = np.linspace(0, np.pi, 300)
            segs  = [(0, 100, "#10b981"), (100, 180, "#f59e0b"),
                     (180, 300, "#ef4444")]
            for s, e, c in segs:
                ax.plot(theta[s:e], np.ones(e - s) * .8, lw=18,
                        color=c, alpha=.85, solid_capstyle="round")

            needle = np.pi * (1 - prob)
            ax.annotate("", xy=(needle, .78), xytext=(needle, .02),
                        arrowprops=dict(arrowstyle="->", color="#f1f5f9",
                                        lw=2.5, mutation_scale=16))
            ax.set_ylim(0, 1); ax.set_xlim(0, np.pi); ax.axis("off")
            ax.set_title(f"  {prob:.1%}  Heart Disease Risk",
                         color="#e2e8f0", fontsize=12,
                         fontweight="bold", pad=12)
            st.pyplot(fig); plt.close(fig)

        with ic:
            st.markdown("<div class='stitle'>🔍 Personalised Insights</div>",
                        unsafe_allow_html=True)
            for ins in insights:
                st.markdown(f"<div class='insight'>{ins}</div>",
                            unsafe_allow_html=True)

        # ── Feature contribution ───────────────────────────────────────
        st.markdown("<div class='stitle' style='margin-top:16px'>"
                    "Feature Contributions (Explainable AI)</div>",
                    unsafe_allow_html=True)

        fi      = model.named_steps["model"].feature_importances_
        top_idx = np.argsort(fi)[::-1][:9]
        clr     = "#ef4444" if prob >= .5 else "#10b981"

        dark_style()
        fig2, ax2 = plt.subplots(figsize=(10, 3.4))
        fig2.patch.set_facecolor("#0a0e1a")
        ax2.set_facecolor("#0f172a")
        ax2.barh(
            [FEATURE_LABELS[FEATURES[i]] for i in top_idx],
            fi[top_idx],
            color=clr, alpha=.82, height=.6, edgecolor="none",
        )
        for i in top_idx:
            pass  # values already in bar
        ax2.set_xlabel("Importance Score", fontsize=9)
        ax2.set_title("Top Feature Importances — Random Forest",
                      color="#e2e8f0", fontsize=11)
        ax2.invert_yaxis()
        ax2.grid(axis="x")
        ax2.tick_params(labelsize=9)
        st.pyplot(fig2); plt.close(fig2)


# ═════════════════════════════════════════════
#  PAGE: DOCTOR DASHBOARD
# ═════════════════════════════════════════════
elif mode == "👨‍⚕️ Doctor Dashboard":

    st.markdown("""
    <div class='hero'>
        <div class='hero-title'>👨‍⚕️ Doctor Dashboard</div>
        <div class='hero-subtitle'>
            Upload a bulk patient CSV for cardiac risk analysis & reporting
        </div>
    </div>""", unsafe_allow_html=True)

  

    uploaded = st.file_uploader("Upload Patient CSV", type=["csv"])

    if uploaded:
        try:
            df_raw = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not parse file: {e}"); st.stop()

        missing = [c for c in FEATURES if c not in df_raw.columns]
        if missing:
            st.error(f"Missing columns: {missing}"); st.stop()

        X_bulk = df_raw[FEATURES]
        probs  = model.predict_proba(X_bulk)[:, 1]
        preds  = (probs >= .5).astype(int)
        risk_l = [classify_risk(p)[0] for p in probs]

        df_out = df_raw.copy()
        df_out["Patient ID"]       = [f"PT-{i+1:04d}" for i in range(len(df_out))]
        df_out["Risk Probability %"] = (probs * 100).round(1)
        df_out["Prediction"]       = np.where(preds == 1, "Disease", "No Disease")
        df_out["Risk Category"]    = risk_l

        n_total = len(df_out)
        n_high  = (df_out["Risk Category"] == "HIGH RISK").sum()
        n_mod   = (df_out["Risk Category"] == "MODERATE RISK").sum()
        n_low   = (df_out["Risk Category"] == "LOW RISK").sum()

        # ── Summary metrics ───────────────────────────────────────────
        s1, s2, s3, s4, s5 = st.columns(5)
        for col, val, lbl in [
            (s1, n_total,           "Total Patients"),
            (s2, n_high,            "🚨 High Risk"),
            (s3, n_mod,             "⚠️ Moderate Risk"),
            (s4, n_low,             "✅ Low Risk"),
            (s5, f"{probs.mean():.1%}", "Avg Risk Score"),
        ]:
            col.markdown(f"""
            <div class='mcard'>
                <div class='mcard-val'>{val}</div>
                <div class='mcard-label'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Three charts ─────────────────────────────────────────────
        v1, v2, v3 = st.columns(3)
        dark_style()

        # Donut
        with v1:
            st.markdown("<div class='stitle'>Risk Distribution</div>",
                        unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            fig.patch.set_facecolor("#0a0e1a")
            ax.set_facecolor("#0a0e1a")
            wedges, _ = ax.pie(
                [n_high, n_mod, n_low],
                colors=["#ef4444", "#f59e0b", "#10b981"],
                startangle=90,
                wedgeprops=dict(width=.55, edgecolor="#0a0e1a", linewidth=2),
            )
            ax.legend(wedges, [f"High ({n_high})", f"Moderate ({n_mod})",
                                f"Low ({n_low})"],
                      loc="lower center", bbox_to_anchor=(.5, -.1),
                      ncol=3, fontsize=8, frameon=False, labelcolor="#94a3b8")
            ax.set_title("Patient Breakdown", color="#e2e8f0", fontsize=10)
            st.pyplot(fig); plt.close(fig)

        # Histogram
        with v2:
            st.markdown("<div class='stitle'>Probability Histogram</div>",
                        unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            fig.patch.set_facecolor("#0a0e1a"); ax.set_facecolor("#0f172a")
            n_hist, bins, patches = ax.hist(probs * 100, bins=20,
                                            edgecolor="#0a0e1a", lw=.5)
            for patch, left in zip(patches, bins[:-1]):
                patch.set_facecolor("#ef4444" if left >= 65 else
                                    "#f59e0b" if left >= 35 else "#10b981")
            ax.set_xlabel("Risk Probability (%)", fontsize=9)
            ax.set_ylabel("Patients", fontsize=9)
            ax.set_title("Risk Score Distribution", color="#e2e8f0", fontsize=10)
            ax.grid(axis="y"); ax.tick_params(labelsize=8)
            st.pyplot(fig); plt.close(fig)

        # Scatter
        with v3:
            st.markdown("<div class='stitle'>Age vs. Risk Score</div>",
                        unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            fig.patch.set_facecolor("#0a0e1a"); ax.set_facecolor("#0f172a")
            sc = ax.scatter(df_out["age"], probs * 100, c=probs,
                            cmap="RdYlGn_r", alpha=.75, s=25, edgecolors="none")
            ax.set_xlabel("Age (years)", fontsize=9)
            ax.set_ylabel("Risk Probability (%)", fontsize=9)
            ax.set_title("Age vs. Risk", color="#e2e8f0", fontsize=10)
            ax.grid(True); ax.tick_params(labelsize=8)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.ax.tick_params(labelsize=7); cbar.set_label("Risk", fontsize=8)
            st.pyplot(fig); plt.close(fig)

        # ── High-risk table ───────────────────────────────────────────
        st.markdown("<div class='stitle' style='margin-top:10px'>"
                    "🚨 High Risk Patients</div>", unsafe_allow_html=True)
        high_df = df_out[df_out["Risk Category"] == "HIGH RISK"].sort_values(
            "Risk Probability %", ascending=False).reset_index(drop=True)

        if high_df.empty:
            st.success("No high-risk patients detected.")
        else:
            show = ["Patient ID", "age", "cholesterol", "resting_bp",
                    "max_heart_rate", "Risk Probability %", "Risk Category"]
            show = [c for c in show if c in high_df.columns]
            st.dataframe(high_df[show], use_container_width=True, height=280)

        # ── Download ──────────────────────────────────────────────────
        st.markdown("---")
        fname = f"cardio_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        st.download_button("⬇️ Download Full Results CSV",
                           df_out.to_csv(index=False).encode(),
                           fname, "text/csv", use_container_width=True)
    else:
        st.markdown("""
        <div class='card' style='text-align:center;padding:50px'>
            <div style='font-size:3rem'>📤</div>
            <div style='font-size:1.05rem;color:#94a3b8;margin-top:12px'>
                Upload a CSV with patient data to begin bulk analysis
            </div>
            <div style='font-size:.82rem;color:#475569;margin-top:8px'>
                Required columns: age · sex · chest_pain_type · resting_bp · cholesterol ·
                fasting_blood_sugar · rest_ecg · max_heart_rate · exercise_angina ·
                st_depression · st_slope · num_vessels · thal
            </div>
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════
#  PAGE: MODEL INSIGHTS
# ═════════════════════════════════════════════
elif mode == "📊 Model Insights":

    st.markdown("""
    <div class='hero'>
        <div class='hero-title'>📊 Model Performance & Explainability</div>
        <div class='hero-subtitle'>
            ROC curve · Confusion matrix · Feature importance · Cross-validation
        </div>
    </div>""", unsafe_allow_html=True)

    acc = metrics["accuracy"]
    auc = metrics["roc_auc"]
    cm  = metrics["confusion_matrix"]
    fi  = model.named_steps["model"].feature_importances_
    fpr = metrics["fpr"]
    tpr = metrics["tpr"]
    cva = cv_metrics["cv_acc"]

    # ── Top metrics ───────────────────────────────────────────────────
    t1, t2, t3, t4 = st.columns(4)
    for col, val, lbl in [
        (t1, f"{acc:.2%}",       "Test Accuracy"),
        (t2, f"{auc:.4f}",       "ROC-AUC Score"),
        (t3, f"{cva.mean():.2%}","CV Mean Accuracy"),
        (t4, f"±{cva.std():.2%}","CV Std Deviation"),
    ]:
        col.markdown(f"""
        <div class='mcard'>
            <div class='mcard-val'>{val}</div>
            <div class='mcard-label'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    dark_style()

    # ── ROC + Confusion Matrix ─────────────────────────────────────────
    r1, r2 = st.columns(2)

    with r1:
        st.markdown("<div class='stitle'>ROC Curve</div>",
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#0a0e1a"); ax.set_facecolor("#0f172a")
        ax.plot(fpr, tpr, color="#3b82f6", lw=2.5,
                label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "--", color="#475569", lw=1.2,
                label="Random")
        ax.fill_between(fpr, tpr, alpha=.08, color="#3b82f6")
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate",  fontsize=9)
        ax.set_title("Receiver Operating Characteristic",
                     color="#e2e8f0", fontsize=11)
        ax.legend(fontsize=9, frameon=False, labelcolor="#94a3b8")
        ax.grid(True)
        st.pyplot(fig); plt.close(fig)

    with r2:
        st.markdown("<div class='stitle'>Confusion Matrix</div>",
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#0a0e1a"); ax.set_facecolor("#0f172a")
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"],
            linewidths=1, linecolor="#1e3a5f",
            cbar_kws={"shrink": .75}, ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=9, color="#94a3b8")
        ax.set_ylabel("Actual",    fontsize=9, color="#94a3b8")
        ax.set_title("Confusion Matrix", color="#e2e8f0", fontsize=11)
        ax.tick_params(colors="#94a3b8", labelsize=9)
        st.pyplot(fig); plt.close(fig)

    # ── Feature Importance full ────────────────────────────────────────
    st.markdown("<div class='stitle'>Feature Importances — Explainable AI</div>",
                unsafe_allow_html=True)
    sorted_idx = np.argsort(fi)
    labels     = [FEATURE_LABELS[FEATURES[i]] for i in sorted_idx]
    grad_clrs  = plt.cm.plasma(np.linspace(.3, .9, len(sorted_idx)))

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0a0e1a"); ax.set_facecolor("#0f172a")
    bars = ax.barh(labels, fi[sorted_idx], color=grad_clrs,
                   height=.65, edgecolor="none")
    for bar, i in zip(bars, sorted_idx):
        ax.text(bar.get_width() + .002, bar.get_y() + bar.get_height()/2,
                f"{fi[i]:.4f}", va="center", ha="left",
                color="#94a3b8", fontsize=9)
    ax.set_xlabel("Importance Score", fontsize=9)
    ax.set_title("All Feature Importances (Random Forest — Gini)",
                 color="#e2e8f0", fontsize=12)
    ax.grid(axis="x"); ax.tick_params(labelsize=9)
    st.pyplot(fig); plt.close(fig)

    # ── CV Bar Chart ───────────────────────────────────────────────────
    st.markdown("<div class='stitle'>5-Fold Stratified Cross Validation</div>",
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    fig.patch.set_facecolor("#0a0e1a"); ax.set_facecolor("#0f172a")
    folds  = [f"Fold {i+1}" for i in range(len(cva))]
    clrs   = ["#3b82f6" if v >= cva.mean() else "#6366f1" for v in cva]
    ax.bar(folds, cva * 100, color=clrs, edgecolor="none", width=.55)
    ax.axhline(cva.mean() * 100, color="#f59e0b", lw=1.5, linestyle="--",
               label=f"Mean = {cva.mean():.2%}")
    for i, v in enumerate(cva):
        ax.text(i, v * 100 + .3, f"{v:.2%}", ha="center", va="bottom",
                color="#e2e8f0", fontsize=9)
    ax.set_ylim(max(0, cva.min() * 100 - 5), min(100, cva.max() * 100 + 5))
    ax.set_ylabel("Accuracy (%)", fontsize=9)
    ax.set_title("Cross-Validation Accuracy per Fold",
                 color="#e2e8f0", fontsize=11)
    ax.legend(fontsize=9, frameon=False, labelcolor="#f59e0b")
    ax.grid(axis="y"); ax.tick_params(labelsize=9)
    st.pyplot(fig); plt.close(fig)

    # ── Architecture card ──────────────────────────────────────────────
    st.markdown("""
    <div class='card' style='margin-top:12px'>
        <div class='stitle'>⚙️ Model Architecture</div>
        <table style='width:100%;font-size:.87rem;color:#94a3b8;border-collapse:collapse'>
            <tr><td style='padding:7px 12px;border-bottom:1px solid rgba(59,130,246,.1)'>
                <b style='color:#60a5fa'>Algorithm</b></td><td>Random Forest Classifier</td></tr>
            <tr><td style='padding:7px 12px;border-bottom:1px solid rgba(59,130,246,.1)'>
                <b style='color:#60a5fa'>Estimators</b></td><td>300 decision trees</td></tr>
            <tr><td style='padding:7px 12px;border-bottom:1px solid rgba(59,130,246,.1)'>
                <b style='color:#60a5fa'>Max Depth</b></td><td>12</td></tr>
            <tr><td style='padding:7px 12px;border-bottom:1px solid rgba(59,130,246,.1)'>
                <b style='color:#60a5fa'>Feature Sampling</b></td><td>sqrt(n_features) per split</td></tr>
            <tr><td style='padding:7px 12px;border-bottom:1px solid rgba(59,130,246,.1)'>
                <b style='color:#60a5fa'>Class Weight</b></td><td>Balanced (handles imbalance)</td></tr>
            <tr><td style='padding:7px 12px;border-bottom:1px solid rgba(59,130,246,.1)'>
                <b style='color:#60a5fa'>Preprocessing</b></td><td>Median Imputation → StandardScaler</td></tr>
            <tr><td style='padding:7px 12px'>
                <b style='color:#60a5fa'>Explainability</b></td><td>Gini-based Feature Importances (XAI)</td></tr>
        </table>
    </div>""", unsafe_allow_html=True)