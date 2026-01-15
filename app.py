import streamlit as st
import joblib
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# =========================
# BACKGROUND IMAGE
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("assets/Copilot_20251221_114700.png");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        filter: contrast(70%) brightness(110%);
    }

    .card {
        background-color: rgba(28, 158, 82, 0.96);
        padding: 0px;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        max-width: 680px;
        margin: 40px auto;
    }

    .question-box {
        background: #f8f9fb;
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 20px;
        border-left: 6px solid #0d6efd;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        font-size: 18px;
        font-weight: 600;
        color: #222;
    }

    .result-box {
        background: #eaf7ee;
        padding: 22px;
        border-radius: 14px;
        border-left: 6px solid #198754;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        font-size: 22px;
        font-weight: bold;
        color: #155724;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# HEADER
# =========================
st.markdown(
    """
    <div class="card" style="text-align:center;">
        <h1>🩺 AI Diabetes Risk Predictor</h1>
        <p style="color:#030303;">
            Educational use only. Not a medical diagnosis.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# LOAD MODEL & FEATURES
# =========================
@st.cache_resource
def load_models():
    model = joblib.load("models/diabetes_model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, feature_names

model, FEATURE_NAMES = load_models()

# =========================
# FEATURE UTILITIES
# =========================
def init_features():
    return {f: 0 for f in FEATURE_NAMES}

def predict_diabetes(features):
    X = np.array([[features[f] for f in FEATURE_NAMES]])
    pred = model.predict(X)[0]

    labels = {
        0: "Gestational",
        1: "No Diabetes",
        2: "Pre-Diabetes",
        3: "Type 1",
        4: "Type 2"
    }
    return labels.get(pred, "Unknown")

# =========================
# SESSION STATE
# =========================
if "step" not in st.session_state:
    st.session_state.step = 0

if "features" not in st.session_state:
    st.session_state.features = init_features()

# =========================
# IMPORTANT LIFESTYLE QUESTIONS ONLY
# =========================
QUESTIONS = [
    {
        "label": "What is your age (years)?",
        "type": "number",
        "feature": "Age"
    },
    {
        "label": "What is your Body Mass Index (BMI)?",
        "type": "number",
        "feature": "bmi"
    },
    {
        "label": "How much physical activity do you do per week?",
        "type": "select",
        "feature": "physical_activity_minutes_per_week",
        "options": {
            "I don’t know": 0,
            "Less than 30 minutes": 30,
            "30–150 minutes": 90,
            "More than 150 minutes": 180
        }
    },
    {
        "label": "What is your smoking status?",
        "type": "select",
        "options": {
            "I don’t know": None,
            "Never smoked": "smoking_status_Never",
            "Former smoker": "smoking_status_Former",
            "Currently smoking": "smoking_status_Current"
        }
    },
    {
        "label": "Alcohol consumption per week?",
        "type": "select",
        "feature": "alcohol_consumption_per_week",
        "options": {
            "I don’t know": 0,
            "None": 0,
            "Moderate": 5,
            "Heavy": 14
        }
    },
    {
        "label": "Do you have a family history of diabetes?",
        "type": "select",
        "feature": "family_history_diabetes",
        "options": {
            "I don’t know": 0,
            "No": 0,
            "Yes": 1
        }
    },
    {
        "label": "Have you ever been told you have high blood pressure?",
        "type": "select",
        "feature": "hypertension_history",
        "options": {
            "I don’t know": 0,
            "No": 0,
            "Yes": 1
        }
    }
]

# =========================
# QUESTION CARD
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)

if st.session_state.step < len(QUESTIONS):
    q = QUESTIONS[st.session_state.step]

    st.markdown(
        f"<div class='question-box'>{q['label']}</div>",
        unsafe_allow_html=True
    )

    if q["type"] == "number":
        value = st.number_input(
            "Enter value (or leave 0 if unknown)",
            min_value=0.0,
            step=1.0
        )

        if st.button("Next ▶", use_container_width=True):
            st.session_state.features[q["feature"]] = value
            st.session_state.step += 1
            st.rerun()

    else:
        choice = st.selectbox("Select one option", list(q["options"].keys()))

        if st.button("Next ▶", use_container_width=True):
            selected = q["options"][choice]

            if isinstance(selected, str):
                st.session_state.features[selected] = 1
            elif selected is not None:
                st.session_state.features[q["feature"]] = selected

            st.session_state.step += 1
            st.rerun()

else:
    result = predict_diabetes(st.session_state.features)

    st.markdown(
        f"<div class='result-box'>🧾 Prediction Result: {result}</div>",
        unsafe_allow_html=True
    )

    st.caption(
        "Prediction is based only on lifestyle information provided and is not a medical diagnosis."
    )

    if st.button("🔄 Start New Assessment", use_container_width=True):
        st.session_state.step = 0
        st.session_state.features = init_features()
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
