# ===============================
#  1. Imports
# ===============================
import streamlit as st
import joblib
import pandas as pd

# ===============================
#  2. Page Config (MUST BE FIRST)
# ===============================
st.set_page_config(
    page_title="Fetal Health Predictor",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Fetal Health Classification")
st.caption("Machine Learning-based CTG Analysis")

# ===============================
#  3. Load Model + Artifacts
# ===============================
model = joblib.load("decision_tree_best_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")
feature_means = joblib.load("feature_means.pkl")
sample_input = joblib.load("sample_input.pkl")

# ==============================
#  4. Sidebar Info
# ===============================
st.sidebar.title("ℹ️ About")
st.sidebar.write("""
This app predicts fetal health using CTG data.

Classes:
- 1 → Normal
- 2 → Suspect
- 3 → Pathologic
""")

# ===============================
#  5. Preset Scenarios
# ===============================

preset = st.selectbox(
    " Select Scenario",
    ["Custom", "Normal", "Suspect", "Pathologic"]
)

if preset == "Normal":
    LB, AC, FM, UC = 120, 2, 1, 2
    ASTV, ALTV, MSTV, MLTV = 20, 0, 1.5, 7

elif preset == "Suspect":
    LB, AC, FM, UC = 140, 1, 1, 3
    ASTV, ALTV, MSTV, MLTV = 40, 20, 1.0, 5

elif preset == "Pathologic":
    LB, AC, FM, UC = 160, 0, 0, 1
    ASTV, ALTV, MSTV, MLTV = 80, 60, 0.3, 1
# ===============================
#  6. UI Layout
# ===============================
st.subheader("📊 Clinical Inputs")


# ===============================
#  7. Build Input Data (CRITICAL)
# ===============================
st.subheader("🩺 Clinical Assessment Inputs")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### ❤️ Heart Rate & Activity")
    LB = st.slider("Baseline FHR (bpm)", 80, 200, 120)
    AC = st.slider("Accelerations", 0, 10, 2)
    FM = st.slider("Fetal Movements", 0, 10, 1)
    UC = st.slider("Uterine Contractions", 0, 10, 2)

with col2:
    st.markdown("### 📉 Variability Indicators")
    ASTV = st.slider("Abnormal Short-Term Variability (%)", 0, 100, 20)
    ALTV = st.slider("Abnormal Long-Term Variability (%)", 0, 100, 0)
    MSTV = st.slider("Mean Short-Term Variability", 0.0, 5.0, 1.5)
    MLTV = st.slider("Mean Long-Term Variability", 0.0, 20.0, 7.0)
# ===============================
#  7. Build Input Data (FIXED)
# ===============================

#  Use a REAL sample from dataset
sample_input = joblib.load("sample_input.pkl")
input_dict = sample_input.copy()

# Override only user inputs
input_dict = sample_input.copy()

input_dict.update({
    "LB": LB,
    "AC": AC,
    "FM": FM,
    "UC": UC,
    "ASTV": ASTV,
    "MSTV": MSTV,
    "ALTV": ALTV,
    "MLTV": MLTV
})
# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Ensure correct order
input_df = input_df[feature_names]

# ===============================
#  8. Prediction
# ===============================
st.markdown("---")

if st.button("🔍 Generate Prediction"):

    #  IMPORTANT: Apply scaling (matches training)
    input_scaled = scaler.transform(input_df)

    pred = model.predict(input_scaled)[0]
    st.write("SCALED INPUT")
    st.write(input_scaled[0][:10])

    label_map = {
        1: "Normal",
        2: "Suspect",
        3: "Pathologic"
    }

    label = label_map.get(pred, str(pred))

    st.subheader("🧠 Prediction Result")

    if pred == 1:
        st.success(f"✅ {label}")
        st.write("Fetal condition appears normal.")

    elif pred == 2:
        st.warning(f"⚠️ {label}")
        st.write("Further monitoring recommended.")

    else:
        st.error(f"🚨 {label}")
        st.write("Immediate medical attention required.")

    # ===============================
    #  9. Confidence Score
    # ===============================
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_scaled)[0]

        st.write("### 📊 Prediction Confidence")
        st.write({
            "Normal": round(probs[0], 3),
            "Suspect": round(probs[1], 3),
            "Pathologic": round(probs[2], 3)
        })
        st.write("Probabilities:", model.predict_proba(input_scaled))
        st.write(model)

# ===============================
#  10. Footer
# ===============================
st.markdown("---")
st.caption("Educational ML application. Not for medical diagnosis.")