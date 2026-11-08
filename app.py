import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Heart Disease Predictor",layout="wide")


st.title("Heart Disease Prediction App")
st.write("A simplified machine learning app for predicting heart disease risk with high accuracy.")

@st.cache_resource
def load_bundle():
    return joblib.load("heart_bundle.joblib")

bundle = load_bundle()

# input value col1,or col2

col1,col2 = st.columns(2)


with col1:
    age = st.number_input("Age (Year)", 18, 100, 50)
    sex = st.selectbox("Gender", ["M", "F"],
                       format_func=lambda x: {"M": "Male", "F": "Female"}[x])
    chest_pain = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"],
                              format_func=lambda x: {
                                  "ASY": "ASY - Asymptomatic",
                                  "ATA": "ATA - Atypical Angina",
                                  "NAP": "NAP - Non-Anginal Pain",
                                  "TA": "TA - Typical Angina"}[x])
    resting_bp = st.number_input("Resting BP (mm Hg)", 80, 220, 130)
    cholesterol = st.number_input("Cholesterol (mg/dl)", 80, 700, 220)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", [0, 1],
                              format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.number_input("Max Heart Rate", 60, 220, 140)
    exercise_angina = st.selectbox("Exercise Angina ?",
                                   ["N", "Y"],
                                   format_func=lambda x: "Yes" if x == "Y" else "No")
    oldpeak = st.number_input("Oldpeak (ST depression)", -3.0, 7.0, 1.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])


# predict 

if st.button("Predict", type="primary",width="stretch"):
    patient = pd.DataFrame([{
        "Age": age, "Sex": sex, "ChestPainType": chest_pain,
        "RestingBP": resting_bp, "Cholesterol": cholesterol,
        "FastingBS": fasting_bs, "RestingECG": resting_ecg,
        "MaxHR": max_hr, "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak, "ST_Slope": st_slope,
    }])


# text ko 0/1 me badlo, aur columns model ke jaise karo

    patient = pd.get_dummies(patient)
    patient = patient.reindex(columns=bundle["columns"], fill_value=0)

    prob = bundle["model"].predict_proba(patient)[0, 1]
    st.markdown("---")

    res_col1,res_col2 = st.columns(2)

    with res_col1:

        st.subheader("Diagnostic Results")
        st.metric("Estimated Disease Probability", f"{prob:.1%}")
    with res_col2:

        
        if prob >= bundle["threshold"]:
            st.error(f"""
            ### ⚠️ Possible Heart Disease Risk
            - **Risk Score:** {prob*100:.1f}%
            - **Meaning:** The model detected signs that may indicate a higher risk of heart disease.
            - **Next Step:** Consult a doctor for proper medical checkup.
            """)

        else:
            st.success(f"""
            ### ✅ Lower Heart Disease Risk
            - **Risk Score:** {prob*100:.1f}%
            - **Meaning:** The model detected a lower risk based on the entered details.
            - **Next Step:** Maintain a healthy lifestyle and get regular health checkups.
            """)

  
