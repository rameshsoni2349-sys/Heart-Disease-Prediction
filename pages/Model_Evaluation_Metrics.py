import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, classification_report

st.set_page_config(
    page_title="Evaluation Metrics - Heart Disease",
    layout="wide"
)


st.title("Model Evaluation Metrics")
st.markdown("Learn Accuracy, Confusion Matrix, Precision, Recall, F1-Score, and ROC-AUC with simple real-world examples")


with st.expander("Why is Accuracy not enough in Medical Machine Learning ?"):

    st.markdown("""
    Imagine a rare disease that only 1 person in 1,000 has:
    - If a lazy model simply predicts Nobody has the disease, it achieves 99.9% Accuracy!
    - But it completely missed the patient who needed treatment. That patient could die.
    - That is why in healthcare, we must check:
      1. Confusion Matrix: Exactly how many diagnoses were correct vs incorrect.
      2. Recall (Sensitivity): Out of all sick patients, how many did the model correctly identify??
      3. Precision: Out of all patients predicted as sick, how many actually have the disease?
    """)

# Load Model & Test Data
@st.cache_resource
def load_eval_data():
    df = pd.read_csv("heart-checkpoint.csv")
    bundle = joblib.load("heart_bundle.joblib")
    
    feature_cols = bundle["columns"]
    chol_median = bundle["chol_median"]
    bp_median = bundle["bp_median"]
    
    X_raw = pd.get_dummies(df.drop("HeartDisease", axis=1), drop_first=True)
    y_raw = df["HeartDisease"]

    _, X_test, _, y_test = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=42, stratify=y_raw
    )

    X_test_clean = X_test.copy()
    X_test_clean["Cholesterol"] = X_test_clean["Cholesterol"].replace(0, chol_median)
    X_test_clean["RestingBP"] = X_test_clean["RestingBP"].replace(0, bp_median)
    X_test_clean = X_test_clean.reindex(columns=feature_cols, fill_value=0)

    model = bundle["model"]
    y_probs = model.predict_proba(X_test_clean)[:, 1]
    return y_test, y_probs, bundle["threshold"]

y_test, y_probs, default_threshold = load_eval_data()

# Calculate default metrics
y_pred_default = (y_probs >= default_threshold).astype(int)
acc = accuracy_score(y_test, y_pred_default)
rec = recall_score(y_test, y_pred_default)
prec = precision_score(y_test, y_pred_default)
f1 = f1_score(y_test, y_pred_default)
auc = roc_auc_score(y_test, y_probs)

cm_default = confusion_matrix(y_test, y_pred_default)
tn, fp, fn, tp = cm_default.ravel()

# Metrics Overview Banner
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Overall Accuracy", f"{acc*100:.2f}%", f"{tp+tn}/{len(y_test)} correct")
with col2:
    st.metric("Recall (Sensitivity)", f"{rec*100:.2f}%", f"{tp}/{tp+fn} caught")
with col3:
    st.metric("Precision", f"{prec*100:.2f}%", f"{tp}/{tp+fp} true alerts")
with col4:
    st.metric("F1-Score", f"{f1*100:.2f}%", "Balanced score")
with col5:
    st.metric("ROC-AUC", f"{auc:.3f}", "Diagnostic power")



# ----------------- TAB 1: CONFUSION MATRIX -----------------
st.subheader("Confusion Matrix Breakdown (Test Set: 184 Patients)")
st.write("A Confusion Matrix compares what the model predicted against the actual reality.")
    
col_cm1, col_cm2 = st.columns(2)
    
with col_cm1:
    fig, ax = plt.subplots(figsize=(6, 5))  
    sns.heatmap(
        cm_default,
        annot=True,
        annot_kws={"size": 18, "weight": "bold"},
        xticklabels=["Healthy (0)", "Disease (1)"],
        yticklabels=["Healthy (0)", "Disease (1)"],
        ax=ax
    )
    ax.set_xlabel("Predicted Diagnosis by Model", fontsize=11, fontweight='bold')
    ax.set_ylabel("Actual Patient Diagnosis", fontsize=11, fontweight='bold')
    ax.set_title("Confusion Matrix Heatmap", fontsize=13, fontweight='bold')
    st.pyplot(fig)
    plt.close(fig)
        
with col_cm2:
    st.markdown(" The 4 Medical Quadrants:")
        
    st.success(f"""
        True Positive (TP = {tp}):
        - Model Predicted: Heart Disease
        - Actual Status: Heart Disease
        - Meaning: Successful diagnosis! Patient receives lifesaving medical care.
        """)
        
    st.info(f"""
        True Negative (TN = {tn}):
        - Model Predicted: Healthy
        - Actual Status: Healthy
        - Meaning: Correct reassurance. Healthy patient is not burdened with unnecessary treatments.
        """)
        
    st.warning(f"""
          False Positive (FP = {fp}):
        - Model Predicted: Heart Disease
        - Actual Status: Healthy
        - Meaning: False Alarm. Patient is asked to do follow-up ECG/blood tests, but is physically safe.
        """)
        
    st.error(f"""
           False Negative (FN = {fn}):
        - Model Predicted: Healthy
        - Actual Status: Heart Disease
        - Meaning: Dangerous Miss! Patient actually has heart disease, but model missed it.
        - Notice: Our model missed only {fn} patients out of 102 actual disease cases!
        """)


    
