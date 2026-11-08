import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Important features",layout="wide")

# Load Dataset & Bundle
@st.cache_resource
def load_assets():
    df = pd.read_csv("heart-checkpoint.csv")
    bundle = joblib.load("heart_bundle.joblib")
    return df, bundle

df, bundle = load_assets()
model = bundle["model"]
feature_cols = bundle["columns"]




st.subheader("Model Weights & Feature Contributions")
st.write("Which symptoms and clinical measurements have the strongest influence on the model's prediction?")
    
coef_df = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=True)
    
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#EF4444' if x > 0 else '#10B981' for x in coef_df['Coefficient']]
ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
ax.set_xlabel("Logistic Regression Coefficient (Weight)", fontsize=11)
ax.set_title("Learned Feature Weights (Red = Increases Risk, Green = Lowers Risk)", fontweight='bold', fontsize=12)
ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
st.pyplot(fig)
plt.close(fig)
    

col_feat1, col_feat2 = st.columns(2)

with col_feat1:
    st.markdown("""
    🔴 Top Factors that INCREASE Heart Disease Risk:
    - ST_Slope_Flat: A flat ST slope during exercise may indicate a higher risk of heart disease.
    - ExerciseAngina_Y: Chest pain during physical activity is linked to a higher risk of heart disease.
    - Sex_M: Male patients showed a higher rate of heart disease in this dataset.
    - Oldpeak: A higher Oldpeak value may indicate a greater risk of heart disease.
    """)

with col_feat2:
    st.markdown("""
    🟢 Top Factors that DECREASE Heart Disease Risk:
    - ST_Slope_Up: An upward ST slope is generally associated with a lower risk of heart disease.
    - ChestPainType_ATA & NAP: These chest pain types showed a lower rate of heart disease in this dataset.
    - MaxHR: A higher maximum heart rate may be associated with a lower risk of heart disease.
    """)



