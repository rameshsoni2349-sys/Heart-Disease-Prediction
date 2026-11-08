import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Heart Disease EDA",layout="wide")

st.title("Data Analysis (EDA) - Exploratory Data Analysis")


df = pd.read_csv("heart-checkpoint.csv")

st.markdown("---")


col1,col2, = st.columns(2)


with col1:
    # 1. Heart Disease Count
    st.subheader("Heart Disease Count")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="HeartDisease", ax=ax)
    st.pyplot(fig)

    st.info("""
    Insight:
    - 0 means the person does not have heart disease.
    - 1 means the person has heart disease.
    - This chart shows how many people have heart disease and how many do not.
    """)

    # 2. Gender
    st.subheader("Gender")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Sex", hue="HeartDisease", ax=ax)
    st.pyplot(fig)

    st.info("""
    Insight:
    - This chart compares heart disease between male and female patients.
    - It helps us understand which gender has more heart disease cases in this dataset.
    """)

with col2:
    # 3. Chest Pain Type
    st.subheader("Chest Pain Type Heart Disease")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="ChestPainType", hue="HeartDisease", ax=ax)
    st.pyplot(fig)

    st.info("""
    Insight:
    - This chart shows the relationship between chest pain types and heart disease.
    - ASY (Asymptomatic) means the patient has no typical chest pain symptoms.
    - It helps us identify which chest pain type has more heart disease cases.
    """)

    # 4. Age Distribution
    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x="Age", hue="HeartDisease", kde=True, ax=ax)
    st.pyplot(fig)

    st.info("""
    Insight:
    - This chart shows the age distribution of patients.
    - It compares people with and without heart disease across different ages.
    - It helps us understand which age groups have more heart disease cases.
    """)

    # 5. Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )
    st.pyplot(fig)

    st.info("""
    Insight:
    - The heatmap shows how numerical features are related to each other.
    - Values close to +1 show a strong positive relationship.
    - Values close to -1 show a strong negative relationship.
    - Values close to 0 show a weak linear relationship.
    """)
