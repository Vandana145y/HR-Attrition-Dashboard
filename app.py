import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="HR Attrition Dashboard", layout="wide")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

df = load_data()

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("Filters")

dept = st.sidebar.multiselect("Department", df["Department"].unique(), df["Department"].unique())
gender = st.sidebar.multiselect("Gender", df["Gender"].unique(), df["Gender"].unique())
jobrole = st.sidebar.multiselect("Job Role", df["JobRole"].unique(), df["JobRole"].unique())

filtered = df[
    (df["Department"].isin(dept)) &
    (df["Gender"].isin(gender)) &
    (df["JobRole"].isin(jobrole))
]

# ---------------- TITLE ----------------
st.title("📊 HR Employee Attrition Dashboard")

# ---------------- KPIs ----------------
total = len(filtered)
attrition = (filtered["Attrition"] == "Yes").sum()
rate = (attrition / total) * 100 if total else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Employees", total)
col2.metric("Attrition Count", attrition)
col3.metric("Attrition Rate", f"{rate:.2f}%")

st.markdown("---")

# ---------------- TABLE ----------------
st.subheader("Dataset Preview")
st.dataframe(filtered)

# ---------------- SIMPLE PREDICTION SECTION ----------------
st.subheader("🔮 Simple Attrition Prediction")

age = st.slider("Age", 18, 60, 30)
income = st.number_input("Monthly Income", 1000, 20000, 5000)
overtime = st.selectbox("OverTime", ["Yes", "No"])

# SIMPLE RULE MODEL (NO ML ERROR)
if st.button("Predict Attrition"):
    score = 0

    if age < 30:
        score += 1
    if income < 5000:
        score += 1
    if overtime == "Yes":
        score += 1

    if score >= 2:
        st.error("⚠️ High chance of Attrition")
    else:
        st.success("✅ Low chance of Attrition")
