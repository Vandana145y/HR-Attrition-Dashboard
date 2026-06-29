import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="HR Attrition Dashboard", page_icon="📊", layout="wide")

st.title("📊 HR Attrition Dashboard + Prediction")
st.markdown("Built using Streamlit + Pandas + ML Model")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\parip\Desktop\SECOND_PROJECT\WA_Fn-UseC_-HR-Employee-Attrition.csv")
    return df

df = load_data()

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("Filters")

departments = st.sidebar.multiselect(
    "Department",
    options=df["Department"].unique(),
    default=df["Department"].unique()
)

genders = st.sidebar.multiselect(
    "Gender",
    options=df["Gender"].unique(),
    default=df["Gender"].unique()
)

job_roles = st.sidebar.multiselect(
    "Job Role",
    options=df["JobRole"].unique(),
    default=df["JobRole"].unique()
)

age_range = st.sidebar.slider(
    "Age Range",
    int(df["Age"].min()),
    int(df["Age"].max()),
    (int(df["Age"].min()), int(df["Age"].max()))
)

filtered = df[
    (df["Department"].isin(departments)) &
    (df["Gender"].isin(genders)) &
    (df["JobRole"].isin(job_roles)) &
    (df["Age"].between(age_range[0], age_range[1]))
]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing {len(filtered)} / {len(df)} employees")

# ---------------- KPIs ----------------
total = len(filtered)
attrition = (filtered["Attrition"] == "Yes").sum()
rate = (attrition / total * 100) if total else 0
income = filtered["MonthlyIncome"].mean() if total else 0

c1, c2, c3, c4 = st.columns(4)

c1.metric("Employees", total)
c2.metric("Attrition", attrition)
c3.metric("Attrition Rate", f"{rate:.1f}%")
c4.metric("Avg Income", f"${income:,.0f}")

st.markdown("---")
# ---------------- ML MODEL ----------------
ml_df = df.copy()

le_dict = {}

for col in ml_df.columns:
    if ml_df[col].dtype == "object":
        le = LabelEncoder()
        ml_df[col] = le.fit_transform(ml_df[col])
        le_dict[col] = le

X = ml_df.drop("Attrition", axis=1)
y = ml_df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------- PREDICTION UI ----------------
st.sidebar.markdown("---")
st.sidebar.header("🔮 Predict Attrition")

age = st.sidebar.number_input("Age", 18, 60, 30)
income_input = st.sidebar.number_input("Monthly Income", 1000, 20000, 5000)
years = st.sidebar.number_input("Years at Company", 0, 40, 5)
overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])

if st.sidebar.button("Predict"):
    
    input_df = pd.DataFrame([[age, income_input, years]],
                            columns=["Age", "MonthlyIncome", "YearsAtCompany"])

    # Add missing columns as 0
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[X.columns]

    result = model.predict(input_df)[0]

    if result == 1:
        st.sidebar.error("⚠ Employee Likely to LEAVE")
    else:
        st.sidebar.success("✅ Employee Likely to STAY")

