import streamlit as st
import pandas as pd
import plotly.express as px
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.title("🎓 Student Performance Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("Simple 100 Student Marks.csv")
    df["total_marks"] = df[
        ["maths_marks", "science_marks", "english_marks", "social_studies_marks", "language_marks"]
    ].sum(axis=1)
    df["pass"] = (df["total_marks"] >= 200).astype(int)
    return df

df = load_data()
features = ["maths_marks", "science_marks", "english_marks", "social_studies_marks", "language_marks"]
X = df[features]
y = df["pass"]

# Load or train model
MODEL_FILE = "student_model.sav"
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_FILE)

st.markdown("## 🔍 Key Metrics")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("👥 Total Students", len(df))
with col2:
    pass_pct = round((df["pass"].sum() / len(df)) * 100, 2)
    st.metric("✅ Pass %", f"{pass_pct}%")
with col3:
    avg_total = round(df["total_marks"].mean(), 2)
    st.metric("📊 Avg Total Marks", avg_total)
with col4:
    top_scorer = df.loc[df["total_marks"].idxmax()]["student_name"]
    st.metric("🏅 Top Scorer", top_scorer)

st.divider()

st.subheader("📈 Visual Insights")

col1, col2 = st.columns(2)

with col1:
    pie_data = df["pass"].value_counts().reset_index()
    pie_data.columns = ["Pass (1) / Fail (0)", "Count"]
    pie_fig = px.pie(pie_data, names="Pass (1) / Fail (0)", values="Count", title="Pass vs Fail Ratio")
    st.plotly_chart(pie_fig, use_container_width=True)

with col2:
    avg_marks = df[features].mean().reset_index()
    avg_marks.columns = ["Subject", "Average Marks"]
    bar_fig = px.bar(avg_marks, x="Subject", y="Average Marks", title="Average Marks per Subject", color="Subject")
    st.plotly_chart(bar_fig, use_container_width=True)

st.subheader("📦 Score Distribution")
box_fig = px.box(df, y=features, title="Subject-wise Score Spread", points="outliers")
st.plotly_chart(box_fig, use_container_width=True)

st.subheader("🏆 Top 10 Students")
top10_df = df.sort_values("total_marks", ascending=False).head(10)
top10_fig = px.bar(top10_df, x="student_name", y="total_marks", title="Top 10 Performers", color="total_marks")
st.plotly_chart(top10_fig, use_container_width=True)

st.subheader("📊 Subject-wise Pass % (≥ 40)")
subject_pass = {subj: (df[subj] >= 40).mean()*100 for subj in features}
subject_pass_df = pd.DataFrame(list(subject_pass.items()), columns=["Subject", "Pass Percentage"])
subj_fig = px.bar(subject_pass_df, x="Subject", y="Pass Percentage", title="Subject-wise Pass Rate", color="Subject")
st.plotly_chart(subj_fig, use_container_width=True)

st.divider()

st.subheader("🧮 Predict Pass/Fail for a Student")

student_name = st.selectbox("Select Student Name", df["student_name"].unique())

if student_name:
    selected_student = df[df["student_name"] == student_name][features]
    prediction = model.predict(selected_student)[0]
    total_marks = selected_student.sum(axis=1).values[0]

    if prediction == 1:
        st.success(f"✅ **{student_name}** is predicted to **PASS** with **{total_marks}** marks.")
    else:
        st.error(f"❌ **{student_name}** is predicted to **FAIL** with **{total_marks}** marks.")

# NEW SECTION: Manual input for unknown student
st.divider()
st.subheader("📝 Enter Marks for a New Student")

with st.form("new_student_form"):
    name_input = st.text_input("Student Name")
    maths = st.number_input("Maths Marks", min_value=0, max_value=100, value=0)
    science = st.number_input("Science Marks", min_value=0, max_value=100, value=0)
    english = st.number_input("English Marks", min_value=0, max_value=100, value=0)
    social = st.number_input("Social Studies Marks", min_value=0, max_value=100, value=0)
    language = st.number_input("Language Marks", min_value=0, max_value=100, value=0)
    
    submitted = st.form_submit_button("Predict Result")

if submitted:
    new_data = pd.DataFrame([{
        "maths_marks": maths,
        "science_marks": science,
        "english_marks": english,
        "social_studies_marks": social,
        "language_marks": language
    }])
    pred = model.predict(new_data)[0]
    total = new_data.sum(axis=1).values[0]

    if pred == 1:
        st.success(f"✅ **{name_input or 'Student'}** is predicted to **PASS** with **{total}** marks.")
    else:
        st.error(f"❌ **{name_input or 'Student'}** is predicted to **FAIL** with **{total}** marks.")