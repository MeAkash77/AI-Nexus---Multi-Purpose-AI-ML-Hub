import pandas as pd
import numpy as np
import os
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay


# ---------------------------------------------------
# PATH SETUP
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Student_performance_data_.csv")
MODEL_PATH = os.path.join(BASE_DIR, "student_gpa_model.pkl")


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


dataset = load_data(DATA_PATH)

# ✅ Create interaction feature automatically
if "StudyGradeInteraction" not in dataset.columns:
    dataset["StudyGradeInteraction"] = (
        dataset["StudyTimeWeekly"] * dataset["GradeClass"]
    )


# ---------------------------------------------------
# VALIDATE REQUIRED COLUMNS
# ---------------------------------------------------
required_cols = [
    "StudentID",
    "Age",
    "StudyTimeWeekly",
    "GradeClass",
    "StudyGradeInteraction",
    "GPA",
]

for col in required_cols:
    if col not in dataset.columns:
        st.error(f"Missing column: {col}")
        st.stop()


# ---------------------------------------------------
# FEATURES & SPLIT
# ---------------------------------------------------
X = dataset[
    ["StudentID", "Age", "StudyTimeWeekly",
     "GradeClass", "StudyGradeInteraction"]
]

y = dataset["GPA"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------
# SCALING
# ---------------------------------------------------
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# ---------------------------------------------------
# TRAIN MODEL (ONLY IF NOT SAVED)
# ---------------------------------------------------
@st.cache_resource
def train_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    param_grid = {"n_neighbors": np.arange(1, 20)}
    knn = KNeighborsRegressor()

    grid_search = GridSearchCV(knn, param_grid, cv=5)
    grid_search.fit(x_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    best_model.fit(x_train_scaled, y_train)

    joblib.dump(best_model, MODEL_PATH)
    return best_model


best_knn_regressor = train_model()


# ---------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------
st.set_page_config(
    page_title="Student GPA Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student GPA Prediction App")

st.markdown("""
Predict student GPA using a K-Nearest Neighbors ML model.
Explore dataset insights and model performance interactively.
""")


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:
    st.header("🔧 Settings")

    show_dataset = st.checkbox("Show Dataset", True)
    show_corr_heatmap = st.checkbox("Correlation Heatmap", True)
    show_scatterplot = st.checkbox("Scatter Plot", True)
    show_performance = st.checkbox("Model Performance", True)


# ---------------------------------------------------
# INPUT FORM
# ---------------------------------------------------
with st.form("prediction_form"):
    st.subheader("🔍 Enter Student Details")

    StudentID = st.number_input("Student ID", 1, value=3392)
    Age = st.number_input("Age", 1, value=16)
    StudyTimeWeekly = st.number_input("Study Time Weekly", 0.0, value=17.8)
    GradeClass = st.number_input("Grade Class", 1.0, value=1.0)

    submit_button = st.form_submit_button("🎓 Predict GPA")


# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
if submit_button:

    interaction = StudyTimeWeekly * GradeClass

    x_input = np.array(
        [[StudentID, Age, StudyTimeWeekly, GradeClass, interaction]]
    )

    x_scaled = scaler.transform(x_input)
    prediction = best_knn_regressor.predict(x_scaled)

    st.success(f"🎉 Predicted GPA: **{prediction[0]:.3f}**")


# ---------------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------------
if show_dataset:
    st.subheader("Dataset Preview")
    st.write(dataset.head())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    sns.histplot(dataset["Age"], ax=axes[0, 0])
    sns.histplot(dataset["StudyTimeWeekly"], ax=axes[0, 1])
    sns.histplot(dataset["GradeClass"], ax=axes[0, 2])
    sns.histplot(dataset["StudyGradeInteraction"], ax=axes[1, 0])
    sns.histplot(dataset["GPA"], ax=axes[1, 1])

    plt.tight_layout()
    st.pyplot(fig)


if show_scatterplot:
    st.subheader("GPA vs Study Time")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=dataset,
        x="StudyTimeWeekly",
        y="GPA",
        hue="GradeClass"
    )
    st.pyplot(plt)


if show_corr_heatmap:
    st.subheader("Correlation Heatmap")

    plt.figure(figsize=(10, 6))
    corr = dataset[
        ["Age", "StudyTimeWeekly",
         "GradeClass", "StudyGradeInteraction", "GPA"]
    ].corr()

    sns.heatmap(corr, annot=True, cmap="coolwarm")
    st.pyplot(plt)


# ---------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------
if show_performance:
    st.subheader("Model Performance")

    cross_scores = cross_val_score(
        best_knn_regressor, x_train_scaled, y_train, cv=5
    )

    y_pred_test = best_knn_regressor.predict(x_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)

    st.write(f"Mean CV R² Score: {np.mean(cross_scores):.2f}")
    st.write(f"Test RMSE: {rmse:.2f}")
    st.write(f"Test R² Score: {r2:.2f}")

    bins = [0, 2, 2.5, 3, 3.5, 4]
    y_test_binned = np.digitize(y_test, bins) - 1
    y_pred_binned = np.digitize(y_pred_test, bins) - 1

    cm = confusion_matrix(y_test_binned, y_pred_binned)

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    st.pyplot(fig)
