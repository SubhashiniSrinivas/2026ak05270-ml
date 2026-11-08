from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, ConfusionMatrixDisplay
)

BASE = Path(__file__).resolve().parent

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

st.set_page_config(page_title="ML Model Comparison", layout="wide")
st.title("Breast Cancer Classification")
st.write("Compare five classifiers on labelled test data.")
st.caption("Target: 0 = Benign | 1 = Malignant. Educational demonstration.")

@st.cache_resource
def load_models():
    return {
        name: joblib.load(BASE / "model" / filename)
        for name, filename in MODEL_FILES.items()
    }

models = load_models()
features = list(models["Logistic Regression"].feature_names_in_)

st.download_button(
    "Download sample test CSV",
    data=(BASE / "test_data.csv").read_bytes(),
    file_name="test_data.csv",
    mime="text/csv"
)

uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
selected = st.selectbox("Select a model", list(models))

st.caption(
    "Upload the 30 original input features and a target column containing "
    "0 or 1. Saved models are used without retraining."
)

if uploaded is None:
    st.info("Download the sample CSV above, then upload it to evaluate the models.")
    st.stop()

try:
    data = pd.read_csv(uploaded)

    if data.empty:
        raise ValueError("The uploaded CSV contains no data rows.")

    required = features + ["target"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    X = data[features].apply(pd.to_numeric, errors="raise")
    y = pd.to_numeric(data["target"], errors="raise")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Input features contain missing or infinite values.")

    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("The target column must contain only 0 and 1.")

    y = y.astype(int)

except (ValueError, TypeError, pd.errors.ParserError) as error:
    st.error(f"Please check your CSV: {error}")
    st.stop()

st.success(f"Loaded {len(X)} test records with {X.shape[1]} features.")

with st.expander("Preview uploaded data"):
    st.dataframe(data.head(10))

if y.nunique() < 2:
    st.warning(
        "Only one target class is present. AUC is undefined and is shown as N/A."
    )

results = []
predictions = {}

for name, model in models.items():
    predicted = model.predict(X)
    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X)[:, positive_index]
    predictions[name] = predicted

    results.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y, predicted),
        "AUC": (
            roc_auc_score(y, probabilities)
            if y.nunique() == 2 else np.nan
        ),
        "Precision": precision_score(y, predicted, zero_division=0),
        "Recall": recall_score(y, predicted, zero_division=0),
        "F1": f1_score(y, predicted, zero_division=0),
        "MCC": matthews_corrcoef(y, predicted)
    })

comparison = pd.DataFrame(results).set_index("ML Model Name")

st.subheader(selected)
columns = st.columns(6)
for column, metric in zip(columns, comparison.columns):
    value = comparison.loc[selected, metric]
    column.metric(metric, "N/A" if pd.isna(value) else f"{value:.4f}")

st.caption("Precision, recall and F1 use malignant (1) as the positive class.")

st.subheader("Confusion matrix")
matrix = confusion_matrix(y, predictions[selected], labels=[0, 1])

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=["Benign", "Malignant"]
).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(selected)
st.pyplot(fig)
plt.close(fig)

st.subheader("Comparison of all five models")
st.dataframe(comparison.style.format("{:.4f}", na_rep="N/A"))

st.download_button(
    "Download evaluation results",
    data=comparison.to_csv().encode("utf-8"),
    file_name="evaluation_results.csv",
    mime="text/csv"
)