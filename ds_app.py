import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from src.data import prepare_features, NUMERIC_COLS
from src.model import TitanicMLP

# i will save it in the "models" folder.
MODELS_DIR = Path("models")

@st.cache_resource
def load_artifacts():
    model_path = MODELS_DIR / "titanic_mlp.pt"
    scaler_path = MODELS_DIR / "scaler.pkl"
    feat_path = MODELS_DIR / "feature_names.json"
    # throw errors if a certain file is missing
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature names file not found: {feat_path}")

    with open(feat_path, "r") as f:
        feature_names = json.load(f)

    scaler = joblib.load(scaler_path)
    input_dim = len(feature_names)
    model = TitanicMLP(input_dim=input_dim)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return model, scaler, feature_names


def preprocess_for_inference(df_raw: pd.DataFrame, scaler, feature_names):

    df_proc = prepare_features(df_raw)

    y_true = None
    if "Survived" in df_proc.columns:
        y_true = df_proc["Survived"].values.astype(int)
        df_features = df_proc.drop("Survived", axis=1)
    else:
        df_features = df_proc

    # i make sure all the columns exist
    for col in feature_names:
        if col not in df_features.columns:
            df_features[col] = 0

    # order
    df_features = df_features[feature_names]
    # matrix for the my model
    X = df_features.values.astype(np.float32)

    numeric_indices = [feature_names.index(col) for col in NUMERIC_COLS if col in feature_names]

    X_scaled = X.copy()
    # normalization
    if numeric_indices:
        X_scaled[:, numeric_indices] = scaler.transform(X_scaled[:, numeric_indices])

    return X_scaled, y_true, df_features


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    return fig


def plot_prob_hist(probs, y_true=None):
    fig, ax = plt.subplots()
    if y_true is not None:
        probs = np.asarray(probs)
        y_true = np.asarray(y_true)
        ax.hist(probs[y_true == 0], bins=20, alpha=0.6, label="True = 0")
        ax.hist(probs[y_true == 1], bins=20, alpha=0.6, label="True = 1")
        ax.legend()
    else:
        ax.hist(probs, bins=20, alpha=0.8)
    ax.set_xlabel("Predicted probability of survival")
    ax.set_ylabel("Count")
    ax.set_title("Predicted probability distribution")
    return fig



def main():
    st.title("Titanic Survival Prediction – PyTorch & Streamlit - My app (Naor)")
    st.markdown(
        """
        This app loads a trained PyTorch model for the Titanic dataset,
        applies the **same preprocessing pipeline** used during training,
        and evaluates the model on a user-provided CSV.
        """
    )
    st.sidebar.header("Input options")

    csv_path = st.sidebar.text_input("Path to CSV file (optional)", value="")
    uploaded_file = st.sidebar.file_uploader("Or upload a CSV file", type=["csv"])

    if not csv_path and not uploaded_file:
        st.info("Please provide a CSV path in the sidebar or upload a CSV file.")
        return

    # loading the csv
    try:
        if uploaded_file is not None:
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    st.subheader("Raw data preview")
    st.dataframe(df_raw.head())

    # loading model and scaler
    try:
        model, scaler, feature_names = load_artifacts()
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        return

    try:
        X_scaled, y_true, df_features = preprocess_for_inference(
            df_raw, scaler, feature_names
        )
    except Exception as e:
        st.error(f"Failed to preprocess data: {e}")
        return

    st.subheader("Processed features (first 5 rows)")
    st.dataframe(df_features.head())

    with torch.no_grad():
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        logits = model(X_t)
        probs = torch.sigmoid(logits).numpy().flatten()
        preds = (probs > 0.5).astype(int)

    st.subheader("Predictions")
    result_df = pd.DataFrame(
        {
            "predicted_survived": preds,
            "predicted_prob": probs,
        }
    )
    if y_true is not None:
        result_df["true_survived"] = y_true

    st.dataframe(result_df.head())

    # evaluation
    if y_true is not None:
        st.subheader("Evaluation Metrics")

        acc = accuracy_score(y_true, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, preds, average="binary", zero_division=0
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{acc:.3f}")
        col2.metric("Precision", f"{prec:.3f}")
        col3.metric("Recall", f"{rec:.3f}")
        col4.metric("F1-score", f"{f1:.3f}")

        st.subheader("Confusion Matrix")
        fig_cm = plot_confusion_matrix(y_true, preds)
        st.pyplot(fig_cm)

        st.subheader("Predicted probability distribution")
        fig_hist = plot_prob_hist(probs, y_true)
        st.pyplot(fig_hist)
    else:
        st.info("No 'Survived' column found in the CSV – showing predictions only.")


if __name__ == "__main__":
    main()
