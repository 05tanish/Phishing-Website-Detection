# ==============================================================================
# utils.py — Shared Utilities for Phishing Detection App
# ==============================================================================
# All helper functions, plotting utilities, and session-state initialization
# shared between Home.py and pages/1_Model_Info.py.
# ==============================================================================

import io
import os
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


# ==============================================================================
# DATA LOADING & CLEANING
# ==============================================================================

def load_data(file_or_path):
    """Load CSV dataset from an uploaded file object or a file path string."""
    try:
        df = pd.read_csv(file_or_path)
        if df.empty:
            st.error("The dataset file is empty.")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading CSV dataset: {e}")
        return None


def get_dataset_info(df, target_col):
    """Return a dictionary summarising the dataset."""
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "numerical_cols": list(df.select_dtypes(include=[np.number]).columns),
        "categorical_cols": list(df.select_dtypes(exclude=[np.number]).columns),
        "duplicates": int(df.duplicated().sum()),
        "missing_total": int(df.isnull().sum().sum()),
        "size_bytes": df.memory_usage(deep=True).sum(),
        "target_col": target_col,
        "feature_count": df.shape[1] - 1,
    }


def clean_data_pipeline(df, target_col="Is_Phishing"):
    """
    Standard data-cleaning pipeline mirroring the project notebook:
      1. Drop exact duplicate rows.
      2. Convert invalid negative values in count/length features to NaN.
      3. Ensure binary columns are strictly {0, 1}.
      4. Ensure ratio columns are within [0, 1].
      5. Median imputation for remaining missing numeric values.
      6. Integer type-casting for appropriate columns.
    Returns (df_clean, duplicates_removed).
    """
    before_len = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True).copy()
    duplicates_removed = before_len - len(df_clean)

    count_length_cols = [
        "URL_Length", "Num_Dots", "Num_Hyphens", "Num_Special_Chars",
        "Num_Subdomains", "Domain_Age_Days", "Domain_Registration_Length",
        "Num_Redirects", "Form_Count", "Iframe_Count", "Popup_Count",
        "Domain_Name_Length", "URL_Entropy",
    ]
    for col in count_length_cols:
        if col in df_clean.columns:
            df_clean.loc[df_clean[col] < 0, col] = np.nan

    binary_cols = [
        "Has_IP_Address", "Has_HTTPS", "Has_Suspicious_Words",
        "Password_Field_Present", "Favicon_External",
    ]
    for col in binary_cols:
        if col in df_clean.columns:
            df_clean.loc[~df_clean[col].isin([0, 1]), col] = np.nan

    ratio_cols = ["External_Link_Ratio", "Image_Link_Ratio"]
    for col in ratio_cols:
        if col in df_clean.columns:
            df_clean.loc[(df_clean[col] < 0) | (df_clean[col] > 1), col] = np.nan

    # Median imputation on all non-target features
    features = [c for c in df_clean.columns if c != target_col]
    for col in features:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Integer casting
    for col in count_length_cols:
        if col in df_clean.columns and col != "URL_Entropy":
            df_clean[col] = df_clean[col].astype(int)
    for col in binary_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(int)
    if target_col in df_clean.columns:
        df_clean[target_col] = df_clean[target_col].astype(int)

    return df_clean, duplicates_removed


def preprocess_data(df, target_col="Is_Phishing", test_size=0.20, random_state=42):
    """
    Split data and fit preprocessing objects.
    StandardScaler is fitted ONLY on the training split to prevent leakage.
    Returns:
        X_train_scaled, X_test_scaled,
        X_train_unscaled, X_test_unscaled,
        y_train, y_test,
        preprocessor (dict with scaler, imputer, feature columns)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_columns = list(X.columns)

    num_cols = list(X.select_dtypes(include=[np.number]).columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    num_imputer = None
    if num_cols:
        num_imputer = SimpleImputer(strategy="median")
        X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
        X_test[num_cols] = num_imputer.transform(X_test[num_cols])

    scaler = StandardScaler()
    all_final_features = list(X_train.columns)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_unscaled = X_train.values
    X_test_unscaled = X_test.values

    preprocessor = {
        "num_imputer": num_imputer,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "all_final_features": all_final_features,
    }

    return (
        X_train_scaled,
        X_test_scaled,
        X_train_unscaled,
        X_test_unscaled,
        y_train.values,
        y_test.values,
        preprocessor,
    )


# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

def get_models(random_state=42):
    """
    Return the 4 core classifiers used in this project:
      1. Logistic Regression   — linear baseline with L2 regularisation
      2. Decision Tree         — non-parametric recursive splitter
      3. Random Forest (Tuned) — ensemble of 100 bagging trees
      4. Naive Bayes           — Gaussian probabilistic classifier
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(
            max_depth=20,
            min_samples_leaf=2,
            min_samples_split=5,
            n_estimators=100,
            random_state=random_state,
        ),
        "Naive Bayes": GaussianNB(),
    }


def needs_scaling(model_name):
    """Return True if the model requires feature scaling."""
    return model_name in {"Logistic Regression", "Naive Bayes"}


def train_and_evaluate_models(
    models,
    X_train_scaled, X_test_scaled,
    X_train_unscaled, X_test_unscaled,
    y_train, y_test,
):
    """
    Train all supplied models and evaluate on the held-out test split.
    Returns:
        results_df       — DataFrame of per-model metrics sorted by F1 Score desc
        trained_models   — {name: fitted_model}
        model_predictions — {name: {y_pred, y_prob, confusion_matrix, classification_report}}
        errors           — {name: error_string} for any models that failed
    """
    results = []
    trained_models = {}
    model_predictions = {}
    errors = {}

    for name, model in models.items():
        t0 = time.time()
        try:
            X_tr, X_te = (
                (X_train_scaled, X_test_scaled)
                if needs_scaling(name)
                else (X_train_unscaled, X_test_unscaled)
            )

            model.fit(X_tr, y_train)
            train_time = time.time() - t0
            y_pred = model.predict(X_te)

            y_prob = None
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_te)[:, 1]

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec  = recall_score(y_test, y_pred, zero_division=0)
            f1   = f1_score(y_test, y_pred, zero_division=0)

            roc = np.nan
            if y_prob is not None:
                try:
                    roc = roc_auc_score(y_test, y_prob)
                except Exception:
                    pass

            cm     = confusion_matrix(y_test, y_pred)
            report = classification_report(
                y_test, y_pred, zero_division=0, output_dict=True
            )

            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "ROC-AUC": roc,
                "Training Time (s)": round(train_time, 3),
            })
            trained_models[name] = model
            model_predictions[name] = {
                "y_pred": y_pred,
                "y_prob": y_prob,
                "confusion_matrix": cm,
                "classification_report": report,
            }

        except Exception as e:
            errors[name] = str(e)

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(
            by="F1 Score", ascending=False
        ).reset_index(drop=True)

    return results_df, trained_models, model_predictions, errors


# ==============================================================================
# SINGLE-SAMPLE PREDICTION
# ==============================================================================

def predict_sample(input_data, preprocessor, model, model_name):
    """
    Preprocess a single feature dict and return (prediction, probability_array).
    prediction: 0 = Legitimate, 1 = Phishing
    probability_array: [P(legit), P(phish)] or None if model has no predict_proba
    """
    input_df = pd.DataFrame([input_data])
    feature_cols = preprocessor["feature_columns"]
    scaler = preprocessor["scaler"]

    if preprocessor["num_imputer"] is not None:
        input_df[feature_cols] = preprocessor["num_imputer"].transform(
            input_df[feature_cols]
        )

    input_df = input_df[feature_cols]

    input_arr = (
        scaler.transform(input_df) if needs_scaling(model_name) else input_df.values
    )

    prediction = int(model.predict(input_arr)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_arr)[0]

    return prediction, probability


# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def plot_target_distribution(df, target_col="Is_Phishing"):
    """Styled bar chart of target class distribution."""
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    counts = df[target_col].value_counts()
    colors = ["#2e7d32", "#c62828"]
    labels = ["Legitimate (0)", "Phishing (1)"]
    vals = [counts.get(0, 0), counts.get(1, 0)]
    bars = ax.bar(labels, vals, color=colors, width=0.45, edgecolor="black", linewidth=1)

    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        ax.annotate(
            f"{h:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
        )

    ax.set_title("Target Class Distribution (Is_Phishing)", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Count", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, title="Confusion Matrix — Final Model"):
    """Styled confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=150)
    cmap = LinearSegmentedColormap.from_list("custom", ["#e3f2fd", "#1565c0"])
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels = ["Legitimate (0)", "Phishing (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("Actual Label", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    thresh = cm.max() / 2.0
    cell_names = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cell_names[i][j]}\n{cm[i, j]:,}",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)
    plt.tight_layout()
    return fig


def plot_roc_curve(y_test, y_prob, model_name, auc_val):
    """ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    ax.plot(fpr, tpr, color="#1565c0", lw=2.5,
            label=f"{model_name} (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="#9e9e9e", lw=1.5, linestyle="--",
            label="Random Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=10, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_precision_recall_curve(y_test, y_prob, model_name):
    """Precision-Recall curve with Average Precision score."""
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    ax.plot(rec, prec, color="#2e7d32", lw=2.5,
            label=f"{model_name} (AP = {ap:.4f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=10, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=10, fontweight="bold")
    ax.set_title("Precision-Recall (PR) Curve", fontsize=11, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_n=20):
    """Horizontal bar chart of Gini-based feature importances."""
    if not hasattr(model, "feature_importances_"):
        return None
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    bars = ax.barh(
        sorted_features[::-1], sorted_importances[::-1],
        color="#0284c7", edgecolor="black", height=0.65,
    )
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}", ha="left", va="center",
                fontsize=8.5, fontweight="bold")
    ax.set_title(
        "Feature Importance Ranking (Gini Impurity Reduction)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Relative Importance", fontsize=10, fontweight="bold")
    ax.set_xlim(0, max(sorted_importances) * 1.15)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df):
    """Full correlation matrix heatmap for all numeric columns."""
    fig, ax = plt.subplots(figsize=(14, 11), dpi=150)
    corr = df.corr()
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cols = df.columns
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center", color=color, fontsize=6.5)
    ax.set_title("Correlation Matrix Across All Features",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def plot_combined_model_comparison(results_df):
    """Grouped bar chart: Accuracy, Precision, Recall, F1 across all 4 models."""
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    models = results_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.18
    colors = ["#1565c0", "#2e7d32", "#e65100", "#6a1b9a"]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    for i, metric in enumerate(metrics):
        vals = results_df[metric].values
        bars = ax.bar(
            x + i * width, vals, width,
            label=metric, color=colors[i],
            edgecolor="black", linewidth=0.6,
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=7.5, rotation=90)

    ax.set_xlabel("Classification Model", fontsize=10, fontweight="bold")
    ax.set_ylabel("Score", fontsize=10, fontweight="bold")
    ax.set_title("Combined Performance Comparison Across 4 Models",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


# ==============================================================================
# SESSION STATE INITIALISATION
# ==============================================================================

def init_session_state():
    """
    Initialise all shared session_state keys and auto-load dataset + pre-trained
    model PKL if they are present on disk.  Safe to call from both pages.
    """
    # ── Raw dataset ────────────────────────────────────────────────────────────
    if "df" not in st.session_state or st.session_state.df is None:
        if os.path.exists("phishing_website_raw.csv"):
            st.session_state.df = load_data("phishing_website_raw.csv")

    if "models_trained" not in st.session_state:
        st.session_state.models_trained = False

    # ── Auto-process dataset if available ──────────────────────────────────────
    if (
        st.session_state.get("df") is not None
        and "df_clean" not in st.session_state
    ):
        df_clean, _ = clean_data_pipeline(st.session_state.df)
        st.session_state.df_clean = df_clean
        (
            X_train_s, X_test_s,
            X_train_u, X_test_u,
            y_train, y_test,
            preprocessor,
        ) = preprocess_data(df_clean, "Is_Phishing", 0.20, 42)
        st.session_state.X_train_scaled   = X_train_s
        st.session_state.X_test_scaled    = X_test_s
        st.session_state.X_train_unscaled = X_train_u
        st.session_state.X_test_unscaled  = X_test_u
        st.session_state.y_train          = y_train
        st.session_state.y_test           = y_test
        st.session_state.preprocessor     = preprocessor

    # ── Auto-load pre-trained PKL if not yet trained ───────────────────────────
    if (
        os.path.exists("api/best_phishing_model.pkl")
        and not st.session_state.models_trained
        and st.session_state.get("preprocessor") is not None
    ):
        try:
            joblib.load("api/best_phishing_model.pkl")   # validate file is readable
            models = get_models(42)
            results_df, trained_models, model_predictions, errors = train_and_evaluate_models(
                models,
                st.session_state.X_train_scaled,
                st.session_state.X_test_scaled,
                st.session_state.X_train_unscaled,
                st.session_state.X_test_unscaled,
                st.session_state.y_train,
                st.session_state.y_test,
            )
            st.session_state.results_df       = results_df
            st.session_state.trained_models   = trained_models
            st.session_state.model_predictions = model_predictions
            st.session_state.best_model_name  = "Random Forest"
            st.session_state.models_trained   = True
        except Exception as e:
            st.warning(f"Could not auto-load pre-trained model: {e}")
