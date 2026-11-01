# ==============================================================================
# pages/1_Model_Info.py — Model Information & Analysis
# ==============================================================================
# Contains: EDA & Visualisations, Preprocessing Pipeline, 4-Model Training,
# Model Comparison, Best Model Deep-Dive (confusion matrix, ROC/PR curves,
# feature importance, classification report, model download).
# ==============================================================================

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    init_session_state,
    plot_combined_model_comparison,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_target_distribution,
    get_models,
    preprocess_data,
    train_and_evaluate_models,
    load_data,
    clean_data_pipeline,
    needs_scaling,
)
import joblib

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Model Info | Phishing Detector",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# SHARED CSS
# ==============================================================================
st.markdown(
    """
<style>
    .page-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
        border-radius: 14px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.6rem;
        box-shadow: 0 6px 24px rgba(0,0,0,0.28);
    }
    .page-header h1 {
        color: #f1f5f9;
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
    }
    .page-header p {
        color: #94a3b8;
        margin: 0;
        font-size: 0.95rem;
    }
    .section-heading {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0.5rem 0 1rem 0;
        padding-left: 0.6rem;
        border-left: 4px solid #3b82f6;
    }
    .best-model-card {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 14px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.25);
    }
    .best-model-card h2 {
        color: #ffd700;
        margin: 0;
        font-size: 1.8rem;
    }
    .best-model-card h3 {
        color: #e0e0e0;
        margin: 0.3rem 0;
        font-weight: 400;
    }
    .metric-row {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        margin-top: 1.2rem;
    }
    .metric-item {
        text-align: center;
        padding: 0.6rem 1.2rem;
        background: rgba(255,255,255,0.08);
        border-radius: 8px;
        margin: 0.3rem;
    }
    .metric-item .value {
        font-size: 1.6rem;
        font-weight: bold;
        color: #4fc3f7;
    }
    .metric-item .label {
        font-size: 0.85rem;
        color: #ddd;
    }
    .model-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.1rem;
        margin-bottom: 0.5rem;
    }
    .model-card h4 {
        color: #60a5fa;
        margin: 0 0 0.3rem 0;
    }
    .model-card p {
        color: #94a3b8;
        font-size: 0.88rem;
        margin: 0;
    }
    .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: bold;
    }
    .badge-success {
        background-color: #14532d;
        color: #86efac;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# SESSION STATE
# ==============================================================================
init_session_state()

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("🛡️ **Phishing Detector**")
    if st.session_state.models_trained:
        st.markdown(
            "<span class='status-badge badge-success'>✓ 4 Models Ready</span>",
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.markdown("### 🗺️ Navigation")
    st.page_link("Home.py", label="🏠 Detection Home", icon="🛡️")
    st.page_link("pages/1_Model_Info.py", label="📊 Model Info & Analysis", icon="ℹ️")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded_file is not None:
        df_new = load_data(uploaded_file)
        if df_new is not None:
            st.session_state.df = df_new
            df_clean, _ = clean_data_pipeline(df_new)
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
            st.session_state.models_trained   = False
            st.success("Custom dataset loaded!")

    st.markdown("---")
    test_size    = st.slider("Test Split Ratio", 0.10, 0.40, 0.20, 0.05)
    random_state = st.number_input("Random Seed", 0, 1000, 42, step=1)
    btn_train    = st.button("🤖 Train 4 Models", use_container_width=True, type="primary")

    if btn_train:
        if st.session_state.get("df_clean") is None:
            st.error("Load a dataset first.")
        else:
            with st.spinner("Training 4 classifiers…"):
                (
                    X_train_s, X_test_s,
                    X_train_u, X_test_u,
                    y_train, y_test,
                    preprocessor,
                ) = preprocess_data(
                    st.session_state.df_clean, "Is_Phishing", test_size, random_state
                )
                st.session_state.X_train_scaled   = X_train_s
                st.session_state.X_test_scaled    = X_test_s
                st.session_state.X_train_unscaled = X_train_u
                st.session_state.X_test_unscaled  = X_test_u
                st.session_state.y_train          = y_train
                st.session_state.y_test           = y_test
                st.session_state.preprocessor     = preprocessor

                models = get_models(random_state)
                results_df, trained_models, model_predictions, errors = train_and_evaluate_models(
                    models, X_train_s, X_test_s, X_train_u, X_test_u, y_train, y_test
                )
                st.session_state.results_df        = results_df
                st.session_state.trained_models    = trained_models
                st.session_state.model_predictions = model_predictions
                st.session_state.best_model_name   = results_df.iloc[0]["Model"]
                st.session_state.models_trained    = True
                st.success(f"✅ Best: **{st.session_state.best_model_name}**")


# ==============================================================================
# PAGE HEADER  +  Back navigation
# ==============================================================================
st.markdown(
    """
<div class="page-header">
    <h1>📊 Model Information &amp; Analysis</h1>
    <p>EDA · Preprocessing Pipeline · 4-Model Training · Comparison · Best Model Deep-Dive</p>
</div>
""",
    unsafe_allow_html=True,
)

st.page_link("Home.py", label="← Back to Detection", icon="🛡️")
st.markdown("---")

# ==============================================================================
# DATA REFERENCES
# ==============================================================================
df_current       = st.session_state.get("df")
df_clean_current = st.session_state.get("df_clean", df_current)

if df_current is None:
    st.warning("⚠️ No dataset loaded. Upload a CSV in the sidebar or ensure `phishing_website_raw.csv` is present.")
    st.stop()

# ==============================================================================
# TABS
# ==============================================================================
(
    tab_overview,
    tab_eda,
    tab_preproc,
    tab_training,
    tab_compare,
    tab_deep,
) = st.tabs([
    "📋 Dataset Overview",
    "📊 EDA & Visualisations",
    "⚙️ Preprocessing Pipeline",
    "🤖 4-Model Training",
    "📈 Model Comparison",
    "🏆 Best Model Deep-Dive",
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: DATASET OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────
with tab_overview:
    st.markdown('<div class="section-heading">📋 Dataset Overview</div>', unsafe_allow_html=True)

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    legit_cnt = int((df_clean_current["Is_Phishing"] == 0).sum())
    phish_cnt = int((df_clean_current["Is_Phishing"] == 1).sum())

    kpi1.metric("Raw Records",     f"{len(df_current):,}")
    kpi2.metric("Clean Records",   f"{len(df_clean_current):,}")
    kpi3.metric("Input Features",  f"{df_clean_current.shape[1] - 1}")
    kpi4.metric("Legitimate (0)",  f"{legit_cnt:,} ({legit_cnt/len(df_clean_current)*100:.1f}%)")
    kpi5.metric("Phishing (1)",    f"{phish_cnt:,} ({phish_cnt/len(df_clean_current)*100:.1f}%)")

    st.markdown("---")
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.subheader("Raw Data Preview")
        n_rows = st.selectbox("Rows to display", [5, 10, 20, 50], index=0)
        st.dataframe(df_current.head(n_rows), use_container_width=True)
    with col_r:
        st.subheader("Target Balance")
        fig_tgt = plot_target_distribution(df_clean_current)
        st.pyplot(fig_tgt)
        plt.close(fig_tgt)

    st.markdown("---")
    st.subheader("Descriptive Statistics")
    st.dataframe(df_clean_current.describe().T, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: EDA & VISUALISATIONS
# ──────────────────────────────────────────────────────────────────────────────
with tab_eda:
    st.markdown('<div class="section-heading">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown("Explore distributions, outliers, and feature-target relationships.")

    eda1, eda2, eda3, eda4 = st.tabs([
        "Feature Distributions",
        "Outlier Detection (Boxplots)",
        "Correlation Heatmap",
        "Feature-Target Correlation",
    ])

    features_list = [c for c in df_clean_current.columns if c != "Is_Phishing"]

    with eda1:
        st.subheader("Distribution of Features")
        selected_feat = st.selectbox("Select feature", features_list, index=0, key="eda_hist")
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        col_data = df_clean_current[selected_feat].dropna()
        ax.hist(col_data, bins=30, color="#1976d2", edgecolor="black", alpha=0.7, density=True)
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(col_data)
            x_grid = np.linspace(col_data.min(), col_data.max(), 200)
            ax.plot(x_grid, kde(x_grid), color="#d32f2f", lw=2, label="KDE Overlay")
            ax.legend()
        except Exception:
            pass
        ax.set_title(f"Distribution: {selected_feat}", fontsize=12, fontweight="bold")
        ax.set_xlabel(selected_feat, fontweight="bold")
        ax.set_ylabel("Density", fontweight="bold")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with eda2:
        st.subheader("Outlier Check — Boxplots")
        box_feat = st.selectbox("Select feature", features_list, index=1, key="eda_box")
        fig, ax = plt.subplots(figsize=(8, 3.5), dpi=150)
        ax.boxplot(
            df_clean_current[box_feat].dropna(),
            vert=False, patch_artist=True,
            boxprops=dict(facecolor="#e0f2fe", color="#0284c7"),
            medianprops=dict(color="#d32f2f", lw=2),
        )
        ax.set_title(f"Boxplot: {box_feat}", fontsize=12, fontweight="bold")
        ax.set_xlabel(box_feat, fontweight="bold")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with eda3:
        st.subheader("Full Correlation Matrix (20 Features)")
        fig_corr = plot_correlation_heatmap(df_clean_current)
        st.pyplot(fig_corr)
        plt.close(fig_corr)

    with eda4:
        st.subheader("Ranked Correlation with Target (Is_Phishing)")
        corr_tgt = df_clean_current.corr()["Is_Phishing"].drop("Is_Phishing").sort_values()
        fig, ax = plt.subplots(figsize=(9, 6.5), dpi=150)
        colors_bar = ["#c62828" if v < 0 else "#2e7d32" for v in corr_tgt]
        bars = ax.barh(corr_tgt.index, corr_tgt.values, color=colors_bar,
                       edgecolor="black", height=0.65)
        for bar in bars:
            w = bar.get_width()
            offset = 0.01 if w >= 0 else -0.01
            ha = "left" if w >= 0 else "right"
            ax.text(w + offset, bar.get_y() + bar.get_height() / 2,
                    f"{w:.3f}", ha=ha, va="center",
                    fontsize=8.5, fontweight="bold")
        ax.set_title("Ranked Pearson Correlation with Is_Phishing",
                     fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Correlation Coefficient", fontweight="bold")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlim(min(corr_tgt.values) - 0.08, max(corr_tgt.values) + 0.08)
        ax.grid(True, axis="x", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: PREPROCESSING PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
with tab_preproc:
    st.markdown('<div class="section-heading">⚙️ Preprocessing Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        """
The dataset passes through a structured, leakage-free pipeline:

| Step | Operation | Detail |
|------|-----------|--------|
| 1 | **Duplicate Removal** | Exact row deduplication |
| 2 | **Invalid Value Clipping** | Negative values in count/length features → NaN |
| 3 | **Binary Validation** | Ensures {0,1} for flag columns |
| 4 | **Ratio Bounding** | Clips ratio columns to [0, 1] |
| 5 | **Median Imputation** | Fills remaining NaN using training-set medians |
| 6 | **Train/Test Split** | 80 % train / 20 % test, stratified |
| 7 | **Feature Scaling** | `StandardScaler` fit on train only (LR & NB only) |
        """
    )

    pp = st.session_state.get("preprocessor")
    if pp:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.subheader("Training Split (80 %)")
            st.info(f"Shape: `{st.session_state.X_train_scaled.shape}`")
        with col_p2:
            st.subheader("Test Split (20 %)")
            st.info(f"Shape: `{st.session_state.X_test_scaled.shape}`")

        st.subheader("Feature Columns (20 Total)")
        st.write(pp["feature_columns"])
    else:
        st.info("Preprocessing info will appear after dataset loads.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4: 4-MODEL TRAINING
# ──────────────────────────────────────────────────────────────────────────────
with tab_training:
    st.markdown('<div class="section-heading">🤖 4-Model Training Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        "The system trains and compares **4 distinct classifiers**. "
        "Click **'Train 4 Models'** in the sidebar to run."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            '<div class="model-card"><h4>1. Logistic Regression</h4>'
            '<p>Linear probability model with L2 regularisation and StandardScaler.</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="model-card"><h4>2. Decision Tree</h4>'
            '<p>Non-parametric tree-based recursive binary splitter (unscaled features).</p></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="model-card"><h4>3. Random Forest (Tuned)</h4>'
            '<p>Ensemble of 100 bagging trees with optimised depth & leaf settings.</p></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            '<div class="model-card"><h4>4. Naive Bayes</h4>'
            '<p>Gaussian probabilistic classifier based on Bayes theorem.</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    if st.session_state.models_trained:
        st.success("✅ All 4 models have been trained and evaluated on the held-out test split.")
    else:
        st.info("Click **'Train 4 Models'** in the sidebar to run the training pipeline.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 5: MODEL COMPARISON
# ──────────────────────────────────────────────────────────────────────────────
with tab_compare:
    st.markdown('<div class="section-heading">📈 Model Performance Comparison</div>', unsafe_allow_html=True)

    if not st.session_state.models_trained:
        st.info("Train the models first using the sidebar button.")
    else:
        res_df = st.session_state.results_df

        st.subheader("Performance Metrics Table")
        formatted_df = res_df.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]:
            formatted_df[col] = formatted_df[col].apply(
                lambda v: f"{v*100:.2f}%" if pd.notnull(v) else "N/A"
            )
        st.dataframe(formatted_df, use_container_width=True)

        st.markdown("---")
        st.subheader("Visual Comparison — All 4 Models")
        fig_comp = plot_combined_model_comparison(res_df)
        st.pyplot(fig_comp)
        plt.close(fig_comp)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 6: BEST MODEL DEEP-DIVE
# ──────────────────────────────────────────────────────────────────────────────
with tab_deep:
    st.markdown('<div class="section-heading">🏆 Best Model Deep-Dive</div>', unsafe_allow_html=True)

    if not st.session_state.models_trained:
        st.info("Train models first to view the detailed evaluation.")
    else:
        best_name  = st.session_state.best_model_name
        best_model = st.session_state.trained_models[best_name]
        best_preds = st.session_state.model_predictions[best_name]
        res_df     = st.session_state.results_df
        best_row   = res_df[res_df["Model"] == best_name].iloc[0]

        # ── Trophy card ────────────────────────────────────────────────────────
        st.markdown(
            f"""
<div class="best-model-card">
    <h2>🏆 Top Performing Model: {best_name}</h2>
    <h3>Optimised with 5-Fold Cross-Validation Hyperparameter Tuning</h3>
    <div class="metric-row">
        <div class="metric-item">
            <div class="value">{best_row['Accuracy']*100:.2f}%</div>
            <div class="label">Accuracy</div>
        </div>
        <div class="metric-item">
            <div class="value">{best_row['Precision']*100:.2f}%</div>
            <div class="label">Precision</div>
        </div>
        <div class="metric-item">
            <div class="value">{best_row['Recall']*100:.2f}%</div>
            <div class="label">Recall</div>
        </div>
        <div class="metric-item">
            <div class="value">{best_row['F1 Score']*100:.2f}%</div>
            <div class="label">F1-Score</div>
        </div>
        <div class="metric-item">
            <div class="value">{best_row['ROC-AUC']:.4f}</div>
            <div class="label">ROC-AUC</div>
        </div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ── Evaluation visuals ─────────────────────────────────────────────────
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.subheader("Confusion Matrix")
            fig_cm = plot_confusion_matrix(best_preds["confusion_matrix"])
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        with col_e2:
            st.subheader("ROC & PR Curves")
            if best_preds["y_prob"] is not None:
                fig_roc = plot_roc_curve(
                    st.session_state.y_test,
                    best_preds["y_prob"],
                    best_name,
                    best_row["ROC-AUC"],
                )
                st.pyplot(fig_roc)
                plt.close(fig_roc)

                fig_pr = plot_precision_recall_curve(
                    st.session_state.y_test, best_preds["y_prob"], best_name
                )
                st.pyplot(fig_pr)
                plt.close(fig_pr)

        st.markdown("---")

        # ── Feature importance ─────────────────────────────────────────────────
        st.subheader("Feature Importance Analysis")
        pp = st.session_state.preprocessor
        fig_imp = plot_feature_importance(best_model, pp["feature_columns"])
        if fig_imp:
            st.pyplot(fig_imp)
            plt.close(fig_imp)
        else:
            st.info("Feature importances not available for this model type.")

        st.markdown("---")

        # ── Classification report ──────────────────────────────────────────────
        st.subheader("Detailed Classification Report")
        cr_dict = best_preds["classification_report"]
        cr_df   = pd.DataFrame(cr_dict).T
        st.dataframe(
            cr_df.style.format({
                "precision": "{:.4f}",
                "recall":    "{:.4f}",
                "f1-score":  "{:.4f}",
                "support":   "{:,.0f}",
            }),
            use_container_width=True,
        )

        st.markdown("---")

        # ── Model download ─────────────────────────────────────────────────────
        st.subheader("💾 Model Artifact Download")
        buf = io.BytesIO()
        joblib.dump(best_model, buf)
        buf.seek(0)
        st.download_button(
            label="⬇️ Download Tuned Random Forest (.pkl)",
            data=buf,
            file_name="best_phishing_model.pkl",
            mime="application/octet-stream",
            use_container_width=True,
        )


# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#64748b; font-size:0.82rem;'>"
    "📊 Model Info &amp; Analysis &nbsp;·&nbsp; Phishing Website Detection &nbsp;·&nbsp; "
    "Built with Streamlit &amp; Scikit-learn"
    "</p>",
    unsafe_allow_html=True,
)
