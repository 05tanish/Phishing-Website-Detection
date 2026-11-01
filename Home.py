# ==============================================================================
# Home.py — Phishing Website Detection · Main Entry Point
# ==============================================================================
#
# DEPLOYMENT NOTES
# ─────────────────────────────────────────────────────────────────────────────
# ⚠️  Streamlit apps CANNOT be deployed on Vercel.
#     Vercel is a static/serverless platform and does not support long-running
#     Python processes or WebSocket connections required by Streamlit.
#
# ✅  Recommended deployment options:
#     1. Streamlit Community Cloud — https://streamlit.io/cloud
#        Push to GitHub → connect repo → deploy in < 5 minutes (free tier).
#
#     2. Railway — https://railway.app
#        Uses the included Procfile:
#          web: streamlit run Home.py --server.port=$PORT --server.address=0.0.0.0
#
#     3. Render — https://render.com
#        Create a "Web Service", set Start Command from the Procfile above.
#
#     4. Docker (any cloud)
#        Use the included Dockerfile. Build & run:
#          docker build -t phishing-detector .
#          docker run -p 8501:8501 phishing-detector
# ==============================================================================

import io
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    clean_data_pipeline,
    get_models,
    init_session_state,
    load_data,
    needs_scaling,
    plot_target_distribution,
    predict_sample,
    preprocess_data,
    train_and_evaluate_models,
)

# ==============================================================================
# PAGE CONFIG  (must be the FIRST Streamlit call)
# ==============================================================================
st.set_page_config(
    page_title="Phishing Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# MODERN CSS THEMING
# ==============================================================================
st.markdown(
    """
<style>
    /* ── Hero section ─────────────────────────────────────────── */
    .hero-section {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0f2d55 100%);
        border-radius: 18px;
        padding: 2.6rem 2.4rem 2rem 2.4rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        position: relative;
        overflow: hidden;
    }
    .hero-section::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 240px; height: 240px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(59,130,246,0.18) 0%, transparent 70%);
    }
    .hero-shield {
        font-size: 4rem;
        display: block;
        filter: drop-shadow(0 0 18px rgba(96,165,250,0.7));
        margin-bottom: 0.4rem;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #f1f5f9;
        letter-spacing: -0.5px;
        margin: 0 0 0.4rem 0;
        text-shadow: 0 0 40px rgba(96,165,250,0.4);
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin: 0 0 1.4rem 0;
    }

    /* ── Section headings ─────────────────────────────────────── */
    .section-heading {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0.5rem 0 1rem 0;
        padding-left: 0.6rem;
        border-left: 4px solid #3b82f6;
    }

    /* ── Detection card ───────────────────────────────────────── */
    .detection-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1.6rem 1.6rem 1rem 1.6rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }

    /* ── Prediction result badges ─────────────────────────────── */
    .prediction-result {
        padding: 1.4rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        margin: 1.2rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    .phishing {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        color: #fca5a5;
        border: 2px solid #ef4444;
    }
    .legitimate {
        background: linear-gradient(135deg, #052e16, #14532d);
        color: #86efac;
        border: 2px solid #22c55e;
    }

    /* ── Stat cards ───────────────────────────────────────────── */
    .stat-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 0.6rem;
    }
    .stat-card .stat-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #60a5fa;
    }
    .stat-card .stat-label {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 0.2rem;
    }

    /* ── Sidebar tweaks ───────────────────────────────────────── */
    .sidebar-brand {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f1f5f9;
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

    /* ── Misc ─────────────────────────────────────────────────── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        border: none;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.65rem;
        border-radius: 10px;
        box-shadow: 0 4px 14px rgba(37,99,235,0.35);
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 22px rgba(37,99,235,0.55);
        transform: translateY(-1px);
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# SESSION STATE — initialise shared state and auto-load data/model
# ==============================================================================
init_session_state()

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown(
        "<span class='sidebar-brand'>🛡️ Phishing Detector</span>", unsafe_allow_html=True
    )
    if st.session_state.models_trained:
        st.markdown(
            "<span class='status-badge badge-success'>✓ 4 Models Ready</span>",
            unsafe_allow_html=True,
        )
    st.markdown("---")

    # ── Page navigation ────────────────────────────────────────────────────────
    st.markdown("### 🗺️ Navigation")
    st.page_link("Home.py", label="🏠 Detection Home", icon="🛡️")
    st.page_link("pages/1_Model_Info.py", label="📊 Model Info & Analysis", icon="ℹ️")
    st.markdown("---")

    # ── Dataset upload ─────────────────────────────────────────────────────────
    st.markdown("### 📂 Dataset")
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

    # ── Training settings ──────────────────────────────────────────────────────
    st.markdown("### ⚙️ Training Settings")
    test_size    = st.slider("Test Split Ratio", 0.10, 0.40, 0.20, 0.05)
    random_state = st.number_input("Random Seed", 0, 1000, 42, step=1)

    st.markdown("---")

    # ── Train button ───────────────────────────────────────────────────────────
    st.markdown("### 🚀 Model Training")
    btn_train = st.button("🤖 Train 4 Models", use_container_width=True, type="primary")

    if btn_train:
        if st.session_state.get("df_clean") is None:
            st.error("Load a dataset first.")
        else:
            with st.spinner("Training Logistic Regression, Decision Tree, Random Forest, Naive Bayes…"):
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
                st.success(
                    f"✅ Training complete! Best model: **{st.session_state.best_model_name}**"
                )

    st.markdown("---")
    if os.path.exists("api/best_phishing_model.pkl"):
        st.info("📦 Pre-trained PKL auto-loaded")


# ==============================================================================
# HERO HEADER
# ==============================================================================
st.markdown(
    """
<div class="hero-section">
    <span class="hero-shield">🛡️</span>
    <div class="hero-title">Phishing Website Detection</div>
    <div class="hero-subtitle">
        Machine Learning Classification Pipeline &nbsp;·&nbsp; 4 Core Models &nbsp;·&nbsp;
        Explainable Cybersecurity Analytics
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# "View Full Analysis" navigation button under the hero
col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
with col_nav2:
    st.page_link(
        "pages/1_Model_Info.py",
        label="📊 View Full Model Analysis →",
        icon="ℹ️",
        use_container_width=True,
    )

st.markdown("---")


# ==============================================================================
# LIVE DETECTION SECTION
# ==============================================================================
st.markdown('<div class="section-heading">🔍 Live Website Detection</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="detection-card">', unsafe_allow_html=True)

    # Resolve current best model
    best_name  = st.session_state.get("best_model_name", "Random Forest")
    best_model = (
        st.session_state.get("trained_models", {}).get(best_name)
        if st.session_state.models_trained
        else None
    )
    if best_model is None and os.path.exists("api/best_phishing_model.pkl"):
        best_model = joblib.load("api/best_phishing_model.pkl")

    pp = st.session_state.get("preprocessor")

    if pp is None:
        st.warning("⚠️ Dataset not loaded. Upload a CSV in the sidebar or ensure `phishing_website_raw.csv` is present.")
    else:
        # ── Preset selector ────────────────────────────────────────────────────
        preset_choice = st.selectbox(
            "⚡ Quick Preset Scenario:",
            [
                "Custom Manual Input",
                "Example 1: Verified Legitimate Banking Portal (HTTPS, Domain Age 3500d, Low Entropy)",
                "Example 2: High-Risk Phishing Page (No HTTPS, Young Domain 12d, High Entropy, Password Field)",
                "Example 3: Malicious Redirect / Popup Scam (Multiple Redirects, Iframes, Popups)",
            ],
        )

        preset_values = {}
        if "Legitimate" in preset_choice:
            preset_values = {
                "URL_Length": 24, "Num_Dots": 1, "Num_Hyphens": 0, "Num_Special_Chars": 1,
                "Num_Subdomains": 0, "Has_IP_Address": 0, "Has_HTTPS": 1,
                "Domain_Age_Days": 3650, "Domain_Registration_Length": 730,
                "Has_Suspicious_Words": 0, "Num_Redirects": 0,
                "External_Link_Ratio": 0.15, "Image_Link_Ratio": 0.20,
                "Form_Count": 1, "Password_Field_Present": 1, "Iframe_Count": 0,
                "Popup_Count": 0, "Favicon_External": 0,
                "Domain_Name_Length": 10, "URL_Entropy": 3.12,
            }
        elif "High-Risk" in preset_choice:
            preset_values = {
                "URL_Length": 115, "Num_Dots": 6, "Num_Hyphens": 4, "Num_Special_Chars": 7,
                "Num_Subdomains": 3, "Has_IP_Address": 1, "Has_HTTPS": 0,
                "Domain_Age_Days": 12, "Domain_Registration_Length": 30,
                "Has_Suspicious_Words": 1, "Num_Redirects": 3,
                "External_Link_Ratio": 0.85, "Image_Link_Ratio": 0.90,
                "Form_Count": 4, "Password_Field_Present": 1, "Iframe_Count": 2,
                "Popup_Count": 3, "Favicon_External": 1,
                "Domain_Name_Length": 28, "URL_Entropy": 4.88,
            }
        elif "Redirect" in preset_choice:
            preset_values = {
                "URL_Length": 88, "Num_Dots": 4, "Num_Hyphens": 3, "Num_Special_Chars": 5,
                "Num_Subdomains": 2, "Has_IP_Address": 0, "Has_HTTPS": 0,
                "Domain_Age_Days": 45, "Domain_Registration_Length": 90,
                "Has_Suspicious_Words": 1, "Num_Redirects": 4,
                "External_Link_Ratio": 0.70, "Image_Link_Ratio": 0.60,
                "Form_Count": 3, "Password_Field_Present": 0, "Iframe_Count": 3,
                "Popup_Count": 4, "Favicon_External": 1,
                "Domain_Name_Length": 22, "URL_Entropy": 4.52,
            }

        # ── Feature input grid (3 columns) ─────────────────────────────────────
        st.markdown("##### 📝 Enter Website Features")
        input_data   = {}
        cols_grid    = st.columns(3)
        feature_cols = pp["feature_columns"]
        binary_cols_set = {
            "Has_IP_Address", "Has_HTTPS", "Has_Suspicious_Words",
            "Password_Field_Present", "Favicon_External",
        }

        for idx, feat in enumerate(feature_cols):
            c_idx = idx % 3
            with cols_grid[c_idx]:
                default_v = preset_values.get(feat, 0.0)
                if feat in binary_cols_set:
                    val = st.selectbox(
                        feat,
                        options=[0, 1],
                        index=int(default_v) if default_v in [0, 1] else 0,
                        key=f"input_{feat}",
                    )
                elif "Ratio" in feat or "Entropy" in feat:
                    val = st.number_input(
                        feat, value=float(default_v),
                        step=0.01, format="%.2f", key=f"input_{feat}",
                    )
                else:
                    val = st.number_input(
                        feat, value=int(default_v),
                        step=1, key=f"input_{feat}",
                    )
                input_data[feat] = val

        st.markdown("---")

        # ── Detect button ──────────────────────────────────────────────────────
        btn_predict = st.button(
            "🔍  DETECT PHISHING STATUS",
            type="primary",
            use_container_width=True,
        )

        if btn_predict:
            if best_model is None:
                st.error("No trained model available. Click 'Train 4 Models' in the sidebar.")
            else:
                pred, proba = predict_sample(input_data, pp, best_model, best_name)

                if pred == 1:
                    st.markdown(
                        '<div class="prediction-result phishing">'
                        "🚨&nbsp; DANGER — PHISHING WEBSITE DETECTED"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="prediction-result legitimate">'
                        "✅&nbsp; SAFE — LEGITIMATE WEBSITE"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                if proba is not None:
                    p_phish = proba[1] * 100
                    p_legit = proba[0] * 100

                    m1, m2 = st.columns(2)
                    m1.metric("Legitimate Probability", f"{p_legit:.2f}%")
                    m2.metric("Phishing Probability",   f"{p_phish:.2f}%")

                    # Stacked horizontal probability bar
                    fig_meter, ax_m = plt.subplots(figsize=(8, 1.4), dpi=150)
                    ax_m.barh([""], [p_legit], color="#16a34a",
                              label=f"Legitimate ({p_legit:.1f}%)")
                    ax_m.barh([""], [p_phish], left=[p_legit], color="#dc2626",
                              label=f"Phishing ({p_phish:.1f}%)")
                    ax_m.set_xlim(0, 100)
                    ax_m.set_xlabel("Probability (%)", fontweight="bold")
                    ax_m.legend(
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.35),
                        ncol=2, fontsize=9,
                    )
                    plt.tight_layout()
                    st.pyplot(fig_meter)
                    plt.close(fig_meter)

                st.caption(f"Model used: **{best_name}**")

    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("---")


# ==============================================================================
# DATASET QUICK STATS
# ==============================================================================
st.markdown('<div class="section-heading">📊 Dataset Quick Stats</div>', unsafe_allow_html=True)

df_current       = st.session_state.get("df")
df_clean_current = st.session_state.get("df_clean", df_current)

if df_current is not None and df_clean_current is not None:
    legit_cnt = int((df_clean_current["Is_Phishing"] == 0).sum())
    phish_cnt = int((df_clean_current["Is_Phishing"] == 1).sum())
    total_clean = len(df_clean_current)

    # ── 5 metric cards ─────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    def stat_card(col, value, label):
        col.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-value">{value}</div>'
            f'<div class="stat-label">{label}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    stat_card(c1, f"{len(df_current):,}",  "Raw Records")
    stat_card(c2, f"{total_clean:,}",       "Clean Records")
    stat_card(c3, f"{df_clean_current.shape[1] - 1}", "Input Features")
    stat_card(c4, f"{legit_cnt:,}",         "Legitimate (0)")
    stat_card(c5, f"{phish_cnt:,}",         "Phishing (1)")

    st.markdown("")
    col_tbl, col_chart = st.columns([3, 2])
    with col_tbl:
        st.markdown("**Sample Data Preview**")
        st.dataframe(df_current.head(8), use_container_width=True)
    with col_chart:
        st.markdown("**Target Balance**")
        fig_tgt = plot_target_distribution(df_clean_current)
        st.pyplot(fig_tgt)
        plt.close(fig_tgt)
else:
    st.info("No dataset loaded. Upload a CSV in the sidebar or add `phishing_website_raw.csv` to the project root.")


# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#64748b; font-size:0.82rem;'>"
    "🛡️ Phishing Website Detection &nbsp;·&nbsp; Built with Streamlit &amp; Scikit-learn"
    "</p>",
    unsafe_allow_html=True,
)
