# 🔐 Phishing Website Detection Using Machine Learning

An end-to-end machine learning system and interactive Streamlit web dashboard for detecting phishing websites with high accuracy ($97.16\%$) and F1-score ($96.50\%$).

---
## 🌟 Key Highlights

- **4 Core Classifiers**: Trained, benchmarked, and evaluated side-by-side:
  1. **Logistic Regression** (Standardized linear baseline)
  2. **Decision Tree** (Non-parametric rule-based tree)
  3. **Random Forest (Tuned)** (Ensemble of 100 bagging trees — **Best Model**)
  4. **Naive Bayes** (Gaussian NB probabilistic classifier)
- **Pre-trained Model Artifacts Included**: Serialized `best_phishing_model.pkl` and `phishing_scaler.pkl` auto-load on launch for instant predictions.
- **Interactive Multi-Tab Dashboard**:
  - 📋 **Dataset Overview**: Dataset stats, target balance, preview, and column dictionary.
  - 📊 **EDA & Visualizations**: Feature distributions with KDE, boxplot outlier checks, full 20-feature correlation matrix, and ranked feature-to-target correlations.
  - ⚙️ **Preprocessing Pipeline**: Leakage-free cleaning, median imputation, stratified train/test split ($80\% / 20\%$), and scaling.
  - 🤖 **4-Model Training**: One-click training and evaluation of the 4 models.
  - 📈 **Model Comparison**: Metric comparison table and grouped visual charts (Accuracy, Precision, Recall, F1-Score, ROC-AUC).
  - 🏆 **Best Model In-Depth**: Confusion matrix, ROC curve, Precision-Recall curve, feature importance ranking, and classification report.
  - 🔮 **Live Website Prediction**: Instant classification with preset scenarios (Legitimate Banking, High-Risk Phishing, Malicious Redirect) and a live probability meter.

---

## 📁 Repository Structure

```
Ml-project/
├── app.py                           # Production Streamlit application
├── api/
│   ├── best_phishing_model.pkl      # Pre-trained Random Forest model artifact
│   └── phishing_scaler.pkl          # Fitted StandardScaler artifact
├── phishing_website_raw.csv         # Structured dataset (25,000 samples, 20 features)
├── phishing_classification.ipynb    # Core research and EDA notebook
├── Final_Report.docx                # Complete project documentation report
├── Project_Documentation.md         # Detailed notebook, feature, and model docs
├── requirements.txt                 # Full Python dependencies (for Streamlit/Docker)
├── Dockerfile & .dockerignore       # Production container definition
├── Procfile                         # PaaS deployment configuration (Render/Heroku)
└── README.md                        # Documentation & deployment guide
```

---

## 🚀 Deployment Guide

### Option 1: Vercel Deployment (100% Serverless & Free)

This project is fully configured for **Vercel** serverless deployment using a Python FastAPI backend and a responsive dark-themed frontend.

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Configure for Vercel deployment with FastAPI"
   git push origin main
   ```

2. **Deploy on Vercel**:
   - Go to [vercel.com](https://vercel.com) and log in.
   - Click **"Add New..."** > **"Project"**.
   - Import your GitHub repository (`phishingwebsitedetection` or `ML-Project`).
   - Keep default settings (`vercel.json` and `api/index.py` are automatically detected).
   - Click **"Deploy"**.

Your web app will be live with full URL prediction and model analysis!

---

### Option 2: Local Execution (FastAPI Server)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the FastAPI server**:
   ```bash
   python3 -m uvicorn api.index:app --reload --port 8000
   ```

3. Open your browser at:
   - Live Detector: `http://localhost:8000`
   - Model Info & Metrics: `http://localhost:8000/model-info`
   - Interactive API Docs: `http://localhost:8000/docs`

---

### Option 3: Streamlit Version (Alternative)

If you prefer running the original or multi-page Streamlit interface:
```bash
pip install -r requirements-streamlit.txt
streamlit run Home.py
```

---

## 📊 Model Performance Summary

Evaluated on a held-out test split of $5,000$ unseen website records:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (Tuned)** | **97.16%** | **97.03%** | **95.98%** | **96.50%** | **0.9942** |
| **Logistic Regression** | 97.10% | 96.97% | 95.88% | 96.42% | 0.9930 |
| **Naive Bayes** | 97.02% | 96.55% | 96.13% | 96.34% | 0.9918 |
| **Decision Tree** | 92.32% | 89.88% | 91.47% | 90.67% | 0.9219 |

---

## 🛠️ Technology Stack

- **Frontend / Framework**: Streamlit
- **Machine Learning**: Scikit-Learn
- **Data Manipulation**: Pandas, NumPy
- **Visualizations**: Matplotlib, Seaborn
- **Persistence**: Joblib
- **Containerization**: Docker

---

## 📄 License & Attribution

This project is built for educational, research, and cybersecurity threat detection demonstration purposes.
