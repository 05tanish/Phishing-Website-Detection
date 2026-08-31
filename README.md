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

### Option 1: Streamlit Community Cloud (Recommended — 100% Free & 1-Click)

1. Push your repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Click **"New app"**.
4. Select your repository: `05tanish/ML-Project`.
5. Set **Main file path**: `app.py`.
6. Click **"Deploy!"** — Streamlit Cloud will install dependencies and launch your live public URL!

---

### Option 2: Docker Deployment

1. **Build the Docker Image**:
   ```bash
   docker build -t phishing-detector-app .
   ```

2. **Run the Container**:
   ```bash
   docker run -d -p 8501:8501 --name phishing-app phishing-detector-app
   ```

3. Open your browser at: `http://localhost:8501`

---

### Option 3: Local Installation & Execution

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/05tanish/ML-Project.git
   cd ML-Project
   ```

2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate       # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Dashboard**:
   ```bash
   streamlit run app.py
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
