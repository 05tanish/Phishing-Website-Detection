#!/usr/bin/env python3
# ==============================================================================
# scripts/precompute_stats.py
# Run this ONCE locally before deploying to generate model_stats.json
# Usage: python3 scripts/precompute_stats.py
# ==============================================================================

import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# Resolve project root (one level up from scripts/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "phishing_website_raw.csv")
OUT_PATH = os.path.join(ROOT, "model_stats.json")

print(f"[INFO] Loading dataset from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
target_col = "Is_Phishing"

# ── Data cleaning ──────────────────────────────────────────────────────────────
df = df.drop_duplicates().reset_index(drop=True)

count_cols = ["URL_Length", "Num_Dots", "Num_Hyphens", "Num_Special_Chars",
              "Num_Subdomains", "Domain_Age_Days", "Domain_Registration_Length",
              "Num_Redirects", "Form_Count", "Iframe_Count", "Popup_Count",
              "Domain_Name_Length", "URL_Entropy"]
binary_cols = ["Has_IP_Address", "Has_HTTPS", "Has_Suspicious_Words",
               "Password_Field_Present", "Favicon_External"]
ratio_cols = ["External_Link_Ratio", "Image_Link_Ratio"]

for col in count_cols:
    if col in df.columns:
        df.loc[df[col] < 0, col] = np.nan
for col in binary_cols:
    if col in df.columns:
        df.loc[~df[col].isin([0, 1]), col] = np.nan
for col in ratio_cols:
    if col in df.columns:
        df.loc[(df[col] < 0) | (df[col] > 1), col] = np.nan

features = [c for c in df.columns if c != target_col]
for col in features:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

df[target_col] = df[target_col].astype(int)

# ── Split ──────────────────────────────────────────────────────────────────────
X = df.drop(columns=[target_col])
y = df[target_col]
feature_columns = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

SCALED_MODELS = {"Logistic Regression", "Naive Bayes"}

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        max_depth=20, min_samples_leaf=2, min_samples_split=5,
        n_estimators=100, random_state=42
    ),
    "Naive Bayes": GaussianNB(),
}

print("[INFO] Training 4 models...")
results = []
best_model_data = {}

for name, model in models.items():
    t0 = time.time()
    X_tr = X_train_scaled if name in SCALED_MODELS else X_train_imp
    X_te = X_test_scaled if name in SCALED_MODELS else X_test_imp

    model.fit(X_tr, y_train.values)
    elapsed = round(time.time() - t0, 3)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = float(roc_auc_score(y_test, y_prob)) if y_prob is not None else None
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    roc_data, pr_data = None, None
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
        ap = float(average_precision_score(y_test, y_prob))
        # Downsample curves to 200 points for JSON size
        idx = np.linspace(0, len(fpr) - 1, min(200, len(fpr)), dtype=int)
        roc_data = {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist(), "auc": roc}
        idx2 = np.linspace(0, len(rec_curve) - 1, min(200, len(rec_curve)), dtype=int)
        pr_data = {"precision": prec_curve[idx2].tolist(), "recall": rec_curve[idx2].tolist(), "ap": ap}

    feat_importance = None
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_.tolist()
        feat_importance = [{"feature": f, "importance": round(v, 6)}
                           for f, v in sorted(zip(feature_columns, fi),
                                              key=lambda x: x[1], reverse=True)]

    row = {
        "model": name,
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "roc_auc": round(roc, 6) if roc else None,
        "train_time": elapsed,
        "confusion_matrix": cm,
        "classification_report": {
            k: {sk: round(sv, 4) if isinstance(sv, float) else sv
                for sk, sv in v.items()} if isinstance(v, dict) else round(v, 4)
            for k, v in report.items()
        },
        "roc_curve": roc_data,
        "pr_curve": pr_data,
        "feature_importance": feat_importance,
    }
    results.append(row)
    print(f"  ✓ {name}: Acc={acc:.4f} F1={f1:.4f} ({elapsed}s)")

results.sort(key=lambda x: x["f1"], reverse=True)
best_name = results[0]["model"]

# ── Dataset stats ──────────────────────────────────────────────────────────────
legit = int((df[target_col] == 0).sum())
phish = int((df[target_col] == 1).sum())
total = len(df)

corr_with_target = df.corr()[target_col].drop(target_col).sort_values()
corr_data = [{"feature": f, "correlation": round(float(v), 4)}
             for f, v in corr_with_target.items()]

# ── Correlation matrix ─────────────────────────────────────────────────────────
corr_matrix = df.corr()
cols = list(corr_matrix.columns)
corr_heatmap = {
    "columns": cols,
    "values": [[round(float(corr_matrix.iloc[i][j]), 3) for j in range(len(cols))]
               for i in range(len(cols))]
}

# ── Feature distributions (histograms) ─────────────────────────────────────────
feature_dists = {}
for col in feature_columns:
    vals = df[col].dropna().tolist()
    counts, bin_edges = np.histogram(vals, bins=30)
    feature_dists[col] = {
        "counts": counts.tolist(),
        "bin_edges": [round(float(e), 4) for e in bin_edges.tolist()],
        "mean": round(float(np.mean(vals)), 4),
        "median": round(float(np.median(vals)), 4),
        "std": round(float(np.std(vals)), 4),
        "min": round(float(np.min(vals)), 4),
        "max": round(float(np.max(vals)), 4),
    }

# ── Assemble output ────────────────────────────────────────────────────────────
output = {
    "dataset": {
        "total_records": total,
        "clean_records": total,
        "features": feature_columns,
        "feature_count": len(feature_columns),
        "legitimate_count": legit,
        "phishing_count": phish,
        "legitimate_pct": round(legit / total * 100, 2),
        "phishing_pct": round(phish / total * 100, 2),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "target_correlation": corr_data,
        "correlation_heatmap": corr_heatmap,
        "feature_distributions": feature_dists,
    },
    "models": results,
    "best_model": best_name,
}

with open(OUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n[✓] model_stats.json written to: {OUT_PATH}")
print(f"[✓] Best model: {best_name} (F1={results[0]['f1']:.4f})")
print(f"[✓] File size: {os.path.getsize(OUT_PATH) / 1024:.1f} KB")

