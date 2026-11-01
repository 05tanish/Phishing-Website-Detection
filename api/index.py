# ==============================================================================
# api/index.py — FastAPI Backend for Phishing Website Detection
# Deployed as a Vercel Serverless Function
# ==============================================================================

import json
import os
import pathlib
import warnings
from functools import lru_cache

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

warnings.filterwarnings("ignore")

app = FastAPI(title="Phishing Detector API", version="1.0.0")

# Allow all origins for Vercel deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# PATH RESOLUTION HELPERS
# ==============================================================================
API_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = API_DIR.parent
STATIC_DIR = ROOT_DIR / "static"
if not STATIC_DIR.exists():
    STATIC_DIR = API_DIR / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

FEATURE_COLUMNS = [
    "URL_Length", "Num_Dots", "Num_Hyphens", "Num_Special_Chars",
    "Num_Subdomains", "Has_IP_Address", "Has_HTTPS", "Domain_Age_Days",
    "Domain_Registration_Length", "Has_Suspicious_Words", "Num_Redirects",
    "External_Link_Ratio", "Image_Link_Ratio", "Form_Count",
    "Password_Field_Present", "Iframe_Count", "Popup_Count",
    "Favicon_External", "Domain_Name_Length", "URL_Entropy",
]

SCALED_MODELS = {"Logistic Regression", "Naive Bayes"}


def _resolve_candidate(filenames):
    for f in filenames:
        candidates = [
            API_DIR / f,
            ROOT_DIR / f,
            ROOT_DIR / "api" / f,
            API_DIR / "api" / f,
            pathlib.Path(f),
            pathlib.Path("api") / f,
        ]
        for c in candidates:
            if c.exists():
                return c
    return None


# ==============================================================================
# LAZY-LOAD MODEL & SCALER (cached so they load only once per cold start)
# ==============================================================================
@lru_cache(maxsize=1)
def load_model():
    p = _resolve_candidate(["best_phishing_model.pkl"])
    if not p:
        raise FileNotFoundError(
            f"best_phishing_model.pkl not found in API_DIR={API_DIR} or ROOT_DIR={ROOT_DIR}"
        )
    return joblib.load(p)


@lru_cache(maxsize=1)
def load_scaler():
    p = _resolve_candidate(["phishing_scaler.pkl"])
    if not p:
        return None
    return joblib.load(p)


@lru_cache(maxsize=1)
def load_stats():
    # Check multiple locations
    candidates = [
        API_DIR / "model_stats.json",
        ROOT_DIR / "model_stats.json",
        STATIC_DIR / "model_stats.json",
        ROOT_DIR / "static" / "model_stats.json",
        pathlib.Path("api/model_stats.json"),
        pathlib.Path("model_stats.json"),
        pathlib.Path("static/model_stats.json"),
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p) as f:
                    data = json.load(f)
                    if data:
                        return data
            except Exception:
                continue
    return {}


# ==============================================================================
# PYDANTIC SCHEMAS
# ==============================================================================
class PredictRequest(BaseModel):
    URL_Length: float = 0
    Num_Dots: float = 0
    Num_Hyphens: float = 0
    Num_Special_Chars: float = 0
    Num_Subdomains: float = 0
    Has_IP_Address: int = 0
    Has_HTTPS: int = 1
    Domain_Age_Days: float = 0
    Domain_Registration_Length: float = 0
    Has_Suspicious_Words: int = 0
    Num_Redirects: float = 0
    External_Link_Ratio: float = 0.0
    Image_Link_Ratio: float = 0.0
    Form_Count: float = 0
    Password_Field_Present: int = 0
    Iframe_Count: float = 0
    Popup_Count: float = 0
    Favicon_External: int = 0
    Domain_Name_Length: float = 0
    URL_Entropy: float = 0.0


# ==============================================================================
# ROUTES
# ==============================================================================
@app.get("/", include_in_schema=False)
def index_page():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Phishing Detector API"}


@app.get("/model-info", include_in_schema=False)
def model_info_page():
    info_file = STATIC_DIR / "model-info.html"
    if info_file.exists():
        return FileResponse(str(info_file))
    return {"message": "Model Info"}


@app.get("/api/health")
def health():
    model_found = _resolve_candidate(["best_phishing_model.pkl"]) is not None
    stats_found = bool(load_stats())
    return {
        "status": "ok",
        "model_loaded": model_found,
        "stats_loaded": stats_found,
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    try:
        model = load_model()
        scaler = load_scaler()

        features = np.array([[getattr(req, col) for col in FEATURE_COLUMNS]], dtype=float)

        # Use scaler if available (model was trained with StandardScaler)
        model_name = type(model).__name__
        if scaler is not None and model_name in {"LogisticRegression", "GaussianNB"}:
            features = scaler.transform(features)

        prediction = int(model.predict(features)[0])
        probability = None

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            probability = {"legitimate": round(float(proba[0]) * 100, 2),
                           "phishing": round(float(proba[1]) * 100, 2)}

        return {
            "prediction": prediction,
            "label": "PHISHING" if prediction == 1 else "LEGITIMATE",
            "probability": probability,
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.get("/api/stats")
def get_stats():
    """Return pre-computed model statistics."""
    stats = load_stats()
    if not stats:
        raise HTTPException(
            status_code=404,
            detail="Model stats not found. Run scripts/precompute_stats.py first."
        )
    return JSONResponse(content=stats)


@app.get("/api/features")
def get_features():
    """Return feature metadata for the frontend form."""
    features = []
    binary = {"Has_IP_Address", "Has_HTTPS", "Has_Suspicious_Words",
               "Password_Field_Present", "Favicon_External"}
    ratio = {"External_Link_Ratio", "Image_Link_Ratio", "URL_Entropy"}

    for col in FEATURE_COLUMNS:
        if col in binary:
            ftype = "binary"
        elif col in ratio:
            ftype = "ratio"
        else:
            ftype = "integer"
        features.append({"name": col, "type": ftype})

    return {"features": features}

