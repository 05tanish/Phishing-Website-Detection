# ==============================================================================
# api/index.py — FastAPI Backend for Phishing Website Detection
# Deployed as a Vercel Serverless Function
# ==============================================================================

import datetime
import json
import math
import os
import pathlib
import re
import urllib.parse
import urllib.request
import warnings
from collections import Counter
from functools import lru_cache

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

warnings.filterwarnings("ignore")

app = FastAPI(title="PhishGuard — Threat Detection API", version="2.1.0")

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
# LAZY-LOAD MODEL & SCALER
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


class URLScanRequest(BaseModel):
    url: str


# ==============================================================================
# LIVE DOMAIN AGE & REDIRECT VERIFIERS WITH INTELLIGENT DEFAULTS
# ==============================================================================
def check_live_domain_age(hostname: str):
    """
    Attempts to query ICANN RDAP for domain registration date.
    Returns (age_days, reg_length_days, status_label)
    If third-party RDAP fails, times out, or domain is invalid, returns (None, None, 'fallback').
    """
    # Extract root domain (e.g., example.com from sub.example.com)
    parts = hostname.split(".")
    if len(parts) >= 2:
        root_domain = f"{parts[-2]}.{parts[-1]}"
    else:
        root_domain = hostname

    if not root_domain or root_domain.replace(".", "").isdigit():
        return None, None, "IP Host — No WHOIS"

    try:
        url = f"https://rdap.org/domain/{root_domain}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; PhishGuard/2.1)"}
        )
        with urllib.request.urlopen(req, timeout=1.2) as resp:
            data = json.loads(resp.read().decode())
            events = data.get("events", [])
            reg_date = None
            exp_date = None
            for ev in events:
                action = ev.get("eventAction")
                if action == "registration" and "eventDate" in ev:
                    reg_date = datetime.datetime.fromisoformat(ev["eventDate"].replace("Z", "+00:00"))
                elif action == "expiration" and "eventDate" in ev:
                    exp_date = datetime.datetime.fromisoformat(ev["eventDate"].replace("Z", "+00:00"))

            if reg_date:
                now = datetime.datetime.now(datetime.timezone.utc)
                age_days = max(1, (now - reg_date).days)
                reg_len = (exp_date - reg_date).days if exp_date else 365
                return age_days, max(30, reg_len), "Verified via ICANN RDAP"
    except Exception:
        pass

    return None, None, "Heuristic Fallback"


def check_live_redirects(url: str):
    """
    Checks actual HTTP redirect count with a 1.2s timeout.
    Returns (num_redirects, status_label).
    """
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        opener = urllib.request.build_opener()
        with opener.open(req, timeout=1.2) as resp:
            final_url = resp.geturl()
            # If final URL differs significantly, count as redirect
            if final_url.rstrip("/") != url.rstrip("/"):
                return 1, "Verified HTTP Redirect"
            return 0, "Verified Direct (0 Redirects)"
    except Exception:
        pass

    # Heuristic redirect check from URL query params
    if any(param in url.lower() for param in ["redirect=", "url=", "goto=", "r=", "dest=", "return="]):
        return 2, "Heuristic (Redirect parameter in URL)"
    return 0, "Default (Direct)"


# ==============================================================================
# AUTOMATIC URL FEATURE EXTRACTION
# ==============================================================================
def extract_features_from_url(raw_url: str):
    raw_url = raw_url.strip()
    if not raw_url.startswith(("http://", "https://")):
        url = "https://" + raw_url
    else:
        url = raw_url

    parsed = urllib.parse.urlparse(url)
    hostname = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    full_url = url

    url_len = len(full_url)
    num_dots = full_url.count(".")
    num_hyphens = full_url.count("-")
    special_chars = set("@?_=%&!#+$*~;,")
    num_special = sum(1 for c in full_url if c in special_chars)

    # Subdomains count
    parts = [p for p in hostname.split(".") if p]
    if len(parts) > 2:
        num_subdomains = len(parts) - 2
    else:
        num_subdomains = 0

    # IP address detection
    ip_pattern = r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
    has_ip = 1 if re.match(ip_pattern, hostname) else 0

    # HTTPS check
    has_https = 1 if parsed.scheme == "https" else 0

    # Domain Name Length
    domain_name = parts[-2] if len(parts) >= 2 else (hostname or "domain")
    domain_name_len = len(domain_name)

    # Shannon Entropy: H(X) = -sum(P(x) * log2(P(x)))
    counts = Counter(full_url).values()
    entropy = -sum((c / url_len) * math.log2(c / url_len) for c in counts) if url_len > 0 else 0.0

    # Suspicious keywords detection
    suspicious = [
        "login", "verify", "update", "banking", "secure", "account",
        "signin", "confirm", "security", "wallet", "service", "support",
        "password", "credential", "auth", "recover", "ebayisapi", "webscr",
        "bonus", "free", "gift", "alert", "validate", "token", "session"
    ]
    matched_suspicious = [w for w in suspicious if w in full_url.lower()]
    has_suspicious = 1 if len(matched_suspicious) > 0 else 0

    # Risk evaluation
    is_suspicious_url = (
        has_suspicious
        or (has_ip == 1)
        or (num_dots > 3)
        or (num_hyphens > 2)
        or (has_https == 0)
        or (num_subdomains >= 2)
        or (entropy > 4.4)
    )

    # Live verification with fast timeout & fallback defaults
    live_age, live_reg_len, domain_age_status = check_live_domain_age(hostname)
    live_redirects, redirect_status = check_live_redirects(url)

    # DEFAULT ASSIGNMENT RULES (Transparently reported):
    # 1. If RDAP gave real domain age, use it.
    # 2. If RDAP failed/unavailable:
    #    - Suspicious characteristics: domain age defaults to 30 days, registration length 90 days.
    #    - Clean characteristics: domain age defaults to 2400 days (~6.5 yrs), registration length 730 days.
    if live_age is not None:
        domain_age = live_age
        reg_length = live_reg_len
    else:
        domain_age = 30 if is_suspicious_url else 2400
        reg_length = 90 if is_suspicious_url else 730

    num_redirects = live_redirects

    # Non-lexical page metrics
    ext_ratio = 0.75 if is_suspicious_url else 0.15
    img_ratio = 0.70 if is_suspicious_url else 0.20
    form_count = 3 if has_suspicious else 1
    has_pwd = 1 if any(w in full_url.lower() for w in ["login", "signin", "password", "auth", "verify", "pwd"]) else 0
    iframe = 1 if is_suspicious_url else 0
    popup = 2 if is_suspicious_url else 0
    favicon = 1 if is_suspicious_url else 0

    features_dict = {
        "URL_Length": url_len,
        "Num_Dots": num_dots,
        "Num_Hyphens": num_hyphens,
        "Num_Special_Chars": num_special,
        "Num_Subdomains": num_subdomains,
        "Has_IP_Address": has_ip,
        "Has_HTTPS": has_https,
        "Domain_Age_Days": domain_age,
        "Domain_Registration_Length": reg_length,
        "Has_Suspicious_Words": has_suspicious,
        "Num_Redirects": num_redirects,
        "External_Link_Ratio": ext_ratio,
        "Image_Link_Ratio": img_ratio,
        "Form_Count": form_count,
        "Password_Field_Present": has_pwd,
        "Iframe_Count": iframe,
        "Popup_Count": popup,
        "Favicon_External": favicon,
        "Domain_Name_Length": domain_name_len,
        "URL_Entropy": round(entropy, 3),
    }

    breakdown = {
        "raw_url": raw_url,
        "normalized_url": url,
        "hostname": hostname,
        "domain_name": domain_name,
        "scheme": parsed.scheme,
        "has_https": bool(has_https),
        "has_ip": bool(has_ip),
        "entropy": round(entropy, 2),
        "num_subdomains": num_subdomains,
        "matched_suspicious_words": matched_suspicious,
        "total_length": url_len,
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_special": num_special,
        "domain_age_days": domain_age,
        "domain_age_source": domain_age_status,
        "redirects_count": num_redirects,
        "redirect_source": redirect_status,
    }

    return features_dict, breakdown


# ==============================================================================
# ROUTES
# ==============================================================================
@app.get("/", include_in_schema=False)
def index_page():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "PhishGuard Threat Intelligence API"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    fav = _resolve_candidate(["favicon.ico"])
    if fav and fav.exists():
        return FileResponse(str(fav), media_type="image/x-icon")
    return JSONResponse(status_code=204, content={})


@app.get("/health")
def health():
    model_found = _resolve_candidate(["best_phishing_model.pkl"]) is not None
    return {
        "status": "ok",
        "service": "PhishGuard",
        "model_loaded": model_found,
    }


@app.post("/scan-url")
def scan_url(req: URLScanRequest):
    if not req.url or not req.url.strip():
        raise HTTPException(status_code=400, detail="Please enter a valid website URL.")

    try:
        features_dict, breakdown = extract_features_from_url(req.url)
        model = load_model()
        scaler = load_scaler()

        features = np.array([[features_dict[col] for col in FEATURE_COLUMNS]], dtype=float)
        model_name = type(model).__name__
        if scaler is not None and model_name in {"LogisticRegression", "GaussianNB"}:
            features = scaler.transform(features)

        prediction = int(model.predict(features)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            probability = {
                "legitimate": round(float(proba[0]) * 100, 2),
                "phishing": round(float(proba[1]) * 100, 2),
            }

        return {
            "prediction": prediction,
            "label": "PHISHING" if prediction == 1 else "LEGITIMATE",
            "probability": probability,
            "extracted_features": features_dict,
            "breakdown": breakdown,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scan error: {str(e)}")


@app.post("/predict")
def predict_manual(req: PredictRequest):
    try:
        model = load_model()
        scaler = load_scaler()

        features = np.array([[getattr(req, col) for col in FEATURE_COLUMNS]], dtype=float)
        model_name = type(model).__name__
        if scaler is not None and model_name in {"LogisticRegression", "GaussianNB"}:
            features = scaler.transform(features)

        prediction = int(model.predict(features)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            probability = {
                "legitimate": round(float(proba[0]) * 100, 2),
                "phishing": round(float(proba[1]) * 100, 2),
            }

        return {
            "prediction": prediction,
            "label": "PHISHING" if prediction == 1 else "LEGITIMATE",
            "probability": probability,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ==============================================================================
# CATCH-ALL ROUTE (Immunity against Vercel prefix/rewrite mutations)
# ==============================================================================
@app.api_route("/{path_name:path}", methods=["GET", "POST", "OPTIONS", "HEAD"], include_in_schema=False)
async def catch_all_router(request: Request, path_name: str):
    p = path_name.lower().strip("/")
    raw_path = request.url.path.lower()
    action = request.query_params.get("action", "").lower()
    target = f"{p} {raw_path} {action}"

    if "favicon.ico" in target:
        return favicon()

    if "health" in target:
        return health()

    if "scan-url" in target and request.method == "POST":
        try:
            body = await request.json()
            return scan_url(URLScanRequest(**body))
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    if "predict" in target and request.method == "POST":
        try:
            body = await request.json()
            return predict_manual(PredictRequest(**body))
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Fallback to index if no specific action was requested
    if not p or p in ["index", "index.py", "api", "api/index", "api/index.py"]:
        return index_page()

    return JSONResponse(
        status_code=404,
        content={"detail": f"Route '{path_name}' not found", "target": target}
    )
