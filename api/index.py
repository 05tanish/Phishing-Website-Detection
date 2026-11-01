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
from html.parser import HTMLParser

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
# FAST HTML DOM PARSER (Live Crawling for Real Webpage Characteristics)
# ==============================================================================
class FastDOMParser(HTMLParser):
    def __init__(self, base_host: str):
        super().__init__()
        self.base_host = (base_host or "").lower().split(":")[0]
        self.form_count = 0
        self.has_password = 0
        self.iframe_count = 0
        self.total_links = 0
        self.external_links = 0
        self.total_images = 0
        self.external_images = 0
        self.favicon_external = 0
        self.popup_count = 0
        self.in_script = False

    def handle_starttag(self, tag, attrs):
        attr_dict = {k.lower(): (v or "") for k, v in attrs}
        tag = tag.lower()

        if tag == "form":
            self.form_count += 1
        elif tag == "input":
            if attr_dict.get("type", "").lower() == "password":
                self.has_password = 1
        elif tag == "iframe":
            self.iframe_count += 1
        elif tag == "a":
            href = attr_dict.get("href", "").strip()
            if href and not href.startswith(("javascript:", "#", "mailto:", "tel:")):
                self.total_links += 1
                parsed = urllib.parse.urlparse(href)
                host = (parsed.netloc or "").lower().split(":")[0]
                if host and host != self.base_host and not host.endswith("." + self.base_host):
                    self.external_links += 1
        elif tag == "img":
            src = attr_dict.get("src", "").strip()
            if src and not src.startswith("data:"):
                self.total_images += 1
                parsed = urllib.parse.urlparse(src)
                host = (parsed.netloc or "").lower().split(":")[0]
                if host and host != self.base_host and not host.endswith("." + self.base_host):
                    self.external_images += 1
        elif tag == "link":
            rel = attr_dict.get("rel", "").lower()
            if "icon" in rel:
                href = attr_dict.get("href", "").strip()
                parsed = urllib.parse.urlparse(href)
                host = (parsed.netloc or "").lower().split(":")[0]
                if host and host != self.base_host and not host.endswith("." + self.base_host):
                    self.favicon_external = 1
        elif tag == "script":
            self.in_script = True

    def handle_endtag(self, tag):
        if tag.lower() == "script":
            self.in_script = False

    def handle_data(self, data):
        if self.in_script and any(k in data for k in ["window.open", "alert(", "confirm(", "prompt("]):
            self.popup_count += 1


# ==============================================================================
# LIVE DOMAIN AGE & WEB CONTENT CRAWLER
# ==============================================================================
def check_live_domain_age(hostname: str, is_suspicious_url: bool):
    """
    Attempts to query ICANN RDAP for domain registration date.
    Returns (age_days, reg_length_days, status_label)
    If RDAP fails, times out, or host is raw IP, uses statistically grounded dataset medians.
    """
    parts = hostname.split(".")
    if len(parts) >= 2:
        root_domain = f"{parts[-2]}.{parts[-1]}"
    else:
        root_domain = hostname

    if not root_domain or root_domain.replace(".", "").isdigit():
        return 30, 90, "IP Host — No Domain WHOIS"

    try:
        url = f"https://rdap.org/domain/{root_domain}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; PhishGuard/2.2)"}
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

    # If domain WHOIS query fails or is not found:
    if is_suspicious_url:
        return 30, 90, "Unverified WHOIS (Flagged Attack Surface)"
    return 1787, 730, "Dataset Benchmark Baseline (Established Domain)"


def check_live_page_and_dom(url: str, hostname: str, is_suspicious_url: bool):
    """
    Choice B Live Scraper:
    Fetches the live HTML source with a fast 1.2s timeout and parses actual DOM structures
    (forms, password fields, iframes, external link & image ratios, favicons, popups).
    If unreachable, times out, or blocked, uses statistically neutral dataset medians.
    """
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )
        with urllib.request.urlopen(req, timeout=1.2) as resp:
            final_url = resp.geturl()
            num_redirects = 1 if final_url.rstrip("/") != url.rstrip("/") else 0
            redirect_status = "Verified HTTP Redirect (1 Hop)" if num_redirects > 0 else "Verified Direct (0 Redirects)"

            raw_bytes = resp.read(131072)  # Read up to 128KB for speed
            charset = resp.headers.get_content_charset() or "utf-8"
            html_text = raw_bytes.decode(charset, errors="ignore")

            parser = FastDOMParser(hostname)
            parser.feed(html_text)

            ext_link_ratio = round(parser.external_links / max(1, parser.total_links), 3) if parser.total_links > 0 else (0.253 if not is_suspicious_url else 0.587)
            img_link_ratio = round(parser.external_images / max(1, parser.total_images), 3) if parser.total_images > 0 else (0.200 if not is_suspicious_url else 0.442)

            dom_metrics = {
                "External_Link_Ratio": ext_link_ratio,
                "Image_Link_Ratio": img_link_ratio,
                "Form_Count": min(parser.form_count, 11),
                "Password_Field_Present": parser.has_password,
                "Iframe_Count": min(parser.iframe_count, 10),
                "Popup_Count": min(parser.popup_count, 11),
                "Favicon_External": parser.favicon_external,
            }
            return num_redirects, redirect_status, dom_metrics, "Live Webpage HTML Extraction"
    except Exception:
        pass

    # Heuristic redirect check from query parameters if live connection was prevented
    if any(param in url.lower() for param in ["redirect=", "url=", "goto=", "r=", "dest=", "return="]):
        num_redirects = 2
        redirect_status = "Query Parameter Redirect (Heuristic)"
    else:
        num_redirects = 0
        redirect_status = "Direct (No Redirect Detected)"

    # Dataset benchmark population medians
    if is_suspicious_url:
        dom_metrics = {
            "External_Link_Ratio": 0.587,
            "Image_Link_Ratio": 0.442,
            "Form_Count": 3,
            "Password_Field_Present": 1 if any(w in url.lower() for w in ["login", "signin", "password", "auth", "token", "pwd"]) else 0,
            "Iframe_Count": 2,
            "Popup_Count": 1,
            "Favicon_External": 1,
        }
    else:
        dom_metrics = {
            "External_Link_Ratio": 0.253,
            "Image_Link_Ratio": 0.200,
            "Form_Count": 1,
            "Password_Field_Present": 0,
            "Iframe_Count": 0,
            "Popup_Count": 0,
            "Favicon_External": 0,
        }
    return num_redirects, redirect_status, dom_metrics, "Dataset Benchmark Baseline (Offline / Unreachable)"


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

    # Risk heuristic evaluation for baseline fallbacks
    is_suspicious_url = (
        has_suspicious
        or (has_ip == 1)
        or (num_dots > 3)
        or (num_hyphens > 2)
        or (has_https == 0)
        or (num_subdomains >= 2)
        or (entropy > 4.4)
    )

    # Real-world network queries:
    # 1. Domain age from ICANN RDAP (or benchmark neutral baseline)
    domain_age, reg_length, domain_age_status = check_live_domain_age(hostname, is_suspicious_url)

    # 2. Choice B: Live HTML Scraper + Redirect Inspector (or benchmark neutral baseline)
    num_redirects, redirect_status, dom_metrics, dom_source = check_live_page_and_dom(url, hostname, is_suspicious_url)

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
        "External_Link_Ratio": dom_metrics["External_Link_Ratio"],
        "Image_Link_Ratio": dom_metrics["Image_Link_Ratio"],
        "Form_Count": dom_metrics["Form_Count"],
        "Password_Field_Present": dom_metrics["Password_Field_Present"],
        "Iframe_Count": dom_metrics["Iframe_Count"],
        "Popup_Count": dom_metrics["Popup_Count"],
        "Favicon_External": dom_metrics["Favicon_External"],
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
        "dom_source": dom_source,
        "form_count": dom_metrics["Form_Count"],
        "has_password": bool(dom_metrics["Password_Field_Present"]),
        "iframe_count": dom_metrics["Iframe_Count"],
        "ext_link_ratio": f"{int(dom_metrics['External_Link_Ratio'] * 100)}%",
        "img_link_ratio": f"{int(dom_metrics['Image_Link_Ratio'] * 100)}%",
        "favicon_external": bool(dom_metrics["Favicon_External"]),
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
@app.get("/api/health")
def health():
    model_found = _resolve_candidate(["best_phishing_model.pkl"]) is not None
    return {
        "status": "ok",
        "service": "PhishGuard",
        "model_loaded": model_found,
    }


@app.post("/scan-url")
@app.post("/api/scan-url")
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
@app.post("/api/predict")
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
    matched = request.headers.get("x-matched-path", "").lower()
    action = request.query_params.get("action", "").lower()
    target = f"{p} {raw_path} {matched} {action}"

    # Handle favicon
    if "favicon.ico" in target:
        return favicon()

    # Handle health check
    if "health" in target:
        return health()

    # ANY POST request reaching here: inspect payload and route to ML engine
    if request.method == "POST":
        try:
            body = await request.json()
            if "url" in body:
                return scan_url(URLScanRequest(**body))
            else:
                return predict_manual(PredictRequest(**body))
        except Exception as e:
            return JSONResponse(status_code=400, content={"detail": str(e)})

    # For GET requests:
    if not p or any(x in p for x in ["index", "api"]):
        return index_page()

    return JSONResponse(
        status_code=404,
        content={"detail": f"Route '{path_name}' not found", "target": target}
    )
