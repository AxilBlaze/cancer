# server.py
import os
import time
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import logging

# --------------------
# Config
# --------------------
MODEL_FOLDER = "./"   # change if models are in different folder
LOG_FILE = "predictions.log"
API_KEY = os.environ.get("API_KEY", "changeme")  # set a real secret in env var on Pi
PORT = int(os.environ.get("PORT", 8000))

# --------------------
# Logging
# --------------------
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# --------------------
# FastAPI app
# --------------------
app = FastAPI(title="Breast-Risk Edge API")

# --------------------
# Pydantic models (input/output)
# --------------------
class PredictRequest(BaseModel):
    gene_features: List[float]  # length must match feature_index length

class PredictResponse(BaseModel):
    risk_percent: float  # Risk score as percentage (0-100)
    will_relapse: bool

# --------------------
# Utilities: load artifacts
# --------------------
def load_artifacts(folder=MODEL_FOLDER):
    # feature index
    fidx_path = os.path.join(folder, "feature_index.pkl")
    scaler_path = os.path.join(folder, "scaler.pkl")
    linear_svc_path = os.path.join(folder, "linear_svc_model.joblib")
    sgd_svm_path = os.path.join(folder, "sgd_svm_model.joblib")

    if not os.path.exists(fidx_path):
        raise FileNotFoundError(f"{fidx_path} not found")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"{scaler_path} not found")

    feature_index = joblib.load(fidx_path)  # list of feature names
    scaler = joblib.load(scaler_path)

    # Load models (optional - at least one should exist)
    log_model = None
    if os.path.exists(linear_svc_path):
        log_model = joblib.load(linear_svc_path)
        logging.info(f"Loaded linear_svc_model.joblib")
    else:
        logging.warning(f"{linear_svc_path} not found, will skip linear SVC model")

    svm_model = None
    if os.path.exists(sgd_svm_path):
        svm_model = joblib.load(sgd_svm_path)
        logging.info(f"Loaded sgd_svm_model.joblib")
    else:
        logging.warning(f"{sgd_svm_path} not found, will skip SGD SVM model")

    if log_model is None and svm_model is None:
        raise FileNotFoundError("At least one model file must be available")

    return feature_index, scaler, log_model, svm_model

# load at startup
try:
    FEATURE_INDEX, SCALER, LOG_MODEL, SVM_MODEL = load_artifacts()
    FEATURE_LEN = len(FEATURE_INDEX)
    models_loaded = []
    if LOG_MODEL is not None:
        models_loaded.append("linear_svc")
    if SVM_MODEL is not None:
        models_loaded.append("sgd_svm")
    logging.info(f"Loaded artifacts. Feature length = {FEATURE_LEN}. Models loaded: {', '.join(models_loaded)}")
except Exception as e:
    logging.exception("Failed to load model artifacts on startup.")
    raise

# If SVM doesn't have predict_proba, we will try to calibrate it at load time (if enough data)
# But calibration requires training data; for simplicity we will use decision_function and map via sigmoid.
def decision_to_prob(score):
    # simple sigmoid mapping to convert decision scores to pseudo-probabilities
    return 1.0 / (1.0 + np.exp(-score))

# mapping labels if using LabelEncoder originally
# If you saved label encoder, load it; else default mapping: 0->class0,1->class1
LABEL_ENCODER_PATH = os.path.join(MODEL_FOLDER, "label_encoder.pkl")
if os.path.exists(LABEL_ENCODER_PATH):
    le = joblib.load(LABEL_ENCODER_PATH)
    LABELS = list(le.classes_)
else:
    # fallback — user should know order (print to logs)
    LABELS = ["class_0", "class_1"]
    logging.info("label_encoder.pkl not found; using fallback labels: %s", LABELS)

# --------------------
# Auth dependency (simple)
# --------------------
def check_api_key(x_api_key: Optional[str]):
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

# --------------------
# Helper: preprocess incoming vector
# --------------------
def preprocess_gene_vector(gene_features: List[float]):
    # 1) verify length
    if len(gene_features) != FEATURE_LEN:
        raise HTTPException(status_code=400, detail=f"gene_features length {len(gene_features)} != expected {FEATURE_LEN}")
    arr = np.array(gene_features, dtype=np.float32).reshape(1, -1)
    # 2) scale
    try:
        arr_scaled = SCALER.transform(arr)
    except Exception as e:
        # If scaler expects different shape, raise readable error
        raise HTTPException(status_code=500, detail=f"Scaler transform failed: {str(e)}")
    return arr_scaled

# --------------------
# ENDPOINTS
# --------------------

@app.get("/health")
async def health():
    return {"status": "ok", "feature_len": FEATURE_LEN}

@app.get("/feature-index")
async def feature_index(x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)
    # return only names (could be large)
    return {"feature_len": FEATURE_LEN, "features": FEATURE_INDEX}

@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest, request: Request, x_api_key: Optional[str] = Header(None)):
    # simple auth
    check_api_key(x_api_key)

    # log incoming request summary
    client = request.client.host if request.client else "unknown"
    logging.info("Incoming predict request from %s", client)

    # preprocess gene vector
    try:
        X_in = preprocess_gene_vector(payload.gene_features)
    except HTTPException as he:
        logging.error("Preprocess error: %s", he.detail)
        raise

    risk_scores = []

    # ---------- Linear SVC model prediction (if available) ----------
    if LOG_MODEL is not None:
        try:
            if hasattr(LOG_MODEL, "predict_proba"):
                prob = float(LOG_MODEL.predict_proba(X_in)[:, 1][0])
            elif hasattr(LOG_MODEL, "decision_function"):
                # fallback: decision_function -> sigmoid
                score = LOG_MODEL.decision_function(X_in)[0]
                prob = float(decision_to_prob(score))
            else:
                pred_label_temp = str(LOG_MODEL.predict(X_in)[0])
                prob = 1.0 if pred_label_temp == LABELS[-1] else 0.0
            risk_scores.append(prob)
        except Exception as e:
            logging.exception("Error during linear SVC prediction")

    # ---------- SGD SVM prediction (if available) ----------
    if SVM_MODEL is not None:
        try:
            # if SVM has predict_proba (rare), use it; else use decision_function -> sigmoid
            if hasattr(SVM_MODEL, "predict_proba"):
                prob_svm = float(SVM_MODEL.predict_proba(X_in)[:, 1][0])
            elif hasattr(SVM_MODEL, "decision_function"):
                score_svm = SVM_MODEL.decision_function(X_in)[0]
                prob_svm = float(decision_to_prob(score_svm))
            else:
                pred_label_svm = str(SVM_MODEL.predict(X_in)[0])
                prob_svm = 1.0 if pred_label_svm == LABELS[-1] else 0.0
            risk_scores.append(prob_svm)
        except Exception as e:
            logging.exception("Error during SGD SVM prediction")

    # ---------- Combined risk (simple average) ----------
    if len(risk_scores) == 0:
        raise HTTPException(status_code=500, detail="No models available for prediction")
    
    combined_risk = float(np.mean(risk_scores))
    risk_percent = combined_risk * 100.0  # Convert to percentage
    will_relapse = combined_risk >= 0.5

    resp = PredictResponse(
        risk_percent=risk_percent,
        will_relapse=will_relapse
    )

    # log full response (avoid storing full gene vector in logs for privacy)
    log_entry = {
        "timestamp": time.time(),
        "client": client,
        "risk_percent": risk_percent,
        "will_relapse": will_relapse
    }
    logging.info("Prediction result: %s", json.dumps(log_entry))

    return resp

# --------------------
# Run with uvicorn externally:
# uvicorn server:app --host 0.0.0.0 --port 8000
# --------------------
