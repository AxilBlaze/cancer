# combined_server.py
# Combines convert.py and server.py into a single FastAPI server
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import time
import os
import json
import io
from typing import List, Optional
from sklearn.preprocessing import StandardScaler
import logging

# ==================== CONFIGURATION ====================
app = FastAPI(title="CSV Converter & Prediction API")

# CORS middleware - allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
MODEL_FOLDER = "./"
FEATURE_INDEX_PATH = os.path.join(MODEL_FOLDER, "feature_index.pkl")
SCALER_PATH = os.path.join(MODEL_FOLDER, "scaler.pkl")
LINEAR_SVC_PATH = os.path.join(MODEL_FOLDER, "linear_svc_model.joblib")
SGD_SVM_PATH = os.path.join(MODEL_FOLDER, "sgd_svm_model.joblib")
LABEL_ENCODER_PATH = os.path.join(MODEL_FOLDER, "label_encoder.pkl")
DATA_CSV_PATH = os.path.join(MODEL_FOLDER, "data.csv")
OUTPUT_JSON_PATH = os.path.join(MODEL_FOLDER, "output.json")

# API Key (optional for internal calls, but keep for /predict endpoint)
API_KEY = os.environ.get("API_KEY", "changeme")
PORT = int(os.environ.get("PORT", 8000))
LOG_FILE = "predictions.log"

# Default desired K
TOP_K = 200

# ==================== LOGGING ====================
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# ==================== PYDANTIC MODELS ====================
class PredictRequest(BaseModel):
    gene_features: List[float]

class PredictResponse(BaseModel):
    risk_percent: float
    will_relapse: bool

# ==================== LOAD ARTIFACTS (from server.py) ====================
def load_artifacts(folder=MODEL_FOLDER):
    """Load models and scaler from server.py"""
    fidx_path = os.path.join(folder, "feature_index.pkl")
    scaler_path = os.path.join(folder, "scaler.pkl")
    linear_svc_path = os.path.join(folder, "linear_svc_model.joblib")
    sgd_svm_path = os.path.join(folder, "sgd_svm_model.joblib")

    if not os.path.exists(fidx_path):
        raise FileNotFoundError(f"{fidx_path} not found")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"{scaler_path} not found")

    feature_index = joblib.load(fidx_path)
    scaler = joblib.load(scaler_path)

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

# Load artifacts at startup
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

# Label encoder
if os.path.exists(LABEL_ENCODER_PATH):
    le = joblib.load(LABEL_ENCODER_PATH)
    LABELS = list(le.classes_)
else:
    LABELS = ["class_0", "class_1"]
    logging.info("label_encoder.pkl not found; using fallback labels: %s", LABELS)

# ==================== UTILITY FUNCTIONS ====================
def decision_to_prob(score):
    """Convert decision scores to pseudo-probabilities"""
    return 1.0 / (1.0 + np.exp(-score))

def load_feature_index():
    """Load feature_index.pkl if present (from convert.py)"""
    if os.path.exists(FEATURE_INDEX_PATH):
        try:
            fi = joblib.load(FEATURE_INDEX_PATH)
            if isinstance(fi, (list, tuple)) and len(fi) >= 1:
                return list(fi)
        except Exception:
            pass
    return None

def normalize_colname(s: str) -> str:
    return str(s).strip().lower()

def extract_by_index_list(row: pd.Series, feature_index: List[str], tolerate_case: bool = True, raise_http: bool = True) -> List[float]:
    """
    Given a pandas Series (one row) and a list of feature names (feature_index),
    try exact matching first, then tolerant lowercase matching. Returns list of floats.
    Raises HTTPException on missing or non-numeric values if raise_http=True, else raises ValueError.
    """
    # Exact match
    missing = [f for f in feature_index if f not in row.index]
    if not missing:
        vals = row[feature_index].astype(float).tolist()
        return vals

    # Build lowercase map
    col_map = {normalize_colname(c): c for c in row.index}
    recovered = []
    still_missing = []
    for f in feature_index:
        if f in row.index:
            recovered.append(row[f])
            continue
        key = normalize_colname(f)
        if key in col_map:
            recovered.append(row[col_map[key]])
        else:
            still_missing.append(f)

    if len(still_missing) > 0:
        # Final attempt: substring match (dangerous but sometimes helpful)
        final_recovered = []
        unresolved = []
        for i,f in enumerate(feature_index):
            if recovered and i < len(recovered) and recovered[i] is not None:
                final_recovered.append(recovered[i])
                continue
            key = normalize_colname(f)
            found = None
            for ck in row.index:
                if key in normalize_colname(ck) or normalize_colname(ck) in key:
                    found = ck
                    break
            if found:
                final_recovered.append(row[found])
            else:
                unresolved.append(f)
        if len(unresolved) > 0:
            error_msg = f"Missing features in CSV: {unresolved[:10]}... (total {len(unresolved)})"
            if raise_http:
                raise HTTPException(status_code=400, detail=error_msg)
            else:
                raise ValueError(error_msg)
        recovered = final_recovered

    # Convert to floats and validate
    numeric = []
    for i, v in enumerate(recovered):
        if pd.isna(v) or v == "" or v is None:
            error_msg = f"Non-numeric/missing value for feature '{feature_index[i]}'"
            if raise_http:
                raise HTTPException(status_code=400, detail=error_msg)
            else:
                raise ValueError(error_msg)
        try:
            # remove commas in numbers like "1,234.56"
            s = str(v).replace(",", "").strip()
            fv = float(s)
        except Exception as e:
            error_msg = f"Failed to convert feature '{feature_index[i]}' value '{v}' to float: {str(e)}"
            if raise_http:
                raise HTTPException(status_code=400, detail=error_msg)
            else:
                raise ValueError(error_msg)
        numeric.append(fv)

    return numeric

def preprocess_gene_vector(gene_features: List[float]):
    """Preprocess gene vector for prediction (from server.py)"""
    if len(gene_features) != FEATURE_LEN:
        raise HTTPException(status_code=400, detail=f"gene_features length {len(gene_features)} != expected {FEATURE_LEN}")
    arr = np.array(gene_features, dtype=np.float32).reshape(1, -1)
    try:
        arr_scaled = SCALER.transform(arr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scaler transform failed: {str(e)}")
    return arr_scaled

# ==================== DIRECT PREDICTION FUNCTION (replaces HTTP call) ====================
def predict_direct(gene_features: List[float], client: str = "internal") -> dict:
    """
    Direct prediction function (replaces call_prediction_server HTTP call).
    Returns prediction result as dict.
    """
    try:
        # Preprocess
        X_in = preprocess_gene_vector(gene_features)
        
        risk_scores = []
        
        # Linear SVC model prediction
        if LOG_MODEL is not None:
            try:
                if hasattr(LOG_MODEL, "predict_proba"):
                    prob = float(LOG_MODEL.predict_proba(X_in)[:, 1][0])
                elif hasattr(LOG_MODEL, "decision_function"):
                    score = LOG_MODEL.decision_function(X_in)[0]
                    prob = float(decision_to_prob(score))
                else:
                    pred_label_temp = str(LOG_MODEL.predict(X_in)[0])
                    prob = 1.0 if pred_label_temp == LABELS[-1] else 0.0
                risk_scores.append(prob)
            except Exception as e:
                logging.exception("Error during linear SVC prediction")
        
        # SGD SVM prediction
        if SVM_MODEL is not None:
            try:
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
        
        if len(risk_scores) == 0:
            raise HTTPException(status_code=500, detail="No models available for prediction")
        
        combined_risk = float(np.mean(risk_scores))
        risk_percent = combined_risk * 100.0
        will_relapse = combined_risk >= 0.5
        
        result = {
            "risk_percent": risk_percent,
            "will_relapse": will_relapse
        }
        
        # Log prediction
        log_entry = {
            "timestamp": time.time(),
            "client": client,
            "risk_percent": risk_percent,
            "will_relapse": will_relapse
        }
        logging.info("Prediction result: %s", json.dumps(log_entry))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error during prediction")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ==================== AUTH ====================
def check_api_key(x_api_key: Optional[str]):
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

# ==================== ENDPOINTS ====================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "top_k": TOP_K,
        "feature_index_loaded": os.path.exists(FEATURE_INDEX_PATH),
        "feature_len": FEATURE_LEN
    }

@app.get("/feature-index")
async def feature_index(x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)
    return {"feature_len": FEATURE_LEN, "features": FEATURE_INDEX}

@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest, request: Request, x_api_key: Optional[str] = Header(None)):
    """Standalone prediction endpoint (from server.py)"""
    check_api_key(x_api_key)
    
    client = request.client.host if request.client else "unknown"
    logging.info("Incoming predict request from %s", client)
    
    result = predict_direct(payload.gene_features, client=client)
    return PredictResponse(**result)

@app.post("/convert")
async def convert_csv(file: UploadFile = File(...)):
    """
    Accepts CSV, extracts features, and gets prediction (from convert.py).
    Now uses direct function call instead of HTTP request.
    """
    # Read feature index if available
    feature_index = load_feature_index()

    # Read uploaded CSV
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), header=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV file: {str(e)}")

    if df.shape[0] < 1:
        raise HTTPException(status_code=400, detail="CSV must contain at least one data row after the header.")

    row = df.iloc[0]

    # If no feature_index saved, fall back to first TOP_K columns
    if feature_index is None:
        possible_drops = ["id", "class", "sample_id"]
        cols = [c for c in df.columns if c.lower() not in possible_drops]
        if len(cols) < TOP_K:
            raise HTTPException(status_code=400, detail=f"CSV has only {len(cols)} candidate columns, need at least {TOP_K}. Consider saving a feature_index.pkl on the server.")
        feature_index = cols[:TOP_K]

    # Ensure we have exactly TOP_K features
    if len(feature_index) < TOP_K:
        raise HTTPException(status_code=400, detail=f"Loaded feature_index length {len(feature_index)} < required {TOP_K}")
    if len(feature_index) > TOP_K:
        feature_index = feature_index[:TOP_K]

    # Extract values
    try:
        gene_features = extract_by_index_list(row, feature_index, raise_http=True)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected extraction error: {str(e)}")

    # Final validation
    if len(gene_features) != TOP_K:
        raise HTTPException(status_code=500, detail=f"Extraction returned {len(gene_features)} values, expected {TOP_K}")

    # Build initial payload
    sample_id = f"uploaded-{int(time.time())}"
    payload = {
        "sample_id": sample_id,
        "feature_version": "v1_top200",
        "gene_features": gene_features
    }

    # Call prediction DIRECTLY (no HTTP call)
    try:
        prediction_result = predict_direct(gene_features, client="csv_upload")
        payload["prediction"] = prediction_result
        payload["status"] = "success"
    except HTTPException as he:
        payload["prediction"] = None
        payload["status"] = "conversion_success_prediction_failed"
        payload["prediction_error"] = he.detail

    return JSONResponse(content=payload)

# Function to convert data.csv to JSON file
def convert_csv_to_json(csv_path: str = DATA_CSV_PATH, output_path: str = OUTPUT_JSON_PATH):
    """
    Reads data.csv, extracts 200 features using feature_index.pkl, and saves to JSON file.
    This function can be called directly or used by the API.
    """
    # Load feature index
    feature_index = load_feature_index()
    if feature_index is None:
        raise ValueError(f"feature_index.pkl not found at {FEATURE_INDEX_PATH}. Cannot extract features.")
    
    # Ensure we have exactly TOP_K features
    if len(feature_index) < TOP_K:
        raise ValueError(f"Loaded feature_index length {len(feature_index)} < required {TOP_K}")
    if len(feature_index) > TOP_K:
        feature_index = feature_index[:TOP_K]
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_path, header=0)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {csv_path}: {str(e)}")
    
    if df.shape[0] < 1:
        raise ValueError(f"CSV file {csv_path} must contain at least one data row after the header.")
    
    # Use the first data row
    row = df.iloc[0]
    
    # Extract values in order
    try:
        gene_features = extract_by_index_list(row, feature_index, raise_http=False)
    except Exception as e:
        raise ValueError(f"Failed to extract features: {str(e)}")
    
    # Final validation
    if len(gene_features) != TOP_K:
        raise ValueError(f"Extraction returned {len(gene_features)} values, expected {TOP_K}")
    
    # Build payload
    sample_id = f"sample-{int(time.time())}"
    payload = {
        "sample_id": sample_id,
        "feature_version": "v1_top200",
        "gene_features": gene_features
    }
    
    # Write to JSON file
    try:
        with open(output_path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"Successfully converted {csv_path} to {output_path}")
        print(f"Extracted {len(gene_features)} features from {len(df.columns)} total columns")
        return payload
    except Exception as e:
        raise ValueError(f"Failed to write JSON file {output_path}: {str(e)}")

# Run conversion if script is executed directly
if __name__ == "__main__":
    if os.path.exists(DATA_CSV_PATH):
        try:
            convert_csv_to_json()
        except Exception as e:
            print(f"Error: {str(e)}")
            exit(1)
    else:
        import uvicorn
        print(f"Starting combined server on port {PORT}")
        uvicorn.run(app, host="0.0.0.0", port=PORT)

