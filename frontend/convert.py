# convert_api.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import time
import os
import json
from typing import List, Optional

app = FastAPI(title="CSV -> Top200 Extractor API")

# Allow the local frontend (and other localhost origins) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path where a feature_index.pkl might be stored (adjust if needed)
MODEL_FOLDER = "./"
FEATURE_INDEX_PATH = os.path.join(MODEL_FOLDER, "feature_index.pkl")
DATA_CSV_PATH = os.path.join(MODEL_FOLDER, "data.csv")
OUTPUT_JSON_PATH = os.path.join(MODEL_FOLDER, "output.json")

# Default desired K
TOP_K = 200

def load_feature_index():
    """
    Load feature_index.pkl if present. It should be a list of names (length TOP_K).
    If missing, return None (caller will fall back to CSV's first TOP_K columns).
    """
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

@app.post("/convert")
async def convert_csv(file: UploadFile = File(...)):
    """
    Accepts an uploaded CSV (header + at least one data row).
    Returns JSON containing the extracted top-K gene_features (length TOP_K).
    """
    # Read feature index if available
    feature_index = load_feature_index()

    # Read uploaded CSV into pandas (only first data row)
    try:
        contents = await file.read()
        # Use pandas read_csv from bytes
        df = pd.read_csv(pd.io.common.BytesIO(contents), header=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV file: {str(e)}")

    if df.shape[0] < 1:
        raise HTTPException(status_code=400, detail="CSV must contain at least one data row after the header.")

    # Use the first data row by default
    row = df.iloc[0]

    # If no feature_index saved, fall back to first TOP_K columns excluding known non-feature columns
    if feature_index is None:
        # try dropping common non-gene columns if present
        possible_drops = ["id", "class", "sample_id"]
        cols = [c for c in df.columns if c.lower() not in possible_drops]
        if len(cols) < TOP_K:
            raise HTTPException(status_code=400, detail=f"CSV has only {len(cols)} candidate columns, need at least {TOP_K}. Consider saving a feature_index.pkl on the server.")
        # choose first TOP_K columns
        feature_index = cols[:TOP_K]

    # Ensure we have exactly TOP_K features (if feature_index bigger, trim; if smaller, error)
    if len(feature_index) < TOP_K:
        raise HTTPException(status_code=400, detail=f"Loaded feature_index length {len(feature_index)} < required {TOP_K}")
    if len(feature_index) > TOP_K:
        feature_index = feature_index[:TOP_K]

    # Extract values in order
    try:
        gene_features = extract_by_index_list(row, feature_index, raise_http=True)
    except HTTPException as he:
        # bubble up meaningful error
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected extraction error: {str(e)}")

    # final validation
    if len(gene_features) != TOP_K:
        raise HTTPException(status_code=500, detail=f"Extraction returned {len(gene_features)} values, expected {TOP_K}")

    # Build response
    sample_id = f"uploaded-{int(time.time())}"
    payload = {
        "sample_id": sample_id,
        "feature_version": "v1_top200",
        "gene_features": gene_features
    }

    return JSONResponse(content=payload)

# Simple health check
@app.get("/health")
def health():
    return {"status": "ok", "top_k": TOP_K, "feature_index_loaded": os.path.exists(FEATURE_INDEX_PATH)}

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
        print(f"Warning: {DATA_CSV_PATH} not found. Run as API server or provide CSV file.")
