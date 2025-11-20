# Breast Cancer Relapse Prediction System

A machine learning system for predicting breast cancer relapse risk using gene expression data. The system consists of two main components: a web-based CSV converter and an IoT device prediction server.

## Project Overview

This project processes gene expression data with ~24,000 features and extracts the top 200 most important features for prediction. The system uses ensemble machine learning models (Linear SVC and SGD SVM) to predict relapse risk.

## Architecture

- **Website Component** (`convert.py`): Converts CSV files with 24k+ features to JSON format with 200 selected features
- **IoT Device Component** (`server.py`): Runs prediction API server that processes the 200 features and returns risk predictions

## Prerequisites

- Python 3.7+
- Required packages:
  ```bash
  pip install fastapi uvicorn pandas scikit-learn joblib numpy requests
  ```

## Setup

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure all model files are present:
   - `feature_index.pkl` - List of 200 feature names
   - `scaler.pkl` - StandardScaler for feature normalization
   - `linear_svc_model.joblib` - Linear SVC model (optional)
   - `sgd_svm_model.joblib` - SGD SVM model (optional)
   - At least one model file must be present

4. Set environment variables (optional):
   ```bash
   export API_KEY="your-secret-api-key"  # For server.py authentication
   export PORT=8000                      # Server port (default: 8000)
   ```

## File Descriptions

### `convert.py` - CSV to JSON Converter (Website Component)

**Purpose**: Converts large CSV files (24k+ features) to JSON format containing only the top 200 features required for prediction.

**Input**:
- `data.csv`: CSV file with header row and at least one data row
  - First column typically contains sample ID
  - Remaining columns contain gene expression values (~24,483 features)
- `feature_index.pkl`: Pickle file containing list of 200 feature names to extract

**Output**:
- `output.json`: JSON file containing:
  ```json
  {
    "sample_id": "sample-<timestamp>",
    "feature_version": "v1_top200",
    "gene_features": [<200 numeric values>]
  }
  ```

**Usage**:

1. **As a standalone script**:
   ```bash
   python convert.py
   ```
   Reads `data.csv` and generates `output.json`

2. **As a FastAPI server** (for website integration):
   ```bash
   uvicorn convert:app --host 0.0.0.0 --port 8001
   ```
   
   **API Endpoints**:
   - `POST /convert`: Upload CSV file and get JSON response
     ```bash
     curl -X POST "http://localhost:8001/convert" \
          -F "file=@data.csv"
     ```
   - `GET /health`: Health check endpoint
     ```bash
     curl http://localhost:8001/health
     ```

**Features**:
- Automatic feature matching (case-insensitive)
- Validates all values are numeric
- Handles missing features gracefully
- Extracts exactly 200 features in the correct order

---

### `server.py` - Prediction API Server (IoT Device Component)

**Purpose**: Runs a FastAPI server that accepts gene feature vectors and returns breast cancer relapse risk predictions.

**Input**:
- `feature_index.pkl`: List of 200 feature names (must match convert.py output)
- `scaler.pkl`: StandardScaler for normalizing input features
- `linear_svc_model.joblib`: Linear SVC model file (optional)
- `sgd_svm_model.joblib`: SGD SVM model file (optional)
- At least one model file must be present

**API Request**:
```json
POST /predict
Headers:
  X-API-Key: <your-api-key>
  Content-Type: application/json
Body:
{
  "gene_features": [<200 float values>]
}
```

**Output**:
```json
{
  "risk_percent": 65.23,
  "will_relapse": true
}
```

**Usage**:

1. **Start the server**:
   ```bash
   uvicorn server:app --host 0.0.0.0 --port 8000
   ```

2. **API Endpoints**:
   - `GET /health`: Health check
     ```bash
     curl http://localhost:8000/health
     ```
   
   - `GET /feature-index`: Get list of required features (requires API key)
     ```bash
     curl -H "X-API-Key: changeme" http://localhost:8000/feature-index
     ```
   
   - `POST /predict`: Get prediction (requires API key)
     ```bash
     curl -X POST "http://localhost:8000/predict" \
          -H "X-API-Key: changeme" \
          -H "Content-Type: application/json" \
          -d '{"gene_features": [<200 values>]}'
     ```

**Features**:
- API key authentication
- Ensemble prediction (averages multiple models if available)
- Automatic feature scaling
- Logs all predictions to `predictions.log`
- Handles models with/without `predict_proba` method

---

### `test_request.py` - API Testing Script

**Purpose**: Helper script to test the prediction API with dummy or real data.

**Input**:
- Optional: `dummy_data.json` - JSON file with `gene_features` array
- Environment variable: `API_KEY` (default: "changeme")
- Configuration: `API_URL` (default: "http://localhost:8000")

**Output**:
- Console output showing:
  - Feature length from API
  - Prediction results (risk_percent, will_relapse)
  - Full JSON response

**Usage**:
```bash
# Set API key if needed
export API_KEY="your-api-key"

# Run test
python test_request.py
```

**Features**:
- Auto-generates dummy data if JSON file not found
- Validates feature length matches API requirements
- Displays formatted prediction results
- Error handling and status reporting

---

### `data.csv` - Input Data File

**Purpose**: Contains gene expression data with ~24,483 features.

**Format**:
- First row: Header with column names
  - First column: `id` (sample identifier)
  - Remaining columns: Gene feature names (e.g., "Contig42006_RC", "AB033032", etc.)
- Subsequent rows: Data rows with numeric values
  - First column: Sample ID (e.g., "95")
  - Remaining columns: Gene expression values (floats)

**Example**:
```csv
id,Contig42006_RC,AB033032,U45975,...
95,-0.167,-0.047,-0.146,...
```

**Note**: This file is processed by `convert.py` to extract the top 200 features.

---

### `feature_index.pkl` - Feature Names List

**Purpose**: Contains the list of 200 feature names that should be extracted from the CSV.

**Format**: Pickle file containing a Python list of strings
- Length: Exactly 200 feature names
- Example: `['Contig42006_RC', 'AB033032', 'U45975', ...]`

**Usage**: 
- Loaded by both `convert.py` and `server.py`
- Ensures consistent feature ordering between conversion and prediction

---

### `scaler.pkl` - Feature Scaler

**Purpose**: StandardScaler object used to normalize input features before prediction.

**Format**: Pickle file containing a scikit-learn StandardScaler object

**Usage**: 
- Loaded by `server.py` at startup
- Applied to input features before model prediction
- Ensures features are on the same scale as training data

---

### `linear_svc_model.joblib` - Linear SVC Model

**Purpose**: Trained Linear Support Vector Classifier model for relapse prediction.

**Format**: Joblib file containing a scikit-learn LinearSVC model

**Usage**:
- Loaded by `server.py` at startup (optional)
- Used for ensemble prediction if available
- If model has `predict_proba`, uses it; otherwise uses `decision_function` with sigmoid mapping

---

### `sgd_svm_model.joblib` - SGD SVM Model

**Purpose**: Trained Stochastic Gradient Descent SVM model for relapse prediction.

**Format**: Joblib file containing a scikit-learn SGDClassifier model

**Usage**:
- Loaded by `server.py` at startup (optional)
- Used for ensemble prediction if available
- If model has `predict_proba`, uses it; otherwise uses `decision_function` with sigmoid mapping

**Note**: At least one model file (`linear_svc_model.joblib` or `sgd_svm_model.joblib`) must be present for the server to start.

---

### `output.json` - Converted Feature Data

**Purpose**: JSON file containing the 200 extracted features from `data.csv`.

**Format**:
```json
{
  "sample_id": "sample-1763577612",
  "feature_version": "v1_top200",
  "gene_features": [0.056, -0.198, -0.212, ...]
}
```

**Usage**:
- Generated by `convert.py`
- Can be used as input to `server.py` prediction endpoint
- Contains exactly 200 numeric feature values

---

### `predictions.log` - Prediction Log File

**Purpose**: Logs all prediction requests and results for audit and debugging.

**Format**: Text log file with entries like:
```
2024-01-15 10:30:45 INFO Incoming predict request from 192.168.1.100
2024-01-15 10:30:45 INFO Prediction result: {"timestamp": 1705312245, "client": "192.168.1.100", "risk_percent": 65.23, "will_relapse": true}
```

**Usage**:
- Automatically created by `server.py`
- Contains timestamp, client IP, and prediction results
- Does not store full gene feature vectors (privacy protection)

---

## Workflow

### Complete Prediction Pipeline

1. **Prepare Data** (Website):
   ```bash
   # Place your CSV file as data.csv
   # Run converter
   python convert.py
   # This generates output.json with 200 features
   ```

2. **Start Prediction Server** (IoT Device):
   ```bash
   uvicorn server:app --host 0.0.0.0 --port 8000
   ```

3. **Send Prediction Request**:
   ```bash
   # Use output.json from step 1
   curl -X POST "http://<iot-device-ip>:8000/predict" \
        -H "X-API-Key: changeme" \
        -H "Content-Type: application/json" \
        -d @output.json
   ```

4. **Get Results**:
   ```json
   {
     "risk_percent": 65.23,
     "will_relapse": true
   }
   ```

### Testing

Test the prediction API:
```bash
python test_request.py
```

---

## Error Handling

### Common Issues

1. **Missing feature_index.pkl**:
   - Error: "feature_index.pkl not found"
   - Solution: Ensure `feature_index.pkl` exists in the project directory

2. **Feature length mismatch**:
   - Error: "gene_features length X != expected 200"
   - Solution: Ensure `convert.py` output has exactly 200 features

3. **Missing model files**:
   - Error: "At least one model file must be available"
   - Solution: Ensure at least one of `linear_svc_model.joblib` or `sgd_svm_model.joblib` exists

4. **API key authentication failure**:
   - Error: "Invalid API key"
   - Solution: Set correct `API_KEY` environment variable or use default "changeme"

5. **CSV parsing errors**:
   - Error: "Failed to read CSV file"
   - Solution: Ensure CSV has proper header row and at least one data row

---

## Security Notes

- Change the default API key in production
- Use environment variables for sensitive configuration
- The prediction server logs requests but not full feature vectors
- Consider adding HTTPS/TLS for production deployments

---

## License

[Specify your license here]

## Contact

[Your contact information]




