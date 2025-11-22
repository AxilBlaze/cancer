const form = document.getElementById("convertForm");
const csvInput = document.getElementById("csvFile");
const apiInput = document.getElementById("apiUrl");
const messageEl = document.getElementById("formMessage");
const resultPreview = document.getElementById("resultPreview");
const downloadBtn = document.getElementById("downloadBtn");
const copyBtn = document.getElementById("copyBtn");
const logList = document.getElementById("logList");
const clearLogBtn = document.getElementById("clearLogBtn");
const predictionPanel = document.getElementById("predictionPanel");
const predictionResult = document.getElementById("predictionResult");

let latestPayload = null;

const pushLog = (text, isError = false) => {
  const entry = document.createElement("li");
  const stamp = new Date().toLocaleTimeString();
  entry.innerHTML = `<time>[${stamp}]</time>${text}`;
  if (isError) {
    entry.style.color = "#f87171";
  }
  logList.prepend(entry);
};

const setMessage = (text, type = "") => {
  messageEl.textContent = text;
  messageEl.className = `form-message ${type}`;
};

const renderPrediction = (payload) => {
  if (payload.prediction) {
    const pred = payload.prediction;
    const riskPercent = pred.risk_percent?.toFixed(2) || "N/A";
    const willRelapse = pred.will_relapse;
    const riskColor = willRelapse ? "#ef4444" : "#10b981";
    
    predictionResult.innerHTML = `
      <div class="prediction-card" style="border-left: 4px solid ${riskColor};">
        <div class="prediction-header">
          <h3>Risk Assessment</h3>
          <span class="risk-badge" style="background-color: ${riskColor}20; color: ${riskColor};">
            ${willRelapse ? "HIGH RISK" : "LOW RISK"}
          </span>
        </div>
        <div class="prediction-details">
          <div class="prediction-item">
            <span class="label">Risk Percentage:</span>
            <span class="value" style="color: ${riskColor}; font-weight: bold; font-size: 1.2em;">
              ${riskPercent}%
            </span>
          </div>
          <div class="prediction-item">
            <span class="label">Will Relapse:</span>
            <span class="value" style="color: ${riskColor}; font-weight: bold;">
              ${willRelapse ? "Yes" : "No"}
            </span>
          </div>
        </div>
      </div>
    `;
    predictionPanel.style.display = "block";
  } else if (payload.prediction_error) {
    predictionResult.innerHTML = `
      <div class="prediction-card" style="border-left: 4px solid #f59e0b;">
        <div class="prediction-header">
          <h3>Prediction Unavailable</h3>
        </div>
        <div class="prediction-details">
          <p style="color: #f59e0b;">${payload.prediction_error}</p>
          <p style="font-size: 0.9em; color: #6b7280; margin-top: 0.5em;">
            Features were extracted successfully, but prediction server could not be reached.
          </p>
        </div>
      </div>
    `;
    predictionPanel.style.display = "block";
  } else {
    predictionPanel.style.display = "none";
  }
};

const renderJSON = (payload) => {
  latestPayload = payload;
  const formatted = JSON.stringify(payload, null, 2);
  resultPreview.textContent = formatted;
  downloadBtn.disabled = false;
  copyBtn.disabled = false;
  renderPrediction(payload);
};

const resetResult = () => {
  latestPayload = null;
  resultPreview.textContent = "// Converted JSON will appear here";
  downloadBtn.disabled = true;
  copyBtn.disabled = true;
  predictionPanel.style.display = "none";
};

const validateInputs = () => {
  const file = csvInput.files?.[0];
  if (!file) {
    throw new Error("Please select a CSV file.");
  }
  if (!file.name.endsWith(".csv")) {
    throw new Error("File must have a .csv extension.");
  }
  const apiUrl = apiInput.value.trim();
  if (!apiUrl) {
    throw new Error("API URL cannot be empty.");
  }
  return { file, apiUrl };
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setMessage("");

  try {
    const { file, apiUrl } = validateInputs();
    const endpoint = new URL("/convert", apiUrl).toString();
    const formData = new FormData();
    formData.append("file", file);

    setMessage("Uploading CSV, extracting features, and getting prediction…");
    pushLog(`Uploading ${file.name} to ${endpoint}`);

    const response = await fetch(endpoint, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `Server responded with ${response.status}`);
    }

    const payload = await response.json();
    renderJSON(payload);
    
    if (payload.prediction) {
      setMessage("Conversion and prediction successful!", "success");
      pushLog(`Prediction: ${payload.prediction.risk_percent?.toFixed(2)}% risk, will relapse: ${payload.prediction.will_relapse}`);
    } else if (payload.status === "conversion_success_prediction_failed") {
      setMessage("Features extracted, but prediction failed. Check server.py connection.", "error");
      pushLog("Conversion succeeded but prediction failed.", true);
    } else {
      setMessage("Conversion successful.", "success");
      pushLog("Conversion succeeded.");
    }
  } catch (error) {
    console.error(error);
    resetResult();
    setMessage(error.message, "error");
    pushLog(error.message, true);
  }
});

downloadBtn.addEventListener("click", () => {
  if (!latestPayload) return;
  const blob = new Blob([JSON.stringify(latestPayload, null, 2)], {
    type: "application/json",
  });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "output.json";
  link.click();
  URL.revokeObjectURL(link.href);
});

copyBtn.addEventListener("click", async () => {
  if (!latestPayload) return;
  try {
    await navigator.clipboard.writeText(JSON.stringify(latestPayload, null, 2));
    setMessage("JSON copied to clipboard.", "success");
  } catch (error) {
    setMessage("Failed to copy JSON.", "error");
  }
});

clearLogBtn.addEventListener("click", () => {
  logList.innerHTML = "";
});


