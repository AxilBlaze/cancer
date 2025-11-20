const form = document.getElementById("convertForm");
const csvInput = document.getElementById("csvFile");
const apiInput = document.getElementById("apiUrl");
const messageEl = document.getElementById("formMessage");
const resultPreview = document.getElementById("resultPreview");
const downloadBtn = document.getElementById("downloadBtn");
const copyBtn = document.getElementById("copyBtn");
const logList = document.getElementById("logList");
const clearLogBtn = document.getElementById("clearLogBtn");

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

const renderJSON = (payload) => {
  latestPayload = payload;
  const formatted = JSON.stringify(payload, null, 2);
  resultPreview.textContent = formatted;
  downloadBtn.disabled = false;
  copyBtn.disabled = false;
};

const resetResult = () => {
  latestPayload = null;
  resultPreview.textContent = "// Converted JSON will appear here";
  downloadBtn.disabled = true;
  copyBtn.disabled = true;
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

    setMessage("Uploading and converting…");
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
    setMessage("Conversion successful.", "success");
    pushLog("Conversion succeeded.");
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


