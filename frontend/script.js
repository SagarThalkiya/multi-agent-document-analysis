const API_BASE = "http://127.0.0.1:8000";

const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("document");
const uploadFeedback = document.getElementById("upload-feedback");
const jobIdInput = document.getElementById("job-id");
const analyzeBtn = document.getElementById("start-analysis");
const analysisFeedback = document.getElementById("analysis-feedback");
const statusBadge = document.getElementById("status-badge");
const timingEl = document.getElementById("timing");

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    uploadFeedback.textContent = "Please select a file first.";
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    uploadFeedback.textContent = "File is larger than 10MB.";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  uploadFeedback.textContent = "Uploading...";
  try {
    const response = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail ?? "Upload failed");
    }
    uploadFeedback.textContent = data.message;
    jobIdInput.value = data.job_id;
    analyzeBtn.disabled = false;
    setStatus("uploaded");
  } catch (error) {
    uploadFeedback.textContent = error.message;
    analyzeBtn.disabled = true;
  }
});

analyzeBtn.addEventListener("click", async () => {
  const jobId = jobIdInput.value.trim();
  if (!jobId) {
    analysisFeedback.textContent = "Upload a document first.";
    return;
  }

  analysisFeedback.textContent = "Starting analysis...";
  setStatus("processing");
  try {
    const response = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: jobId }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail ?? "Unable to start analysis.");
    }
    analysisFeedback.textContent = data.message;
    await pollResults(jobId);
  } catch (error) {
    analysisFeedback.textContent = error.message;
    setStatus("error");
  }
});

async function pollResults(jobId) {
  let attempts = 0;
  while (attempts < 30) {
    attempts += 1;
    const response = await fetch(`${API_BASE}/results/${jobId}`);
    const data = await response.json();
    setStatus(data.status);
    renderResults(data);
    if (["completed", "partial", "failed"].includes(data.status)) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
  analysisFeedback.textContent = "Timed out waiting for results.";
}

function setStatus(status) {
  statusBadge.textContent = status;
  statusBadge.dataset.status = status;
}

function renderResults(data) {
  const results = data.results || {};
  renderSummary(results.summary || {});
  renderEntities(results.entities || {});
  renderSentiment(results.sentiment || {});

  const total = data.total_processing_time_seconds;
  const timingParts = [
    total ? `Total wall time: ${total.toFixed(2)}s` : null,
    `Agents complete: ${data.agents_completed ?? 0}`,
    `Agents failed: ${data.agents_failed ?? 0}`,
  ].filter(Boolean);
  timingEl.textContent = timingParts.join(" • ");

  if (data.warning) {
    analysisFeedback.textContent = data.warning;
  }
}

function renderSummary(summary) {
  const summaryText = document.getElementById("summary-text");
  const summaryPoints = document.getElementById("summary-points");
  const summaryConfidence = document.getElementById("summary-confidence");

  if (summary.error) {
    summaryText.textContent = summary.error;
    summaryPoints.innerHTML = "";
    summaryConfidence.textContent = "";
    return;
  }

  summaryText.textContent = summary.text || "No summary generated.";
  summaryPoints.innerHTML = "";
  (summary.key_points || []).forEach((point) => {
    const li = document.createElement("li");
    li.textContent = point;
    summaryPoints.appendChild(li);
  });
  summaryConfidence.textContent = summary.confidence
    ? `Confidence: ${(summary.confidence * 100).toFixed(1)}%`
    : "";
}

function renderEntities(entities) {
  populateList("people-list", entities.people, "role");
  populateList("orgs-list", entities.organizations, "type");
  populateList("dates-list", entities.dates, "context");
  populateList("locations-list", entities.locations, "context");

  if (entities?.error) {
    analysisFeedback.textContent = entities.error;
  }
}

function populateList(elementId, items = [], extraField) {
  const container = document.getElementById(elementId);
  container.innerHTML = "";
  if (!items || items.length === 0) {
    const li = document.createElement("li");
    li.textContent = "—";
    container.appendChild(li);
    return;
  }
  items.forEach((item) => {
    const li = document.createElement("li");
    const extras = extraField && item[extraField] ? ` (${item[extraField]})` : "";
    li.textContent = `${item.name} — mentions: ${item.mentions}${extras}`;
    container.appendChild(li);
  });
}

function renderSentiment(sentiment) {
  const sentimentText = document.getElementById("sentiment-text");
  const phrasesList = document.getElementById("sentiment-phrases");

  if (sentiment.error) {
    sentimentText.textContent = sentiment.error;
    phrasesList.innerHTML = "";
    return;
  }

  sentimentText.textContent = sentiment.tone
    ? `Tone: ${sentiment.tone} (${sentiment.formality || "n/a"})`
    : "No sentiment detected.";

  phrasesList.innerHTML = "";
  (sentiment.key_phrases || []).forEach((phrase) => {
    const li = document.createElement("li");
    li.textContent = phrase;
    phrasesList.appendChild(li);
  });
}

