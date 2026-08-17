const uploadForm = document.querySelector("#upload-form");
const uploadAdapter = document.querySelector("#upload-adapter");
const uploadStatus = document.querySelector("#upload-status");
const uploadResult = document.querySelector("#upload-result");

function uploadElement(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function readableNumber(value, digits = 0) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "not available";
  return Number(value).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

async function loadUploadAdapters() {
  const response = await fetch("/api/bootstrap");
  const payload = await response.json();
  for (const adapter of payload.adapters) {
    const option = uploadElement("option", "", adapter.label);
    option.value = adapter.id;
    uploadAdapter.append(option);
  }
  const preferred = payload.adapters.find((adapter) => adapter.task === "future_value" && adapter.rank === 4);
  if (preferred) uploadAdapter.value = preferred.id;
}

function resultMetric(label, value) {
  const metric = uploadElement("div", "");
  metric.append(uploadElement("span", "", label), uploadElement("strong", "", value));
  return metric;
}

function renderUploadResult(data) {
  uploadResult.classList.remove("is-hidden");
  uploadResult.replaceChildren();
  uploadResult.append(uploadElement("h2", "", data.prediction.kind === "future_value" ? "Future transaction-volume output" : "Cash-flow stress output"));
  const grid = uploadElement("div", "upload-result-grid");
  if (data.prediction.kind === "future_value") {
    grid.append(resultMetric("Predicted log1p(volume)", readableNumber(data.prediction.prediction_log1p, 3)), resultMetric("Displayed volume", readableNumber(data.prediction.prediction_volume)), resultMetric("Events sent to model", `${data.input.model_event_count} of ${data.input.event_count}`));
  } else {
    grid.append(resultMetric("Stress probability", `${(data.prediction.stress_probability * 100).toFixed(1)}%`), resultMetric("Raw classifier logit", readableNumber(data.prediction.raw_logit, 3)), resultMetric("Events sent to model", `${data.input.model_event_count} of ${data.input.event_count}`));
  }
  uploadResult.append(grid);
  const note = uploadElement("p", "upload-note", data.input.context_truncated ? `The selected window had ${data.input.event_count} rows. The model retained its final ${data.input.context_limit} events.` : "The whole selected window fit inside the model context.");
  uploadResult.append(note);
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  uploadStatus.textContent = "Validating the CSV and running the selected adapter.";
  uploadStatus.classList.remove("error");
  try {
    const response = await fetch("/api/upload-prediction", { method: "POST", body: new FormData(uploadForm) });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || "The upload could not be scored.");
    renderUploadResult(payload);
    uploadStatus.textContent = "Prediction ready. The uploaded rows were not saved.";
  } catch (error) {
    uploadStatus.textContent = error.message || "The upload could not be scored.";
    uploadStatus.classList.add("error");
  }
});

loadUploadAdapters().catch((error) => { uploadStatus.textContent = error.message || "Adapters could not be loaded."; uploadStatus.classList.add("error"); });
