/* PRAGMA-lite explorer: presentation and interaction only. All predictions
 * come from the small fixed Flask API; no model state lives in the browser. */

const state = {
  bootstrap: null,
  accounts: [],
  current: null,
  hitRegions: [],
  selectedEvent: null,
};

const $ = (selector) => document.querySelector(selector);
const adapterSelect = $("#adapter-select");
const accountSelect = $("#account-select");
const cutoffSelect = $("#cutoff-select");
const runButton = $("#run-prediction");
const randomButton = $("#random-account");
const status = $("#load-status");
const canvas = $("#timeline");
const windowStartInput = $("#window-start");

function setStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle("error", isError);
}

async function getJSON(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || "The request could not be completed.");
  return payload;
}

function makeElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function formatNumber(value, decimals = 0) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(Number(value));
}

function formatSigned(value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
  return `${Number(value) >= 0 ? "+" : "−"}${formatNumber(Math.abs(Number(value)), 2)}`;
}

function formatPercent(value) {
  return value === null || value === undefined ? "—" : `${(Number(value) * 100).toFixed(1)}%`;
}

function rankAdapterForTask(task) {
  const choices = state.bootstrap.adapters.filter((adapter) => adapter.task === task);
  const selected = choices.find((adapter) => adapter.rank === (task === "future_value" ? 4 : 16));
  return selected || choices[0];
}

function selectedAdapter() {
  return state.bootstrap.adapters.find((adapter) => adapter.id === adapterSelect.value);
}

function populateAdapters() {
  adapterSelect.replaceChildren();
  for (const adapter of state.bootstrap.adapters) {
    const option = makeElement("option", "", adapter.label);
    option.value = adapter.id;
    adapterSelect.append(option);
  }
  const defaultAdapter = rankAdapterForTask("future_value");
  if (defaultAdapter) adapterSelect.value = defaultAdapter.id;
}

function renderEvidence(results) {
  $("#protocol-copy").textContent = results.protocol;
  const host = $("#result-cards");
  host.replaceChildren(
    resultCard({
      title: "Cash-flow stress · next 60 days",
      question: results.cashflow_stress.task_label,
      models: results.cashflow_stress.models,
      metricA: { key: "roc_auc", label: "ROC-AUC" },
      metricB: { key: "average_precision", label: "Average precision" },
      selected: results.cashflow_stress.models.at(-1),
      intervalKey: "average_precision_interval",
      intervalLabel: "AP",
      interpretation: results.cashflow_stress.interpretation,
    }),
    resultCard({
      title: "Future volume · next 180 days",
      question: results.future_value.task_label,
      models: results.future_value.models,
      metricA: { key: "mae", label: "MAE ↓" },
      metricB: { key: "r2", label: "R² ↑" },
      selected: results.future_value.models.at(-1),
      intervalKey: "mae_interval",
      intervalLabel: "MAE",
      interpretation: results.future_value.interpretation,
    }),
  );
}

function resultCard(config) {
  const card = makeElement("article", "result-card");
  card.append(makeElement("h3", "", config.title));
  card.append(makeElement("p", "result-question", config.question));
  const header = makeElement("div", "metric-row header");
  header.append(makeElement("span", "", "Model"), makeElement("span", "", config.metricA.label), makeElement("span", "", config.metricB.label));
  card.append(header);
  for (const model of config.models) {
    const row = makeElement("div", `metric-row${model === config.selected ? " best" : ""}`);
    row.append(
      makeElement("span", "", model.name),
      makeElement("span", "", formatNumber(model[config.metricA.key], 3)),
      makeElement("span", "", formatNumber(model[config.metricB.key], 3)),
    );
    card.append(row);
  }
  const interval = config.selected[config.intervalKey];
  if (interval) {
    card.append(makeElement(
      "p",
      "result-callout",
      `Selected rank-${config.selected.selected_rank} LoRA: ${config.intervalLabel} ${formatNumber(config.selected[config.intervalLabel === "AP" ? "average_precision" : "mae"], 3)} (account-clustered 95% interval ${formatNumber(interval[0], 3)}–${formatNumber(interval[1], 3)}).`,
    ));
  }
  card.append(makeElement("p", "result-interpretation", config.interpretation));
  return card;
}

async function refreshAccounts({ random = false } = {}) {
  const adapter = selectedAdapter();
  if (!adapter) return;
  setStatus("Finding cutoff-safe account snapshots…");
  accountSelect.disabled = true;
  cutoffSelect.disabled = true;
  runButton.disabled = true;
  const payload = await getJSON(`/api/accounts?task=${encodeURIComponent(adapter.task)}`);
  state.accounts = payload.accounts;
  accountSelect.replaceChildren();
  for (const account of state.accounts) {
    const option = makeElement("option", "", `Account ${account.account_id} · ${account.eligible_history_events} eligible events · ${account.split}`);
    option.value = String(account.account_id);
    accountSelect.append(option);
  }
  if (random && state.accounts.length) accountSelect.selectedIndex = Math.floor(Math.random() * state.accounts.length);
  await refreshCutoffs();
}

async function refreshCutoffs() {
  const adapter = selectedAdapter();
  if (!adapter || !accountSelect.value) return;
  cutoffSelect.disabled = true;
  runButton.disabled = true;
  setStatus("Checking the available prediction cutoffs…");
  const payload = await getJSON(`/api/cutoffs?task=${encodeURIComponent(adapter.task)}&account_id=${encodeURIComponent(accountSelect.value)}`);
  cutoffSelect.replaceChildren();
  for (const cutoff of payload.cutoffs) {
    const option = makeElement("option", "", cutoff.slice(0, 10));
    option.value = cutoff;
    cutoffSelect.append(option);
  }
  if (payload.cutoffs.length) cutoffSelect.selectedIndex = payload.cutoffs.length - 1;
  if (windowStartInput && payload.cutoffs.length) {
    windowStartInput.max = cutoffSelect.value.slice(0, 10);
    if (windowStartInput.value && windowStartInput.value > windowStartInput.max) {
      windowStartInput.value = "";
    }
  }
  accountSelect.disabled = false;
  cutoffSelect.disabled = payload.cutoffs.length === 0;
  runButton.disabled = payload.cutoffs.length === 0;
  setStatus(payload.cutoffs.length ? "Snapshot ready. Run the model when you are ready." : "This account has no eligible cutoff for the selected task.", !payload.cutoffs.length);
}

async function requestPrediction() {
  const adapter = selectedAdapter();
  if (!adapter || !accountSelect.value || !cutoffSelect.value) return;
  runButton.disabled = true;
  setStatus("Binding the adapter and scoring the cutoff-safe history…");
  try {
    const params = new URLSearchParams({ adapter: adapter.id, account_id: accountSelect.value, cutoff: cutoffSelect.value });
    if (windowStartInput?.value) params.set("start", windowStartInput.value);
    state.current = await getJSON(`/api/prediction?${params}`);
    renderPrediction(state.current);
    setStatus("Prediction ready. Click an event in the timeline to inspect its input representation.");
  } catch (error) {
    setStatus(error.message || "Unable to run this prediction.", true);
  } finally {
    runButton.disabled = false;
  }
}

function metric(label, value) {
  const container = makeElement("div", "outcome-metric");
  container.append(makeElement("span", "", label), makeElement("strong", "", value));
  return container;
}

function renderPrediction(data) {
  $("#prediction-workspace").classList.remove("is-hidden");
  $("#snapshot-title").textContent = `Account ${data.account.id} · ${data.adapter.label}`;
  $("#snapshot-meta").textContent = `${data.history_event_count} observed events · cutoff ${data.cutoff} · held-out horizon ends ${data.window_end}`;
  const output = data.outcome;
  const metrics = $("#outcome-metrics");
  metrics.replaceChildren();
  if (output.kind === "future_value") {
    $("#outcome-title").textContent = "Forecasted volume against the held-out 180 days";
    $("#outcome-copy").textContent = output.message;
    metrics.append(
      metric("Model output · log1p(volume)", formatNumber(output.prediction_log1p, 3)),
      metric("Displayed inverse · absolute volume", formatNumber(output.prediction_volume)),
      metric("Held-out actual · absolute volume", formatNumber(output.actual_volume)),
    );
  } else {
    $("#outcome-title").textContent = "Forecasted stress against the held-out 60 days";
    $("#outcome-copy").textContent = output.message;
    metrics.append(
      metric("Stress probability", formatPercent(output.stress_probability)),
      metric("Held-out outcome", output.actual_crossed_threshold ? "Crossed threshold" : "Did not cross"),
      metric("Held-out minimum balance", formatNumber(output.actual_minimum_balance, 2)),
    );
  }
  const preferred = data.history.findLast((event) => event.model_visible) || data.history.at(-1) || data.future[0];
  drawTimeline(data, preferred?.id);
  renderEvent(preferred);
  renderTrace(data);
}

function renderTrace(data) {
  const host = $("#trace-steps");
  const taskLabel = data.task === "future_value" ? "180-day volume estimate" : "60-day stress estimate";
  const steps = [
    ["History supplied", `${data.model_event_count} events at or before ${data.cutoff}`],
    ["Context encoded", data.context_truncated ? `Trailing ${data.context_limit} events used` : "Entire history used"],
    ["Future held out", `${taskLabel} observed through ${data.window_end}`],
  ];
  host.replaceChildren(...steps.map(([label, value]) => {
    const step = makeElement("div", "trace-step");
    step.append(makeElement("span", "", label), makeElement("strong", "", value));
    return step;
  }));
}

function renderEvent(event) {
  const host = $("#event-details");
  if (!event) {
    host.replaceChildren(makeElement("p", "muted", "No event is available for this snapshot."));
    return;
  }
  $("#event-title").textContent = event.model_visible ? "Model-visible transaction" : "Held-out transaction";
  const chip = makeElement("span", `context-chip${event.model_visible ? "" : " future"}`, event.model_visible ? "In model context" : "Observed after cutoff");
  const date = makeElement("p", "event-date", event.date);
  const value = makeElement("div", `event-value${event.amount < 0 ? " negative" : ""}`, formatSigned(event.amount));
  const list = makeElement("dl", "detail-list");
  const details = [
    ["Balance after", formatNumber(event.balance, 2)],
    ["Transaction type", event.type || "—"],
    ["Operation", event.operation || "—"],
    ["Category", event.category || "—"],
  ];
  if (event.model_visible) {
    details.push(["Amount input token", event.amount_bucket || "—"], ["Balance input token", event.balance_bucket || "—"]);
  }
  for (const [key, valueText] of details) {
    const row = document.createElement("div");
    row.append(makeElement("dt", "", key), makeElement("dd", "", valueText));
    list.append(row);
  }
  host.replaceChildren(chip, date, value, list);
}

function drawTimeline(data, selectedId) {
  const all = [...data.history, ...data.future];
  const empty = $("#timeline-empty");
  if (!all.length) {
    empty.classList.remove("is-hidden");
    return;
  }
  empty.classList.add("is-hidden");
  const bounds = canvas.getBoundingClientRect();
  const scale = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.floor(bounds.width * scale));
  canvas.height = Math.max(1, Math.floor(bounds.height * scale));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  const width = bounds.width;
  const height = bounds.height;
  ctx.clearRect(0, 0, width, height);
  const pad = { left: 44, right: 18, top: 24, bottom: 48 };
  const balanceTop = pad.top;
  const balanceBottom = height * .63;
  const amountTop = height * .73;
  const amountBottom = height - pad.bottom;
  const times = all.map((event) => new Date(event.date).getTime());
  const minTime = Math.min(...times);
  const maxTime = Math.max(...times, minTime + 86400000);
  const balances = all.map((event) => Number(event.balance)).filter(Number.isFinite);
  const balanceMin = Math.min(...balances);
  const balanceMax = Math.max(...balances);
  const balanceRange = Math.max(balanceMax - balanceMin, 1);
  const amounts = all.map((event) => Math.abs(Number(event.amount))).filter(Number.isFinite);
  const amountMax = Math.max(...amounts, 1);
  const xOf = (event) => pad.left + ((new Date(event.date).getTime() - minTime) / (maxTime - minTime)) * (width - pad.left - pad.right);
  const yBalance = (event) => balanceBottom - ((Number(event.balance) - balanceMin) / balanceRange) * (balanceBottom - balanceTop);
  const yAmount = (event) => amountBottom - (Math.abs(Number(event.amount)) / amountMax) * (amountBottom - amountTop);
  const cutoffTime = new Date(data.cutoff).getTime();
  const cutoffX = pad.left + ((cutoffTime - minTime) / (maxTime - minTime)) * (width - pad.left - pad.right);

  ctx.strokeStyle = "rgba(174, 187, 209, .16)";
  ctx.lineWidth = 1;
  for (let i = 0; i < 3; i += 1) {
    const y = balanceTop + ((balanceBottom - balanceTop) * i / 2);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
  }
  ctx.fillStyle = "#91a0bb";
  ctx.font = "11px Inter, system-ui, sans-serif";
  ctx.fillText(formatNumber(balanceMax), 4, balanceTop + 4);
  ctx.fillText(formatNumber(balanceMin), 4, balanceBottom + 3);
  ctx.fillText("balance", 4, balanceTop - 8);
  ctx.fillText("transaction amount", 4, amountTop - 7);
  ctx.fillText(all[0].date, pad.left, height - 16);
  ctx.textAlign = "right";
  ctx.fillText(all.at(-1).date, width - pad.right, height - 16);
  ctx.textAlign = "left";

  for (const event of all) {
    const x = xOf(event);
    const y = yAmount(event);
    ctx.fillStyle = event.model_visible ? "rgba(221, 154, 64, .72)" : "rgba(113, 128, 160, .42)";
    ctx.fillRect(x - 1.5, y, 3, amountBottom - y);
  }
  const paths = [data.history, data.future];
  for (const [index, events] of paths.entries()) {
    if (!events.length) continue;
    ctx.beginPath();
    events.forEach((event, i) => {
      const x = xOf(event); const y = yBalance(event);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = index === 0 ? "#65d9c1" : "#7180a0";
    ctx.lineWidth = index === 0 ? 2.2 : 1.7;
    if (index === 1) ctx.setLineDash([5, 5]);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.strokeStyle = "rgba(215, 245, 106, .68)";
  ctx.setLineDash([4, 5]);
  ctx.beginPath(); ctx.moveTo(cutoffX, balanceTop); ctx.lineTo(cutoffX, amountBottom); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = "#d7f56a"; ctx.fillText("cutoff", Math.min(cutoffX + 6, width - 53), balanceTop + 11);

  state.hitRegions = [];
  for (const event of all) {
    const x = xOf(event); const y = yBalance(event); const isSelected = event.id === selectedId;
    ctx.beginPath(); ctx.arc(x, y, isSelected ? 5.4 : 3.2, 0, Math.PI * 2);
    ctx.fillStyle = isSelected ? "#d7f56a" : (event.model_visible ? "#65d9c1" : "#7180a0");
    ctx.fill();
    state.hitRegions.push({ event, x, y, barY: yAmount(event) });
  }
}

function chooseEventAt(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  let closest = null;
  for (const hit of state.hitRegions) {
    const distance = Math.min(Math.hypot(hit.x - x, hit.y - y), Math.hypot(hit.x - x, hit.barY - y));
    if (!closest || distance < closest.distance) closest = { ...hit, distance };
  }
  if (closest && closest.distance <= 18) {
    state.selectedEvent = closest.event;
    renderEvent(closest.event);
    drawTimeline(state.current, closest.event.id);
  }
}

async function initialize() {
  try {
    state.bootstrap = await getJSON("/api/bootstrap");
    if ($("#result-cards")) renderEvidence(state.bootstrap.results);
    populateAdapters();
    await refreshAccounts();
  } catch (error) {
    setStatus(error.message || "The explorer could not initialise.", true);
  }
}

adapterSelect.addEventListener("change", () => refreshAccounts().catch((error) => setStatus(error.message, true)));
accountSelect.addEventListener("change", () => refreshCutoffs().catch((error) => setStatus(error.message, true)));
cutoffSelect.addEventListener("change", () => {
  if (windowStartInput) {
    windowStartInput.max = cutoffSelect.value.slice(0, 10);
    if (windowStartInput.value && windowStartInput.value > windowStartInput.max) {
      windowStartInput.value = "";
    }
  }
});
randomButton.addEventListener("click", () => refreshAccounts({ random: true }).catch((error) => setStatus(error.message, true)));
runButton.addEventListener("click", requestPrediction);
canvas.addEventListener("click", (event) => chooseEventAt(event.clientX, event.clientY));
canvas.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && state.hitRegions.length) {
    const eventItem = state.hitRegions.at(-1).event;
    renderEvent(eventItem); drawTimeline(state.current, eventItem.id);
  }
});
window.addEventListener("resize", () => { if (state.current) drawTimeline(state.current, state.selectedEvent?.id); });
initialize();
