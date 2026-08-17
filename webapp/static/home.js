const select = (query) => document.querySelector(query);

function number(value, decimals = 3) {
  return Number(value).toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

function element(tag, className, text) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== undefined) item.textContent = text;
  return item;
}

function renderResult(title, question, metric, models, takeaway) {
  const card = element("article", "home-result-card");
  card.append(element("p", "eyebrow", "Downstream experiment"), element("h3", "", title), element("p", "result-question", question));
  const leader = models.at(-1);
  const highlight = element("div", "result-highlight");
  highlight.append(element("span", "", metric.label), element("strong", "", number(leader[metric.key])));
  card.append(highlight);
  const comparison = element("div", "result-comparison");
  for (const model of models.slice(-3)) {
    const line = element("div", model === leader ? "chosen" : "");
    line.append(element("span", "", model.name), element("b", "", number(model[metric.key])));
    comparison.append(line);
  }
  card.append(comparison, element("p", "result-interpretation", takeaway));
  return card;
}

async function renderResults() {
  const host = select("#home-results");
  try {
    const response = await fetch("/api/bootstrap");
    const { results } = await response.json();
    host.replaceChildren(
      renderResult(
        "Cash-flow stress", results.cashflow_stress.task_label,
        { key: "average_precision", label: "Average precision" },
        results.cashflow_stress.models, results.cashflow_stress.interpretation,
      ),
      renderResult(
        "Future transaction volume", results.future_value.task_label,
        { key: "mae", label: "MAE · lower is better" },
        results.future_value.models, results.future_value.interpretation,
      ),
    );
  } catch (_) {
    host.textContent = "Results will appear when the local model release is available.";
  }
}

renderResults();
