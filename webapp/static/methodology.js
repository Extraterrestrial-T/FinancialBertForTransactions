const chartCanvas = document.querySelector("#loss-chart");
const chartSummary = document.querySelector("#loss-summary");

function drawLossChart(points) {
  const rect = chartCanvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  chartCanvas.width = Math.floor(rect.width * ratio);
  chartCanvas.height = Math.floor(rect.height * ratio);
  const context = chartCanvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  const width = rect.width;
  const height = rect.height;
  const pad = { left: 43, right: 16, top: 22, bottom: 33 };
  const allLosses = points.flatMap((point) => [point.train_loss, point.validation_loss]);
  const minimum = Math.min(...allLosses);
  const maximum = Math.max(...allLosses);
  const yRange = Math.max(maximum - minimum, .1);
  const x = (point) => pad.left + ((point.epoch - points[0].epoch) / (points.at(-1).epoch - points[0].epoch)) * (width - pad.left - pad.right);
  const y = (loss) => height - pad.bottom - ((loss - minimum) / yRange) * (height - pad.top - pad.bottom);
  context.clearRect(0, 0, width, height);
  context.strokeStyle = "rgba(174,185,207,.18)";
  context.fillStyle = "#9dabC4";
  context.font = "11px system-ui";
  for (let index = 0; index < 4; index += 1) {
    const yPos = pad.top + ((height - pad.top - pad.bottom) * index / 3);
    const value = maximum - yRange * index / 3;
    context.beginPath(); context.moveTo(pad.left, yPos); context.lineTo(width - pad.right, yPos); context.stroke();
    context.fillText(value.toFixed(2), 4, yPos + 4);
  }
  const trace = (field, color) => {
    context.beginPath();
    points.forEach((point, index) => { if (index === 0) context.moveTo(x(point), y(point[field])); else context.lineTo(x(point), y(point[field])); });
    context.strokeStyle = color; context.lineWidth = 2; context.stroke();
  };
  trace("train_loss", "#7895ff");
  trace("validation_loss", "#d7f56a");
  context.fillStyle = "#aeb9cf";
  context.fillText(`epoch ${points[0].epoch}`, pad.left, height - 10);
  context.textAlign = "right"; context.fillText(`epoch ${points.at(-1).epoch}`, width - pad.right, height - 10); context.textAlign = "left";
  context.fillStyle = "#7895ff"; context.fillRect(width - 176, 15, 10, 3); context.fillStyle = "#d9e8a0"; context.fillText("train", width - 160, 19); context.fillStyle = "#d7f56a"; context.fillRect(width - 103, 15, 10, 3); context.fillStyle = "#d9e8a0"; context.fillText("validation", width - 87, 19);
}

async function loadLoss() {
  try {
    const response = await fetch("/api/training-history");
    if (!response.ok) throw new Error("training history unavailable");
    const payload = await response.json();
    drawLossChart(payload.epochs);
    const best = payload.epochs.reduce((previous, current) => current.validation_loss < previous.validation_loss ? current : previous);
    chartSummary.textContent = `Lowest validation loss: ${payload.best_validation_loss.toFixed(4)} at epoch ${best.epoch}.`;
    window.addEventListener("resize", () => drawLossChart(payload.epochs));
  } catch (error) {
    chartSummary.textContent = "The committed training trace could not be loaded.";
  }
}

loadLoss();
