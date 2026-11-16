/**
 * Industrial Predictive Maintenance Dashboard Client Application
 * Communicates strictly with FastAPI backend endpoints.
 * ZERO HARDCODED PREDICTIONS OR METRICS. All charts rendered from API payloads.
 */

// Global Chart Instances to prevent canvas reuse errors
let sensorChartInstance = null;
let shapChartInstance = null;
let uploadChartInstance = null;
let costChartInstance = null;
let modelChartInstance = null;
let simInterval = null;

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
  setupNavigation();
  loadFleetOverview();
  setupUploadZone();
  setupCostSimulator();
  setupSimulationView();
});

// =====================================================================
// 1. Navigation Controller
// =====================================================================
function setupNavigation() {
  const buttons = document.querySelectorAll(".nav-btn");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetView = btn.dataset.view;
      switchView(targetView);
    });
  });
}

function switchView(viewName) {
  document.querySelectorAll(".nav-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.view === viewName);
  });
  document.querySelectorAll(".view-section").forEach((sec) => {
    sec.classList.toggle("active", sec.id === `view-${viewName}`);
  });

  // Lazy trigger loads per view
  if (viewName === "fleet") loadFleetOverview();
  if (viewName === "models") loadModelComparison();
  if (viewName === "cost") runCostSimulation();
}

// =====================================================================
// 2. Fleet Overview View (GET /fleet)
// =====================================================================
async function loadFleetOverview() {
  try {
    const res = await fetch("/fleet");
    if (!res.ok) throw new Error("Failed to load fleet data");
    const data = await res.json();

    // Populate KPI cards
    document.getElementById("kpi-total-engines").innerText = data.summary.total_engines;
    document.getElementById("kpi-critical").innerText = data.summary.critical;
    document.getElementById("kpi-warning").innerText = data.summary.warning;
    document.getElementById("kpi-healthy").innerText = data.summary.healthy;
    document.getElementById("kpi-avg-rul").innerText = `${data.summary.avg_fleet_rul} cycles`;

    // Populate Table
    const tbody = document.getElementById("fleet-table-body");
    tbody.innerHTML = "";

    data.fleet.forEach((eng) => {
      const tr = document.createElement("tr");

      let badgeClass = "badge-healthy";
      if (eng.status === "CRITICAL") badgeClass = "badge-critical";
      else if (eng.status === "WARNING") badgeClass = "badge-warning";

      tr.innerHTML = `
        <td><strong>#${eng.engine_id}</strong></td>
        <td>${eng.current_cycles}</td>
        <td><strong>${eng.predicted_rul}</strong> cycles</td>
        <td>[${eng.lower_q10} – ${eng.upper_q90}] cycles</td>
        <td><span class="badge ${badgeClass}">${eng.status}</span></td>
        <td style="font-size: 0.8rem; color: #94a3b8;">${eng.recommended_action}</td>
        <td>
          <button class="btn-primary" style="padding: 0.3rem 0.6rem; font-size: 0.75rem;" onclick="inspectEngine(${eng.engine_id})">
            Inspect
          </button>
        </td>
      `;
      tbody.appendChild(tr);
    });

    // Populate Engine selector dropdowns
    populateEngineSelectors(data.fleet);
  } catch (err) {
    console.error("Fleet load error:", err);
  }
}

function populateEngineSelectors(fleet) {
  const selectors = ["detail-engine-select", "sim-engine-select"];
  selectors.forEach((selId) => {
    const sel = document.getElementById(selId);
    if (!sel || sel.children.length > 1) return;
    sel.innerHTML = "";
    fleet.forEach((eng) => {
      const opt = document.createElement("option");
      opt.value = eng.engine_id;
      opt.innerText = `Engine #${eng.engine_id} (RUL: ${eng.predicted_rul} cyc, ${eng.status})`;
      sel.appendChild(opt);
    });
  });
}

function inspectEngine(engineId) {
  document.getElementById("detail-engine-select").value = engineId;
  switchView("detail");
  loadEngineDetail(engineId);
}

// =====================================================================
// 3. Engine Detail View (GET /engine/{id})
// =====================================================================
async function loadEngineDetail(engineId) {
  if (!engineId) {
    engineId = document.getElementById("detail-engine-select").value || 1;
  }
  try {
    const res = await fetch(`/engine/${engineId}`);
    if (!res.ok) throw new Error(`Failed to load engine #${engineId}`);
    const data = await res.json();

    document.getElementById("detail-engine-title").innerText = `Engine #${data.engine_id} Prognostic Telemetry`;
    document.getElementById("detail-current-cycle").innerText = data.current_cycle;
    document.getElementById("detail-pred-rul").innerText = `${data.predicted_rul} cycles`;
    document.getElementById("detail-interval").innerText = `[${data.prediction_interval.lower_q10} to ${data.prediction_interval.upper_q90}] cycles`;
    document.getElementById("detail-action").innerText = data.recommended_action;

    const badge = document.getElementById("detail-status-badge");
    badge.innerText = data.status;
    badge.className = `badge ${data.status === "CRITICAL" ? "badge-critical" : data.status === "WARNING" ? "badge-warning" : "badge-healthy"}`;

    // Render Sensor Trend Chart
    renderSensorTrendChart(data.history);

    // Render Local SHAP Waterfall
    renderShapChart(data.shap_explanation);
  } catch (err) {
    console.error("Engine detail error:", err);
  }
}

function renderSensorTrendChart(history) {
  const ctx = document.getElementById("sensorTrendChart").getContext("2d");
  if (sensorChartInstance) sensorChartInstance.destroy();

  sensorChartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: history.cycles,
      datasets: [
        {
          label: "LPT Outlet Temp (s_4)",
          data: history.sensor_s4_lpt_temp,
          borderColor: "#ef4444",
          backgroundColor: "rgba(239, 68, 68, 0.1)",
          yAxisID: "y1",
          tension: 0.2
        },
        {
          label: "HPC Static Press (s_11)",
          data: history.sensor_s11_hpc_press,
          borderColor: "#06b6d4",
          backgroundColor: "rgba(6, 182, 212, 0.1)",
          yAxisID: "y2",
          tension: 0.2
        },
        {
          label: "LPC Outlet Temp (s_2)",
          data: history.sensor_s2_lpc_temp,
          borderColor: "#f59e0b",
          backgroundColor: "rgba(245, 158, 11, 0.1)",
          yAxisID: "y1",
          tension: 0.2
        }
      ]
    },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { title: { display: true, text: "Operational Flight Cycles", color: "#94a3b8" }, grid: { color: "#334155" }, ticks: { color: "#94a3b8" } },
        y1: { type: "linear", display: true, position: "left", title: { display: true, text: "Temperature (°R)", color: "#ef4444" }, grid: { color: "#334155" }, ticks: { color: "#94a3b8" } },
        y2: { type: "linear", display: true, position: "right", title: { display: true, text: "Pressure (psia)", color: "#06b6d4" }, grid: { drawOnChartArea: false }, ticks: { color: "#94a3b8" } }
      },
      plugins: {
        legend: { labels: { color: "#f8fafc" } }
      }
    }
  });
}

function renderShapChart(shapData) {
  const ctx = document.getElementById("shapChart").getContext("2d");
  if (shapChartInstance) shapChartInstance.destroy();

  const labels = shapData.top_features.map((f) => f.feature);
  const impacts = shapData.top_features.map((f) => f.shap_impact);
  const colors = impacts.map((v) => (v < 0 ? "#ef4444" : "#10b981"));

  shapChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "SHAP Impact on RUL (Cycles)",
          data: impacts,
          backgroundColor: colors,
          borderColor: "#334155",
          borderWidth: 1
        }
      ]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      scales: {
        x: { title: { display: true, text: "SHAP Contribution (Cycles)", color: "#94a3b8" }, grid: { color: "#334155" }, ticks: { color: "#94a3b8" } },
        y: { ticks: { color: "#f8fafc" }, grid: { color: "#334155" } }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

// =====================================================================
// 4. Upload & Predict View (POST /predict)
// =====================================================================
function setupUploadZone() {
  const dropzone = document.getElementById("upload-dropzone");
  const fileInput = document.getElementById("file-input");

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "#38bdf8";
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.style.borderColor = "#0284c7";
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "#0284c7";
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      handleFileUpload(fileInput.files[0]);
    }
  });
}

async function handleFileUpload(file) {
  const statusDiv = document.getElementById("upload-status");
  const resultsCard = document.getElementById("upload-results-card");

  statusDiv.innerHTML = `<span style="color: #38bdf8;">Uploading & running live inference on <strong>${file.name}</strong>...</span>`;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/predict", {
      method: "POST",
      body: formData
    });

    const data = await res.json();

    if (!res.ok) {
      statusDiv.innerHTML = `<span style="color: #ef4444;">Error: ${data.detail || "Upload validation failed"}</span>`;
      resultsCard.style.display = "none";
      return;
    }

    statusDiv.innerHTML = `<span style="color: #10b981;">Inference complete for ${data.filename}!</span>`;
    resultsCard.style.display = "block";

    document.getElementById("upload-pred-rul").innerText = `${data.predicted_rul} cycles`;
    document.getElementById("upload-pred-interval").innerText = `[${data.prediction_interval.lower_q10} – ${data.prediction_interval.upper_q90}] cycles`;
    document.getElementById("upload-action").innerText = data.recommended_action;

    const badge = document.getElementById("upload-status-badge");
    badge.innerText = data.status;
    badge.className = `badge ${data.status === "CRITICAL" ? "badge-critical" : data.status === "WARNING" ? "badge-warning" : "badge-healthy"}`;

    renderUploadChart(data.history);
  } catch (err) {
    statusDiv.innerHTML = `<span style="color: #ef4444;">Network or parsing error: ${err.message}</span>`;
  }
}

function renderUploadChart(history) {
  const ctx = document.getElementById("uploadHistoryChart").getContext("2d");
  if (uploadChartInstance) uploadChartInstance.destroy();

  uploadChartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: history.cycles,
      datasets: [
        {
          label: "Predicted Remaining Useful Life (RUL)",
          data: history.predicted_rul_series,
          borderColor: "#38bdf8",
          backgroundColor: "rgba(56, 189, 248, 0.15)",
          fill: true,
          tension: 0.1
        }
      ]
    },
    options: {
      responsive: true,
      scales: {
        x: { title: { display: true, text: "Operational Cycle", color: "#94a3b8" }, grid: { color: "#334155" }, ticks: { color: "#94a3b8" } },
        y: { title: { display: true, text: "RUL Prediction (Cycles)", color: "#94a3b8" }, grid: { color: "#334155" }, ticks: { color: "#94a3b8" }, min: 0 }
      },
      plugins: {
        legend: { labels: { color: "#f8fafc" } }
      }
    }
  });
}

// =====================================================================
// 5. Live Simulation View (GET /simulate/{id})
// =====================================================================
function setupSimulationView() {
  const startBtn = document.getElementById("sim-start-btn");
  const stopBtn = document.getElementById("sim-stop-btn");
  const resetBtn = document.getElementById("sim-reset-btn");
  const engineSelect = document.getElementById("sim-engine-select");

  let currentCycle = 30;
  let maxCycles = 150;

  startBtn.addEventListener("click", () => {
    startBtn.disabled = true;
    stopBtn.disabled = false;
    const engineId = engineSelect.value;
    const speed = parseInt(document.getElementById("sim-speed-select").value) || 1;

    simInterval = setInterval(async () => {
      try {
        const res = await fetch(`/simulate/${engineId}?cycle=${currentCycle}`);
        if (!res.ok) throw new Error("Simulation fetch failed");
        const data = await res.json();

        maxCycles = data.total_cycles_in_history;

        // Update Gauge & Texts
        document.getElementById("sim-current-cycle").innerText = data.current_cycle;
        document.getElementById("sim-rul-display").innerText = `${data.predicted_rul} cyc`;
        document.getElementById("sim-max-cycle").innerText = maxCycles;

        const badge = document.getElementById("sim-status-badge");
        badge.innerText = data.status;
        badge.className = `badge ${data.status === "CRITICAL" ? "badge-critical" : data.status === "WARNING" ? "badge-warning" : "badge-healthy"}`;

        currentCycle += 2;
        if (currentCycle > maxCycles) {
          clearInterval(simInterval);
          startBtn.disabled = false;
          stopBtn.disabled = true;
        }
      } catch (e) {
        console.error(e);
        clearInterval(simInterval);
        startBtn.disabled = false;
        stopBtn.disabled = true;
      }
    }, 1000 / speed);
  });

  stopBtn.addEventListener("click", () => {
    clearInterval(simInterval);
    startBtn.disabled = false;
    stopBtn.disabled = true;
  });

  resetBtn.addEventListener("click", () => {
    clearInterval(simInterval);
    currentCycle = 30;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    document.getElementById("sim-current-cycle").innerText = "30";
    document.getElementById("sim-rul-display").innerText = "--";
    document.getElementById("sim-status-badge").innerText = "READY";
  });
}

// =====================================================================
// 6. Cost Simulator View (POST /cost)
// =====================================================================
function setupCostSimulator() {
  const inputs = ["cost-unplanned", "cost-planned", "cost-waste", "cost-threshold"];

  inputs.forEach((id) => {
    const el = document.getElementById(id);
    el.addEventListener("input", () => {
      document.getElementById(`${id}-val`).innerText = id === "cost-threshold" ? `${el.value} cycles` : `$${parseInt(el.value).toLocaleString()}`;
      runCostSimulation();
    });
  });
}

async function runCostSimulation() {
  const payload = {
    cost_unplanned: parseFloat(document.getElementById("cost-unplanned").value),
    cost_planned: parseFloat(document.getElementById("cost-planned").value),
    cost_waste_per_cycle: parseFloat(document.getElementById("cost-waste").value),
    threshold: parseFloat(document.getElementById("cost-threshold").value)
  };

  try {
    const res = await fetch("/cost", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error("Cost calculation failed");
    const data = await res.json();

    document.getElementById("cost-savings-rtf").innerText = `$${data.savings.savings_vs_run_to_failure.toLocaleString()}`;
    document.getElementById("cost-savings-rtf-pct").innerText = `(${data.savings.pct_savings_vs_rtf}% Cost Avoidance)`;
    document.getElementById("cost-savings-fixed").innerText = `$${data.savings.savings_vs_fixed_schedule.toLocaleString()}`;
    document.getElementById("cost-savings-fixed-pct").innerText = `(${data.savings.pct_savings_vs_fixed}% vs Fixed Overhaul)`;

    renderCostChart(data);
  } catch (err) {
    console.error("Cost simulation error:", err);
  }
}

function renderCostChart(data) {
  const ctx = document.getElementById("costComparisonChart").getContext("2d");
  if (costChartInstance) costChartInstance.destroy();

  const labels = ["Run-to-Failure", "Fixed-Schedule (130c)", `Predictive (T*=${data.inputs.threshold}c)`];
  const unplanned = [data.run_to_failure.unplanned_cost, data.fixed_schedule.unplanned_cost, data.predictive_maintenance.unplanned_cost];
  const planned = [data.run_to_failure.planned_cost, data.fixed_schedule.planned_cost, data.predictive_maintenance.planned_cost];
  const waste = [data.run_to_failure.wasted_life_cost, data.fixed_schedule.wasted_life_cost, data.predictive_maintenance.wasted_life_cost];

  costChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        { label: "Unplanned Failures", data: unplanned, backgroundColor: "#ef4444" },
        { label: "Planned Overhauls", data: planned, backgroundColor: "#0284c7" },
        { label: "Wasted Component Life", data: waste, backgroundColor: "#f59e0b" }
      ]
    },
    options: {
      responsive: true,
      scales: {
        x: { stacked: true, ticks: { color: "#f8fafc" }, grid: { color: "#334155" } },
        y: {
          stacked: true,
          ticks: {
            color: "#94a3b8",
            callback: (v) => `$${(v / 1e6).toFixed(1)}M`
          },
          title: { display: true, text: "Total Fleet Expenditure ($)", color: "#94a3b8" },
          grid: { color: "#334155" }
        }
      },
      plugins: {
        legend: { labels: { color: "#f8fafc" } }
      }
    }
  });
}

// =====================================================================
// 7. Model Comparison View (GET /metrics)
// =====================================================================
async function loadModelComparison() {
  try {
    const res = await fetch("/metrics");
    if (!res.ok) throw new Error("Failed to load metrics");
    const metrics = await res.json();

    const fd1Models = metrics.dataset_fd001.models;
    const tbody = document.getElementById("models-table-body");
    tbody.innerHTML = "";

    const modelNames = Object.keys(fd1Models);
    const rmses = [];
    const maes = [];
    const scores = [];

    modelNames.forEach((name) => {
      const m = fd1Models[name];
      rmses.push(m.rmse);
      maes.push(m.mae);
      scores.push(m.nasa_score);

      const isChosen = name === metrics.dataset_fd001.chosen_model;
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${name}</strong> ${isChosen ? '<span class="badge badge-healthy">Production</span>' : ""}</td>
        <td><strong>${m.rmse}</strong></td>
        <td>${m.mae}</td>
        <td><strong>${m.nasa_score}</strong></td>
        <td>${m.pct_early}% / ${m.pct_late}%</td>
        <td>+${m.max_late_cycles} cyc</td>
      `;
      tbody.appendChild(tr);
    });

    renderModelComparisonChart(modelNames, rmses, scores);
  } catch (err) {
    console.error("Model load error:", err);
  }
}

function renderModelComparisonChart(names, rmses, scores) {
  const ctx = document.getElementById("modelBenchmarkChart").getContext("2d");
  if (modelChartInstance) modelChartInstance.destroy();

  modelChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: names,
      datasets: [
        {
          label: "Test RMSE (Cycles, Lower is Better)",
          data: rmses,
          backgroundColor: "#0284c7",
          yAxisID: "y"
        },
        {
          label: "NASA Asymmetric Score (Lower is Better)",
          data: scores,
          backgroundColor: "#f59e0b",
          yAxisID: "y1"
        }
      ]
    },
    options: {
      responsive: true,
      scales: {
        x: { ticks: { color: "#f8fafc" }, grid: { color: "#334155" } },
        y: { type: "linear", position: "left", title: { display: true, text: "RMSE (Cycles)", color: "#0284c7" }, ticks: { color: "#94a3b8" }, grid: { color: "#334155" } },
        y1: { type: "linear", position: "right", title: { display: true, text: "NASA Asymmetric Score", color: "#f59e0b" }, ticks: { color: "#94a3b8" }, grid: { drawOnChartArea: false } }
      },
      plugins: {
        legend: { labels: { color: "#f8fafc" } }
      }
    }
  });
}
