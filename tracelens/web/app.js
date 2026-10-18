"use strict";
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const escapeHTML = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const fmt = (x, digits = 3) =>
  x == null
    ? "—"
    : Number(x).toLocaleString("en-US", {
        maximumFractionDigits: digits,
        minimumFractionDigits: digits,
      });
const pct = (x) => (x == null ? "—" : `${fmt(x * 100, 1)}%`);
const int = (x) => Number(x).toLocaleString("en-US");
const state = {
  report: null,
  model: null,
  timeline: null,
  threshold: 0,
  filter: "all",
  offset: 0,
  limit: 12,
  query: "",
  page: "overview",
  generation: 0,
  listGeneration: 0,
};

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let body;
    try {
      body = await response.json();
    } catch {
      body = { detail: `Request failed (${response.status})` };
    }
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : "The request could not be processed. Check your input.",
    );
  }
  return response.json();
}
function toast(message) {
  $("#toast").textContent = message;
  $("#toast").classList.remove("hidden");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => $("#toast").classList.add("hidden"), 3500);
}
function run() {
  return state.report.runs.find((r) => r.id === state.model);
}
function metricCard(label, value, caption, icon) {
  return `<div class="stat"><div class="stat-label">${label}<span class="stat-icon" aria-hidden="true">${icon}</span></div><div class="stat-value">${value}</div><div class="stat-caption">${caption}</div></div>`;
}
function compute() {
  let tp = 0,
    fp = 0,
    fn = 0,
    tn = 0;
  for (const p of state.timeline.points) {
    const flag = p.score > state.threshold;
    if (flag && p.label) tp++;
    else if (flag) fp++;
    else if (p.label) fn++;
    else tn++;
  }
  return {
    tp,
    fp,
    fn,
    tn,
    recall: tp / Math.max(tp + fn, 1),
    precision: tp / Math.max(tp + fp, 1),
    fpr: (fp / state.timeline.points.length) * 1000,
    flagged: tp + fp,
  };
}
function updateThreshold() {
  const q = Number($("#sensitivity").value) / 1000;
  const normals = state.timeline.normal_calibration;
  state.threshold = normals[Math.ceil((normals.length - 1) * q)];
  $("#quantile-value").textContent =
    `${fmt(q * 100, 1).replace(".0", "")}th percentile · score ${fmt(state.threshold, 3)}`;
  renderStats();
  renderSignal();
  state.offset = 0;
  loadTraces();
}
function renderStats() {
  const m = compute();
  $("#stats").innerHTML =
    metricCard(
      "Average precision",
      pct(run().metrics.average_precision),
      "Threshold-independent · held-out test",
      "↗",
    ) +
    metricCard(
      "Anomaly recall",
      pct(m.recall),
      `${int(m.tp)} of ${int(m.tp + m.fn)} labeled windows detected`,
      "⌁",
    ) +
    metricCard(
      "False alerts / 1k",
      fmt(m.fpr, 1),
      `${int(m.fp)} false positives · all test windows`,
      "⚑",
    ) +
    metricCard(
      "Flagged windows",
      int(m.flagged),
      `<span class="em">${pct(m.precision)} precision</span> at current threshold`,
      "≋",
    );
}
function renderSignal() {
  const points = state.timeline.points,
    w = Math.max(280, $("#signal-chart").clientWidth || 790),
    h = 205,
    left = 34,
    right = 10,
    top = 12,
    bottom = 27;
  const iw = w - left - right,
    ih = h - top - bottom,
    max =
      Math.max(state.threshold * 1.15, ...points.map((p) => p.score)) * 1.08;
  const x = (i) => left + (i / Math.max(points.length - 1, 1)) * iw,
    y = (v) => top + ih - (v / max) * ih;
  let grid = "";
  for (let i = 0; i <= 3; i++) {
    const v = (max * i) / 3;
    grid += `<line class="chart-gridline" x1="${left}" y1="${y(v)}" x2="${w - right}" y2="${y(v)}"/><text class="axis-label" x="${left - 8}" y="${y(v) + 3}" text-anchor="end">${fmt(v, 1)}</text>`;
  }
  let bands = "",
    segments = [];
  const bins = 140,
    bucket = Math.ceil(points.length / bins);
  for (let i = 0; i < points.length; i += bucket) {
    const chunk = points.slice(i, i + bucket);
    const v = Math.max(...chunk.map((p) => p.score));
    const px = x(i + Math.floor(chunk.length / 2));
    segments.push([px, y(v)]);
    if (chunk.some((p) => p.label))
      bands += `<rect x="${x(i)}" y="${top}" width="${Math.max(2, (chunk.length / points.length) * iw)}" height="${ih}" fill="#f5e9df" opacity=".75"/>`;
  }
  const line = segments
    .map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`)
    .join(" ");
  const area = `${line} L${segments.at(-1)[0]},${top + ih} L${segments[0][0]},${top + ih}Z`;
  let labels = "";
  for (let i = 0; i <= 4; i++) {
    const id = Math.round(((points.length - 1) * i) / 4);
    labels += `<text class="axis-label" x="${x(id)}" y="${h - 5}" text-anchor="${i === 0 ? "start" : i === 4 ? "end" : "middle"}">${int(id + 1)}</text>`;
  }
  $("#signal-chart").innerHTML =
    `<svg viewBox="0 0 ${w} ${h}" role="img" aria-labelledby="chart-title chart-desc"><title id="chart-title">Test stream anomaly scores</title><desc id="chart-desc">${int(points.length)} non-overlapping test windows. Green line: maximum surprise per plotted bin. Shaded bands: bins containing labeled anomalies. Threshold ${fmt(state.threshold)}. Use the sequence table to inspect individual windows.</desc><defs><linearGradient id="area" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#7ead91" stop-opacity=".25"/><stop offset="1" stop-color="#7ead91" stop-opacity=".01"/></linearGradient></defs>${bands}${grid}<path d="${area}" fill="url(#area)"/><line x1="${left}" x2="${w - right}" y1="${y(state.threshold)}" y2="${y(state.threshold)}" stroke="#ad865b" stroke-width="1.1" stroke-dasharray="5 4"/><path d="${line}" stroke="#548e74" stroke-width="1.4" fill="none" stroke-linejoin="round"/>${labels}</svg>`;
  $("#window-count").textContent = `${int(points.length)} WINDOWS`;
}
async function selectModel(id) {
  const generation = ++state.generation;
  state.model = id;
  state.offset = 0;
  ["#model-select", "#analysis-model-select", "#lab-model-select"].forEach(
    (selector) => ($(selector).value = id),
  );
  try {
    const timeline = await api(`/api/timeline?model=${encodeURIComponent(id)}`);
    if (generation !== state.generation) return;
    state.timeline = timeline;
    const r = run();
    $("#model-name").textContent = r.name;
    $("#model-description").textContent = r.description;
    $("#model-facts").innerHTML =
      `<div><dt>Parameters</dt><dd>${int(r.parameters)}</dd></div><div><dt>Checkpoint seed</dt><dd>${r.seed ?? "Deterministic"}</dd></div><div><dt>Context window</dt><dd>20 events</dd></div><div><dt>Training</dt><dd>${r.history.length ? r.history.length + " epochs" : "Normal-event counts"}</dd></div>`;
    updateThreshold();
    renderLearning();
  } catch (error) {
    toast(error.message);
  }
}
async function loadTraces() {
  const version = ++state.listGeneration;
  const params = new URLSearchParams({
    model: state.model,
    kind: state.filter,
    q: state.query,
    offset: state.offset,
    limit: state.limit,
    threshold: state.threshold,
  });
  $("#trace-rows").setAttribute("aria-busy", "true");
  try {
    const data = await api(`/api/traces?${params}`);
    if (version !== state.listGeneration) return;
    $("#trace-rows").innerHTML = data.items.length
      ? data.items
          .map((t) => {
            const alert = t.score > state.threshold;
            return `<tr data-id="${t.id}"><td><span class="trace-id">Window ${int(t.id + 1)}</span><span class="source-lines">L${int(t.start_line)}–${int(t.end_line)}</span></td><td><div class="trace-preview" title="${escapeHTML(t.preview)}">${escapeHTML(t.preview)}</div>${t.unknowns ? `<span class="live-chip">${t.unknowns} unknown events</span>` : ""}</td><td class="score-cell">${fmt(t.score)}</td><td><span class="badge ${alert ? "alert" : "normal"}">${alert ? "FLAGGED" : "BELOW THRESHOLD"}</span></td><td><span class="badge ${t.label ? "miss" : "neutral"}">${t.label ? "ANOMALOUS" : "NORMAL"}</span></td><td><button class="icon-button inspect-window" data-id="${t.id}" aria-label="Inspect window ${t.id + 1}">↗</button></td></tr>`;
          })
          .join("")
      : '<tr><td colspan="6" class="empty">No windows match this view. Try another filter or search.</td></tr>';
    $("#table-count").textContent = data.total
      ? `Showing ${int(data.offset + 1)}–${int(Math.min(data.offset + state.limit, data.total))} of ${int(data.total)} windows`
      : "0 matching windows";
    $("#prev-page").disabled = state.offset === 0;
    $("#next-page").disabled = state.offset + state.limit >= data.total;
  } catch (error) {
    $("#trace-rows").innerHTML =
      `<tr><td colspan="6" class="empty">${escapeHTML(error.message)}</td></tr>`;
  } finally {
    if (version === state.listGeneration)
      $("#trace-rows").removeAttribute("aria-busy");
  }
}
async function inspect(id) {
  $("#trace-title").textContent = `Window ${int(id + 1)}`;
  $("#trace-detail").innerHTML =
    '<p class="detail-note">Loading checkpoint evidence…</p>';
  $("#trace-dialog").showModal();
  try {
    const t = await api(
      `/api/traces/${id}?model=${encodeURIComponent(state.model)}`,
    );
    const score = t.scores[state.model],
      max = Math.max(...t.event_scores.filter((v) => v != null), 0.01);
    $("#trace-detail").innerHTML =
      `<div class="detail-summary"><span class="badge ${score > state.threshold ? "alert" : "normal"}">${score > state.threshold ? "MODEL FLAGGED" : "BELOW THRESHOLD"}</span><span class="badge ${t.label ? "miss" : "neutral"}">DATASET: ${t.label ? "ANOMALOUS" : "NORMAL"}</span><span class="mono">Score ${fmt(score)} · threshold ${fmt(state.threshold)}</span></div><p class="detail-note">Source lines ${int(t.start_line)}–${int(t.end_line)}. Each score is −log P(observed event | preceding events). The first event supplies context and is not scored. Event surprise shows where the model struggled; it does not establish root cause.</p>` +
      t.events
        .map(
          (e, i) =>
            `<div class="event-row ${t.event_scores[i] > max * 0.65 ? "high" : ""}"><span class="line-number">${int(e.line)}</span><div><div class="event-template">${escapeHTML(e.template)}</div><div class="event-meta">${!e.known ? '<span class="badge neutral">UNKNOWN TO TRAINING VOCABULARY</span>' : ""}${e.label ? '<span class="badge miss">DATASET ALERT LABEL</span>' : ""}</div></div><div class="event-score">${t.event_scores[i] == null ? "context" : fmt(t.event_scores[i], 2)}${t.event_scores[i] != null ? `<div class="event-bar"><i style="width:${(t.event_scores[i] / max) * 100}%"></i></div>` : ""}</div></div>`,
        )
        .join("");
  } catch (error) {
    $("#trace-detail").textContent = error.message;
  }
}
function renderModels() {
  $("#model-rows").innerHTML = state.report.runs
    .map(
      (r) =>
        `<tr><td>${escapeHTML(r.name)}${r.id === state.report.selected_model ? ' <span class="badge mint">DEFAULT</span>' : ""}</td><td>${r.seed ?? "—"}</td><td>${fmt(r.metrics.average_precision, 4)}</td><td>${pct(r.metrics.precision)}</td><td>${pct(r.metrics.recall)}</td><td>${fmt(r.metrics.f1)}</td><td>${fmt(r.metrics.false_alerts_per_1000_windows, 1)}</td><td>${int(r.parameters)}</td></tr>`,
    )
    .join("");
  const base = state.report.runs[0],
    lstm = state.report.runs.filter((r) => /^lstm-s\d+$/.test(r.id)),
    tcn = state.report.runs.filter((r) => /^tcn-s\d+$/.test(r.id));
  const mean = (rows) =>
    rows.reduce((s, r) => s + r.metrics.average_precision, 0) / rows.length;
  $("#model-summary").innerHTML =
    `<strong>The baseline gets a fair hearing.</strong><p>Test average precision: event frequency <strong>${fmt(base.metrics.average_precision, 4)}</strong>; LSTM seed mean <strong>${fmt(mean(lstm), 4)}</strong>; temporal CNN seed mean <strong>${fmt(mean(tcn), 4)}</strong>. Anomaly prevalence is ${pct(base.metrics.prevalence)}. These are one dataset prefix and one split, not evidence of universal superiority.</p><p>The default checkpoint was selected using normal-validation predictive loss. Average precision summarizes ranking; the operating-point metrics use the fixed normal-validation threshold.</p>`;
}
function renderLearning() {
  const history = run().history;
  if (!history.length) {
    $("#learning-chart").innerHTML =
      '<p class="empty">The count-based baseline has no gradient-based training curve.</p>';
    return;
  }
  const w = Math.max(280, $("#learning-chart").clientWidth - 46 || 520),
    h = 190,
    l = 36,
    r = 10,
    t = 12,
    b = 27,
    max =
      Math.max(...history.flatMap((p) => [p.train_loss, p.validation_loss])) *
      1.1;
  const x = (i) => l + (i / Math.max(history.length - 1, 1)) * (w - l - r),
    y = (v) => t + (1 - v / max) * (h - t - b);
  let grid = "";
  for (let i = 0; i <= 3; i++) {
    const v = (max * i) / 3;
    grid += `<line class="chart-gridline" x1="${l}" x2="${w - r}" y1="${y(v)}" y2="${y(v)}"/><text class="axis-label" x="${l - 7}" y="${y(v) + 3}" text-anchor="end">${fmt(v, 1)}</text>`;
  }
  const line = (key) =>
    history.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p[key])}`).join(" ");
  $("#learning-chart").innerHTML =
    `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Training and normal-validation cross-entropy by epoch for ${escapeHTML(run().name)}">${grid}<path d="${line("train_loss")}" fill="none" stroke="#548e74" stroke-width="2"/><path d="${line("validation_loss")}" fill="none" stroke="#8b78ac" stroke-width="2"/>${history.map((p, i) => `<text class="axis-label" x="${x(i)}" y="${h - 7}" text-anchor="middle">${p.epoch}</text>`).join("")}</svg><span class="live-chip">Epoch → · active model: ${escapeHTML(state.model)}</span>`;
}
function renderMethod() {
  const r = state.report,
    d = r.dataset;
  $("#method-content").innerHTML =
    `<div class="method-grid"><section class="panel"><div class="panel-top"><h2>Evaluation protocol</h2></div><ol class="method-list"><li><div><strong>Split before windowing.</strong><br>First ${int(d.lines)} source records. Contiguous 60/20/20 train, validation, and test partitions; non-overlapping ${d.window_size}-event windows.</div></li><li><div><strong>Keep the answer out of the input.</strong><br>Remove BGL alert category and metadata, including severity. Deterministic masking forms content templates. Learn the vocabulary on training records only.</div></li><li><div><strong>Learn normal behavior.</strong><br>Fit weights on normal training windows. Select each epoch checkpoint by cross-entropy on normal validation windows.</div></li><li><div><strong>Calibrate once, evaluate honestly.</strong><br>Threshold at the 99th percentile of normal-validation scores. Evaluate unadjusted test windows using strict score &gt; threshold. The overview slider is exploratory.</div></li></ol></section><section class="panel"><div class="panel-top"><h2>Data boundaries</h2></div><dl class="protocol-dl">${Object.entries(
      d.splits,
    )
      .map(
        ([k, v]) =>
          `<div><dt>${escapeHTML(k)} records</dt><dd>L${int(v.first_line)}–L${int(v.last_line)}</dd></div><div><dt>${escapeHTML(k)} anomalous windows</dt><dd>${int(v.anomalous_windows)} / ${int(v.windows)}</dd></div><div><dt>${escapeHTML(k)} unknown events</dt><dd>${pct(v.unknown_event_fraction)}</dd></div>`,
      )
      .join(
        "",
      )}<div><dt>Vocabulary</dt><dd>${int(d.vocab_size)} event identities</dd></div></dl></section><section class="panel"><div class="panel-top"><h2>Limits worth making visible</h2></div><div class="prose"><p>BGL labels indicate alert messages. A window is positive if any of its events has an alert label. This is retrospective detection, not proof of future failure prediction.</p><p>File-order windows can mix messages from different nodes. The first-million-record experiment is a bounded subset with time-dependent class balance. Results do not establish transfer to modern services.</p><p>Unknown templates can dominate scores. The parser is a deterministic masking baseline, not a production log parser. Ground truth is used to filter normal training/calibration windows: the fitting protocol is semi-supervised.</p><p>Per-run AP intervals in the exported JSON use a 50-window moving-block bootstrap (200 resamples). These are descriptive and cannot remove dataset-selection bias.</p></div></section><section class="panel"><div class="panel-top"><h2>D802 → demonstrated work</h2></div><div class="prose"><p><strong>Neural architectures:</strong> learned embeddings, LSTM gates and temporal context, residual causal convolutions, GELU, and dropout.</p><p><strong>Model analysis:</strong> precision–recall metrics, a count-based baseline, confusion matrices, seed variability, unknown-event rates, and a sequence-order ablation.</p><p><strong>Optimization:</strong> AdamW, weight decay, gradient clipping, minibatches, validation-based checkpoint selection, and recorded training dynamics.</p><p>Course mapping follows the supplied course overview. No assessment rubric was supplied.</p></div></section><section class="panel wide"><div class="panel-top"><h2>Provenance & reproducibility</h2></div><div class="prose"><p>Original data: Loghub / Blue Gene/L. Cite Oliner & Stearley (2007) and Zhu et al. (2023). Dataset terms are separate from the project's code license.</p><p>Parser: <span class="mono">${escapeHTML(d.parser)}</span> · Python ${escapeHTML(r.environment.python)} · PyTorch ${escapeHTML(r.environment.torch)} · CPU · ${r.environment.threads} threads</p><p class="hash">Archive SHA-256: ${escapeHTML(d.sha256)}</p><div class="method-links"><a href="https://github.com/logpai/loghub/tree/master/BGL" target="_blank" rel="noopener">Dataset documentation ↗</a><a href="/api/report" target="_blank" rel="noopener">Full experiment JSON ↗</a></div></div></section></div>`;
}
function navigate() {
  const page = location.hash.slice(1) || "overview";
  state.page = ["overview", "models", "analyze", "method"].includes(page)
    ? page
    : "overview";
  $$(".page").forEach((p) =>
    p.classList.toggle("hidden", p.id !== `page-${state.page}`),
  );
  $$("nav a").forEach((a) => {
    a.classList.toggle("active", a.dataset.page === state.page);
    if (a.dataset.page === state.page) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  $("#breadcrumb").textContent = {
    overview: "Signal overview",
    models: "Model lab",
    analyze: "Analyze logs",
    method: "Method & evidence",
  }[state.page];
  document.title = `TraceLens / ${$("#breadcrumb").textContent}`;
  if (state.timeline && state.page === "overview")
    requestAnimationFrame(renderSignal);
  if (state.timeline && state.page === "models")
    requestAnimationFrame(renderLearning);
}
async function analyze() {
  const button = $("#analyze-button"),
    text = $("#log-input").value;
  $("#analyze-status").textContent = "";
  $("#analyze-results").innerHTML = "";
  button.disabled = true;
  button.textContent = "Analyzing…";
  try {
    const data = await api("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, model: state.model }),
    });
    $("#analyze-results").innerHTML =
      `<section class="panel"><div class="panel-top"><div><h2>Inference results</h2><p>${int(data.analyzed_lines)} lines · ${data.windows.length} windows · ${data.discarded_tail_lines} trailing lines unused · ${pct(data.unknown_fraction)} unknown events</p></div><span class="badge neutral">${escapeHTML(data.model)}</span></div><div class="prose"><p>${escapeHTML(data.note)}</p></div>${data.windows.map((w) => `<details class="analysis-window"><summary><strong>Window ${w.id + 1}</strong><span class="badge ${w.alert ? "alert" : "normal"}">${w.alert ? "FLAGGED" : "BELOW THRESHOLD"}</span><span class="mono">score ${fmt(w.score)}</span></summary><pre>${w.events.map((e) => `${String(e.line).padStart(4)}  ${escapeHTML(e.template)}${!e.known ? "  [UNKNOWN]" : ""}`).join("\n")}</pre></details>`).join("")}</section>`;
    $("#analyze-status").textContent = "Analysis complete.";
    $("#analyze-status").style.color = "var(--green)";
  } catch (error) {
    $("#analyze-status").textContent = error.message;
    $("#analyze-status").style.color = "var(--red)";
  } finally {
    button.disabled = false;
    button.textContent = "Analyze sequence →";
  }
}
function lineCount() {
  $("#input-count").textContent = `${int(
    $("#log-input")
      .value.split(/\r?\n/)
      .filter((l) => l.trim()).length,
  )} lines`;
}

async function init() {
  try {
    state.report = await api("/api/report");
    state.model = state.report.selected_model;
    const options = state.report.runs
      .map(
        (r) =>
          `<option value="${escapeHTML(r.id)}">${escapeHTML(r.name)}${r.seed ? " · seed " + r.seed : ""}</option>`,
      )
      .join("");
    ["#model-select", "#analysis-model-select", "#lab-model-select"].forEach(
      (selector) => ($(selector).innerHTML = options),
    );
    $("#dataset-meta").textContent =
      `${int(state.report.dataset.lines)} records / 60–20–20 split`;
    renderModels();
    renderMethod();
    await selectModel(state.model);
    $("#loading").classList.add("hidden");
    navigate();
  } catch (error) {
    $("#loading").classList.add("hidden");
    $("#fatal").classList.remove("hidden");
    $("#fatal").innerHTML =
      `<h1>Workspace could not load.</h1><p>${escapeHTML(error.message)}</p><p>Check the local server and evaluation artifacts, then reload.</p>`;
    return;
  }
  ["#model-select", "#analysis-model-select", "#lab-model-select"].forEach(
    (selector) =>
      $(selector).addEventListener("change", (e) =>
        selectModel(e.target.value),
      ),
  );
  $("#sensitivity").addEventListener("input", updateThreshold);
  $$(".tabs button").forEach((button) =>
    button.addEventListener("click", () => {
      state.filter = button.dataset.filter;
      state.offset = 0;
      $$(".tabs button").forEach((b) => {
        b.classList.toggle("active", b === button);
        b.setAttribute("aria-pressed", String(b === button));
      });
      loadTraces();
    }),
  );
  let searchTimer;
  $("#search").addEventListener("input", (e) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      state.query = e.target.value;
      state.offset = 0;
      loadTraces();
    }, 180);
  });
  $("#next-page").addEventListener("click", () => {
    state.offset += state.limit;
    loadTraces();
  });
  $("#prev-page").addEventListener("click", () => {
    state.offset = Math.max(0, state.offset - state.limit);
    loadTraces();
  });
  $("#trace-rows").addEventListener("click", (e) => {
    const row = e.target.closest("[data-id]");
    if (row) inspect(Number(row.dataset.id));
  });
  $("#close-dialog").addEventListener("click", () =>
    $("#trace-dialog").close(),
  );
  $("#trace-dialog").addEventListener("click", (e) => {
    if (e.target === $("#trace-dialog")) {
      const rect = e.target.getBoundingClientRect();
      if (
        e.clientX < rect.left ||
        e.clientX > rect.right ||
        e.clientY < rect.top ||
        e.clientY > rect.bottom
      )
        e.target.close();
    }
  });
  window.addEventListener("hashchange", navigate);
  window.addEventListener("resize", () => {
    if (!state.timeline) return;
    if (state.page === "overview") renderSignal();
    if (state.page === "models") renderLearning();
  });
  $("#export-report").addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(state.report, null, 2)], {
        type: "application/json",
      }),
      url = URL.createObjectURL(blob),
      a = document.createElement("a");
    a.href = url;
    a.download = "tracelens-experiment-report.json";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast("Experiment report exported.");
  });
  $("#log-input").addEventListener("input", lineCount);
  $("#analyze-button").addEventListener("click", analyze);
  $("#load-sample").addEventListener("click", async () => {
    try {
      const id = state.timeline.points.find((p) => p.label)?.id ?? 0;
      const t = await api(
        `/api/traces/${id}?model=${encodeURIComponent(state.model)}`,
      );
      $("#log-input").value = t.events.map((e) => e.template).join("\n");
      lineCount();
      toast("Loaded normalized messages from the held-out test set.");
    } catch (error) {
      toast(error.message);
    }
  });
  $("#log-file").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (file.size > 500000) {
      toast("Choose a file smaller than 500 KB.");
      return;
    }
    $("#log-input").value = await file.text();
    lineCount();
  });
}
init();
