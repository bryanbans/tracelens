"use strict";
const $ = (s) => document.querySelector(s),
  $$ = (s) => [...document.querySelectorAll(s)];
const esc = (x) =>
  String(x ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const number = (x, d = 0) =>
  Number(x).toLocaleString("en-US", {
    minimumFractionDigits: d,
    maximumFractionDigits: d,
  });
const percent = (x, d = 1) => `${number(x * 100, d)}%`;
const S = {
  report: null,
  model: "calibrated",
  split: "all",
  timeline: null,
  points: [],
  filter: "alerts",
  search: "",
  offset: 0,
  limit: 8,
  selected: null,
  generation: 0,
  listGeneration: 0,
  detailGeneration: 0,
  page: "overview",
};
async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let e;
    try {
      e = await response.json();
    } catch {
      e = { detail: `Request failed (${response.status})` };
    }
    throw Error(
      typeof e.detail === "string"
        ? e.detail
        : "Check the request and try again.",
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
function entry(id = S.model) {
  return S.report.runs.find((r) => r.id === id);
}
function modelName(id = S.model) {
  return {calibrated: 'Word CNN ensemble', recalibrated: 'LSTM with a new threshold', legacy: 'Original LSTM'}[id] ?? id;
}
function baselineMetrics() {
  return S.split === "all"
    ? entry("legacy").metrics
    : S.report.audits[S.split].metrics.legacy;
}
function metrics() {
  let tp = 0,
    fp = 0,
    tn = 0,
    fn = 0;
  for (const p of S.points) {
    const flag = p.score > S.timeline.threshold;
    if (flag && p.label) tp++;
    else if (flag) fp++;
    else if (p.label) fn++;
    else tn++;
  }
  return {
    tp,
    fp,
    tn,
    fn,
    recall: tp / Math.max(tp + fn, 1),
    precision: tp / Math.max(tp + fp, 1),
    fpr: fp / Math.max(fp + tn, 1),
  };
}
function renderMetrics() {
  const m = metrics(),
    old = baselineMetrics();
  $("#hero-recall").innerHTML = `${number(m.recall * 100, 1)}<small>%</small>`;
  $("#hero-precision").textContent = percent(m.precision);
  $("#hero-fpr").textContent = percent(m.fpr, 2);
  $("#drift-warning").textContent = m.fpr > S.report.config.normal_fpr_budget
    ? `FALSE-ALARM TARGET MISSED / ${percent(m.fpr,2)} of normal windows were flagged in this range, above the ${percent(S.report.config.normal_fpr_budget,0)} calibration target. The threshold was fixed before evaluation.`
    : `FIXED-THRESHOLD RESULT / ${percent(m.fpr,2)} of normal windows were flagged in this range. This observed rate does not guarantee future performance.`;
  $('#outcome-summary').textContent = `${modelName()} caught ${number(m.tp)} of ${number(m.tp + m.fn)} labeled anomalous windows, missed ${number(m.fn)}, and flagged ${number(m.fp)} of ${number(m.fp + m.tn)} normal windows in this range.`;
  const delta = (m.recall - old.recall) * 100;
  $("#hero-comparison").innerHTML =
    S.model === "legacy"
      ? "Original LSTM · same evaluation windows"
      : `<strong>${delta >= 0 ? "+" : ""}${number(delta, 1)} percentage points</strong><br>recall vs. the original LSTM on the same range`;
  $("#signal-summary").textContent =
    `${number(S.points.length)} non-overlapping windows · ${number(m.tp + m.fn)} labeled anomalous · ${modelName()}`;
  $("#threshold-label").textContent =
    `THRESHOLD ${number(S.timeline.threshold, 4)} / ${entry().model_type === "semantic" ? "ACTIVATION" : "SURPRISE"}`;
  $("#source-range").textContent = S.points.length
    ? `LINES ${number(S.points[0].line)}—${number(S.points.at(-1).line + 19)}`
    : "";
}
function plot() {
  if (!S.points.length) return;
  const points = S.points,
    w = Math.max(300, $("#signal").clientWidth - 40),
    h = 235,
    l = 37,
    r = 12,
    t = 12,
    b = 28,
    iw = w - l - r,
    ih = h - t - b;
  let max = S.timeline.threshold;
  for (const p of points) max = Math.max(max, p.score);
  max = Math.max(max * 1.1, 0.01);
  const x = (i) => l + (i / Math.max(points.length - 1, 1)) * iw,
    y = (v) => t + ih - (v / max) * ih;
  let grid = "",
    bands = "",
    dots = [];
  for (let i = 0; i < 4; i++) {
    const v = (max * i) / 3;
    grid += `<line class="plot-grid" x1="${l}" x2="${w - r}" y1="${y(v)}" y2="${y(v)}"/><text class="plot-label" x="${l - 8}" y="${y(v) + 3}" text-anchor="end">${number(v, max < 2 ? 2 : 1)}</text>`;
  }
  const bucket = Math.ceil(points.length / Math.min(550, iw));
  for (let i = 0; i < points.length; i += bucket) {
    const chunk = points.slice(i, i + bucket),
      peak = chunk.reduce((a, p) => (p.score > a.score ? p : a), chunk[0]);
    dots.push([x(i + chunk.length / 2), y(peak.score)]);
    if (chunk.some((p) => p.label))
      bands += `<rect x="${x(i)}" y="${t}" width="${Math.max(1, (chunk.length / points.length) * iw)}" height="${ih}" fill="#71444a" opacity=".27"/>`;
  }
  const path = dots
    .map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(2)},${p[1].toFixed(2)}`)
    .join(" ");
  const area = `${path}L${dots.at(-1)[0]},${t + ih}L${dots[0][0]},${t + ih}Z`;
  let labels = "";
  for (let i = 0; i <= 4; i++) {
    const j = Math.round(((points.length - 1) * i) / 4);
    labels += `<text class="plot-label" x="${x(j)}" y="${h - 5}" text-anchor="${i === 0 ? "start" : i === 4 ? "end" : "middle"}">${number(points[j].line / 1e6, 2)}m</text>`;
  }
  $("#signal").innerHTML =
    `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${number(points.length)} evaluation windows. Cyan score line, orange fixed threshold, shaded labeled anomalies. Each bin retains its maximum score; use the incident list for exact windows."><defs><linearGradient id="fill" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#72dfcf" stop-opacity=".18"/><stop offset="1" stop-color="#72dfcf" stop-opacity="0"/></linearGradient></defs>${bands}${grid}<path d="${area}" fill="url(#fill)"/><line x1="${l}" x2="${w - r}" y1="${y(S.timeline.threshold)}" y2="${y(S.timeline.threshold)}" stroke="#ffa882" stroke-dasharray="5 5" stroke-width="1.2"/><path d="${path}" fill="none" stroke="#72dfcf" stroke-width="1.25"/>${labels}</svg>`;
}
async function modelChange(id, initial = false) {
  const g = ++S.generation;
  S.model = id;
  S.offset = 0;
  S.selected = null;
  ++S.detailGeneration;
  $("#audit-range").disabled = true;
  $("#signal-summary").textContent = "Loading saved evaluation scores…";
  $("#inspector").innerHTML = '<div class="empty"><h3>Loading messages and model scores…</h3></div>';
  $$("[data-model]").forEach((b) => {
    b.classList.toggle("active", b.dataset.model === id);
    b.setAttribute("aria-pressed", String(b.dataset.model === id));
  });
  try {
    const timeline = await api(`/api/timeline?model=${id}`);
    if (g !== S.generation) return;
    S.timeline = timeline;
    S.points = S.timeline.points.filter(
      (p) => S.split === "all" || p.split === S.split,
    );
    renderMetrics();
    requestAnimationFrame(plot);
    await list();
  } catch (e) {
    if (initial) throw e;
    $('#signal-summary').textContent = 'Detector could not load. Select a detector to retry.';
    toast(e.message);
  } finally {
    if (g === S.generation) $("#audit-range").disabled = false;
  }
}
async function list() {
  const version = ++S.listGeneration;
  ++S.detailGeneration;
  $('#inspector').removeAttribute('aria-busy');
  const params = new URLSearchParams({
    model: S.model,
    split: S.split,
    kind: S.filter,
    q: S.search,
    offset: S.offset,
    limit: S.limit,
  });
  $("#incidents").setAttribute("aria-busy", "true");
  try {
    const data = await api(`/api/traces?${params}`);
    if (version !== S.listGeneration) return;
    $("#count").textContent = number(data.total);
    $("#incidents").innerHTML = data.items.length
      ? data.items
          .map(
            (t) =>
              `<button class="incident ${t.id === S.selected ? "selected" : ""}" data-id="${t.id}" aria-label="Inspect window at line ${t.start_line}"><div class="incident-top"><span>L${number(t.start_line)}</span><span class="incident-score">${number(t.score, 3)}</span></div><div class="incident-text">${esc(t.preview)}</div><div class="incident-bottom"><span><i class="dot"></i>${t.label ? "LABEL: ANOMALOUS" : "LABEL: NORMAL"}</span><span>${esc(t.split.replace("_", " ").toUpperCase())}</span></div></button>`,
          )
          .join("")
      : '<div class="empty"><h3>No matching windows.</h3><p>Change the filter or search.</p></div>';
    $("#page-count").textContent = data.total
      ? `${number(S.offset + 1)}—${number(Math.min(S.offset + S.limit, data.total))} / ${number(data.total)}`
      : "0 windows";
    $("#previous").disabled = S.offset === 0;
    $("#next").disabled = S.offset + S.limit >= data.total;
    if (data.items.length)
      inspect(data.items.some(t => t.id === S.selected) ? S.selected : data.items[0].id);
    if (!data.items.length)
      $("#inspector").innerHTML =
        '<div class="empty"><h3>No evidence to inspect.</h3><p>Choose another filter or evaluation range.</p></div>';
  } catch (e) {
    toast(e.message);
  } finally {
    if (version === S.listGeneration)
      $("#incidents").removeAttribute("aria-busy");
  }
}
async function inspect(id) {
  const generation = ++S.detailGeneration;
  S.selected = id;
  $$(".incident").forEach((b) =>
    b.classList.toggle("selected", Number(b.dataset.id) === id),
  );
  $("#inspector").setAttribute("aria-busy", "true");
  try {
    const model = S.model,
      t = await api(`/api/traces/${id}?model=${model}`);
    if (generation !== S.detailGeneration || model !== S.model) return;
    const score = t.scores[model],
      semantic = entry(model).model_type === "semantic",
      values = t.event_scores,
      max = Math.max(...values.filter((v) => v != null), 0.001);
    $("#inspector").innerHTML =
      `<div class="detail-top"><div class="section-caption">WINDOW ${number(id + 1)} / ${esc(t.split ?? "TEST")}</div><h3>Lines ${number(t.start_line)}—${number(t.end_line)}</h3><div class="detail-tags"><span class="tag ${score > S.timeline.threshold ? "hot" : "cool"}">${score > S.timeline.threshold ? "MODEL FLAGGED" : "BELOW THRESHOLD"}</span><span class="tag truth">LABEL: ${t.label ? "ANOMALOUS" : "NORMAL"}</span><span class="tag">SCORE ${number(score, 4)}</span></div><p class="detail-note">${semantic ? "Each number is a message activation: a stronger response to a learned pattern. Three CNNs average their activations for each message; the highest average is the window score. Message order across the window is not modeled." : "Each number measures how unexpected a message is given earlier event identities. The window score averages 19 next-event surprise values. The first message provides context and has no score."} Dataset labels are shown separately to check the decision. Scores do not identify a root cause or estimate outage probability. Message text is normalized, so numbers and identifiers may appear as placeholders.</p></div>` +
      t.events
        .map(
          (e, i) =>
            `<div class="event ${values[i] > max * 0.65 ? "high" : ""}"><span class="ln">${number(e.line)}</span><div><div class="event-code">${esc(e.template)}</div><div class="event-meta">${e.label ? "DATASET: ABNORMAL MESSAGE" : ""}${!e.known ? (e.label ? " · " : "") + "UNKNOWN TO LSTM EVENT VOCABULARY" : ""}</div></div><div class="activation">${values[i] == null ? "—" : number(values[i], 3)}${values[i] != null ? `<div class="activation-bar"><i style="width:${Math.max(0, values[i] / max) * 100}%"></i></div>` : ""}</div></div>`,
        )
        .join("");
    $("#inspector").scrollTop = 0;
  } catch (e) {
    if (generation !== S.detailGeneration) return;
    $("#inspector").innerHTML = `<div class="empty">${esc(e.message)}</div>`;
  } finally {
    if (generation === S.detailGeneration)
      $("#inspector").removeAttribute("aria-busy");
  }
}
function renderValidation() {
  const r = S.report,
    rows = r.runs;
  const old = entry("legacy").metrics,
    now = entry("calibrated").metrics;
  $("#validation-content").innerHTML =
    `<div class="report-intro">These models are compared on the same 112,398 later log windows. Read recall together with precision and false positives: catching more anomalies can also create more alerts to review. <strong>The word CNN exceeded its 1% normal false-positive target in every evaluation range.</strong></div><div class="report-heading"><h2>What changes when the model or threshold changes?</h2><span>SAME WINDOWS / FIXED THRESHOLDS</span></div><div class="table-wrap"><table><thead><tr><th>Detector</th><th>Recall</th><th>Precision</th><th>Normal false-positive rate</th><th>Average precision</th><th>False alerts / 1,000 windows</th></tr></thead><tbody>${rows.map((e) => `<tr class="${e.id === "calibrated" ? "emphasis" : ""}"><td>${esc(modelName(e.id))}</td><td>${percent(e.metrics.recall)}</td><td>${percent(e.metrics.precision)}</td><td>${percent(e.metrics.normal_false_positive_rate, 2)}</td><td>${number(e.metrics.average_precision, 4)}</td><td>${number(e.metrics.false_alerts_per_1000_windows, 2)}</td></tr>`).join("")}</tbody></table></div><div class="prose metric-definitions"><p><strong>Recall:</strong> anomalous windows flagged / all anomalous windows. <strong>Precision:</strong> anomalous windows flagged / all flagged windows. <strong>Normal false-positive rate:</strong> normal windows flagged / all normal windows.</p><p><strong>Average precision:</strong> summarizes precision as recall increases across score thresholds; compare it on the same data. <strong>False alerts / 1,000 windows:</strong> normal windows flagged per 1,000 total windows, including both classes. Each 20-message window is counted once; detections do not give credit to neighboring windows.</p></div><div class="report-heading"><h2>Does the CNN keep working on later logs?</h2><span>WORD CNN VS. ORIGINAL LSTM / THREE TIME RANGES</span></div><div class="audit-grid">${Object.entries(
      r.audits,
    )
      .map(([name, a]) => {
        const m = a.metrics.calibrated,
          prior = a.metrics.legacy,
          ci = a.paired_intervals;
        return `<article class="audit-block"><span class="section-caption">${name.replace("_", " ").toUpperCase()}</span><div class="audit-number">${percent(m.recall)}</div><p>CNN recall · original LSTM ${percent(prior.recall)}<br>Precision ${percent(m.precision)} · normal false-positive rate ${percent(m.normal_false_positive_rate, 2)}</p><p class="interval">Recall gain · 95% bootstrap interval:<br>${ci ? ci.recall_difference_95.map((v) => `${v >= 0 ? "+" : ""}${number(v * 100, 1)} pp`).join(" to ") : "unavailable"}<br>${number(a.windows)} windows · ${number(a.anomalous_windows)} labeled anomalous</p></article>`;
      })
      .join(
        "",
      )}</div><div class="confirm-callout"><strong>What each comparison means.</strong> The two LSTM rows use the same saved model with different thresholds. The word CNN also changes the input representation, training objective, supervision, and model ensemble. It learns from labeled windows; the LSTM learns from normal windows only. A similarly supervised simple text classifier is still needed to isolate the benefit of deep learning.</div><div class="prose"><p><strong>How uncertainty was estimated.</strong> Paired moving-block bootstrap resampling compares models on the same windows in 250-window blocks, with 500 resamples. The intervals describe recall differences conditional on this data and procedure. Blocks preserve nearby dependence but cannot account for all longer-range correlation or dataset bias. “pp” means percentage points. These results concern abnormal messages in one historical system; they do not establish advance warning of outages.</p><p><strong>Evaluation history.</strong> Coarse label counts across the full archive were inspected earlier. Later predictions were not used for model or threshold selection. These are temporal audits of one dataset, not independent external replication. All three ranges have now been inspected; further tuning needs new evaluation data.</p><details><summary>Technical selection and provenance details</summary><p>${esc(r.disclosure)}</p></details><p>Development selected <strong>${esc(r.selection.id)}</strong>. Final threshold: <strong>${number(entry("calibrated").metrics.threshold, 6)}</strong>. The original threshold and all audit predictions remain recorded in the exported report.</p></div>`;
  $("#finding").innerHTML =
    `<span>WHAT THIS COMPARISON SHOWS</span><p><strong>Recall changed from ${percent(old.recall)} to ${percent(now.recall)} on the same combined audit windows.</strong> Of its flagged windows, ${percent(now.precision)} were labeled anomalous. It also flagged ${percent(now.normal_false_positive_rate, 2)} of normal windows. More anomalies were found, with more false alarms to review.</p>`;
}
function renderProtocol() {
  const r = S.report;
  const stages = [
    [
      "01",
      "Train the models",
      "Sequence models retain their original checkpoints and vocabulary. Word CNNs learn content on lines 1–600,000, with epoch selection on 600,001–1,000,000.",
    ],
    [
      "02",
      "Development",
      "Fit candidate thresholds on lines 1,000,001–1,500,000. Choose the model, scoring rule, and threshold policy on 1,500,001–2,000,000, using F2 (which emphasizes recall) subject to a 1% normal false-positive constraint.",
    ],
    [
      "03",
      "Set the final threshold",
      "Fit the chosen threshold policy on lines 2,000,001–2,500,000. Fix weights, mappings, scoring rules, threshold, and evaluation boundaries before opening later predictions. The original checksum record has a mapping omission disclosed below.",
    ],
    [
      "04",
      "Evaluate on later logs",
      "Score 2,500,001–3,250,000 and 3,250,001–4,000,000 without adapting thresholds.",
    ],
    [
      "05",
      "Confirmation",
      "Score 4,000,001–4,747,963 with the same frozen detector. Discard three incomplete trailing events, and report that exclusion.",
    ],
    [
      "06",
      "Compare the same windows",
      "Count detections, misses, and false alarms for each model on identical windows. Report recall, precision, normal false-positive rate, average precision, confusion matrices, and paired moving-block intervals.",
    ],
  ];
  $("#protocol-content").innerHTML =
    `<div class="protocol-steps">${stages.map(([n, title, text]) => `<article class="protocol-step"><span>${n} /</span><h3>${title}</h3><p>${text}</p></article>`).join("")}</div><div class="report-heading"><h2>What do the models learn from?</h2><span>CONTENT ONLY / SOURCE LABEL STRIPPED</span></div><div class="prose"><p>The original LSTM predicts event identities. An unfamiliar template maps to one unknown event. The word CNN instead embeds content words, applies convolutional phrase filters and max pooling, and learns from window-level alert labels. Three seeded checkpoints average their per-event activations; the maximum event activation becomes the window score.</p><p>Only message content enters either model. Source alert tags, severity, timestamps, and node metadata are excluded. A new event template can still contain words the CNN learned during training. The first 16 word tokens per event are retained; this truncation is a documented limitation.</p><p><strong>D802 evidence:</strong> embeddings, LSTM state, causal convolutions, phrase convolutions, GELU, max pooling, backpropagation, dropout, AdamW, clipped gradients, held-out checkpoint selection, controlled comparisons, and calibration.</p><p><strong>Evaluation history:</strong> full-archive label counts were inspected earlier, so these ranges were prediction-blind rather than pristine label-blind tests. They are now inspected and cannot be reused as untouched evaluation data for further tuning.</p><details><summary>Read the recorded development disclosure</summary><p>${esc(r.disclosure)}</p></details></div><div class="report-heading"><h2>Inspect the evaluation record.</h2></div><div class="prose"><p>The evaluation plan was recorded at ${esc(r.frozen_at)}. Checkpoints and source checksums bind the recorded procedure to specific files. The audit command refuses to overwrite an existing report. The evaluated weights, scores, and thresholds are preserved.</p><p><strong>Known provenance gap:</strong> the first CNN freeze omitted an independent checksum for the word vocabulary. The retained mapping matches the training copy and is covered by release checksums, but this cannot prove the missing pre-evaluation commitment. Original files and source snapshots are preserved; future freezes require that checksum.</p></div><p class="hash">EVALUATION PLAN SHA-256<br>${esc(r.freeze_sha256)}<br><br>ORIGINAL ARCHIVE SHA-256<br>${esc(r.dataset.sha256)}</p><div class="source-links"><a href="https://github.com/logpai/loghub/tree/master/BGL" target="_blank" rel="noopener">BGL data and attribution ↗</a><a href="/api/report" target="_blank" rel="noopener">Full evaluation JSON ↗</a><a href="/api/docs" target="_blank" rel="noopener">Local API reference ↗</a></div>`;
}
function navigate() {
  const page = location.hash.slice(1) || "overview";
  S.page = ["overview", "validation", "replay", "protocol"].includes(page)
    ? page
    : "overview";
  $$(".page").forEach((e) =>
    e.classList.toggle("hidden", e.id !== `page-${S.page}`),
  );
  $$("nav a").forEach((a) => {
    a.classList.toggle("active", a.dataset.page === S.page);
    if (a.dataset.page === S.page) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  document.title = `TraceLens / ${({overview: "Explore logs", validation: "Compare models", replay: "Try a sample", protocol: "Method"})[S.page]}`;
  if (S.page === "overview" && S.timeline) requestAnimationFrame(plot);
}
function countLines() {
  $("#line-count").textContent = `${number(
    $("#log-input")
      .value.split(/\r?\n/)
      .filter((x) => x.trim()).length,
  )} lines`;
}
async function analyze() {
  const button = $("#analyze");
  button.disabled = true;
  button.textContent = "Scoring…";
  $("#analysis-status").textContent = "Running the saved models locally…";
  $("#analysis-results").innerHTML = "";
  try {
    const data = await api("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: $("#replay-model").value,
        text: $("#log-input").value,
      }),
    });
    $("#analysis-status").textContent =
      `Complete. ${number(data.analyzed_lines)} nonempty lines scored; ${data.discarded_tail_lines} trailing lines unused; ${data.ignored_blank_lines} blank lines ignored. Original line numbers retained.`;
    $("#analysis-results").innerHTML =
      `<div class="report-heading"><h2>How did the model respond?</h2><span>NO VERIFIED LABELS / ACCURACY CANNOT BE MEASURED HERE</span></div>` +
      `<p class="replay-disclosure">A flag means the sample score exceeds the saved threshold; it is not a verified anomaly. Unsupported windows receive no decision.<br>Messages absent from the LSTM event vocabulary: ${percent(data.legacy_template_unknown_fraction)}.</p>` +
      data.windows
        .map(
          (w) =>
            `<details class="replay-result"><summary><span>Window ${w.id + 1}</span><span class="tag ${w.alert === null ? "truth" : w.alert ? "hot" : "cool"}">${w.alert === null ? "NO DECISION" : w.alert ? "FLAGGED" : "BELOW THRESHOLD"}</span><span class="micro">SCORE ${number(w.score, 4)}</span></summary><p class="replay-disclosure">${w.coverage.status === 'unscorable' ? 'No decision: this window has insufficient recognizable input for the selected model. The raw score is shown only for debugging.' : 'Input domain unverified: this threshold has only been evaluated on BGL logs.'} ${w.coverage.known_word_fraction == null ? '' : `Retained words found in the training vocabulary: ${percent(w.coverage.known_word_fraction)}.`} ${w.coverage.truncated_events} messages shortened to their first 16 word tokens; ${w.coverage.empty_token_events} messages without usable word tokens. ${entry(data.model).model_type === "semantic" ? "Each line shows its source line number, normalized message, and CNN activation. A larger activation is a stronger model response, not an outage probability." : "Each line shows its source line number, normalized message, and next-event surprise. CONTEXT marks the first message, which has no prediction score."}</p><pre>${w.events.map((e) => `${String(e.line).padStart(3)}  ${esc(e.template)}  ${e.surprise == null ? "CONTEXT" : number(e.surprise, 3)}`).join("\n")}</pre></details>`,
        )
        .join("");
  } catch (e) {
    $("#analysis-status").textContent = e.message;
  } finally {
    button.disabled = false;
    button.textContent = "Score messages →";
  }
}
async function init() {
  try {
    S.report = await api("/api/report");
    if (S.report.schema_version !== 2)
      throw Error(
        "Start the app with artifacts/validated to explore the current models and evaluation.",
      );
    renderValidation();
    renderProtocol();
    await modelChange("calibrated", true);
    $("#loading").classList.add("hidden");
    navigate();
  } catch (e) {
    $("#loading").classList.add("hidden");
    $("#error").classList.remove("hidden");
    $("#error").textContent = e.message;
    return;
  }
  $$("[data-model]").forEach((b) =>
    b.addEventListener("click", () => modelChange(b.dataset.model)),
  );
  $("#audit-range").addEventListener("change", (e) => {
    S.split = e.target.value;
    S.offset = 0;
    S.points = S.timeline.points.filter(
      (p) => S.split === "all" || p.split === S.split,
    );
    renderMetrics();
    plot();
    list();
  });
  $("#filter").addEventListener("change", (e) => {
    S.filter = e.target.value;
    S.offset = 0;
    list();
  });
  let searchTimer;
  $("#search").addEventListener("input", (e) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      S.search = e.target.value;
      S.offset = 0;
      list();
    }, 180);
  });
  $("#previous").addEventListener("click", () => {
    S.offset = Math.max(0, S.offset - S.limit);
    list();
  });
  $("#next").addEventListener("click", () => {
    S.offset += S.limit;
    list();
  });
  $("#incidents").addEventListener("click", (e) => {
    const button = e.target.closest("[data-id]");
    if (button) inspect(Number(button.dataset.id));
  });
  window.addEventListener("hashchange", navigate);
  window.addEventListener("resize", () => {
    if (S.timeline && S.page === "overview") plot();
  });
  $("#export").addEventListener("click", () => {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(S.report, null, 2)], {
        type: "application/json",
      }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = "tracelens-validated-audit.json";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast("Evaluation report downloaded.");
  });
  $("#sample").addEventListener("click", async () => {
    try {
      const model = $("#replay-model").value;
      const result = await api(
        `/api/traces?model=${model}&kind=alerts&limit=1`,
      );
      if (!result.items.length)
        throw Error("No flagged examples for this detector.");
      const trace = await api(
        `/api/traces/${result.items[0].id}?model=${model}`,
      );
      $("#log-input").value = trace.events.map((e) => e.template).join("\n");
      countLines();
      toast("Loaded a flagged evaluation window, chosen as an example. It is not a random sample.");
    } catch (e) {
      toast(e.message);
    }
  });
  $("#log-input").addEventListener("input", countLines);
  $("#analyze").addEventListener("click", analyze);
  $("#file").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    if (f.size > 500000) {
      toast("Choose a file smaller than 500 KB.");
      return;
    }
    $("#log-input").value = await f.text();
    countLines();
  });
}
init();
