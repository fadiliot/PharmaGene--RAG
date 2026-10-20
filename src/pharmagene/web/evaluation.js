import { api, errorBanner, fixed, h, pct, s, table, tip } from "./util.js";

const METRICS = [
  ["precision@5", "Precision@5", "series-1", "var(--c1)"],
  ["mrr", "MRR", "series-2", "var(--c2)"],
  ["ndcg@10", "nDCG@10", "series-3", "var(--c3)"],
];

// 4px-rounded data end, square at the baseline
function barPath(x, y, w, hgt, r = 4) {
  if (w <= 0.5) return "";
  const rr = Math.min(r, w, hgt / 2);
  return `M${x} ${y}H${x + w - rr}A${rr} ${rr} 0 0 1 ${x + w} ${y + rr}V${y + hgt - rr}A${rr} ${rr} 0 0 1 ${x + w - rr} ${y + hgt}H${x}Z`;
}

function legend(items) {
  return h("div", { class: "legend" }, items.map(([label, color]) =>
    h("span", { class: "key" }, h("span", { class: "sw", style: { "--s": color } }), label)));
}

function tipRows(title, rows) {
  return [h("b", null, title), ...rows.map(([label, color, value]) =>
    h("div", { class: "row" }, h("span", null, h("span", { class: "sw", style: { background: color } }), label), h("b", null, value)))];
}

// ---------------------------------------------------------------- grouped bars (retrieval ablation)

function barChart(rows) {
  const W = 760, L = 150, R = 52, plotW = W - L - R, bh = 12, gap = 2, groupGap = 16, caption = 26;
  const groupH = METRICS.length * bh + (METRICS.length - 1) * gap;
  const svg = s("svg", { viewBox: "0 0 10 10", role: "img", "aria-label": "Retrieval metrics by configuration; a table view is available" });
  const grid = s("g", { class: "grid" }), marks = s("g");
  let y = 6, lastKg = null;

  for (const row of rows) {
    const kg = row.config.endsWith("+kg");
    if (kg !== lastKg) {
      if (lastKg !== null) y += 6;
      marks.append(s("text", { x: 0, y: y + 12, class: "lbl", style: "font-weight:700" }, kg ? "With knowledge-graph phenotype + metadata filter" : "Retrieval only"));
      y += caption;
      lastKg = kg;
    }
    const top = y;
    marks.append(s("text", { x: L - 10, y: top + groupH / 2 + 4, "text-anchor": "end", class: "lbl" }, row.config.replace("+kg", "").replace("+", " + ")));
    METRICS.forEach(([key, , cls], i) => {
      const v = row[key] ?? 0, by = top + i * (bh + gap), w = v * plotW;
      marks.append(s("path", { class: `bar-mark ${cls}`, d: barPath(L, by, w, bh) }));
      marks.append(s("text", { x: L + w + 6, y: by + bh - 2, class: "lbl" }, v.toFixed(2)));
    });
    const hit = s("rect", { class: "hit", x: 0, y: top - 4, width: W, height: groupH + 8 });
    const show = (ev) => tip.show(tipRows(row.config, [
      ...METRICS.map(([key, label, , color]) => [label, color, fixed(row[key], 3)]),
      ["Recall@5", "var(--muted)", fixed(row["recall@5"], 3)],
      ["Latency (ms)", "var(--muted)", fixed(row.mean_latency_ms, 2)],
    ]), ev);
    hit.addEventListener("mousemove", show);
    hit.addEventListener("mouseleave", () => tip.hide());
    marks.append(hit);
    y += groupH + groupGap;
  }
  const bottom = y + 4;
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    const x = L + v * plotW;
    grid.append(s("line", { x1: x, x2: x, y1: 0, y2: bottom }));
    marks.append(s("text", { x, y: bottom + 16, "text-anchor": "middle" }, v.toFixed(2)));
  }
  svg.setAttribute("viewBox", `0 0 ${W} ${bottom + 26}`);
  svg.append(grid, marks);
  return h("div", { class: "chart" }, svg);
}

// ---------------------------------------------------------------- lines (tau calibration)

function lineChart(points, suggested) {
  const W = 760, H = 330, ml = 44, mr = 96, mt = 16, mb = 52, pw = W - ml - mr, ph = H - mt - mb;
  const X = (t) => ml + t * pw, Y = (v) => mt + (1 - v) * ph;
  const series = [
    { key: "coverage", label: "Coverage", cls: "series-1", color: "var(--c1)" },
    { key: "accuracy", label: "Accuracy", cls: "series-2", color: "var(--c2)" },
  ];
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Coverage and accuracy as the abstention threshold changes; a table view is available" });
  const grid = s("g", { class: "grid" });
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    grid.append(s("line", { x1: ml, x2: W - mr, y1: Y(v), y2: Y(v) }));
    svg.append(s("text", { x: ml - 8, y: Y(v) + 4, "text-anchor": "end" }, `${Math.round(v * 100)}%`));
  }
  for (const t of [0, 0.25, 0.5, 0.75, 1]) svg.append(s("text", { x: X(t), y: H - 28, "text-anchor": "middle" }, t.toFixed(2)));
  svg.append(grid, s("text", { x: ml + pw / 2, y: H - 6, "text-anchor": "middle", class: "axis-label" }, "Abstention threshold τ"));

  const endLabels = [];
  for (const sr of series) {
    let d = "", pen = false, last = null;
    for (const p of points) {
      if (p[sr.key] == null) { pen = false; continue; }
      d += `${pen ? "L" : "M"}${X(p.tau).toFixed(1)} ${Y(p[sr.key]).toFixed(1)}`;
      pen = true; last = p;
    }
    if (d) svg.append(s("path", { class: `line ${sr.cls}`, d }));
    if (last) endLabels.push({ sr, x: X(last.tau), y: Y(last[sr.key]) });
  }
  endLabels.sort((a, b) => a.y - b.y);
  if (endLabels.length === 2 && endLabels[1].y - endLabels[0].y < 14) endLabels[1].ly = endLabels[0].y + 14;
  for (const e of endLabels) {
    svg.append(s("circle", { class: `dot ${e.sr.cls}`, cx: e.x, cy: e.y, r: 4.5 }));
    svg.append(s("text", { x: e.x + 10, y: (e.ly ?? e.y) + 4, class: "lbl" }, e.sr.label));
  }
  if (suggested != null) {
    svg.append(s("line", { x1: X(suggested), x2: X(suggested), y1: mt, y2: mt + ph, class: "crosshair", style: "opacity:.9" }),
      s("text", { x: X(suggested) + 6, y: mt + ph - 8, class: "lbl" }, `suggested τ = ${suggested}`));
  }

  const cross = s("line", { class: "crosshair", y1: mt, y2: mt + ph, visibility: "hidden" });
  const dots = series.map((sr) => s("circle", { class: `dot ${sr.cls}`, r: 4.5, visibility: "hidden" }));
  const overlay = s("rect", { class: "hit", x: ml, y: mt, width: pw, height: ph });
  overlay.addEventListener("mousemove", (ev) => {
    const r = svg.getBoundingClientRect();
    const t = Math.max(0, Math.min(1, ((ev.clientX - r.left) / r.width * W - ml) / pw));
    const p = points.reduce((best, q) => (Math.abs(q.tau - t) < Math.abs(best.tau - t) ? q : best));
    cross.setAttribute("x1", X(p.tau)); cross.setAttribute("x2", X(p.tau)); cross.setAttribute("visibility", "visible");
    series.forEach((sr, i) => {
      if (p[sr.key] == null) { dots[i].setAttribute("visibility", "hidden"); return; }
      dots[i].setAttribute("cx", X(p.tau)); dots[i].setAttribute("cy", Y(p[sr.key])); dots[i].setAttribute("visibility", "visible");
    });
    tip.show(tipRows(`τ = ${p.tau.toFixed(2)}`, series.map((sr) => [sr.label, sr.color, pct(p[sr.key])])), ev);
  });
  overlay.addEventListener("mouseleave", () => { tip.hide(); cross.setAttribute("visibility", "hidden"); dots.forEach((d) => d.setAttribute("visibility", "hidden")); });
  svg.append(cross, ...dots, overlay);
  return h("div", { class: "chart" }, svg);
}

// ---------------------------------------------------------------- sections

function seg(options, current, onPick) {
  return h("div", { class: "seg", role: "group" }, options.map(([val, label]) =>
    h("button", { type: "button", class: val === current ? "on" : "", onclick: () => onPick(val) }, label)));
}

function retrievalCard(data, rerun) {
  const body = h("div");
  let mode = "chart";
  const ran = (data?.rows || []).filter((r) => !r.skipped);
  const skipped = (data?.rows || []).filter((r) => r.skipped);
  const head = h("div", { class: "card-title" }, h("h2", null, "Retrieval ablation"));
  const runBtn = h("button", { class: "ghost", type: "button", onclick: async () => {
    runBtn.disabled = true; runBtn.textContent = "Running…";
    try { await rerun(); } catch (err) { body.prepend(errorBanner(err)); runBtn.disabled = false; runBtn.textContent = "Run ablation now"; }
  } }, "Run ablation now");

  function paint() {
    if (!ran.length) {
      body.replaceChildren(h("div", { class: "empty" }, h("h2", null, "No retrieval results yet"),
        h("p", null, "Click “Run ablation now”, or run ", h("code", null, "pharmagene eval-retrieval"), " from the terminal.")));
      return;
    }
    const best = ran.reduce((a, b) => (b["ndcg@10"] > a["ndcg@10"] ? b : a));
    const stats = h("div", { class: "stat-row" },
      h("div", { class: "stat" }, h("div", { class: "v" }, String(data.n_scenarios ?? ran[0].n ?? "–")), h("div", { class: "l" }, "scenarios (answerable)")),
      h("div", { class: "stat" }, h("div", { class: "v" }, best["ndcg@10"].toFixed(3)), h("div", { class: "l" }, `best nDCG@10 · ${best.config}`)),
      h("div", { class: "stat" }, h("div", { class: "v", style: { fontSize: "15px", wordBreak: "break-all" } }, data.embedder || "unknown"), h("div", { class: "l" }, "embedder")));
    const view = mode === "chart"
      ? [legend(METRICS.map(([, label, , color]) => [label, color])), barChart(ran)]
      : [table([
        { key: "config", label: "Configuration" }, { key: "precision@5", label: "P@5", render: (r) => fixed(r["precision@5"]) },
        { key: "recall@5", label: "Recall@5", render: (r) => fixed(r["recall@5"]) }, { key: "mrr", label: "MRR", render: (r) => fixed(r.mrr) },
        { key: "ndcg@10", label: "nDCG@10", render: (r) => fixed(r["ndcg@10"]) }, { key: "mean_latency_ms", label: "Latency (ms)", render: (r) => fixed(r.mean_latency_ms, 2) },
      ], ran, { numeric: ["precision@5", "recall@5", "mrr", "ndcg@10", "mean_latency_ms"] })];
    body.replaceChildren(stats,
      h("div", { class: "chart-controls" }, h("span", { class: "hint" }, "Precision@5 is capped by how many chunks are relevant per query in this small corpus."),
        seg([["chart", "Chart"], ["table", "Table"]], mode, (m) => { mode = m; paint(); })),
      ...view, ...skipped.map((r) => h("p", { class: "hint" }, `${r.config}: ${r.skipped}`)));
  }
  head.append(runBtn);
  paint();
  return h("section", { class: "card" }, head, body);
}

function e2eCard(e2e) {
  const head = h("div", { class: "card-title" }, h("h2", null, "End-to-end: LLM only vs RAG"));
  if (!e2e?.metrics) {
    return h("section", { class: "card" }, head, h("div", { class: "empty" }, h("h2", null, "No end-to-end results yet"),
      h("p", null, "Run ", h("code", null, "pharmagene eval-e2e --limit 20"), " (uses your LLM; with Gemini this makes API calls)."),
    ));
  }
  const done = e2e.metrics.filter((m) => !m.skipped), skipped = e2e.metrics.filter((m) => m.skipped);
  const P = (k) => (r) => pct(r[k]);
  const cols = [
    { key: "system", label: "System" }, { key: "answered", label: "Answered" },
    { key: "action_accuracy", label: "Action accuracy", render: P("action_accuracy") },
    { key: "abstention_recall", label: "Abstain recall", render: P("abstention_recall") },
    { key: "abstention_precision", label: "Abstain precision", render: P("abstention_precision") },
    { key: "unsafe_answer_rate", label: "Unsafe answers", render: P("unsafe_answer_rate") },
    { key: "hallucination_rate", label: "Hallucination rate", render: P("hallucination_rate") },
    { key: "unsupported_statement_rate", label: "Unsupported statements", render: P("unsupported_statement_rate") },
    { key: "macro_f1", label: "Macro F1", render: (r) => fixed(r.macro_f1, 2) },
    { key: "mean_latency_ms", label: "Latency (ms)", render: (r) => fixed(r.mean_latency_ms, 0) },
  ];
  return h("section", { class: "card" }, head,
    h("p", { class: "hint" }, `LLM: ${e2e.llm || "unknown"} · embedder: ${e2e.embedder || "unknown"} · ${e2e.n_scenarios ?? "?"} scenarios`),
    table(cols, done, { numeric: cols.slice(1).map((c) => c.key) }),
    skipped.map((m) => h("p", { class: "hint" }, `${m.system}: ${m.skipped}`)),
    h("p", { class: "hint" }, "Hallucination rate = answered recommendations that contain an unsupported claim or the wrong action. Unsafe answers = low-evidence queries that were answered instead of abstained."));
}

function calibrationCard(cal) {
  const head = h("div", { class: "card-title" }, h("h2", null, "Abstention threshold calibration"));
  if (!cal?.table) {
    return h("section", { class: "card" }, head, h("div", { class: "empty" }, h("h2", null, "Not calibrated yet"),
      h("p", null, "Run ", h("code", null, "pharmagene calibrate"), " to see how coverage and accuracy trade off as τ changes.")));
  }
  let mode = "chart";
  const body = h("div");
  function paint() {
    body.replaceChildren(
      h("div", { class: "chart-controls" },
        h("span", { class: "hint" }, `${cal.n_scored} scored scenarios · suggested τ for ≥ ${pct(cal.target_accuracy)} accuracy: ${cal.suggested_tau ?? "none reached"}`),
        seg([["chart", "Chart"], ["table", "Table"]], mode, (m) => { mode = m; paint(); })),
      ...(mode === "chart"
        ? [legend([["Coverage (answered share)", "var(--c1)"], ["Accuracy (of answered)", "var(--c2)"]]), lineChart(cal.table, cal.suggested_tau)]
        : [table([{ key: "tau", label: "τ" }, { key: "coverage", label: "Coverage", render: (r) => pct(r.coverage) }, { key: "accuracy", label: "Accuracy", render: (r) => pct(r.accuracy) }],
          cal.table, { numeric: ["tau", "coverage", "accuracy"] })]));
  }
  paint();
  return h("section", { class: "card" }, head, body);
}

export async function mount(root) {
  const results = await api("/results");
  const norm = (r) => (Array.isArray(r) ? { rows: r } : r);
  const host = h("div");
  const state = { retrieval: results.retrieval && norm(results.retrieval), e2e: results.e2e, calibration: results.calibration };

  async function rerun() {
    state.retrieval = await api("/eval/retrieval", {});
    paint();
  }
  function paint() {
    host.replaceChildren(retrievalCard(state.retrieval, rerun), e2eCard(state.e2e), calibrationCard(state.calibration));
  }

  const embedder = state.retrieval?.embedder || state.e2e?.embedder || "";
  root.replaceChildren(
    h("div", { class: "view-head" }, h("h1", null, "Evaluation"),
      h("p", null, "Retrieval quality (Precision@5, MRR, nDCG@10) across dense, BM25, hybrid and reranked configurations, end-to-end safety against an unmodified LLM, and abstention calibration.")),
    h("div", { class: "banner review" }, h("div", null, h("b", null, "Read these numbers with care"),
      "Gold labels are derived from the same corpus and rules the system uses, so they test plumbing and robustness, not clinical validity. ",
      embedder.startsWith("hashing") ? "The hashing embedder is a lexical stand-in; run with sentence-transformers or Gemini embeddings for meaningful dense scores. " : "",
      "Add clinician-written scenarios before quoting any figure.")),
    host);
  paint();
}
