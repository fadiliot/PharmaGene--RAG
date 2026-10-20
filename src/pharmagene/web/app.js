"use strict";

const $ = (sel, el = document) => el.querySelector(sel);

// Every string from the server (including LLM output) goes through textContent, never innerHTML.
function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    el.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return el;
}

const ACTION_LABELS = {
  standard: "Standard dosing",
  standard_monitor: "Standard dosing, monitor",
  reduce_dose: "Reduce dose",
  alternative_agent: "Use alternative agent",
  avoid: "Avoid",
};
const ACTION_TONE = { standard: "ok", standard_monitor: "neutral", reduce_dose: "abstain", alternative_agent: "abstain", avoid: "avoid" };

function apiKeyHeader() {
  const key = $("#opt-key").value.trim();
  return key ? { "X-API-Key": key } : {};
}

function options() {
  return { mode: $("#opt-mode").value, use_kg: $("#opt-kg").checked, rerank: $("#opt-rerank").checked };
}

async function call(path, body) {
  const resp = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...apiKeyHeader() },
    body: JSON.stringify({ ...body, ...options() }),
  });
  if (!resp.ok) {
    let detail = resp.statusText;
    try { detail = (await resp.json()).detail || detail; } catch (_) { /* keep statusText */ }
    if (resp.status === 401) detail = "This server requires an API key. Enter it under Retrieval settings.";
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return resp.json();
}

function bar(value, tau) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const el = h("div", { class: "bar" + (tau != null ? " total" : "") }, h("i", { style: `width:${pct}%` }));
  if (tau != null) el.append(h("span", { class: "tau", style: `left:${Math.min(1, tau) * 100}%`, title: `Abstention threshold ${tau}` }));
  return el;
}

function confidence(c) {
  const rows = [
    ["Relevance", c.relevance], ["Alignment", c.alignment], ["Coverage", c.coverage], ["Verification", c.verification],
  ];
  const grid = h("div", { class: "conf" });
  for (const [label, v] of rows) grid.append(h("span", null, label), bar(v), h("span", { class: "num" }, v.toFixed(2)));
  grid.append(
    h("span", { class: "total-label" }, "Confidence"), bar(c.score, c.tau),
    h("span", { class: "num total-label" }, c.score.toFixed(2)),
  );
  return [h("h3", null, "Confidence"), grid, h("div", { class: "hint", style: "margin-top:6px" }, `Black marker = abstention threshold (${c.tau.toFixed(2)})`)];
}

function safeUrl(u) {
  try { const p = new URL(u); return p.protocol === "https:" || p.protocol === "http:" ? p.href : null; } catch (_) { return null; }
}

function evidenceItem(e, uid) {
  const url = e.source_url && safeUrl(e.source_url);
  return h("div", { class: "ev", id: `${uid}-${e.handle}` },
    h("div", { class: "ev-head" },
      h("span", { class: "handle" }, e.handle),
      h("span", { class: "src" }, e.source || "Source n/a"),
      h("span", { class: "id" }, e.chunk_id),
    ),
    e.citation ? h("p", null, e.citation, " ", url ? h("a", { href: url, target: "_blank", rel: "noopener noreferrer" }, "source") : null) : null,
    e.snippet ? h("p", null, e.snippet) : null,
  );
}

function jumpTo(uid, handle) {
  const el = document.getElementById(`${uid}-${handle}`);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.add("flash");
  setTimeout(() => el.classList.remove("flash"), 1400);
}

function card(rec, idx) {
  const uid = `r${idx}`;
  const abstain = rec.decision === "abstain";
  const head = h("div", { class: "card-head" }, h("h2", null, rec.drug));
  if (abstain) head.append(h("span", { class: "badge abstain" }, "Abstained"));
  else head.append(h("span", { class: `badge ${ACTION_TONE[rec.action] || "neutral"}` }, ACTION_LABELS[rec.action] || rec.action || "Recommendation"));

  const meta = h("div", { class: "meta-line" });
  if (rec.gene) {
    meta.append(h("code", null, `${rec.gene} ${rec.genotype || ""}`.trim()));
    if (rec.phenotype) meta.append(" → ", h("strong", null, rec.phenotype));
  } else meta.append("No gene could be matched");

  const body = [head, meta];

  if (abstain) {
    body.push(h("div", { class: "banner abstain", role: "alert" },
      h("div", null, h("b", null, "No recommendation. Flagged for clinician review."), rec.abstain_reason || "Evidence insufficient.")));
  } else {
    if (rec.requires_clinician_review) {
      body.push(h("div", { class: "banner review" }, h("div", null, h("b", null, "Clinician verification recommended"),
        rec.used_fallback ? "The model's draft failed verification, so guideline text is shown verbatim." : "Confidence is close to the abstention threshold.")));
    }
    const list = h("ul", { class: "claims" });
    for (const c of rec.claims) {
      list.append(h("li", null, c.text.replace(/\s*\[[A-Z]+\d+\]/g, ""),
        c.citations.map((cid) => h("button", { type: "button", class: "cite", title: "Show evidence", onclick: () => jumpTo(uid, cid) }, cid))));
    }
    body.push(list);
    if (rec.evidence.length) body.push(h("h3", null, "Evidence"), h("div", { class: "evidence" }, rec.evidence.map((e) => evidenceItem(e, uid))));
  }

  if (rec.confidence) body.push(confidence(rec.confidence));

  const tech = [];
  if (rec.verification_issues.length) {
    tech.push(h("h3", null, "Verifier findings"), h("ul", { class: "notes" }, rec.verification_issues.map((i) => h("li", null, i))));
  }
  if (rec.notes.length) tech.push(h("h3", null, "Pipeline notes"), h("ul", { class: "notes" }, rec.notes.map((n) => h("li", null, n))));
  if (tech.length) body.push(h("details", { class: "technical" }, h("summary", null, "Technical details"), tech));

  body.push(h("div", { class: "foot-meta" }, `${rec.llm_backend} · ${rec.latency_ms} ms`));
  return h("article", { class: "card" }, body);
}

function render(report) {
  const box = $("#results");
  box.replaceChildren(...report.recommendations.map(card));
}

async function submit(path, body, button) {
  const box = $("#results");
  button.disabled = true;
  box.replaceChildren(h("div", { class: "skeleton", "aria-label": "Loading" }));
  try {
    render(await call(path, body));
  } catch (err) {
    box.replaceChildren(h("div", { class: "error", role: "alert" }, `Request failed: ${err.message}`));
  } finally {
    button.disabled = false;
  }
}

function setTab(name) {
  for (const t of document.querySelectorAll(".tab")) {
    const on = t.dataset.tab === name;
    t.classList.toggle("active", on);
    t.setAttribute("aria-selected", String(on));
  }
  $("#form-ask").classList.toggle("hidden", name !== "ask");
  $("#form-structured").classList.toggle("hidden", name !== "structured");
}

async function init() {
  for (const t of document.querySelectorAll(".tab")) t.addEventListener("click", () => setTab(t.dataset.tab));
  for (const c of document.querySelectorAll("[data-example]")) {
    c.addEventListener("click", () => { $("#ask-text").value = c.dataset.example; $("#ask-text").focus(); });
  }
  $("#form-ask").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const text = $("#ask-text").value.trim();
    if (text) submit("/ask", { text }, ev.submitter);
  });
  $("#form-structured").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const medications = $("#s-drugs").value.split(",").map((s) => s.trim()).filter(Boolean);
    const genotype = $("#s-genotype").value.trim();
    if (genotype && medications.length) submit("/recommend", { genotype, medications, context: $("#s-context").value.trim() }, ev.submitter);
  });

  const status = $("#status");
  try {
    const [health, meta] = await Promise.all([fetch("/health").then((r) => r.json()), fetch("/meta").then((r) => r.json())]);
    status.replaceChildren(h("span", { class: "dot" }), h("b", null, health.llm), ` · ${health.embedder} · ${health.chunks} chunks`);
    $("#drug-list").replaceChildren(...meta.drugs.map((d) => h("option", { value: d.name }, d.genes.join(", "))));
    if (!meta.reranker) {
      $("#opt-rerank").disabled = true;
      $("#opt-rerank").parentElement.append(h("span", { class: "hint" }, "(unavailable)"));
    }
  } catch (_) {
    status.replaceChildren(h("span", { class: "dot off" }), "Server unreachable");
  }
}

init();
