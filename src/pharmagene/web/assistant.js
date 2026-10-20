import { $, ACTION_LABELS, ACTION_TONE, api, errorBanner, h, retrievalOpts, safeUrl, shared } from "./util.js";

const history = []; // kept in memory only: genotypes are sensitive, nothing is written to storage
let composerMode = "free";
let nextId = 1;

const EXAMPLES = [
  ["CYP2C19 *2/*2 · clopidogrel", "Patient with CYP2C19 *2/*2 genotype prescribed clopidogrel following percutaneous coronary intervention."],
  ["SLCO1B1 c.521T>C · simvastatin", "Patient carries the SLCO1B1 c.521T>C variant (heterozygous) and is being evaluated for simvastatin therapy."],
  ["DPYD *1/*2A · capecitabine", "DPYD *1/*2A, starting capecitabine for colorectal cancer."],
  ["CYP2D6 *1xN/*1 · codeine", "CYP2D6 *1xN/*1, considering codeine for postoperative pain."],
  ["Ambiguous copy number", "CYP2D6 *1xN/*4, considering codeine for postoperative pain."],
  ["Unsupported drug", "CYP2C19 *2/*2 considering warfarin."],
];

// ---------------------------------------------------------------- result card

function bar(value, tau) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const el = h("div", { class: "bar" + (tau != null ? " total" : "") }, h("i", { style: { width: `${pct}%` } }));
  if (tau != null) el.append(h("span", { class: "tau", style: { left: `${Math.min(1, tau) * 100}%` }, title: `Abstention threshold ${tau}` }));
  return el;
}

function confidence(c) {
  const grid = h("div", { class: "conf" });
  for (const [label, v] of [["Relevance", c.relevance], ["Alignment", c.alignment], ["Coverage", c.coverage], ["Verification", c.verification]]) {
    grid.append(h("span", null, label), bar(v), h("span", { class: "num" }, v.toFixed(2)));
  }
  grid.append(h("span", { class: "strong" }, "Confidence"), bar(c.score, c.tau), h("span", { class: "num strong" }, c.score.toFixed(2)));
  return [h("h3", { class: "eyebrow" }, "Confidence"), grid,
    h("div", { class: "hint", style: { marginTop: "6px" } }, `Vertical marker = abstention threshold (${c.tau.toFixed(2)})`)];
}

function evidenceItem(e, uid) {
  const url = e.source_url && safeUrl(e.source_url);
  return h("div", { class: "ev", id: `${uid}-${e.handle}` },
    h("div", { class: "ev-head" }, h("span", { class: "handle" }, e.handle), h("span", { class: "src" }, e.source || "Source n/a"), h("span", { class: "id" }, e.chunk_id)),
    e.citation ? h("p", null, e.citation, " ", url ? h("a", { href: url, target: "_blank", rel: "noopener noreferrer" }, "source") : null) : null,
    e.snippet ? h("p", null, e.snippet) : null);
}

function jumpTo(uid, handle) {
  const el = document.getElementById(`${uid}-${handle}`);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.add("flash");
  setTimeout(() => el.classList.remove("flash"), 1400);
}

export function resultCard(rec, uid) {
  const abstain = rec.decision === "abstain";
  const head = h("div", { class: "card-head" }, h("h2", null, rec.drug),
    abstain ? h("span", { class: "badge abstain" }, "Abstained")
      : h("span", { class: `badge ${ACTION_TONE[rec.action] || "neutral"}` }, ACTION_LABELS[rec.action] || rec.action || "Recommendation"));
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
    if (rec.evidence.length) body.push(h("h3", { class: "eyebrow" }, "Evidence"), h("div", { class: "evidence" }, rec.evidence.map((e) => evidenceItem(e, uid))));
  }
  if (rec.confidence) body.push(confidence(rec.confidence));

  const tech = [];
  if (rec.verification_issues.length) tech.push(h("h3", { class: "eyebrow" }, "Verifier findings"), h("ul", { class: "notes" }, rec.verification_issues.map((i) => h("li", null, i))));
  if (rec.notes.length) tech.push(h("h3", { class: "eyebrow" }, "Pipeline notes"), h("ul", { class: "notes" }, rec.notes.map((n) => h("li", null, n))));
  if (tech.length) body.push(h("details", { class: "technical" }, h("summary", null, "Technical details"), tech));
  body.push(h("div", { class: "foot-meta" }, h("span", null, `${rec.llm_backend} · ${rec.latency_ms} ms`)));
  return h("article", { class: "card rcard" }, body);
}

// ---------------------------------------------------------------- chat

function renderEntry(entry) {
  const node = h("div", { class: "msg-bot", id: `entry-${entry.id}` });
  if (entry.status === "loading") node.append(h("div", { class: "skeleton", "aria-label": "Loading" }));
  else if (entry.status === "error") node.append(errorBanner(entry.error));
  else {
    node.append(...entry.report.recommendations.map((r, i) => resultCard(r, `e${entry.id}r${i}`)));
    node.append(h("div", { class: "foot-meta" },
      h("button", { type: "button", class: "link-btn", onclick: () => { shared.traceText = entry.traceText; location.hash = "#/trace"; } }, "Inspect pipeline for this query →")));
  }
  return node;
}

function hero(send) {
  return h("div", { class: "hero" },
    h("h1", null, "Evidence-grounded pharmacogenomic guidance"),
    h("p", null, "Enter a genotype and a medication. Every recommendation cites its guideline source, or the system abstains and flags the case for a clinician."),
    h("div", { class: "examples" }, EXAMPLES.map(([label, text]) => h("button", { type: "button", class: "chip", onclick: () => send(text) }, label))));
}

export function mount(root) {
  const thread = h("div", { class: "thread", "aria-live": "polite" });
  const free = h("textarea", { rows: 2, placeholder: "e.g. Patient with CYP2C19 *2/*2 genotype prescribed clopidogrel following PCI.", "aria-label": "Clinical question" });
  const sGeno = h("input", { type: "text", placeholder: "Genotype: CYP2C19 *2/*2; DPYD *1/*2A", "aria-label": "Genotype" });
  const sDrugs = h("input", { type: "text", list: "drug-list", placeholder: "Medications: clopidogrel, capecitabine", "aria-label": "Medications" });
  const sCtx = h("input", { type: "text", placeholder: "Context (optional)", "aria-label": "Clinical context" });
  const datalist = h("datalist", { id: "drug-list" }, (shared.meta?.drugs || []).map((d) => h("option", { value: d.name }, d.genes.join(", "))));
  const struct = h("div", { class: "struct-grid" }, sGeno, sDrugs, sCtx);
  const send = h("button", { class: "primary", type: "submit" }, "Get recommendation");
  const clear = h("button", { class: "ghost", type: "button", onclick: () => { history.length = 0; paint(); } }, "Clear conversation");
  const segFree = h("button", { type: "button", onclick: () => setMode("free") }, "Free text");
  const segStruct = h("button", { type: "button", onclick: () => setMode("struct") }, "Structured");

  function setMode(m) {
    composerMode = m;
    free.classList.toggle("hidden", m !== "free");
    struct.classList.toggle("hidden", m !== "struct");
    segFree.classList.toggle("on", m === "free");
    segStruct.classList.toggle("on", m === "struct");
  }

  function paint() {
    thread.replaceChildren(...(history.length
      ? history.flatMap((e) => [h("div", { class: "msg-user" }, e.userText), renderEntry(e)])
      : [hero(submitText)]));
    clear.classList.toggle("hidden", !history.length);
  }

  async function run(userText, traceText, request) {
    const entry = { id: nextId++, userText, traceText, status: "loading" };
    history.push(entry);
    paint();
    send.disabled = true;
    thread.lastElementChild.scrollIntoView({ behavior: "smooth", block: "end" });
    try {
      entry.report = await request();
      entry.status = "done";
    } catch (err) {
      entry.status = "error";
      entry.error = err;
    } finally {
      send.disabled = false;
    }
    document.getElementById(`entry-${entry.id}`)?.replaceWith(renderEntry(entry));
    thread.lastElementChild.scrollIntoView({ behavior: "smooth", block: "end" });
  }

  function submitText(text) {
    run(text, text, () => api("/ask", { text, ...retrievalOpts() }));
  }

  const form = h("form", { class: "composer", autocomplete: "off", onsubmit: (ev) => {
    ev.preventDefault();
    if (composerMode === "free") {
      const text = free.value.trim();
      if (!text) return;
      free.value = "";
      submitText(text);
    } else {
      const medications = sDrugs.value.split(",").map((x) => x.trim()).filter(Boolean);
      const genotype = sGeno.value.trim();
      if (!genotype || !medications.length) return;
      const context = sCtx.value.trim();
      const shown = `${genotype} · ${medications.join(", ")}${context ? ` · ${context}` : ""}`;
      run(shown, `${genotype}; ${medications.join(", ")} ${context}`.trim(),
        () => api("/recommend", { genotype, medications, context, ...retrievalOpts() }));
    }
  } },
  h("div", { class: "composer-box" }, free, struct, datalist,
    h("div", { class: "composer-row" }, h("div", { class: "seg", role: "group", "aria-label": "Input mode" }, segFree, segStruct), h("div", { class: "kv" }, clear, send))));

  free.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); form.requestSubmit(); }
  });

  root.replaceChildren(h("div", { class: "chat" }, thread, form));
  setMode(composerMode);
  paint();
  free.focus();
}
