import { api, errorBanner, fixed, h, retrievalOpts, shared, table } from "./util.js";

const SERIES = [["dense", "Dense (cosine)", "var(--c1)"], ["bm25", "BM25 (scaled)", "var(--c2)"], ["hybrid", "Hybrid", "var(--c3)"]];

function mini(value, color) {
  return h("div", { class: "mini", style: { "--s": color } },
    h("div", { class: "bar" }, h("i", { style: { width: `${Math.max(0, Math.min(1, value)) * 100}%` } })),
    h("span", null, value.toFixed(2)));
}

function step(n, title, ...body) {
  return h("div", { class: "step" }, h("div", { class: "step-n" }, String(n)),
    h("div", null, h("h3", null, title), h("div", { class: "body" }, body)));
}

function legend() {
  return h("div", { class: "legend" }, SERIES.map(([, label, color]) =>
    h("span", { class: "key" }, h("span", { class: "sw", style: { "--s": color } }), label)));
}

function retrievalTable(items) {
  const maxBm = Math.max(...items.map((r) => r.bm25), 1e-9);
  const tbody = h("tbody");
  for (const r of items) {
    const detail = h("tr", { class: "expanded-row hidden" }, h("td", { colspan: 7 }, r.text));
    const row = h("tr", { class: "clickable", tabindex: 0, title: "Click to show the chunk text" },
      h("td", { class: "num" }, r.rank),
      h("td", null, h("code", null, r.chunk_id)),
      h("td", null, r.type || "–", r.phenotype ? h("div", { class: "hint" }, r.phenotype) : null),
      h("td", null, mini(Math.max(0, r.dense), "var(--c1)")),
      h("td", null, mini(r.bm25 / maxBm, "var(--c2)")),
      h("td", null, mini(r.hybrid, "var(--c3)")),
      h("td", { class: "num" }, r.rerank == null ? "–" : fixed(r.rerank, 2)));
    const toggle = () => detail.classList.toggle("hidden");
    row.addEventListener("click", toggle);
    row.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } });
    tbody.append(row, detail);
  }
  const head = h("tr", null, ["#", "Chunk", "Type", "Dense", "BM25", "Hybrid", "Rerank"].map((t, i) => h("th", { class: i === 0 || i === 6 ? "num" : "" }, t)));
  return h("div", { class: "table-wrap" }, h("table", null, h("thead", null, head), tbody));
}

function drugCard(item, cfgLabel) {
  const steps = [];
  const entities = h("div", { class: "kv" }, h("span", { class: "tag" }, `drug: ${item.drug}`));
  if (item.gene) entities.append(h("span", { class: "tag" }, `gene: ${item.gene}`), h("span", { class: "tag" }, `genotype: ${item.genotype}`));
  steps.push(step(1, "Entity linking", entities,
    h("div", { class: "hint" }, "Genes, alleles and drugs are matched with the lexicon (brand names and aliases included); an LLM never invents them.")));

  const res = item.resolution;
  if (res) {
    const phen = h("div", { class: "kv" }, res.phenotypes.map((p) => h("span", { class: `badge ${res.ambiguous ? "abstain" : "neutral"}` }, p)));
    if (res.activity_score != null) phen.append(h("span", { class: "tag" }, `activity score ${res.activity_lower_bound ? "≥ " : ""}${res.activity_score}`));
    steps.push(step(2, "Genotype → phenotype (deterministic)", phen, h("div", null, res.explanation)));
  } else if (!item.error) {
    steps.push(step(2, "Genotype → phenotype", h("div", { class: "hint" }, "Knowledge-graph translation is off in settings, so retrieval alone must identify the phenotype.")));
  }

  if (item.error) {
    steps.push(step(res ? 3 : 2, "Stopped before retrieval",
      h("div", { class: "banner abstain" }, h("div", null, h("b", null, "The pipeline abstains here."), item.error))));
  } else {
    const filters = item.filters
      ? h("div", { class: "kv" }, Object.entries(item.filters).filter(([, v]) => v).map(([k, v]) => h("span", { class: "tag" }, `${k} = ${v}`)))
      : h("div", { class: "hint" }, "No metadata filter (paper-style retrieval over the whole corpus).");
    steps.push(step(3, `Retrieval (${cfgLabel})`, h("div", { class: "qbox" }, item.query), filters,
      item.note ? h("div", { class: "hint" }, item.note) : null, legend(), retrievalTable(item.retrieved),
      item.expanded.length ? h("div", { class: "hint" }, `Graph expansion guarantees these chunks accompany the answer: ${item.expanded.join(", ")}`) : null));

    const path = h("div", { class: "path" });
    item.graph_path.forEach((id, i) => {
      if (i) path.append(h("span", { class: "arrow" }, "→"));
      path.append(h("span", { class: "tag" }, id.split(":").slice(1).join(" · ")));
    });
    const btn = h("button", { class: "ghost", type: "button", onclick: () => {
      shared.highlight = { label: item.drug, nodes: item.graph_path, chunks: [...item.retrieved.map((r) => r.chunk_id), ...item.expanded] };
      location.hash = "#/graph";
    } }, "Show in knowledge graph");
    steps.push(step(4, "Knowledge-graph path", path, h("div", null, btn)));
  }
  return h("section", { class: "card" }, h("div", { class: "card-title" }, h("h2", { style: { textTransform: "capitalize" } }, item.drug)), h("div", { class: "steps" }, steps));
}

export function mount(root) {
  const ta = h("textarea", { rows: 2, placeholder: "e.g. CYP2C19 *2/*2, prescribed clopidogrel after PCI", "aria-label": "Query to trace" });
  const btn = h("button", { class: "primary", type: "submit" }, "Trace");
  const out = h("div", { style: { marginTop: "16px" } });

  async function run() {
    const text = ta.value.trim();
    if (!text) return;
    btn.disabled = true;
    out.replaceChildren(h("div", { class: "skeleton" }));
    try {
      const data = await api("/trace", { text, ...retrievalOpts() });
      out.replaceChildren(...(data.drugs.length
        ? data.drugs.map((d) => drugCard(d, data.config))
        : [h("div", { class: "banner abstain" }, h("div", null, h("b", null, "No supported medication found."), "Nothing to trace: add a medication such as clopidogrel or codeine."))]));
    } catch (err) {
      out.replaceChildren(errorBanner(err));
    } finally {
      btn.disabled = false;
    }
  }

  root.replaceChildren(
    h("div", { class: "view-head" }, h("h1", null, "Pipeline trace"),
      h("p", null, "See exactly what happens before an answer is generated: entity linking, deterministic genotype translation, the retrieval query and filters, per-chunk dense, BM25 and hybrid scores, and the knowledge-graph path. No LLM is called here.")),
    h("form", { class: "card", onsubmit: (e) => { e.preventDefault(); run(); } }, h("div", { class: "trace-input" }, ta, btn)),
    out);

  // text arrives from the Assistant ("Inspect pipeline") or from a deep link such as #/trace?q=...
  const fromLink = new URLSearchParams(location.hash.split("?")[1] || "").get("q");
  const initial = shared.traceText || fromLink;
  shared.traceText = null;
  if (initial) {
    ta.value = initial;
    run();
  }
}
