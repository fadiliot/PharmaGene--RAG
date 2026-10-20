import { api, h } from "./util.js";

export async function mount(root) {
  const chunks = await api("/corpus");
  const uniq = (get) => [...new Set(chunks.flatMap((c) => [].concat(get(c) ?? [])).filter(Boolean))].sort();
  const select = (label, values) => h("select", { "aria-label": label },
    h("option", { value: "" }, `All ${label.toLowerCase()}`), values.map((v) => h("option", { value: v }, v)));

  const gene = select("Genes", uniq((c) => c.metadata.gene));
  const drug = select("Drugs", uniq((c) => c.metadata.drugs));
  const type = select("Types", uniq((c) => c.metadata.type));
  const q = h("input", { type: "text", placeholder: "Search chunk text…", "aria-label": "Search chunk text" });
  const count = h("p", { class: "hint" });
  const body = h("tbody");

  function paint() {
    const needle = q.value.trim().toLowerCase();
    const rows = chunks.filter((c) => {
      const m = c.metadata;
      return (!gene.value || m.gene === gene.value) && (!drug.value || (m.drugs || []).includes(drug.value))
        && (!type.value || m.type === type.value) && (!needle || c.text.toLowerCase().includes(needle));
    });
    count.textContent = `Showing ${rows.length} of ${chunks.length} chunks`;
    body.replaceChildren(...rows.flatMap((c) => {
      const m = c.metadata;
      const detail = h("tr", { class: "expanded-row hidden" },
        h("td", { colspan: 6 }, h("div", { class: "chunk-text" }, c.text), m.citation ? h("p", { class: "hint" }, `${m.citation}${m.status ? ` · status: ${m.status}` : ""}`) : null));
      const row = h("tr", { class: "clickable", tabindex: 0, title: "Click to show the chunk text" },
        h("td", null, h("code", null, c.id)), h("td", null, m.gene || "–"), h("td", null, (m.drugs || []).join(", ") || "–"),
        h("td", null, m.type || "–"), h("td", null, m.phenotype || "–"), h("td", null, m.action || "–"));
      const toggle = () => detail.classList.toggle("hidden");
      row.addEventListener("click", toggle);
      row.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } });
      return [row, detail];
    }));
  }
  for (const el of [gene, drug, type]) el.addEventListener("change", paint);
  q.addEventListener("input", paint);

  const head = h("tr", null, ["Chunk", "Gene", "Drug", "Type", "Phenotype", "Action"].map((t) => h("th", null, t)));
  root.replaceChildren(
    h("div", { class: "view-head" }, h("h1", null, "Corpus"),
      h("p", null, "The guideline chunks the retriever searches, with the metadata used for filtering and citations. Documents are chunked by section; per-section comments in the source files set gene, drug, phenotype, type and action.")),
    h("div", { class: "card" }, h("div", { class: "filters" }, q, gene, drug, type), count,
      h("div", { class: "table-wrap" }, h("table", null, h("thead", null, head), body))));
  paint();
}
