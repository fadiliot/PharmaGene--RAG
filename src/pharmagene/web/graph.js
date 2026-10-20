import { ACTION_LABELS, api, errorBanner, h, s, shared, table, tip } from "./util.js";
import { forceLayout, layeredLayout, nodeSize } from "./graphlayout.js";

const TYPE_LABEL = { gene: "Gene", phenotype: "Phenotype", drug: "Drug", chunk: "Guideline chunk" };
const ACTIONS = ["standard", "standard_monitor", "reduce_dose", "alternative_agent", "avoid"];

function edgeClass(e) {
  if (e.rel === "RECOMMENDS") return `rec act-${e.action}`;
  if (e.rel === "HAS_PHENOTYPE") return "has-phenotype";
  if (e.rel === "AFFECTED_BY") return "affected";
  return "about";
}

function shapeFor(n) {
  const { w, h: hh } = nodeSize(n);
  if (n.type === "chunk") return s("circle", { class: "shape", r: 6 });
  if (n.type === "gene") return s("rect", { class: "shape", x: -w / 2, y: -hh / 2, width: w, height: hh, rx: 8 });
  if (n.type === "phenotype") return s("rect", { class: "shape", x: -w / 2, y: -hh / 2, width: w, height: hh, rx: hh / 2 });
  const c = 11, hw = w / 2, hh2 = hh / 2;
  return s("polygon", { class: "shape", points: `${-hw + c},${-hh2} ${hw - c},${-hh2} ${hw},0 ${hw - c},${hh2} ${-hw + c},${hh2} ${-hw},0` });
}

function legendCard() {
  const nodeSample = (type, label) => h("div", { class: "key" },
    s("svg", { width: 64, height: 26, viewBox: "-42 -17 84 34", "aria-hidden": "true" },
      s("g", { class: `node ${type}` }, shapeFor({ type, label: "x" }))),
    label);
  const lineSample = (action) => h("div", { class: "key" },
    s("svg", { width: 54, height: 12, viewBox: "0 0 54 12", "aria-hidden": "true" }, s("path", { class: `edge rec act-${action}`, d: "M2 6 H52" })),
    ACTION_LABELS[action]);
  return h("div", { class: "panel legend-card" },
    h("h3", { class: "eyebrow", style: { marginTop: 0 } }, "Nodes"),
    h("div", { class: "legend", style: { flexDirection: "column", alignItems: "flex-start" } },
      nodeSample("gene", "Gene"), nodeSample("phenotype", "Phenotype"), nodeSample("drug", "Drug"), nodeSample("chunk", "Guideline chunk")),
    h("h3", { class: "eyebrow" }, "Recommendation (phenotype → drug)"),
    h("div", { class: "legend", style: { flexDirection: "column", alignItems: "flex-start" } }, ACTIONS.map(lineSample)),
    h("p", { class: "hint" }, "Line style repeats the colour, so the action is readable without relying on colour alone."));
}

export async function mount(root) {
  const st = { chunks: false, layout: "layered", hidden: new Set(), query: "", selected: null, table: false, hl: shared.highlight };
  shared.highlight = null;
  let data = null, pos = new Map(), adj = new Map(), nodeById = new Map();
  const view = { k: 1, x: 0, y: 0 };

  // ---- structure
  const svg = s("svg", { role: "group", "aria-label": "Knowledge graph of genes, phenotypes and drugs" });
  const vp = s("g");
  const edgesG = s("g"), nodesG = s("g");
  vp.append(edgesG, nodesG);
  svg.append(vp);
  const wrap = h("div", { class: "canvas-wrap" }, svg,
    h("div", { class: "zoom" },
      h("button", { type: "button", "aria-label": "Zoom in", onclick: () => zoomBy(1.25) }, "+"),
      h("button", { type: "button", "aria-label": "Zoom out", onclick: () => zoomBy(0.8) }, "−")));
  const tableHost = h("div", { class: "hidden" });
  const panel = h("aside", { class: "panel", "aria-live": "polite" });
  const hlBar = h("div", { class: "banner review hidden" });

  const search = h("input", { type: "text", placeholder: "Search nodes…", "aria-label": "Search nodes", oninput: (e) => { st.query = e.target.value.trim().toLowerCase(); draw(); } });
  const layoutBtns = ["layered", "force"].map((name) => h("button", { type: "button", "data-l": name, onclick: () => setLayout(name) }, name === "layered" ? "Layered" : "Force"));
  const viewBtns = ["graph", "table"].map((name) => h("button", { type: "button", "data-v": name, onclick: () => { st.table = name === "table"; paintMode(); } }, name === "graph" ? "Graph" : "Table"));
  const chunkChk = h("input", { type: "checkbox", id: "chk-chunks", onchange: async (e) => { st.chunks = e.target.checked; if (st.chunks) st.layout = "force"; await load(); } });
  const typeChecks = ["gene", "phenotype", "drug"].map((t) => h("label", { class: "check" },
    h("input", { type: "checkbox", checked: true, onchange: (e) => { e.target.checked ? st.hidden.delete(t) : st.hidden.add(t); draw(); } }), TYPE_LABEL[t]));
  const toolbar = h("div", { class: "graph-toolbar" }, search,
    h("div", { class: "seg", role: "group", "aria-label": "Layout" }, layoutBtns),
    h("div", { class: "seg", role: "group", "aria-label": "View" }, viewBtns),
    typeChecks, h("label", { class: "check", for: "chk-chunks" }, chunkChk, "Guideline chunks"),
    h("button", { type: "button", class: "ghost", onclick: () => fit() }, "Fit to view"));

  root.replaceChildren(
    h("div", { class: "view-head" }, h("h1", null, "Knowledge graph"),
      h("p", null, "Genes, their metabolizer / function phenotypes, and the drugs they affect. Each coloured line is a guideline recommendation (phenotype → drug) and carries its action. Drag nodes, scroll to zoom, click a node for details.")),
    toolbar, hlBar,
    h("div", { class: "graph-layout" }, h("div", null, wrap, tableHost), h("div", null, panel, legendCard())));

  // ---- data
  async function load() {
    try {
      data = await api(`/graph?chunks=${st.chunks}`);
    } catch (err) {
      root.append(errorBanner(err));
      return;
    }
    nodeById = new Map(data.nodes.map((n) => [n.id, n]));
    adj = new Map(data.nodes.map((n) => [n.id, []]));
    for (const e of data.edges) {
      adj.get(e.source)?.push({ edge: e, other: e.target, dir: "out" });
      adj.get(e.target)?.push({ edge: e, other: e.source, dir: "in" });
    }
    chunkChk.checked = st.chunks;
    relayout();
    paintMode();
    renderPanel();
  }

  function relayout() {
    pos = st.layout === "layered" && !st.chunks
      ? layeredLayout(data.nodes, data.edges)
      : forceLayout(data.nodes, data.edges);
    for (const b of layoutBtns) b.classList.toggle("on", b.dataset.l === st.layout);
    requestAnimationFrame(() => { draw(); fit(); });
  }

  function setLayout(name) {
    st.layout = name;
    if (name === "layered" && st.chunks) { st.chunks = false; load(); return; }
    relayout();
  }

  function paintMode() {
    wrap.classList.toggle("hidden", st.table);
    tableHost.classList.toggle("hidden", !st.table);
    for (const b of viewBtns) b.classList.toggle("on", (b.dataset.v === "table") === st.table);
    if (st.table) renderTable();
    else draw();
  }

  // ---- emphasis (selection > search > trace highlight)
  function emphasis() {
    if (st.selected) return new Set([st.selected, ...(adj.get(st.selected) || []).map((a) => a.other)]);
    if (st.query) return new Set(data.nodes.filter((n) => n.label.toLowerCase().includes(st.query) || (n.gene || "").toLowerCase() === st.query).map((n) => n.id));
    if (st.hl) return new Set([...st.hl.nodes, ...st.hl.chunks.map((c) => `chunk:${c}`)]);
    return null;
  }

  function draw() {
    if (!data) return;
    const hidden = (n) => st.hidden.has(n.type);
    const visible = new Set(data.nodes.filter((n) => !hidden(n)).map((n) => n.id));
    const emph = emphasis();
    const layered = st.layout === "layered" && !st.chunks;
    const pathNodes = st.hl ? new Set(st.hl.nodes) : new Set();

    edgesG.replaceChildren(...data.edges.filter((e) => {
      if (!visible.has(e.source) || !visible.has(e.target)) return false;
      return !(layered && (e.rel === "AFFECTED_BY" || e.rel.startsWith("ABOUT")));
    }).map((e) => {
      const a = pos.get(e.source), b = pos.get(e.target);
      const sa = nodeSize(nodeById.get(e.source)), sb = nodeSize(nodeById.get(e.target));
      let d;
      if (layered) {
        const x1 = a.x + sa.w / 2, x2 = b.x - sb.w / 2, mx = (x1 + x2) / 2;
        d = `M${x1} ${a.y} C${mx} ${a.y} ${mx} ${b.y} ${x2} ${b.y}`;
      } else d = `M${a.x} ${a.y} L${b.x} ${b.y}`;
      const dim = emph && !(emph.has(e.source) && emph.has(e.target));
      const title = e.rel === "RECOMMENDS" ? `${nodeById.get(e.source).label} → ${nodeById.get(e.target).label}: ${ACTION_LABELS[e.action] || e.action} (${e.chunk_id})` : e.rel.replace(/_/g, " ").toLowerCase();
      return s("path", { class: `edge ${edgeClass(e)}${dim ? " dim-out" : ""}`, d }, s("title", null, title));
    }));

    nodesG.replaceChildren(...data.nodes.filter((n) => visible.has(n.id)).map((n) => {
      const p = pos.get(n.id);
      const dim = emph && !emph.has(n.id);
      const g = s("g", {
        class: `node ${n.type}${st.selected === n.id ? " selected" : ""}${pathNodes.has(n.id) ? " path" : ""}${dim ? " dim-out" : ""}`,
        transform: `translate(${p.x} ${p.y})`, tabindex: 0, role: "button", "aria-label": `${TYPE_LABEL[n.type]} ${n.label}`,
      }, shapeFor(n), n.type === "chunk" ? s("text", { y: 16 }, n.label.split("#").pop()) : s("text", null, n.label));
      g.addEventListener("pointerdown", (ev) => startNodeDrag(ev, n));
      g.addEventListener("mouseenter", (ev) => tip.show(tooltipFor(n), ev));
      g.addEventListener("mousemove", (ev) => tip.show(tooltipFor(n), ev));
      g.addEventListener("mouseleave", () => tip.hide());
      g.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); select(n.id); } });
      return g;
    }));

    if (st.hl) {
      hlBar.classList.remove("hidden");
      hlBar.replaceChildren(h("div", null, h("b", null, `Highlighting the path for ${st.hl.label}`), "From the Pipeline trace. Select a node or search to override."),
        h("button", { class: "ghost", type: "button", style: { marginLeft: "auto" }, onclick: () => { st.hl = null; draw(); } }, "Clear"));
    } else hlBar.classList.add("hidden");
    applyView();
  }

  function tooltipFor(n) {
    const lines = [h("b", null, n.label), h("div", { class: "muted" }, TYPE_LABEL[n.type] + (n.gene && n.type !== "gene" ? ` · ${n.gene}` : ""))];
    if (n.type === "chunk") lines.push(h("div", null, `${n.ctype}${n.heading ? ` · ${n.heading}` : ""}`));
    const recs = (adj.get(n.id) || []).filter((a) => a.edge.rel === "RECOMMENDS");
    for (const a of recs.slice(0, 5)) lines.push(h("div", null, `${a.dir === "out" ? "→" : "←"} ${nodeById.get(a.other).label}: ${ACTION_LABELS[a.edge.action] || a.edge.action}`));
    return lines;
  }

  // ---- pan / zoom / drag
  function applyView() { vp.setAttribute("transform", `translate(${view.x} ${view.y}) scale(${view.k})`); }
  function fit() {
    if (!data) return;
    const vis = data.nodes.filter((n) => !st.hidden.has(n.type) && pos.has(n.id));
    if (!vis.length) return;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const n of vis) {
      const p = pos.get(n.id), sz = nodeSize(n);
      x0 = Math.min(x0, p.x - sz.w / 2); x1 = Math.max(x1, p.x + sz.w / 2);
      y0 = Math.min(y0, p.y - sz.h / 2 - 12); y1 = Math.max(y1, p.y + sz.h / 2 + 12);
    }
    const r = svg.getBoundingClientRect(), W = r.width || 800, H = r.height || 600;
    view.k = Math.min((W - 60) / (x1 - x0), (H - 60) / (y1 - y0), 1.4);
    view.x = (W - (x1 - x0) * view.k) / 2 - x0 * view.k;
    view.y = (H - (y1 - y0) * view.k) / 2 - y0 * view.k;
    applyView();
  }
  function zoomAt(mx, my, factor) {
    const k2 = Math.max(0.25, Math.min(3, view.k * factor));
    view.x = mx - (mx - view.x) * (k2 / view.k);
    view.y = my - (my - view.y) * (k2 / view.k);
    view.k = k2;
    applyView();
  }
  function zoomBy(f) { const r = svg.getBoundingClientRect(); zoomAt(r.width / 2, r.height / 2, f); }
  svg.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    const r = svg.getBoundingClientRect();
    zoomAt(ev.clientX - r.left, ev.clientY - r.top, Math.exp(-ev.deltaY * 0.0015));
  }, { passive: false });

  svg.addEventListener("pointerdown", (ev) => {
    if (ev.target !== svg) return;
    const start = { x: ev.clientX, y: ev.clientY, vx: view.x, vy: view.y };
    let moved = false;
    svg.classList.add("panning");
    svg.setPointerCapture(ev.pointerId);
    const move = (e) => {
      if (Math.hypot(e.clientX - start.x, e.clientY - start.y) > 4) moved = true;
      view.x = start.vx + e.clientX - start.x;
      view.y = start.vy + e.clientY - start.y;
      applyView();
    };
    const up = () => {
      svg.classList.remove("panning");
      svg.removeEventListener("pointermove", move);
      svg.removeEventListener("pointerup", up);
      if (!moved && st.selected) { st.selected = null; renderPanel(); draw(); } // a plain background click clears selection
    };
    svg.addEventListener("pointermove", move);
    svg.addEventListener("pointerup", up);
  });

  function startNodeDrag(ev, n) {
    ev.stopPropagation();
    tip.hide();
    const start = { x: ev.clientX, y: ev.clientY };
    const origin = { ...pos.get(n.id) };
    let moved = false;
    const move = (e) => {
      const dx = (e.clientX - start.x) / view.k, dy = (e.clientY - start.y) / view.k;
      if (Math.hypot(e.clientX - start.x, e.clientY - start.y) > 4) moved = true;
      if (moved) { pos.set(n.id, { x: origin.x + dx, y: origin.y + dy }); draw(); }
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      if (!moved) select(n.id);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }

  // ---- panel + table
  function select(id) {
    st.selected = st.selected === id ? null : id;
    renderPanel();
    draw();
  }

  function linkBtn(id) {
    const n = nodeById.get(id);
    return h("button", { type: "button", onclick: () => { st.selected = id; renderPanel(); draw(); } }, n.label);
  }

  function renderPanel() {
    const n = st.selected && nodeById.get(st.selected);
    if (!n) {
      const counts = data ? ["gene", "phenotype", "drug", "chunk"].map((t) => [t, data.nodes.filter((x) => x.type === t).length]).filter(([, c]) => c) : [];
      panel.replaceChildren(h("h2", null, "Select a node"), h("p", { class: "sub" }, "Click a gene, phenotype or drug to see how it connects."),
        h("ul", null, counts.map(([t, c]) => h("li", null, `${c} × ${TYPE_LABEL[t].toLowerCase()}${c > 1 ? "s" : ""}`))));
      return;
    }
    const out = [h("h2", null, n.label), h("div", { class: "sub" }, `${TYPE_LABEL[n.type]}${n.gene && n.type !== "gene" ? ` · ${n.gene}` : ""}`)];
    const links = adj.get(n.id) || [];
    if (n.type === "gene") {
      const alleles = shared.meta?.genes.find((g) => g.symbol === n.label)?.alleles;
      if (alleles) out.push(h("h3", { class: "eyebrow" }, "Alleles in knowledge base"), h("div", { class: "kv" }, alleles.map((a) => h("span", { class: "tag" }, a))));
      out.push(h("h3", { class: "eyebrow" }, "Phenotypes"), h("ul", null, links.filter((l) => l.edge.rel === "HAS_PHENOTYPE").map((l) => h("li", null, linkBtn(l.other)))));
      out.push(h("h3", { class: "eyebrow" }, "Drugs"), h("ul", null, links.filter((l) => l.edge.rel === "AFFECTED_BY").map((l) => h("li", null, linkBtn(l.other)))));
    } else if (n.type === "phenotype" || n.type === "drug") {
      const recs = links.filter((l) => l.edge.rel === "RECOMMENDS");
      out.push(h("h3", { class: "eyebrow" }, "Recommendations"), recs.length
        ? h("ul", null, recs.map((l) => h("li", null, linkBtn(l.other), ` → ${ACTION_LABELS[l.edge.action] || l.edge.action}`, h("div", { class: "hint" }, l.edge.chunk_id))))
        : h("p", { class: "hint" }, "None in the corpus."));
      const genes = links.filter((l) => ["AFFECTED_BY"].includes(l.edge.rel) && l.dir === "out" || l.edge.rel === "HAS_PHENOTYPE" && l.dir === "in");
      if (genes.length) out.push(h("h3", { class: "eyebrow" }, "Gene"), h("ul", null, genes.map((l) => h("li", null, linkBtn(l.other)))));
    } else {
      out.push(h("div", { class: "kv" }, h("span", { class: "tag" }, n.ctype || "chunk")), h("p", { class: "snippet" }, n.snippet || ""),
        h("h3", { class: "eyebrow" }, "About"), h("ul", null, links.map((l) => h("li", null, linkBtn(l.other)))));
    }
    out.push(h("p", null, h("button", { class: "ghost", type: "button", onclick: () => { st.selected = null; renderPanel(); draw(); } }, "Clear selection")));
    panel.replaceChildren(...out);
  }

  function renderTable() {
    const rows = data.edges.filter((e) => e.rel === "RECOMMENDS").map((e) => {
      const ph = nodeById.get(e.source), dr = nodeById.get(e.target);
      return { gene: ph.gene, phenotype: ph.label, drug: dr.label, action: ACTION_LABELS[e.action] || e.action, chunk: e.chunk_id };
    }).sort((a, b) => (a.gene + a.drug + a.phenotype).localeCompare(b.gene + b.drug + b.phenotype));
    tableHost.replaceChildren(table([
      { key: "gene", label: "Gene" }, { key: "phenotype", label: "Phenotype" }, { key: "drug", label: "Drug" },
      { key: "action", label: "Recommendation" }, { key: "chunk", label: "Source chunk", render: (r) => h("code", null, r.chunk) },
    ], rows));
  }

  await load();
}
