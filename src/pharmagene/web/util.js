// Shared helpers. Every server string (including LLM output) is rendered with textContent, never innerHTML.

export const $ = (sel, el = document) => el.querySelector(sel);

export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style" && typeof v === "object") {
      for (const [prop, val] of Object.entries(v)) {
        if (prop.startsWith("--")) el.style.setProperty(prop, val); else el.style[prop] = val;
      }
    }
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  append(el, children);
  return el;
}

const SVG_NS = "http://www.w3.org/2000/svg";
export function s(tag, attrs, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v);
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
}

export const ACTION_LABELS = {
  standard: "Standard dosing",
  standard_monitor: "Standard dosing, monitor",
  reduce_dose: "Reduce dose",
  alternative_agent: "Use alternative agent",
  avoid: "Avoid",
};
export const ACTION_TONE = { standard: "ok", standard_monitor: "neutral", reduce_dose: "abstain", alternative_agent: "abstain", avoid: "avoid" };

export const settings = { mode: "hybrid", use_kg: true, rerank: false, key: "" };
export const retrievalOpts = () => ({ mode: settings.mode, use_kg: settings.use_kg, rerank: settings.rerank });

// State handed between views (e.g. Assistant -> Pipeline trace -> Knowledge graph).
export const shared = { traceText: null, highlight: null, meta: null };

export async function api(path, body) {
  const headers = settings.key ? { "X-API-Key": settings.key } : {};
  const opts = { headers };
  if (body !== undefined) {
    opts.method = "POST";
    opts.headers = { ...headers, "Content-Type": "application/json" };
    opts.body = JSON.stringify(body);
  }
  const resp = await fetch(path, opts);
  if (!resp.ok) {
    let detail = resp.statusText;
    try { detail = (await resp.json()).detail || detail; } catch (_) { /* keep statusText */ }
    if (resp.status === 401) detail = "This server requires an API key. Add it under Retrieval settings (the sliders icon).";
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return resp.json();
}

export function safeUrl(u) {
  try {
    const p = new URL(u);
    return p.protocol === "https:" || p.protocol === "http:" ? p.href : null;
  } catch (_) { return null; }
}

export const pct = (v) => (v == null ? "–" : `${(v * 100).toFixed(1)}%`);
export const fixed = (v, n = 3) => (v == null ? "–" : Number(v).toFixed(n));

export function errorBanner(err) {
  return h("div", { class: "banner error", role: "alert" }, h("div", null, h("b", null, "Request failed"), err.message || String(err)));
}

// ---- tooltip (one shared element) ----
const tipEl = () => document.getElementById("tooltip");
export const tip = {
  show(content, evt) {
    const el = tipEl();
    el.replaceChildren(...[].concat(content));
    el.hidden = false;
    const pad = 14, r = el.getBoundingClientRect();
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - pad;
    if (y + r.height > window.innerHeight - 8) y = evt.clientY - r.height - pad;
    el.style.left = `${Math.max(8, x)}px`;
    el.style.top = `${Math.max(8, y)}px`;
  },
  hide() { tipEl().hidden = true; },
};

export function table(columns, rows, { numeric = [] } = {}) {
  const head = h("tr", null, columns.map((c) => h("th", { class: numeric.includes(c.key) ? "num" : "" }, c.label)));
  const body = rows.map((r) => h("tr", null, columns.map((c) => {
    const v = c.render ? c.render(r) : r[c.key];
    return h("td", { class: numeric.includes(c.key) ? "num" : "" }, v == null ? "–" : v);
  })));
  return h("div", { class: "table-wrap" }, h("table", null, h("thead", null, head), h("tbody", null, body)));
}
