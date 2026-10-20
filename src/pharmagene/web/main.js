import { $, api, h, settings, shared } from "./util.js";
import * as assistant from "./assistant.js";
import * as trace from "./trace.js";
import * as graph from "./graph.js";
import * as evaluation from "./evaluation.js";
import * as corpus from "./corpus.js";

const ROUTES = { assistant, trace, graph, evaluation, corpus };

function currentRoute() {
  const name = location.hash.replace(/^#\/?/, "").split(/[/?]/)[0];
  return ROUTES[name] ? name : "assistant";
}

async function render() {
  const name = currentRoute();
  for (const a of document.querySelectorAll(".nav a")) {
    const on = a.dataset.route === name;
    a.classList.toggle("active", on);
    if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  }
  const view = $("#view");
  view.replaceChildren();
  try {
    await ROUTES[name].mount(view);
  } catch (err) {
    view.replaceChildren(h("div", { class: "banner error", role: "alert" }, h("div", null, h("b", null, "Could not load this view"), err.message)));
  }
  window.scrollTo(0, 0);
}

function initTheme() {
  const root = document.documentElement;
  try { const saved = localStorage.getItem("pg-theme"); if (saved) root.dataset.theme = saved; } catch (_) { /* storage blocked */ }
  $("#theme-btn").addEventListener("click", () => {
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("pg-theme", root.dataset.theme); } catch (_) { /* ignore */ }
  });
}

function initSettings() {
  const dlg = $("#settings");
  $("#opt-mode").value = settings.mode;
  $("#opt-kg").checked = settings.use_kg;
  $("#settings-btn").addEventListener("click", () => dlg.showModal());
  $("#opt-mode").addEventListener("change", (e) => { settings.mode = e.target.value; });
  $("#opt-kg").addEventListener("change", (e) => { settings.use_kg = e.target.checked; });
  $("#opt-rerank").addEventListener("change", (e) => { settings.rerank = e.target.checked; });
  $("#opt-key").addEventListener("input", (e) => { settings.key = e.target.value.trim(); });
}

async function initStatus() {
  const status = $("#status");
  try {
    const [health, meta] = await Promise.all([fetch("/health").then((r) => r.json()), fetch("/meta").then((r) => r.json())]);
    shared.meta = meta;
    status.replaceChildren(h("span", { class: "dot" }), h("b", null, health.llm), ` · ${health.embedder} · ${health.chunks} chunks`);
    if (!meta.reranker) {
      $("#opt-rerank").disabled = true;
      $("#rerank-note").textContent = "(unavailable: install sentence-transformers)";
    }
  } catch (_) {
    status.replaceChildren(h("span", { class: "dot off" }), "Server unreachable");
  }
}

initTheme();
initSettings();
window.addEventListener("hashchange", render);
initStatus().then(render);
