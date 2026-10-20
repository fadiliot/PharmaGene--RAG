// Pure layout functions (no DOM) so they can be unit-tested in Node.

export const ROW_H = 42;
export const COL_X = { gene: 0, phenotype: 300, drug: 620 };

export function nodeSize(n) {
  if (n.type === "chunk") return { w: 14, h: 14 };
  return { w: Math.max(74, n.label.length * 7 + 28), h: 30 };
}

function mulberry32(seed) {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Three columns (gene | phenotype | drug); each gene owns a horizontal band holding its phenotypes and drugs. */
export function layeredLayout(nodes, edges) {
  const pos = new Map();
  const drugGene = new Map();
  for (const e of edges) if (e.rel === "AFFECTED_BY") drugGene.set(e.source, e.target);
  let cursor = 0;
  for (const gene of nodes.filter((n) => n.type === "gene")) {
    const phenotypes = nodes.filter((n) => n.type === "phenotype" && n.gene === gene.label);
    const drugs = nodes.filter((n) => n.type === "drug" && drugGene.get(n.id) === gene.id);
    const height = Math.max(phenotypes.length, drugs.length, 1) * ROW_H;
    const place = (list, x) => list.forEach((n, i) => pos.set(n.id, { x, y: cursor + (height / list.length) * (i + 0.5) }));
    place(phenotypes, COL_X.phenotype);
    place(drugs, COL_X.drug);
    pos.set(gene.id, { x: COL_X.gene, y: cursor + height / 2 });
    cursor += height + 26;
  }
  // anything not placed above (e.g. a drug without a gene) goes below the bands
  let extra = 0;
  for (const n of nodes) {
    if (n.type !== "chunk" && !pos.has(n.id)) pos.set(n.id, { x: COL_X[n.type] ?? 0, y: cursor + 40 * extra++ });
  }
  return pos;
}

/** Fruchterman-Reingold style simulation with box collision, seeded so the layout is stable between loads. */
export function forceLayout(nodes, edges, seed = 7) {
  const rand = mulberry32(seed);
  const n = nodes.length;
  const P = nodes.map((nd, i) => {
    const a = (i / Math.max(n, 1)) * Math.PI * 2;
    const r = 260 + rand() * 120;
    return { id: nd.id, x: Math.cos(a) * r, y: Math.sin(a) * r, vx: 0, vy: 0, size: nodeSize(nd) };
  });
  const index = new Map(P.map((p, i) => [p.id, i]));
  const links = edges
    .filter((e) => index.has(e.source) && index.has(e.target))
    .map((e) => [index.get(e.source), index.get(e.target), e.rel.startsWith("ABOUT") ? 70 : 150]);

  const iterations = 420;
  for (let it = 0; it < iterations; it++) {
    const temp = 1 - it / iterations;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let dx = P[i].x - P[j].x, dy = P[i].y - P[j].y;
        const d2 = dx * dx + dy * dy + 0.01, d = Math.sqrt(d2);
        const f = 22000 / d2;
        dx = (dx / d) * f; dy = (dy / d) * f;
        P[i].vx += dx; P[i].vy += dy; P[j].vx -= dx; P[j].vy -= dy;
        // collision: push apart when the boxes (plus margin) overlap
        const ox = (P[i].size.w + P[j].size.w) / 2 + 10 - Math.abs(P[i].x - P[j].x);
        const oy = (P[i].size.h + P[j].size.h) / 2 + 8 - Math.abs(P[i].y - P[j].y);
        if (ox > 0 && oy > 0) {
          if (ox < oy) { const s = Math.sign(P[i].x - P[j].x) || 1; P[i].vx += s * ox * 0.5; P[j].vx -= s * ox * 0.5; }
          else { const s = Math.sign(P[i].y - P[j].y) || 1; P[i].vy += s * oy * 0.5; P[j].vy -= s * oy * 0.5; }
        }
      }
    }
    for (const [a, b, len] of links) {
      const dx = P[b].x - P[a].x, dy = P[b].y - P[a].y;
      const d = Math.sqrt(dx * dx + dy * dy) || 1;
      const f = (d - len) * 0.05;
      P[a].vx += (dx / d) * f; P[a].vy += (dy / d) * f;
      P[b].vx -= (dx / d) * f; P[b].vy -= (dy / d) * f;
    }
    const maxStep = 30 * temp + 1;
    for (const p of P) {
      p.vx -= p.x * 0.004; p.vy -= p.y * 0.004;
      const sp = Math.hypot(p.vx, p.vy) || 1, cap = Math.min(sp, maxStep);
      p.x += (p.vx / sp) * cap; p.y += (p.vy / sp) * cap;
      p.vx *= 0.5; p.vy *= 0.5;
    }
  }
  // final relaxation: resolve any remaining box overlaps without moving things far
  for (let pass = 0; pass < 80; pass++) {
    let moved = false;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const ox = (P[i].size.w + P[j].size.w) / 2 + 8 - Math.abs(P[i].x - P[j].x);
        const oy = (P[i].size.h + P[j].size.h) / 2 + 6 - Math.abs(P[i].y - P[j].y);
        if (ox > 0 && oy > 0) {
          moved = true;
          if (oy <= ox) { const s = Math.sign(P[i].y - P[j].y) || 1; P[i].y += (s * oy) / 2; P[j].y -= (s * oy) / 2; }
          else { const s = Math.sign(P[i].x - P[j].x) || 1; P[i].x += (s * ox) / 2; P[j].x -= (s * ox) / 2; }
        }
      }
    }
    if (!moved) break;
  }
  return new Map(P.map((p) => [p.id, { x: p.x, y: p.y }]));
}

export function overlaps(nodes, pos) {
  const boxes = nodes.map((n) => ({ id: n.id, ...pos.get(n.id), ...nodeSize(n) }));
  let count = 0;
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      const a = boxes[i], b = boxes[j];
      if (Math.abs(a.x - b.x) < (a.w + b.w) / 2 && Math.abs(a.y - b.y) < (a.h + b.h) / 2) count++;
    }
  }
  return count;
}
