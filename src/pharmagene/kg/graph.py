from __future__ import annotations

import networkx as nx

from ..schemas import Chunk
from .lexicon import Lexicon


class KnowledgeGraph:
    """Gene - phenotype - drug graph linking guideline chunks to the entities they describe.

    Edges: gene -HAS_PHENOTYPE-> phenotype -RECOMMENDS(action)-> drug, drug -AFFECTED_BY-> gene,
    and chunk -ABOUT_*-> gene/drug/phenotype. It is used to (a) expand retrieved evidence with the
    chunks that must accompany a recommendation (the matching phenotype recommendation and the
    genotype translation table), (b) answer "does the knowledge base cover this gene-drug pair"
    without touching the vector index, and (c) feed the graph explorer in the web UI.
    """

    def __init__(self, chunks: list[Chunk], lexicon: Lexicon):
        g = nx.MultiDiGraph()
        for symbol, gene in lexicon.genes.items():
            g.add_node(f"gene:{symbol}", kind="gene", label=symbol)
            for phenotype in gene.phenotypes:
                node = f"phenotype:{symbol}:{phenotype}"
                g.add_node(node, kind="phenotype", label=phenotype, gene=symbol)
                g.add_edge(f"gene:{symbol}", node, rel="HAS_PHENOTYPE")
        for name, drug in lexicon.drugs.items():
            g.add_node(f"drug:{name}", kind="drug", label=name)
            for symbol in drug.genes:
                g.add_edge(f"drug:{name}", f"gene:{symbol}", rel="AFFECTED_BY")

        self._order: dict[str, int] = {}
        self._chunks: dict[str, Chunk] = {}
        for i, chunk in enumerate(chunks):
            meta = chunk.metadata
            node = f"chunk:{chunk.id}"
            self._order[chunk.id] = i
            self._chunks[chunk.id] = chunk
            g.add_node(
                node, kind="chunk", label=chunk.id, ctype=meta.get("type"), heading=meta.get("heading"),
                gene=meta.get("gene"), snippet=chunk.text.split("\n", 1)[-1][:280],
            )
            if meta.get("gene"):
                g.add_edge(node, f"gene:{meta['gene']}", rel="ABOUT_GENE")
            for drug in meta.get("drugs", []):
                g.add_edge(node, f"drug:{drug}", rel="ABOUT_DRUG")
            if meta.get("phenotype") and meta.get("gene"):
                pheno = f"phenotype:{meta['gene']}:{meta['phenotype']}"
                g.add_edge(node, pheno, rel="ABOUT_PHENOTYPE")
                if meta.get("type") == "recommendation":
                    for drug in meta.get("drugs", []):
                        g.add_edge(pheno, f"drug:{drug}", rel="RECOMMENDS",
                                   action=meta.get("action"), chunk_id=chunk.id)
        self.g = g

    def _chunks_about(self, node: str) -> set[str]:
        if node not in self.g:
            return set()
        return {
            n[len("chunk:"):] for n in self.g.predecessors(node) if self.g.nodes[n].get("kind") == "chunk"
        }

    def covers(self, gene: str, drug: str) -> bool:
        return bool(self._chunks_about(f"gene:{gene}") & self._chunks_about(f"drug:{drug}"))

    def evidence_for(self, gene: str, drug: str, phenotype: str | None) -> list[str]:
        """Chunk ids needed to support a recommendation: matching recommendation + translation table."""
        base = self._chunks_about(f"gene:{gene}") & self._chunks_about(f"drug:{drug}")
        selected: set[str] = set()
        for cid in base:
            meta = self._chunks[cid].metadata
            if meta.get("type") == "translation":
                selected.add(cid)
            elif meta.get("type") == "recommendation" and phenotype and meta.get("phenotype") == phenotype:
                selected.add(cid)
        return sorted(selected, key=self._order.__getitem__)

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def view(self, include_chunks: bool = False) -> dict:
        """JSON-serialisable nodes/edges for the explorer. Chunk nodes are opt-in (they dominate the layout)."""
        nodes = []
        for node_id, data in self.g.nodes(data=True):
            if data["kind"] == "chunk" and not include_chunks:
                continue
            nodes.append({"id": node_id, "type": data["kind"], **{k: v for k, v in data.items() if k != "kind"}})
        keep = {n["id"] for n in nodes}
        edges = [
            {"source": u, "target": v, **d}
            for u, v, d in self.g.edges(data=True)
            if u in keep and v in keep
        ]
        return {"nodes": nodes, "edges": edges}
