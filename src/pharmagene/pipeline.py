from __future__ import annotations

import re
import time
from dataclasses import replace

import numpy as np

from .audit import AuditLog
from .config import Settings
from .confidence import score_confidence
from .generation import EvidenceItem, GenOutput, extractive_answer, generate_grounded, verify_generation
from .ingestion import chunk_documents, load_documents
from .kg import GenotypeError, KnowledgeGraph, Lexicon, parse_genotypes, resolve
from .kg.genotype import _genotype_in
from .llm import LLMClient, make_llm
from .retrieval import (
    HybridRetriever,
    RetrievalConfig,
    load_or_build_vectors,
    make_embedder,
    make_reranker,
)
from .schemas import Chunk, Claim, DrugRecommendation, EvidenceRef, PGxReport

_CITE_TAGS = re.compile(r"\s*\[[A-Z]+\d+\]")


class PharmaGenePipeline:
    """Algorithm 1: genotype G, medications M, context C -> evidence-grounded recommendation R.

    Stages: KG genotype translation -> metadata-filtered hybrid retrieval (+ optional cross-encoder
    rerank) -> evidence-constrained generation -> citation/support verification -> confidence
    scoring -> abstain and flag for clinician review if confidence < tau.
    """

    def __init__(self, settings: Settings, lexicon: Lexicon, chunks: list[Chunk], retriever: HybridRetriever,
                 graph: KnowledgeGraph, llm: LLMClient | None, audit: AuditLog | None = None):
        self.settings = settings
        self.lexicon = lexicon
        self.chunks = chunks
        self.retriever = retriever
        self.graph = graph
        self.llm = llm
        self.audit = audit or AuditLog(settings.audit_log, settings.audit_log_raw)
        self._reranker_tried = retriever.reranker is not None

    @classmethod
    def from_settings(cls, settings: Settings | None = None, *, llm: LLMClient | None | str = "auto",
                      embedder=None, reranker=None) -> "PharmaGenePipeline":
        settings = settings or Settings()
        lexicon = Lexicon.from_dir(settings.kb_dir)
        chunks = chunk_documents(load_documents(settings.corpus_dir), lexicon)
        if not chunks:
            raise RuntimeError(f"No guideline chunks found in {settings.corpus_dir}.")
        embedder = embedder or make_embedder(settings)
        vectors = load_or_build_vectors(chunks, embedder, settings.index_dir)
        retriever = HybridRetriever(chunks, embedder, vectors, settings.vector_backend, reranker)
        return cls(settings, lexicon, chunks, retriever, KnowledgeGraph(chunks, lexicon),
                   make_llm(settings) if llm == "auto" else llm)

    # ------------------------------------------------------------------ public API

    def reranker_available(self) -> bool:
        if self.retriever.reranker is None and not self._reranker_tried:
            self._reranker_tried = True
            self.retriever.reranker = make_reranker(self.settings)
        return self.retriever.reranker is not None

    def default_config(self, **overrides) -> RetrievalConfig:
        base = RetrievalConfig(
            alpha=self.settings.alpha, top_k=self.settings.top_k, candidate_k=self.settings.candidate_k
        )
        return replace(base, **overrides)

    def ask(self, text: str, cfg: RetrievalConfig | None = None) -> PGxReport:
        """Free-text entry point: parses genotype(s) and medication(s) with the lexicon."""
        drugs = self.lexicon.find_drugs(text)
        if not drugs:
            rec = DrugRecommendation(
                drug="(none recognised)", decision="abstain", llm_backend=self._backend_name(),
                abstain_reason="No supported medication was found in the query. Supported: "
                + ", ".join(self.lexicon.drugs) + ".",
            )
            return PGxReport(query=text, recommendations=[rec])
        return self.recommend(parse_genotypes(text, self.lexicon) or text, drugs, context=text, cfg=cfg, query=text)

    def recommend(self, genotype: str | dict[str, str], medications: list[str], context: str = "",
                  cfg: RetrievalConfig | None = None, query: str | None = None) -> PGxReport:
        cfg = cfg or self.default_config()
        if isinstance(genotype, dict):
            genotypes, bare = genotype, None
        else:
            genotypes = parse_genotypes(genotype, self.lexicon)
            bare = None if genotypes else _genotype_in(genotype)
        recs = [self._recommend_one(m, genotypes, bare, context, cfg, query) for m in medications]
        return PGxReport(query=query, recommendations=recs)

    # ------------------------------------------------------------------ internals

    def _backend_name(self) -> str:
        return self.llm.name if self.llm else "none (extractive)"

    def _abstain(self, rec: DrugRecommendation, reason: str, t0: float) -> DrugRecommendation:
        rec.decision, rec.abstain_reason = "abstain", reason
        rec.action, rec.summary, rec.claims = None, "", []
        rec.requires_clinician_review = True
        rec.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return rec

    def _recommend_one(self, drug_name: str, genotypes: dict[str, str], bare: str | None, context: str,
                       cfg: RetrievalConfig, query: str | None) -> DrugRecommendation:
        t0 = time.perf_counter()
        drug = self.lexicon.canonical_drug(drug_name)
        rec = DrugRecommendation(drug=drug or drug_name, decision="abstain", llm_backend=self._backend_name())
        result = self._run(rec, drug, genotypes, bare, context, cfg, query, t0)
        self._audit(result, cfg, context, query)
        return result

    def _run(self, rec, drug, genotypes, bare, context, cfg, query, t0) -> DrugRecommendation:
        if drug is None:
            return self._abstain(rec, f"'{rec.drug}' is not covered by the knowledge base.", t0)

        genes = self.lexicon.drugs[drug].genes
        gene = next((g for g in genes if g in genotypes), None)
        if gene:
            genotype = genotypes[gene]
        elif bare:
            gene, genotype = genes[0], bare
        else:
            return self._abstain(rec, f"No {'/'.join(genes)} genotype was provided for {drug}.", t0)
        rec.gene, rec.genotype = gene, genotype
        if not self.graph.covers(gene, drug):
            return self._abstain(rec, f"No guideline evidence for {gene}-{drug} is indexed.", t0)

        resolution = None
        phenotype = None
        if cfg.use_kg:
            try:
                resolution = resolve(self.lexicon.genes[gene], genotype)
            except GenotypeError as exc:
                return self._abstain(rec, str(exc), t0)
            if resolution.ambiguous:
                return self._abstain(rec, resolution.explanation, t0)
            rec.genotype, phenotype = resolution.genotype, resolution.phenotype
            rec.phenotype = phenotype

        if cfg.use_kg:
            retrieval_query = f"{gene} {resolution.genotype} {phenotype} {drug} {context}".strip()
            filters = {"gene": gene, "drug": drug, "phenotype": phenotype}
        else:
            retrieval_query = query or f"{genotype} {drug} {context}".strip()
            filters = None
        if cfg.rerank and not self.reranker_available():
            cfg = replace(cfg, rerank=False)
            rec.notes.append("Reranker unavailable (sentence-transformers not installed); reranking skipped.")
        retrieved = self.retriever.retrieve(retrieval_query, cfg, filters)

        items: list[EvidenceItem] = []
        if resolution is not None:
            items.append(EvidenceItem("KG1", resolution.explanation,
                                      {"type": "kg_translation", "source": "PharmaGene gene definitions"}))
        seen: set[str] = set()
        for r in retrieved:
            seen.add(r.chunk.id)
            items.append(EvidenceItem(f"C{len(seen)}", r.chunk.text, r.chunk.metadata, r.chunk.id))
        if cfg.use_kg:
            for cid in self.graph.evidence_for(gene, drug, phenotype):
                if cid not in seen:
                    chunk = self.graph.get(cid)
                    seen.add(cid)
                    items.append(EvidenceItem(f"C{len(seen)}", chunk.text, chunk.metadata, cid))
                    rec.notes.append(f"Graph expansion added {cid}.")
        rec.notes.append("retrieved: " + ", ".join(r.chunk.id for r in retrieved))

        gen, used_fallback, llm_issues, final_issues = self._generate(
            rec, items, drug, phenotype, gene, genotype, context
        )
        rec.used_fallback, rec.verification_issues = used_fallback, llm_issues + final_issues
        if gen is None:
            return self._abstain(rec, "No verified recommendation could be produced from the retrieved evidence.", t0)
        if gen.abstain:
            return self._abstain(rec, gen.abstain_reason or "The model abstained: evidence insufficient.", t0)
        if final_issues:  # even the extractive answer failed verification: never show it
            return self._abstain(rec, "The final answer failed verification: " + "; ".join(final_issues), t0)

        by_handle = {e.handle: e for e in items}
        cited = [h for h in dict.fromkeys(h for c in gen.claims for h in c.citations) if h in by_handle]
        chunk_cited = [by_handle[h] for h in cited if by_handle[h].chunk_id]
        embedder = self.retriever.embedder
        query_vec = embedder.embed([retrieval_query], kind="query")[0]
        claim_vecs = []
        claim_embeds = embedder.embed([c.text for c in gen.claims], kind="query")
        for claim, vec in zip(gen.claims, claim_embeds):
            claim_vecs.append((vec, [self.retriever.chunk_vector(by_handle[h].chunk_id)
                                     for h in claim.citations if h in by_handle and by_handle[h].chunk_id]))
        verification = 0.7 if used_fallback else 1.0
        rec.confidence = score_confidence(
            query_vec=query_vec,
            claim_vecs=claim_vecs,
            cited_vecs=[self.retriever.chunk_vector(e.chunk_id) for e in chunk_cited],
            cited_meta=[e.metadata for e in chunk_cited],
            gene=gene, drug=drug, phenotype=phenotype,
            verification=verification, tau=self.settings.tau,
        )

        rec.claims = [Claim(text=c.text, citations=c.citations) for c in gen.claims]
        rec.action = gen.action
        rec.phenotype = rec.phenotype or gen.phenotype
        rec.summary = " ".join(f"{_CITE_TAGS.sub('', c.text)} [{', '.join(c.citations)}]" for c in gen.claims)
        rec.evidence = [
            EvidenceRef(
                handle=e.handle, chunk_id=e.chunk_id or e.handle,
                source=e.metadata.get("source"), citation=e.metadata.get("citation"),
                source_url=e.metadata.get("source_url"), type=e.metadata.get("type"),
                snippet=e.text.split("\n", 1)[-1][:300],
            )
            for e in (by_handle[h] for h in cited)
        ]
        rec.decision, rec.abstain_reason = "recommend", None
        rec.requires_clinician_review = used_fallback or rec.confidence.score < self.settings.tau + 0.2
        rec.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        if rec.confidence.score < self.settings.tau:
            tau = self.settings.tau
            return self._abstain(
                rec, f"Confidence {rec.confidence.score:.2f} is below the threshold {tau:.2f}; "
                     "flagged for clinician review.", t0,
            )
        return rec

    def _generate(self, rec, items, drug, phenotype, gene, genotype, context
                  ) -> tuple[GenOutput | None, bool, list[str], list[str]]:
        """Returns (answer, used_fallback, issues found in rejected LLM drafts, issues in the final answer)."""
        by_handle = {e.handle: e for e in items}
        kwargs = dict(lexicon=self.lexicon, drug=drug, phenotype=phenotype,
                      threshold=self.settings.support_threshold)
        if self.llm is None:
            gen = extractive_answer(items, drug, phenotype)
            if gen is None:
                return None, False, [], []
            return gen, False, [], verify_generation(gen, by_handle, **kwargs).issues

        patient = (
            f"Gene: {gene}\nGenotype: {genotype}\nMedication: {drug}\n"
            f"Resolved phenotype: {phenotype or 'not resolved - determine only from the evidence'}\n"
            f"Clinical context: {context or 'not provided'}"
        )
        question = f"What is the recommended management of {drug} for this patient?"
        feedback: list[str] | None = None
        issues: list[str] = []
        for _ in range(1 + self.settings.max_generation_retries):
            try:
                out = generate_grounded(self.llm, patient, question, items, feedback)
            except Exception as exc:  # network/API failure must never produce an unverified answer
                rec.notes.append(f"LLM call failed ({type(exc).__name__}); used extractive fallback.")
                break
            if out is None:
                feedback = issues = ["The output was not valid JSON in the required format."]
                continue
            if out.abstain:
                return out, False, [], []
            result = verify_generation(out, by_handle, **kwargs)
            if result.passed:
                return out, False, [], []
            feedback = issues = result.issues
        rec.notes.append("LLM output failed verification; replaced with extractive answer from the guideline.")
        gen = extractive_answer(items, drug, phenotype)
        if gen is None:
            return None, True, issues, []
        return gen, True, issues, verify_generation(gen, by_handle, **kwargs).issues

    def _audit(self, rec: DrugRecommendation, cfg: RetrievalConfig, context: str, query: str | None) -> None:
        self.audit.write(
            {
                "request": AuditLog.digest(rec.gene or "", rec.genotype or "", rec.drug, context),
                "drug": rec.drug, "gene": rec.gene, "phenotype": rec.phenotype,
                "decision": rec.decision, "action": rec.action,
                "abstain_reason": rec.abstain_reason,
                "confidence": rec.confidence.score if rec.confidence else None,
                "evidence": [e.chunk_id for e in rec.evidence],
                "llm": rec.llm_backend, "retrieval": cfg.label, "used_fallback": rec.used_fallback,
            },
            raw={"genotype": rec.genotype, "context": context, "query": query},
        )
