# PharmaGene-RAG

Evidence-grounded pharmacogenomic decision support. Implements the pipeline from *PharmaGene-RAG:
Reducing Hallucinations in Personalized Drug Recommendations* (hybrid retrieval, cross-encoder
reranking, evidence-constrained generation, confidence-based abstention), plus the knowledge-graph
and metadata-filtering layer described in the project write-up.

> Research prototype. Not a medical device. The bundled corpus is a condensed, **paraphrased and
> unverified** seed written for development; check every recommendation against the current source
> guideline before using it for anything beyond experimentation.

## Pipeline

```
genotype G + medications M + context C
        |
        v
 1. KG translation     deterministic genotype -> phenotype (activity score / allele-function rules)
        |              refuses incomplete, unknown or ambiguous genotypes (e.g. *1xN/*4)
        v
 2. Hybrid retrieval   S = a*S_dense + (1-a)*S_BM25  (min-max normalised, allele-aware BM25 tokens)
        |              + metadata filter (gene, drug, phenotype) + optional cross-encoder rerank
        v
 3. Graph expansion    adds the matching phenotype recommendation + translation table if missing
        |
        v
 4. Constrained gen    LLM sees ONLY the cited evidence, answers as JSON claims with citations
        |              (Gemini, or a local Ollama model for on-prem)
        v
 5. Verification       every claim cited; per-sentence lexical support; no new numbers, drugs or
        |              phenotypes; action must match the cited recommendation's action
        |              fail -> one retry with feedback -> extractive fallback (verbatim guideline text)
        v
 6. Confidence         relevance, claim/evidence alignment, entity coverage, verification
        |
        v
   confidence < tau, or any hard refusal  ->  ABSTAIN and flag for clinician review
```

Why the KG step exists: dense embeddings barely separate `*1/*1` from `*1xN/*1` (almost identical
text), but they map to Normal and Ultrarapid metabolizers. Translation is therefore done with
explicit rules, and the result is used for the retrieval query, the metadata filter and the
verifier. Run with `--no-kg` to get the paper-style retrieval-only behaviour for ablations.

## Quick start

```bash
pip install -e ".[dev]"            # core: numpy, networkx, pydantic, python-dotenv
pytest                             # 95 tests, fully offline

# offline smoke test (hashing embedder, no LLM, answers extracted verbatim from the guideline)
pharmagene ask "CYP2C19 *2/*2 prescribed clopidogrel after PCI" --no-llm --embeddings hashing
```

### With Gemini

```bash
pip install -e ".[gemini]"
cp .env.example .env               # then put GEMINI_API_KEY=... in .env (never commit it)
pharmagene doctor                  # shows the resolved LLM / embedding backends
pharmagene ask "Patient with CYP2C19 *2/*2 genotype prescribed clopidogrel following PCI."
```

`GEMINI_MODEL` defaults to `gemini-2.5-flash`; change it in `.env` to whichever model you have access
to. **Gemini sends prompts (including genotypes) to Google.** Use synthetic or de-identified data
with it. The paper's on-prem claim only holds with local components:

```bash
pip install -e ".[local]"          # sentence-transformers + faiss-cpu (embeddings, reranker)
# in .env: PHARMAGENE_LLM=ollama  OLLAMA_MODEL=llama3.1  PHARMAGENE_EMBEDDINGS=sbert  VECTOR_BACKEND=faiss
```

## Web UI

```bash
pharmagene serve          # then open http://127.0.0.1:8000
```

A dependency-free frontend (`src/pharmagene/web/`) served by the FastAPI app: free-text and structured
queries, per-drug cards with action, phenotype, clickable citations, evidence, confidence breakdown against
the abstention threshold, and clear abstain / clinician-review states. No CDN or external fonts, so it works
air-gapped.

## Commands

| Command | Purpose |
|---|---|
| `pharmagene ask "<text>"` | free-text query; parses genotypes and drugs |
| `pharmagene recommend --genotype "CYP2C19 *2/*2; DPYD *1/*2A" --drug clopidogrel --drug capecitabine` | structured query |
| `pharmagene build-eval` | generate scenarios (answerable, paper cases, abstention cases) |
| `pharmagene eval-retrieval` | Precision@5, Recall@5, MRR, nDCG@10 for dense / BM25 / hybrid / hybrid+rerank, with and without KG |
| `pharmagene eval-e2e` | `llm_only` vs `rag_paper` vs `rag_kg`: action accuracy, macro P/R/F1, hallucination rate, abstention recall/precision; also writes `results/ratings_*.csv` for human Likert scoring |
| `pharmagene calibrate` | sweeps the abstention threshold tau |
| `pharmagene serve` | FastAPI: `POST /recommend`, `POST /ask`, `GET /health` (set `PHARMAGENE_API_KEYS` for auth) |

`--mode dense|bm25|hybrid`, `--rerank`, `--no-kg`, `--no-llm` and `--embeddings` work on query and eval commands.
`pharmagene.evaluation.cohens_kappa` computes inter-rater agreement for the exported ratings.

## Adding guidelines

Drop `.md`, `.txt` or `.pdf` (needs `pypdf`) files into `data/corpus/`. Markdown with front matter and
per-section comments gives exact metadata (see `data/corpus/cyp2c19_clopidogrel.md`); anything without
it is chunked by heading and tagged with the lexicon. Add new genes and drugs in `data/kb/genes.json`
and `data/kb/drugs.json`. PharmGKB, FDA label and KNMP content each have their own licence terms; this
repo ships none of it.

## Deviations from the paper, and what is not reproduced

- **LLM:** the paper uses Llama 3.1; this repo defaults to Gemini and supports Ollama (Llama 3.1) for local runs.
- **Sources:** the paper cites CPIC, PharmGKB, FDA and KNMP. The seed corpus covers only CPIC-style summaries of
  CYP2C19-clopidogrel, SLCO1B1-simvastatin, TPMT-thiopurines, DPYD-fluoropyrimidines and CYP2D6-codeine.
- **Unspecified in the paper, chosen here:** score normalisation (min-max), alpha = 0.5, the confidence formula
  and its weights, tau = 0.5. All are configurable and tau should be calibrated on your own data.
- **No numbers are carried over.** Table I of the paper, the resume figures and the slide claims are not
  produced by this code. Numbers from `eval-*` depend on the embedder and the corpus. Evaluation labels here
  are derived from the same corpus and rules the system uses, so they test plumbing and robustness, not clinical
  validity; run with real embeddings and add clinician-written scenarios before quoting anything. Only cite
  figures you have measured.
- **Verifier limits:** support checking is lexical. It catches added numbers, drugs, phenotypes and unsupported
  sentences, but it cannot detect a wrong guideline or a paraphrase that distorts meaning.
- **Security:** the audit log omits raw genotypes by default. Encryption at rest, network isolation and RBAC
  beyond an API key are deployment concerns this repo does not implement.

## Layout

```
src/pharmagene/
  kg/          gene definitions, genotype resolver, lexicon, gene-drug-phenotype graph
  ingestion/   loaders (md/txt/pdf) and section-aware chunker with metadata
  retrieval/   embedders (sbert | gemini | hashing), BM25, dense index (numpy | faiss), hybrid, reranker
  llm/         Gemini, Ollama clients
  generation/  prompts, evidence-constrained generator, verifier
  confidence.py, audit.py, pipeline.py, cli.py, api.py
  evaluation/  scenarios, retrieval metrics, baselines, ablation + end-to-end runner
data/kb/       genes.json, drugs.json        data/corpus/   guideline documents
data/eval/     generated scenarios           tests/         pytest suite (offline)
```
