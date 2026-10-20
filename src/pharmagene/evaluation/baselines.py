from __future__ import annotations

from ..generation.generator import GenOutput
from ..llm import LLMClient, parse_json
from ..schemas import ACTIONS

SYSTEM = (
    "You are a pharmacogenomics assistant. Answer from your own knowledge. "
    "If you cannot give a reliable recommendation, abstain."
)


def llm_only(llm: LLMClient, query: str) -> GenOutput:
    """Unmodified-LLM baseline: no retrieval, no knowledge graph, no verification."""
    prompt = (
        f"QUESTION:\n{query}\n\nReturn JSON with keys: abstain (boolean), abstain_reason (string|null), "
        f'phenotype (string|null), action (one of {list(ACTIONS)} | null), claims (list of {{"text": string}}). '
        "Action meanings: standard = standard dosing; standard_monitor = standard dosing with monitoring; "
        "reduce_dose = lower dose; alternative_agent = prefer another drug; avoid = do not use."
    )
    try:
        return GenOutput.model_validate(parse_json(llm.generate(SYSTEM, prompt, json_mode=True)))
    except Exception:
        return GenOutput(abstain=True, abstain_reason="unparseable model output")
