from dataclasses import replace

import pytest

from pharmagene.config import Settings
from pharmagene.pipeline import PharmaGenePipeline
from pharmagene.retrieval import HashingEmbedder


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory):
    """Offline pipeline: hashing embedder, no LLM (extractive answers), isolated index and audit log."""
    tmp = tmp_path_factory.mktemp("pharmagene")
    settings = replace(Settings(), index_dir=tmp / "index", audit_log=tmp / "audit.jsonl", audit_log_raw=False)
    pipe = PharmaGenePipeline.from_settings(settings, llm=None, embedder=HashingEmbedder())
    pipe.reranker_available = lambda: False  # never download a model during tests
    return pipe


class FakeLLM:
    """Scripted LLM. `payload` may be a dict/str, or a callable(prompt) -> dict/str."""

    is_local = True
    name = "fake"

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def generate(self, system, prompt, json_mode=True):
        import json
        import re

        self.calls += 1
        if isinstance(self.payload, Exception):
            raise self.payload
        data = self.payload(prompt) if callable(self.payload) else self.payload
        return data if isinstance(data, str) else json.dumps(data)

    @staticmethod
    def rec_handle(prompt: str) -> str:
        import re

        return re.search(r"\[(C\d+)\] \(type: recommendation", prompt).group(1)
