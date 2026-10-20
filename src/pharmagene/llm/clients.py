from __future__ import annotations

import json
import re
import time
import urllib.request
from typing import Any, Protocol

from ..config import Settings


class LLMClient(Protocol):
    name: str
    is_local: bool

    def generate(self, system: str, prompt: str, json_mode: bool = True) -> str: ...


def parse_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating code fences and surrounding prose."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(cleaned[start:end + 1])


class GeminiClient:
    """Gemini via the google-genai SDK. Note: prompts leave the machine; use only de-identified or
    synthetic patient data, or switch to the local Ollama backend for on-prem deployments."""

    is_local = False

    def __init__(self, api_key: str, model: str):
        from google import genai

        self.name = f"gemini:{model}"
        self._model = model
        self._client = genai.Client(api_key=api_key)

    def generate(self, system: str, prompt: str, json_mode: bool = True) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            response_mime_type="application/json" if json_mode else None,
        )
        for attempt in range(4):
            try:
                resp = self._client.models.generate_content(
                    model=self._model, contents=prompt, config=config
                )
                return resp.text or ""
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        return ""


class OllamaClient:
    """Locally hosted model (e.g. Llama 3.1) through Ollama's HTTP API; no data leaves the host."""

    is_local = True

    def __init__(self, url: str, model: str):
        self.name = f"ollama:{model}"
        self._url = url.rstrip("/")
        self._model = model

    def generate(self, system: str, prompt: str, json_mode: bool = True) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0},
        }
        if json_mode:
            payload["format"] = "json"
        req = urllib.request.Request(
            f"{self._url}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())["message"]["content"]


def make_llm(settings: Settings) -> LLMClient | None:
    """Returns None when no backend is configured; the pipeline then answers extractively."""
    backend = settings.resolved_llm_backend()
    if backend == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("PHARMAGENE_LLM=gemini but GEMINI_API_KEY is not set.")
        return GeminiClient(settings.gemini_api_key, settings.gemini_model)
    if backend == "ollama":
        return OllamaClient(settings.ollama_url, settings.ollama_model)
    if backend == "none":
        return None
    raise ValueError(f"Unknown LLM backend: {backend}")
