"""Provider-neutral LLM boundary.

The ranking pipeline depends only on ``LLMClient.json_call``. Configure separate
light and heavy clients with environment variables or replace one factory branch
below to connect another provider.
"""
from __future__ import annotations

import json
import os
import re
import hashlib
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv

load_dotenv()

_CACHE_PATH = Path(__file__).with_name("llm_cache.json")
_CACHE_LOCK = threading.Lock()


class LLMClient(Protocol):
    """Contract implemented by every hosted or local model adapter."""

    def json_call(self, prompt: str, use_cache: bool = True) -> dict:
        """Return the model response as a parsed JSON object."""


def _extract_json(text: str) -> dict:
    """Parse JSON while tolerating fenced model output."""
    if not text:
        raise ValueError("The model returned an empty response.")
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            return json.loads(candidate[start : end + 1])
        raise ValueError("The model response did not contain valid JSON.")


def _cache_key(provider: str, model: str, prompt: str) -> str:
    """Create a stable key for one model and exact prompt."""
    content = f"{provider}\0{model}\0{prompt}".encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _load_cache() -> dict:
    """Read cached structured responses, tolerating a missing cache file."""
    if not _CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _cached_response(key: str) -> dict | None:
    """Return a defensive copy of a cached response when available."""
    if os.getenv("LLM_CACHE_ENABLED", "1") != "1":
        return None
    with _CACHE_LOCK:
        value = _load_cache().get(key)
    return dict(value) if isinstance(value, dict) else None


def _store_response(key: str, value: dict) -> None:
    """Persist one response atomically so concurrent workers cannot corrupt it."""
    if os.getenv("LLM_CACHE_ENABLED", "1") != "1":
        return
    with _CACHE_LOCK:
        data = _load_cache()
        data[key] = value
        temporary = _CACHE_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(_CACHE_PATH)


def clear_response_cache() -> None:
    """Delete reviewed model responses before an intentional fresh analysis."""
    with _CACHE_LOCK:
        _CACHE_PATH.unlink(missing_ok=True)


@lru_cache(maxsize=4)
def _compatible_client(base_url: str, api_key: str):
    """Reuse an OpenAI-compatible client and its HTTP connection pool."""
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key, timeout=180)


@dataclass(frozen=True)
class OpenAICompatibleLLM:
    """Adapter for hosted Hugging Face, Colab, vLLM, or compatible APIs."""

    model: str
    base_url: str
    api_key: str

    def json_call(self, prompt: str, use_cache: bool = True) -> dict:
        """Call a hosted OpenAI-compatible endpoint and parse its response."""
        key = _cache_key("openai_compatible", self.model, prompt)
        if use_cache and (cached := _cached_response(key)) is not None:
            return cached
        client = _compatible_client(self.base_url, self.api_key)
        options = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        if self.model.lower().startswith(("gpt-5", "o1", "o3", "o4")):
            options.update(max_completion_tokens=12000, reasoning_effort="low")
        else:
            options.update(temperature=0)
        response = client.chat.completions.create(**options)
        value = _extract_json(response.choices[0].message.content or "")
        if use_cache:
            _store_response(key, value)
        return value


# ============================================================================
# CHANGE PROVIDER HERE
# Add an adapter that implements LLMClient.json_call(), then add one branch to
# build_llm(). Ranking, parsing, prompts, and the UI do not need to change.
#
# Examples:
# - Hugging Face: load transformers.pipeline and return _extract_json(output).
# - Colab/API: POST prompt + model to the hosted endpoint and parse its response.
# - OpenAI-compatible service: construct its SDK client in a new adapter class.
# ============================================================================
def build_llm(
    provider: str,
    model: str,
    base_url: str = "",
    api_key: str = "",
) -> LLMClient:
    """Construct a provider adapter for a model role."""
    if provider.lower() == "openai_compatible":
        return OpenAICompatibleLLM(model, base_url, api_key)
    raise ValueError(
        f"Unsupported provider {provider!r}. Add its adapter in llm_provider.py "
        "under the CHANGE PROVIDER HERE section."
    )


def configured_models() -> tuple[LLMClient, LLMClient]:
    """Return independently configurable light-parser and heavy-scorer models."""
    light_model = os.getenv("LIGHT_LLM_MODEL", "gpt-4o-mini")
    heavy_model = os.getenv("HEAVY_LLM_MODEL", "gpt-5.2")
    shared_base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    shared_api_key = os.getenv("LLM_API_KEY", "")
    light = build_llm(
        os.getenv("LIGHT_LLM_PROVIDER", "openai_compatible"),
        light_model,
        os.getenv("LIGHT_LLM_BASE_URL", shared_base_url),
        os.getenv("LIGHT_LLM_API_KEY", shared_api_key),
    )
    heavy = build_llm(
        os.getenv("HEAVY_LLM_PROVIDER", "openai_compatible"),
        heavy_model,
        os.getenv("HEAVY_LLM_BASE_URL", shared_base_url),
        os.getenv("HEAVY_LLM_API_KEY", shared_api_key),
    )
    return light, heavy