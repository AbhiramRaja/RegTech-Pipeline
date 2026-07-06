"""
src/llm/provider.py

LLM provider abstraction layer.

Architecture rule: all LLM calls go through this interface.
Swapping providers (Groq → OpenAI, etc.) only requires a new class here —
no changes in agent nodes.

GroqProvider:
  - Uses langchain-groq ChatGroq for structured output.
  - Expects JSON output from the model.
  - Retries up to MAX_RETRIES times on malformed JSON.
  - Raises LLMProviderError after exhausting retries.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """Raised when the LLM provider fails after all retries."""
    pass


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> dict:
        """
        Send a prompt and return a parsed JSON dict.

        Args:
            prompt:        The user message / instruction.
            system_prompt: Optional system/role instruction.

        Returns:
            Parsed dict from model JSON output.

        Raises:
            LLMProviderError: After all retries are exhausted.
        """
        ...


class GroqProvider(LLMProvider):
    """
    Groq-backed LLM provider using langchain-groq.

    Instructs the model to respond in JSON and retries on parse failures.
    """

    def __init__(self, model_name: Optional[str] = None, max_retries: Optional[int] = None):
        try:
            from langchain_groq import ChatGroq
        except ImportError as e:
            raise ImportError("langchain-groq required. Run: pip install langchain-groq") from e

        from config import settings

        self._model_name = model_name or settings.llm_model_name
        self._max_retries = max_retries if max_retries is not None else settings.max_retries
        self._client = ChatGroq(
            model=self._model_name,
            api_key=settings.groq_api_key,
            temperature=0.0,  # deterministic for compliance tasks
        )
        logger.info("GroqProvider initialized with model: %s", self._model_name)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> dict:
        """
        Call Groq and parse JSON response. Retries on JSON parse errors.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.invoke(messages)
                raw_text: str = response.content.strip()

                # Strip markdown code fences if present
                if raw_text.startswith("```"):
                    lines = raw_text.split("\n")
                    # Remove first line (```json or ```) and last line (```)
                    raw_text = "\n".join(lines[1:-1]) if len(lines) > 2 else raw_text

                parsed = json.loads(raw_text)
                return parsed

            except json.JSONDecodeError as exc:
                last_exc = exc
                logger.warning(
                    "Attempt %d/%d — JSON parse error: %s. Raw: %.200s",
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                    response.content if "response" in dir() else "(no response)",
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Attempt %d/%d — LLM call error: %s",
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )

        raise LLMProviderError(
            f"GroqProvider failed after {self._max_retries + 1} attempts. "
            f"Last error: {last_exc}"
        )


# ── Singleton factory ──────────────────────────────────────────────────────────

_provider_instance: Optional[LLMProvider] = None


def get_provider() -> LLMProvider:
    """
    Return the default LLM provider singleton.
    Swap provider class here to change the backend globally.
    """
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = GroqProvider()
    return _provider_instance


def reset_provider(provider: Optional[LLMProvider] = None) -> None:
    """
    Override the provider singleton — used in tests to inject mocks.
    Call with no args to reset to the real provider.
    """
    global _provider_instance
    _provider_instance = provider
