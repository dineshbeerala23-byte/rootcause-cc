"""
Groq implementation of the LLMProvider interface.

The `groq` package is imported lazily inside _get_client() rather than at
module load time. This means the FastAPI app (and its /docs, /health, and
every endpoint unrelated to extraction) still starts and runs fine even in
an environment where GROQ_API_KEY isn't set yet or the groq package isn't
installed — the failure only surfaces when an extraction is actually
attempted, as a clean LLMProviderError rather than an import-time crash.
"""

import logging
from typing import Optional

from app.core.exceptions import LLMProviderError
from app.services.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    """Calls the Groq chat completions API. API key comes from settings only."""

    def __init__(
        self,
        api_key: Optional[str],
        model: str,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = None  # lazily constructed real Groq SDK client

    @property
    def name(self) -> str:
        return "groq"

    @property
    def model(self) -> str:
        return self._model

    def _get_client(self):
        if not self._api_key:
            raise LLMProviderError(
                "GROQ_API_KEY is not configured. Set it in your .env file to use the extraction endpoint."
            )
        if self._client is None:
            try:
                from groq import AsyncGroq
            except ImportError as exc:  # pragma: no cover - depends on local env
                raise LLMProviderError(
                    "The 'groq' package is not installed. Run: pip install -r requirements.txt"
                ) from exc
            self._client = AsyncGroq(api_key=self._api_key, timeout=self._timeout)
        return self._client

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_client()

        last_error: Optional[Exception] = None
        attempts = self._max_retries + 1

        for attempt in range(1, attempts + 1):
            try:
                response = await client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                return content or ""
            except Exception as exc:  # noqa: BLE001 - vendor SDK errors are intentionally caught broadly here
                last_error = exc
                logger.warning(
                    "Groq completion attempt %s/%s failed: %s", attempt, attempts, exc
                )

        logger.error("Groq completion failed after %s attempt(s).", attempts, exc_info=last_error)
        raise LLMProviderError(f"Failed to get a response from Groq: {last_error}")