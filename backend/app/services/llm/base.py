"""
LLM provider abstraction.

The rest of the application (extraction service, API layer) depends only on
this interface, never on a specific vendor SDK. This is what lets Phase 2
swap or add providers later without touching app/services/extraction_service.py
or app/api — and lets tests substitute a fake provider instead of calling a
real API.
"""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Minimal interface every LLM provider implementation must satisfy."""

    @property
    def name(self) -> str:
        """Short identifier for the provider (used in API responses/logs)."""
        return self.__class__.__name__

    @property
    def model(self) -> str:
        """Identifier of the specific model this provider instance uses."""
        return "unknown"

    @abstractmethod
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a system + user prompt to the LLM and return its raw text
        response.

        Implementations must raise app.core.exceptions.LLMProviderError (or
        a subclass) on any failure — missing configuration, timeout, rate
        limiting, transport errors — rather than letting a vendor-specific
        exception escape to the caller.
        """
        raise NotImplementedError