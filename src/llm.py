"""Gemini API client wrapper using the official google-genai SDK."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional, Type, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class GeminiClientError(RuntimeError):
    """Raised when Gemini API calls fail after retries."""


class GeminiClient:
    """Wrapper around google.genai.Client with retry and structured output support."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", logger: Optional[logging.Logger] = None):
        if not api_key or api_key == "your_gemini_api_key_here":
            raise ValueError("A valid GEMINI_API_KEY is required.")
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.logger = logger or logging.getLogger("research_system.llm")

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_retries: int = 3,
    ) -> str:
        """Generate a text response from Gemini with exponential backoff."""
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
        )
        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                return (response.text or "").strip()
            except Exception as exc:
                last_err = exc
                self.logger.warning("Gemini text call failed (attempt %d/%d): %s", attempt, max_retries, exc)
                if attempt < max_retries:
                    time.sleep(1.5 * attempt)
        raise GeminiClientError(f"Gemini API failed after {max_retries} attempts: {last_err}") from last_err

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        max_retries: int = 3,
    ) -> T:
        """Generate a structured Pydantic response from Gemini."""
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=response_schema,
        )
        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                raw = (response.text or "").strip()
                data: Any = json.loads(raw)
                return response_schema.model_validate(data)
            except Exception as exc:
                last_err = exc
                self.logger.warning("Gemini structured call failed (attempt %d/%d): %s", attempt, max_retries, exc)
                if attempt < max_retries:
                    time.sleep(1.5 * attempt)
        raise GeminiClientError(f"Gemini structured API failed after {max_retries} attempts: {last_err}") from last_err
