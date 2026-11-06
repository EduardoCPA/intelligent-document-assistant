import os
from typing import Any

from google import genai

DEFAULT_MODEL_NAME = "gemini-2.5-flash"


class GeminiClient:
    """Client for generating text with Google's Gemini API."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        client: Any | None = None,
    ) -> None:
        resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")

        if client is None and not resolved_api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        self._model_name = model_name
        self._client = (
            client if client is not None else genai.Client(api_key=resolved_api_key)
        )

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt using Gemini."""
        if not prompt.strip():
            raise ValueError("Prompt cannot be blank.")

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        return response.text.strip()
