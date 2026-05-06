# src/ai_integration/openclaw_client.py
from typing import Any

import requests


class OpenClawClient:
    """Client for OpenClaw (Claude) API."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.base_url = "https://api.anthropic.com/v1/messages"

    def chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Send message request.

        Args:
            messages: List of {'role': str, 'content': str}.

        Returns:
            dict[str, Any]: Parsed JSON response.
        """
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
        }
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
