# src/ai_integration/kimi_client.py
from typing import Any

import requests


class KimiClient:
    """Client for Kimi k-2.5 API."""

    def __init__(
        self,
        api_key: str,
        model: str = "kimi-k2-5",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.base_url = "https://api.moonshot.cn/v1/chat/completions"

    def chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Send chat completion request.

        Args:
            messages: List of {'role': str, 'content': str}.

        Returns:
            dict[str, Any]: Parsed JSON response.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
