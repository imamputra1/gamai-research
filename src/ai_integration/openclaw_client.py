# src/ai_integration/openclaw_client.py
from __future__ import annotations

from typing import Any

import requests


class OpenClawClient:
    """Client for OpenClaw (Claude) API via OpenRouter proxy."""

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.base_url = "https://openrouter.ai/api/v1"
        self.chat_endpoint = f"{self.base_url}/chat/completions"

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "ThesisSynthPipeline",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
        }
        response = requests.post(
            self.chat_endpoint,
            headers=headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

