# src/ai_integration/open_router_client.py
from __future__ import annotations

import os
from typing import Any

import requests
import yaml


class OpenRouterClient:
    """Unified client for OpenRouter API with multi-model fallback."""

    def __init__(
        self,
        api_key: str,
        models: list[str] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        self.api_key = api_key
        self.models = models or ["google/gemma-2-9b-it:free"]
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.base_url = base_url.rstrip("/")
        self.chat_endpoint = f"{self.base_url}/chat/completions"

    @classmethod
    def from_config(cls, config_path: str = "config/pipeline_config.yaml") -> "OpenRouterClient":
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        ai_cfg = cfg["ai"]
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        return cls(
            api_key=api_key,
            models=ai_cfg.get("models", ["google/gemma-2-9b-it:free"]),
            max_tokens=ai_cfg.get("max_tokens", 4096),
            temperature=ai_cfg.get("temperature", 0.7),
            base_url=ai_cfg.get("kimi_base_url", "https://openrouter.ai/api/v1"),
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        target = model or self.models[0]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "ThesisSynthPipeline",
        }
        payload = {
            "model": target,
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

    def chat_with_fallback(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        last_err: Exception | None = None
        for model in self.models:
            try:
                return self.chat(
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except requests.HTTPError as exc:
                last_err = exc
                continue
        raise RuntimeError(f"All models exhausted. Last error: {last_err}")

