# src/ai_integration/open_router_client.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.ai_integration.constants import (
    CHEAP_PING_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SECONDS,
    ENV_OPENROUTER_API_KEY,
    ERROR_CONNECTION_FAILED,
    ERROR_INSUFFICIENT_BALANCE,
    ERROR_INVALID_CREDENTIALS,
    HTTP_BAD_GATEWAY,
    HTTP_OK,
    HTTP_PAYMENT_REQUIRED,
    HTTP_SERVICE_UNAVAILABLE,
    HTTP_TOO_MANY_REQUESTS,
    OPENROUTER_CHAT_ENDPOINT,
    PING_MAX_TOKENS,
    PING_PROMPT,
    REASONING_MAX_TOKENS,
    REASONING_MODELS,
    REASONING_SYSTEM_PROMPT,
    REASONING_TEMPERATURE,
    REASONING_USER_TEMPLATE,
    RETRY_MAX_ATTEMPTS,
    RETRY_WAIT_MAX,
    RETRY_WAIT_MIN,
    RETRY_WAIT_MULTIPLIER,
)
from src.ai_integration.prompt_builder import build_reasoning_payload


class InsufficientBalanceError(Exception):
    """Raised when the API returns HTTP 402 (Payment Required)."""
    pass


class InvalidCredentialsError(Exception):
    """Raised when the API key is rejected or connection fails."""
    pass


class OpenRouterClient:
    """HTTP bridge to OpenRouter API with exponential backoff resilience."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = OPENROUTER_CHAT_ENDPOINT,
        models: List[str] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.api_key: str = api_key or os.getenv(ENV_OPENROUTER_API_KEY, "")
        if not self.api_key:
            raise ValueError(
                f"API key tidak ditemukan. Set environment variable "
                f"'{ENV_OPENROUTER_API_KEY}' atau berikan via constructor."
            )

        self.base_url: str = base_url
        self.models: List[str] = models or REASONING_MODELS
        self.max_tokens: int = max_tokens
        self.temperature: float = temperature
        self.timeout: int = timeout

    def _build_headers(self) -> Dict[str, str]:
        """Construct HTTP headers for OpenRouter requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "ThesisSynthPipeline",
        }

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Dict[str, Any]:
        """Construct JSON payload for chat completion request."""
        return {
            "model": model or self.models[0],
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
        }

    @retry(
        stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        wait=wait_exponential(
            multiplier=RETRY_WAIT_MULTIPLIER,
            min=RETRY_WAIT_MIN,
            max=RETRY_WAIT_MAX,
        ),
        retry=retry_if_exception_type(
            (requests.exceptions.Timeout, requests.exceptions.ConnectionError)
        ),
        reraise=True,
    )
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Dict[str, Any]:
        """Send chat completion request with exponential backoff."""
        headers: Dict[str, str] = self._build_headers()
        payload: Dict[str, Any] = self._build_payload(
            messages, model, max_tokens, temperature
        )

        try:
            response: requests.Response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.ConnectionError as exc:
            raise InvalidCredentialsError(ERROR_CONNECTION_FAILED) from exc

        if response.status_code == HTTP_PAYMENT_REQUIRED:
            raise InsufficientBalanceError(ERROR_INSUFFICIENT_BALANCE)

        if response.status_code in (HTTP_BAD_GATEWAY, HTTP_SERVICE_UNAVAILABLE):
            response.raise_for_status()

        response.raise_for_status()
        return response.json()

    def chat_single(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful research assistant.",
        **kwargs: Any,
    ) -> str:
        """Convenience wrapper for single-turn chat."""
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        result: Dict[str, Any] = self.chat(messages, **kwargs)
        return str(result["choices"][0]["message"]["content"])

    def generate_reasoning(
        self,
        context_dict: Dict[str, Any],
        temperature: float = REASONING_TEMPERATURE,
        max_tokens: int = REASONING_MAX_TOKENS,
    ) -> str:
        """Generate academic narrative from structured computation results.

        Args:
            context_dict: Dictionary of computed results (VAF, coefficients, etc.).
            temperature: Low temperature for deterministic output (default 0.2).
            max_tokens: Token limit for reasoning output.

        Returns:
            str: Academic narrative in Indonesian.

        Raises:
            TypeError: If context_dict is not JSON-serializable.
        """
        payload: Dict[str, str] = build_reasoning_payload(
            context_dict=context_dict,
            system_prompt=REASONING_SYSTEM_PROMPT,
            user_template=REASONING_USER_TEMPLATE,
        )
        result: Dict[str, Any] = self.chat(
            messages=payload["messages"],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return str(result["choices"][0]["message"]["content"])

    def test_connection(self) -> bool:
        """Diagnostic ping using the cheapest available model.

        Sends a minimal PING prompt to verify API key validity,
        balance sufficiency, and server responsiveness.

        Returns:
            bool: True if connection is healthy.

        Raises:
            InsufficientBalanceError: If balance is depleted.
            InvalidCredentialsError: If API key is invalid.
        """
        try:
            reply: str = self.chat_single(
                prompt=PING_PROMPT,
                model=CHEAP_PING_MODEL,
                max_tokens=PING_MAX_TOKENS,
                temperature=0.0,
            )
            return len(reply.strip()) > 0
        except InsufficientBalanceError:
            raise
        except Exception as exc:
            raise InvalidCredentialsError(ERROR_INVALID_CREDENTIALS) from exc
