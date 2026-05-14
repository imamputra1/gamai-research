# tests/test_ai_integration.py
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.ai_integration.constants import (
    CHEAP_PING_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    ENV_OPENROUTER_API_KEY,
    HTTP_PAYMENT_REQUIRED,
    OPENROUTER_CHAT_ENDPOINT,
    PING_MAX_TOKENS,
    PING_PROMPT,
    REASONING_MODELS,
    REASONING_TEMPERATURE,
)
from src.ai_integration.open_router_client import (
    InsufficientBalanceError,
    InvalidCredentialsError,
    OpenRouterClient,
)


def test_client_loads_api_key_from_env() -> None:
    with patch.dict("os.environ", {ENV_OPENROUTER_API_KEY: "sk-test-123"}):
        client: OpenRouterClient = OpenRouterClient()
        assert client.api_key == "sk-test-123"


def test_client_uses_provided_api_key_over_env() -> None:
    with patch.dict("os.environ", {ENV_OPENROUTER_API_KEY: "sk-env-456"}):
        client: OpenRouterClient = OpenRouterClient(api_key="sk-explicit-789")
        assert client.api_key == "sk-explicit-789"


def test_client_raises_without_api_key() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError):
            OpenRouterClient()


def test_client_default_parameters() -> None:
    with patch.dict("os.environ", {ENV_OPENROUTER_API_KEY: "sk-test"}):
        client: OpenRouterClient = OpenRouterClient()

        assert client.base_url == OPENROUTER_CHAT_ENDPOINT
        assert client.models == REASONING_MODELS
        assert client.max_tokens == DEFAULT_MAX_TOKENS
        assert client.temperature == DEFAULT_TEMPERATURE


def test_build_headers_contains_bearer_token() -> None:
    client: OpenRouterClient = OpenRouterClient(api_key="sk-test")
    headers: Dict[str, str] = client._build_headers()

    assert headers["Authorization"] == "Bearer sk-test"
    assert headers["Content-Type"] == "application/json"


def test_build_payload_uses_defaults() -> None:
    client: OpenRouterClient = OpenRouterClient(api_key="sk-test")
    payload: Dict[str, Any] = client._build_payload(
        messages=[{"role": "user", "content": "Hello"}]
    )

    assert payload["model"] == REASONING_MODELS[0]
    assert payload["max_tokens"] == DEFAULT_MAX_TOKENS
    assert payload["temperature"] == DEFAULT_TEMPERATURE


def test_build_payload_uses_overrides() -> None:
    client: OpenRouterClient = OpenRouterClient(api_key="sk-test")
    payload: Dict[str, Any] = client._build_payload(
        messages=[{"role": "user", "content": "Hello"}],
        model="custom/model",
        max_tokens=1024,
        temperature=0.3,
    )

    assert payload["model"] == "custom/model"
    assert payload["max_tokens"] == 1024
    assert payload["temperature"] == 0.3


def test_chat_raises_insufficient_balance_on_402() -> None:
    client: OpenRouterClient = OpenRouterClient(api_key="sk-test")

    mock_response: MagicMock = MagicMock()
    mock_response.status_code = HTTP_PAYMENT_REQUIRED

    with patch("requests.post", return_value=mock_response):
        with pytest.raises(InsufficientBalanceError):
            client.chat([{"role": "user", "content": "test"}])


def test_chat_retries_on_timeout() -> None:
    client: OpenRouterClient = OpenRouterClient(api_key="sk-test")

    with patch("requests.post") as mock_post:
        mock_post.side_effect = [
            requests.exceptions.Timeout(),
            MagicMock(status_code=200, json=lambda: {"choices": [{"message": {"content": "ok"}}]}),
        ]

        result: Dict[str, Any] = client.chat([{"role": "user", "content": "test"}])
        assert result["choices"][0]["message"]["content"] == "ok"
        assert mock_post.call_count == 2


def test_chat_single_returns_string() -> None:
    client: OpenRouterClient = OpenRouterClient(api_key="sk-test")

    mock_response: MagicMock = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Hello, researcher!"}}]
    }

    with patch("requests.post", return_value=mock_response):
        reply: str = client.chat_single("Say hello")

    assert reply == "Hello, researcher!"


def test_generate_reasoning_accepts_dict_and_returns_string() -> None:
    """Must accept dictionary, use low temperature, and return narrative."""
    client: OpenRouterClient = OpenRouterClient(api_key="sk-test")

    mock_response: MagicMock = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Narasi akademik hasil analisis..."}}]
    }

    with patch("requests.post", return_value=mock_response) as mock_post:
        context: Dict[str, Any] = {
            "VAF": 85.7,
            "Direct_Effect": 0.45,
            "Indirect_Effect": 2.65,
        }
        reply: str = client.generate_reasoning(context)

        assert "Narasi akademik" in reply
        # Verify low temperature was used
        call_payload: Dict[str, Any] = mock_post.call_args[1]["json"]
        assert call_payload["temperature"] == REASONING_TEMPERATURE


def test_generate_reasoning_rejects_non_serializable() -> None:
    """Must raise TypeError when context contains non-serializable objects."""
    client: OpenRouterClient = OpenRouterClient(api_key="sk-test")

    with pytest.raises(TypeError):
        client.generate_reasoning({"data": object()})


def test_test_connection_uses_cheap_model_and_ping_prompt() -> None:
    """Must use gemma free model, PING prompt, and 5 max_tokens."""
    client: OpenRouterClient = OpenRouterClient(api_key="sk-test")

    mock_response: MagicMock = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "PONG"}}]
    }

    with patch("requests.post", return_value=mock_response) as mock_post:
        result: bool = client.test_connection()

        assert result is True
        call_payload: Dict[str, Any] = mock_post.call_args[1]["json"]
        assert call_payload["model"] == CHEAP_PING_MODEL
        assert call_payload["max_tokens"] == PING_MAX_TOKENS
        assert call_payload["messages"][1]["content"] == PING_PROMPT


def test_test_connection_raises_on_invalid_credentials() -> None:
    """Must raise InvalidCredentialsError on connection failure."""
    client: OpenRouterClient = OpenRouterClient(api_key="sk-test")

    with patch("requests.post", side_effect=requests.exceptions.ConnectionError()):
        with pytest.raises(InvalidCredentialsError):
            client.test_connection()


def test_test_connection_raises_on_insufficient_balance() -> None:
    """Must propagate InsufficientBalanceError if ping returns 402."""
    client: OpenRouterClient = OpenRouterClient(api_key="sk-test")

    mock_response: MagicMock = MagicMock()
    mock_response.status_code = HTTP_PAYMENT_REQUIRED

    with patch("requests.post", return_value=mock_response):
        with pytest.raises(InsufficientBalanceError):
            client.test_connection()
