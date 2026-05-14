# src/ai_integration/prompt_builder.py
from __future__ import annotations

import json
from typing import Any, Dict


def build_reasoning_payload(
    context_dict: Dict[str, Any],
    system_prompt: str,
    user_template: str,
) -> Dict[str, str]:
    """Serialize context dictionary into a structured reasoning prompt.

    Args:
        context_dict: Dictionary of computed results (VAF, coefficients, etc.).
        system_prompt: System role message.
        user_template: User message template with {context_json} placeholder.

    Returns:
        dict: Messages payload ready for chat().

    Raises:
        TypeError: If context_dict is not serializable.
    """
    try:
        context_json: str = json.dumps(context_dict, indent=2, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"context_dict harus serializable ke JSON. Error: {exc}"
        ) from exc

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_template.format(context_json=context_json)},
        ]
    }
