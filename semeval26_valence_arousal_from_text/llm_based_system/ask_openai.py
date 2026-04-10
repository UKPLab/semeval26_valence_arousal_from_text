"""Minimal OpenAI client helpers used by the experiment runners."""

from __future__ import annotations

import os

from openai import OpenAI

from configs import OPENAI_API_KEY_ENV


def ask_openai_chat(prompt: str, model_name: str, api_key: str | None = None) -> str:
    """Send a chat-style labeling request through the Responses API."""
    key = api_key or os.getenv(OPENAI_API_KEY_ENV)
    if not key:
        raise ValueError(f"Set {OPENAI_API_KEY_ENV} before using the OpenAI provider.")

    client = OpenAI(api_key=key)
    response = client.responses.create(
        model=model_name,
        input=[
            {"role": "assistant", "content": prompt},
            {"role": "user", "content": "Produce the labels now."},
        ],
    )
    return response.output_text
