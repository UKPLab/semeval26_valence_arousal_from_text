"""Minimal Ollama client helpers used by the experiment runners."""

from __future__ import annotations

import requests

from configs import OLLAMA_DEFAULT_BASE_URL


def ask_ollama_chat(
    prompt: str,
    model_name: str,
    base_url: str = OLLAMA_DEFAULT_BASE_URL,
    temperature: float = 0.1,
    timeout: int = 300,
) -> str:
    """Send a non-streaming chat-style completion request to Ollama."""
    response = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model": model_name,
            "messages": [
                {"role": "assistant", "content": prompt},
                {"role": "user", "content": "Produce the labels now."},
            ],
            "stream": False,
            "temperature": temperature,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["message"]["content"].strip()


def print_available_models(base_url: str = OLLAMA_DEFAULT_BASE_URL, timeout: int = 30) -> None:
    """Print model names exposed by the configured Ollama server."""
    response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
    response.raise_for_status()
    for model in response.json().get("models", []):
        print(model["name"])
