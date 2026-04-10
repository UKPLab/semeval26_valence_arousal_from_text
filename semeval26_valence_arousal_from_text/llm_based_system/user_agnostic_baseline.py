"""User-agnostic baseline runner and shared response parsing helpers."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import configs
from ask_ollama import ask_ollama_chat
from ask_openai import ask_openai_chat
from help_functions import build_not_shuffled_buckets, split_essays
from words_classification import get_all_texts


def sanitize_json_like(text: str) -> dict[str, object] | None:
    """Parse JSON-like model output and normalize top-level keys to strings."""
    normalized = re.sub(r"{(\s*)(\d+)(\s*):", r'{"\2":', text)
    normalized = re.sub(r",(\s*)(\d+)(\s*):", r',"\2":', normalized)
    try:
        parsed = json.loads(normalized)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(normalized)
        except Exception:
            return None
    return {str(key): value for key, value in parsed.items()}


def _run_model(
    prompt: str,
    provider: str,
    model_name: str,
    openai_api_key: str | None = None,
    ollama_base_url: str | None = None,
) -> str:
    """Dispatch a prompt to the selected provider and return raw text output."""
    if provider == "openai":
        return ask_openai_chat(prompt=prompt, model_name=model_name, api_key=openai_api_key)
    if provider == "ollama":
        return ask_ollama_chat(prompt=prompt, model_name=model_name, base_url=ollama_base_url or configs.OLLAMA_DEFAULT_BASE_URL)
    raise ValueError(f"Unsupported provider: {provider}")


def get_user_agnostic_buckets(
    test_data: bool = False,
    shuffled: bool = True,
    num_of_buckets: int = 70,
    length_mode: str = "text_len",
) -> list[dict[int, str]]:
    """Build buckets for the user-agnostic flow."""
    if shuffled:
        all_texts = get_all_texts(test_data=test_data)
        return split_essays(
            n_buckets=num_of_buckets,
            essays_dict=all_texts,
            respect_users=False,
            shuffle=True,
        )
    return build_not_shuffled_buckets(num_buckets=num_of_buckets, test_data=test_data, length_mode=length_mode)


def run_user_agnostic_prompt(
    prompt_template: str = configs.PROMPT_15SHOT,
    provider: str = "ollama",
    model_name: str = configs.MODEL_GPT_OSS_120B,
    test_data: bool = False,
    shuffled: bool = True,
    length_mode: str = "text_len",
    num_of_buckets: int = 70,
    openai_api_key: str | None = None,
    ollama_base_url: str | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Run the user-agnostic baseline over all buckets and collect predictions."""
    results: dict[str, object] = {}
    failures: list[dict[str, object]] = []
    for bucket_index, bucket in enumerate(
        get_user_agnostic_buckets(
            test_data=test_data,
            shuffled=shuffled,
            num_of_buckets=num_of_buckets,
            length_mode=length_mode,
        ),
        start=1,
    ):
        if not bucket:
            continue
        raw_response = _run_model(
            prompt=prompt_template.format(bucket),
            provider=provider,
            model_name=model_name,
            openai_api_key=openai_api_key,
            ollama_base_url=ollama_base_url,
        )
        parsed = sanitize_json_like(raw_response)
        if parsed is None:
            failures.append({"bucket": bucket_index, "response": raw_response})
            continue
        results.update(parsed)
    return results, failures


def save_results(output_path: str | Path, results: dict[str, object], failures: list[dict[str, object]] | None = None) -> tuple[Path, Path | None]:
    """Persist predictions to JSON and optionally save unparsable responses."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    failure_path = None
    if failures:
        failure_path = path.with_name(f"{path.stem}_BAD.txt")
        with failure_path.open("w", encoding="utf-8") as handle:
            for item in failures:
                handle.write(f"Bucket {item['bucket']}: {item['response']}\n")
    return path, failure_path
