"""User-agnostic runner that predicts word lists and essays separately."""

from __future__ import annotations

import configs
from help_functions import split_essays_words
from user_agnostic_baseline import _run_model, sanitize_json_like


def _run_buckets(
    buckets: list[dict[int, str]],
    prompt_template: str,
    provider: str,
    model_name: str,
    openai_api_key: str | None = None,
    ollama_base_url: str | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Run a prompt template over an already prepared list of buckets."""
    results = {}
    failures = []
    for bucket_index, bucket in enumerate(buckets, start=1):
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


def run_uag_prompt_essays_and_words(
    provider: str = "ollama",
    model_name: str = configs.MODEL_GPT_OSS_120B,
    n_bucket_essays: int = 80,
    n_bucket_words: int = 50,
    words_limit: int = 70,
    openai_api_key: str | None = None,
    ollama_base_url: str | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Run the split words-versus-essays variant and merge both result sets."""
    essay_buckets, word_buckets = split_essays_words(
        n_buckets_essays=n_bucket_essays,
        n_buckets_words=n_bucket_words,
        words_limit=words_limit,
    )
    word_results, word_failures = _run_buckets(
        word_buckets,
        configs.PROMPT_15SHOT_WORDS,
        provider,
        model_name,
        openai_api_key=openai_api_key,
        ollama_base_url=ollama_base_url,
    )
    essay_results, essay_failures = _run_buckets(
        essay_buckets,
        configs.PROMPT_15SHOT_ESSAYS,
        provider,
        model_name,
        openai_api_key=openai_api_key,
        ollama_base_url=ollama_base_url,
    )
    merged = {}
    merged.update(word_results)
    merged.update(essay_results)
    return merged, [*word_failures, *essay_failures]
