"""Dynamic runners that update user history after each prediction chunk."""

from __future__ import annotations

import pandas as pd

import configs
from user_agnostic_baseline import _run_model, sanitize_json_like


def get_texts_split_by_user(source_file: str | None = None) -> dict[int, dict[str, dict[int, dict[str, object]]]]:
    """Group enriched texts by user and by words-versus-essays bucket."""
    source = source_file or configs.TRAIN_DATA_EMOTION
    data = pd.read_csv(source)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values(by=["user_id", "timestamp"], ascending=True)

    grouped: dict[int, dict[str, dict[int, dict[str, object]]]] = {}
    for _, row in data.iterrows():
        user_id = int(row["user_id"])
        bucket = "words" if bool(row["is_words"]) else "essays"
        grouped.setdefault(user_id, {}).setdefault(bucket, {})[int(row["text_id"])] = {
            "text": str(row["text"]),
            "timestamp": str(row["timestamp"]),
            "is_words": bool(row["is_words"]),
            "valence": int(row["valence"]),
            "arousal": int(row["arousal"]),
            "emotion": str(row["emotion"]),
        }
    return grouped


def split_by_user_train_predict(
    full_test: bool = False,
    n_pred: int = 5,
    part1: int = 2,
    part2: int = 3,
    n_shot: int = 10,
    source_file: str | None = None,
) -> dict[int, dict[str, dict[str, object]]]:
    """Split each user history into seed examples and rolling prediction chunks."""
    grouped = get_texts_split_by_user(source_file=source_file)
    for _, buckets in grouped.items():
        for _, items in buckets.items():
            item_ids = list(items.keys())
            item_len = len(item_ids)
            if item_len == 2:
                train_len = 1
            elif item_len == 3:
                train_len = 2
            elif item_len == 4:
                train_len = 3
            else:
                train_len = min(round(item_len * part1 / part2), n_shot)

            train_texts = {}
            test_texts = {0: {}}
            test_bucket = 0
            for index, text_id in enumerate(item_ids):
                if index < train_len and not full_test:
                    train_texts[text_id] = items[text_id]
                    continue
                if len(test_texts[test_bucket]) >= n_pred:
                    test_bucket += 1
                    test_texts[test_bucket] = {}
                test_texts[test_bucket][text_id] = items[text_id]

            items["train_texts"] = train_texts
            items["test_texts"] = test_texts
    return grouped


def run_dynamic_prompt(
    provider: str = "ollama",
    model_name: str = configs.MODEL_GPT_OSS_120B,
    full_text: bool = True,
    n_pred: int = 5,
    source_file: str | None = None,
    openai_api_key: str | None = None,
    ollama_base_url: str | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Run the dynamic prediction flow and update history after each chunk."""
    grouped = split_by_user_train_predict(full_test=full_text, n_pred=n_pred, source_file=source_file)
    results = {}
    failures = []

    for user_id, buckets in grouped.items():
        for bucket_name, items in buckets.items():
            if full_text:
                history = list(configs.FAKE_WORDS_HISTORY if bucket_name == "words" else configs.FAKE_ESSAYS_HISTORY)
            else:
                history = [
                    f"{payload['text']} -> {payload['emotion']}"
                    for payload in items["train_texts"].values()
                ]

            for test_bucket, payloads in items["test_texts"].items():
                prediction_texts = {text_id: value["text"] for text_id, value in payloads.items()}
                if not prediction_texts:
                    continue
                prompt = configs.PROMPT_DYNAMIC.format(train=history, predict=prediction_texts)
                raw_response = _run_model(
                    prompt=prompt,
                    provider=provider,
                    model_name=model_name,
                    openai_api_key=openai_api_key,
                    ollama_base_url=ollama_base_url,
                )
                parsed = sanitize_json_like(raw_response)
                if parsed is None:
                    failures.append({"user_id": user_id, "bucket": bucket_name, "test_bucket": test_bucket, "response": raw_response})
                    continue
                results.update(parsed)
                history = history[n_pred:]
                for text_id, text in prediction_texts.items():
                    history.append(f"{text} -> {parsed[str(text_id)]}")
    return results, failures
