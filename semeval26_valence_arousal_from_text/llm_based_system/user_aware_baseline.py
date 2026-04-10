"""User-aware static runners for emotion and numerical VA prediction."""

from __future__ import annotations

import pandas as pd

import configs
from user_agnostic_baseline import _run_model, sanitize_json_like


def get_texts_split_by_user_collection(
    split_words_essays: bool = False,
    source_file: str | None = None,
) -> dict[int, dict[int, dict[int, dict[str, object]]]]:
    """Group enriched training texts by user and collection phase."""
    source = source_file or configs.TRAIN_DATA_EMOTION
    data = pd.read_csv(source)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values(by=["user_id", "collection_phase", "timestamp"], ascending=True)

    grouped: dict[int, dict[int, dict[int, dict[str, object]]]] = {}
    for _, row in data.iterrows():
        user_id = int(row["user_id"])
        phase = int(row["collection_phase"])
        if split_words_essays and bool(row["is_words"]):
            phase *= 10
        grouped.setdefault(user_id, {}).setdefault(phase, {})[int(row["text_id"])] = {
            "text": str(row["text"]),
            "timestamp": str(row["timestamp"]),
            "is_words": bool(row["is_words"]),
            "valence": int(row["valence"]),
            "arousal": int(row["arousal"]),
            "emotion": str(row["emotion"]),
        }
    return grouped


def split_by_user_collection_train_predict(
    part1: int = 2,
    part2: int = 3,
    n_shot: int = 15,
    split_words_essays: bool = False,
    source_file: str | None = None,
) -> dict[int, dict[int, dict[int, dict[str, object]]]]:
    """Split each user-phase history into prompt examples and prediction targets."""
    grouped = get_texts_split_by_user_collection(split_words_essays=split_words_essays, source_file=source_file)
    for _, phases in grouped.items():
        for _, items in phases.items():
            phase_ids = list(items.keys())
            phase_len = len(phase_ids)
            if phase_len == 2:
                train_len = 1
            elif phase_len == 3:
                train_len = 2
            elif phase_len == 4:
                train_len = 3
            else:
                train_len = min(round(phase_len * part1 / part2), n_shot)

            train_ids = phase_ids[:train_len]
            test_ids = phase_ids[train_len:]
            items["train_texts"] = {text_id: items[text_id] for text_id in train_ids}
            items["test_texts"] = {text_id: items[text_id] for text_id in test_ids}
    return grouped


def _format_user_aware_training_example(payload: dict[str, object], prompt_type: str) -> str | dict[str, dict[str, int]]:
    """Convert one labeled example into the format expected by a prompt type."""
    text = str(payload["text"])
    if prompt_type == "emotion":
        return f"{text} -> {payload['emotion']}"
    if prompt_type == "valence":
        return f"{text} -> {int(payload['valence'])}"
    if prompt_type == "arousal":
        return f"{text} -> {int(payload['arousal'])}"
    if prompt_type == "val_aro":
        return {text: {"valence": int(payload["valence"]), "arousal": int(payload["arousal"])}}
    raise ValueError(f"Unsupported prompt_type: {prompt_type}")


def run_user_aware_prompt_static(
    provider: str = "ollama",
    model_name: str = configs.MODEL_GPT_OSS_120B,
    prompt_type: str = "emotion",
    split_words_essays: bool = False,
    part1: int = 2,
    part2: int = 3,
    n_shot: int = 15,
    source_file: str | None = None,
    openai_api_key: str | None = None,
    ollama_base_url: str | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Run the user-aware static flow for emotion, valence, arousal, or joint VA."""
    grouped = split_by_user_collection_train_predict(
        part1=part1,
        part2=part2,
        n_shot=n_shot,
        split_words_essays=split_words_essays,
        source_file=source_file,
    )

    if prompt_type not in configs.USER_AWARE_PROMPTS:
        raise ValueError(f"Unsupported prompt_type: {prompt_type}")

    results: dict[str, object] = {}
    failures = []
    for user_id, phases in grouped.items():
        for phase, items in phases.items():
            train_texts = [
                _format_user_aware_training_example(payload, prompt_type)
                for payload in items["train_texts"].values()
            ]
            prediction_texts = {
                text_id: payload["text"]
                for text_id, payload in items["test_texts"].items()
            }
            if not prediction_texts:
                continue
            prompt = configs.USER_AWARE_PROMPTS[prompt_type].format(train=train_texts, predict=prediction_texts)
            raw_response = _run_model(
                prompt=prompt,
                provider=provider,
                model_name=model_name,
                openai_api_key=openai_api_key,
                ollama_base_url=ollama_base_url,
            )
            parsed = sanitize_json_like(raw_response)
            if parsed is None:
                failures.append({"user_id": user_id, "phase": phase, "response": raw_response})
                continue
            results.update(parsed)
    return results, failures
