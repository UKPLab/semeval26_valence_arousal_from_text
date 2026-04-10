"""Reusable helpers for bucket construction and JSON utilities."""

from __future__ import annotations

import json
import random
from typing import Dict

import pandas as pd

import configs
from words_classification import split_words_and_essays


def split_essays(
    n_buckets: int,
    essays_dict: Dict[int, str],
    user_map: dict[int, int] | None = None,
    respect_users: bool = True,
    shuffle: bool = True,
) -> list[dict[int, str]]:
    """Balance texts across buckets, optionally keeping each user intact."""
    items = list(essays_dict.items())
    if not respect_users:
        if shuffle:
            random.seed(42)
            random.shuffle(items)
        else:
            items.sort(key=lambda item: len(item[1]), reverse=True)

        buckets = [{} for _ in range(n_buckets)]
        bucket_lengths = [0] * n_buckets
        for text_id, text in items:
            bucket_index = bucket_lengths.index(min(bucket_lengths))
            buckets[bucket_index][text_id] = text
            bucket_lengths[bucket_index] += len(text)
        return buckets

    if user_map is None:
        raise ValueError("user_map is required when respect_users=True")

    user_to_items: dict[int, list[tuple[int, str]]] = {}
    user_lengths: dict[int, int] = {}
    for text_id, text in items:
        user_id = user_map[text_id]
        user_to_items.setdefault(user_id, []).append((text_id, text))
        user_lengths[user_id] = user_lengths.get(user_id, 0) + len(text)

    buckets = [{} for _ in range(n_buckets)]
    bucket_lengths = [0] * n_buckets
    for user_id, _ in sorted(user_lengths.items(), key=lambda item: item[1], reverse=True):
        bucket_index = bucket_lengths.index(min(bucket_lengths))
        for text_id, text in user_to_items[user_id]:
            buckets[bucket_index][text_id] = text
            bucket_lengths[bucket_index] += len(text)
    return buckets


def split_essays_words(
    n_buckets_essays: int,
    n_buckets_words: int,
    shuffle: bool = True,
    words_limit: int = 70,
) -> tuple[list[dict[int, str]], list[dict[int, str]]]:
    """Create separate balanced bucket sets for essays and word-list entries."""
    _, _, words, essays, _ = split_words_and_essays(limit=words_limit)
    essay_buckets = split_essays(n_buckets_essays, essays, respect_users=False, shuffle=shuffle)
    word_items = {text_id: " , ".join(parts) for text_id, parts in words.items()}
    word_buckets = split_essays(n_buckets_words, word_items, respect_users=False, shuffle=shuffle)
    return essay_buckets, word_buckets


def build_not_shuffled_buckets(num_buckets: int, test_data: bool = False, length_mode: str = "text_len") -> list[dict[int, str]]:
    """Build chronological buckets without shuffling, keeping user order intact."""
    source = configs.RAW_TEST_DATA if test_data else configs.RAW_TRAIN_DATA
    data = pd.read_csv(source).sort_values(by=["user_id", "timestamp"], ascending=True).reset_index(drop=True)
    data["item_len"] = data["text"].astype(str).str.len() if length_mode == "text_len" else 1
    target = data["item_len"].sum() / num_buckets

    buckets: dict[int, dict[int, str]] = {}
    bucket_index = 0
    current_len = 0
    for _, row in data.iterrows():
        if current_len >= target and bucket_index < num_buckets - 1:
            bucket_index += 1
            current_len = 0
        buckets.setdefault(bucket_index, {})[int(row["text_id"])] = str(row["text"])
        current_len += int(row["item_len"])
    return [buckets[index] for index in sorted(buckets)]


def load_json_int_keys(filename: str) -> dict[int, object]:
    """Load JSON and normalize top-level keys to integers."""
    with open(filename, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {int(key): value for key, value in data.items()}
