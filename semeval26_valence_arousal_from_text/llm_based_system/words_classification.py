"""Dataset loading helpers for all text, word-list, and essay splits."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

import configs


def _read_dataset(path: Path) -> pd.DataFrame:
    """Load a CSV dataset into a pandas DataFrame."""
    return pd.read_csv(path)


def get_all_texts(test_data: bool = False, exclude_seen_users: bool = True) -> dict[int, str]:
    """Return a mapping from text id to raw text for the selected split."""
    source = configs.RAW_TEST_DATA if test_data else configs.RAW_TRAIN_DATA
    data = _read_dataset(source)
    if test_data and exclude_seen_users and "is_seen_user" in data.columns:
        data = data.loc[~data["is_seen_user"]].reset_index(drop=True)
    return dict(zip(data["text_id"].astype(int), data["text"].astype(str)))


def split_words_and_essays(
    test_data: bool = False,
    stats: bool = False,
    limit: int = 70,
) -> tuple[set[str], list[str], dict[int, list[str]], dict[int, str], dict[int, int]]:
    """Split the dataset into short feeling-word lists and essay-like entries."""
    source = configs.RAW_TEST_DATA if test_data else configs.RAW_TRAIN_DATA
    data = _read_dataset(source)
    words_df = data[data["is_words"] == True].copy()

    if stats and not words_df.empty:
        print(f"Average words-entry length: {words_df['text'].astype(str).str.len().mean():.2f}")

    words_df_short = words_df[words_df["text"].astype(str).str.len() <= limit]
    all_words_dict: dict[int, list[str]] = {}
    all_words: list[str] = []
    for text_id, text in zip(words_df_short["text_id"], words_df_short["text"]):
        pieces = str(text).lower().split(" , ")
        all_words_dict[int(text_id)] = pieces
        all_words.extend(pieces)

    essays_df = pd.concat(
        [
            data[data["is_words"] == False],
            words_df[words_df["text"].astype(str).str.len() > limit],
        ],
        ignore_index=True,
    )
    essays = dict(zip(essays_df["text_id"].astype(int), essays_df["text"].astype(str)))
    user_map = dict(zip(data["text_id"].astype(int), data["user_id"].astype(int)))

    if stats:
        print(f"Feeling-word entries: {len(all_words_dict)}")
        print(f"Essay entries: {len(essays)}")

    return set(all_words), all_words, all_words_dict, essays, user_map


def most_common_feeling_words(limit: int = 70) -> list[tuple[str, int]]:
    """Return the most common tokenized feeling words in short word-list entries."""
    _, all_words, _, _, _ = split_words_and_essays(limit=limit)
    return Counter(all_words).most_common()
