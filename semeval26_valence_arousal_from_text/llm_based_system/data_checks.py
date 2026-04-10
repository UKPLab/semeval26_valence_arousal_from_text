"""Validation and repair helpers for prediction outputs."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import configs
from words_classification import get_all_texts


def check_missing_ids(json_file_name: str, test_data: bool = False) -> list[int]:
    """Return dataset ids that are missing from a prediction file."""
    with open(json_file_name, "r", encoding="utf-8") as handle:
        predictions = {int(key) for key in json.load(handle).keys()}
    all_ids = set(get_all_texts(test_data=test_data).keys())
    return sorted(all_ids - predictions)


def check_naming_consistency(json_file_name: str) -> dict[int, str]:
    """Return predictions whose labels are outside the allowed emotion set."""
    with open(json_file_name, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    invalid = {}
    allowed = {emotion.lower() for emotion in configs.ALLOWED_EMOTIONS}
    for key, value in data.items():
        if str(value).lower() not in allowed:
            invalid[int(key)] = value
    return invalid


def merge_json_files(output_path: str, input_files: list[str]) -> str:
    """Merge multiple prediction JSON files into a single output file."""
    merged = {}
    for file_name in input_files:
        with open(file_name, "r", encoding="utf-8") as handle:
            merged.update(json.load(handle))
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2, ensure_ascii=False, sort_keys=True)
    return output_path


def repair_dicts_from_txt(input_file: str, output_json: str) -> tuple[dict[str, object], list[str]]:
    """Repair line-delimited Python-style dictionaries captured in a text file."""
    repaired = {}
    failed = []
    with open(input_file, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            prefix, payload = line.split(":", 1)
            try:
                repaired[prefix.strip()] = ast.literal_eval(payload)
            except Exception:
                failed.append(line)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(repaired, handle, indent=2, ensure_ascii=False)
    return repaired, failed


def from_bad_txt_to_json(txt_path: str, json_path: str) -> dict[str, dict[str, int]]:
    """Extract joint valence/arousal predictions from a malformed text dump."""
    text = Path(txt_path).read_text(encoding="utf-8")
    pattern = re.compile(r"(\d+)\s*:\s*\{\s*valence\s*:\s*'?(-?\d+)'?\s*,\s*arousal\s*:\s*'?(-?\d+)'?\s*\}")
    result = {
        text_id: {"valence": int(valence), "arousal": int(arousal)}
        for text_id, valence, arousal in pattern.findall(text)
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result
