"""Central configuration for the neural regression package."""

from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

DEFAULT_MODEL_NAME = "FacebookAI/roberta-base"
DEFAULT_CHECKPOINT_NAME = "best_model.pth"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "outputs"

DEFAULT_TRAIN_PATH = PROJECT_ROOT / "train_data_enriched_2301_label_5padded.csv"
DEFAULT_TEST_PATH = PROJECT_ROOT / "semeval_test_subtask2_5padded.csv"
DEFAULT_TRAIN_LIKE_INFERENCE_PATH = PROJECT_ROOT / "train_data_padded.csv"
DEFAULT_TEST_LIKE_INFERENCE_PATH = PROJECT_ROOT / "test_data_padded.csv"

DEFAULT_WINDOW_SIZE = 4
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_EPOCHS = 10
DEFAULT_HIDDEN_DIM = 128
DEFAULT_USER_EMB_DIM = 8
DEFAULT_LR = 5e-4
DEFAULT_MAX_TEXT_LENGTH = 128

DEFAULT_PREDICT_TARGET = "both"
DEFAULT_USE_TEXT = False
DEFAULT_USE_WORDS = False
DEFAULT_TRAIN_DATASET_FLAG = False


def build_default_output_path(
    predict_target: str,
    window_size: int,
    model_name: str,
    user_emb_dim: int,
    use_text: bool,
    use_words: bool,
    action: str,
) -> Path:
    """Build a descriptive default output filename for inference results."""
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if use_text:
        suffix = "withwords" if use_words else "withtext"
    else:
        suffix = "notext"
    model_slug = model_name.replace("/", "_")
    filename = f"{action}_window_{window_size}_{model_slug}_user{user_emb_dim}_{predict_target}_{suffix}.csv"
    return DEFAULT_OUTPUT_DIR / filename
