"""Command-line entry point for experiment package."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import configs
from user_agnostic_baseline import run_user_agnostic_prompt, save_results
from user_agnostic_words_and_essays import run_uag_prompt_essays_and_words
from user_aware_baseline import run_user_aware_prompt_static
from user_dynamic import run_dynamic_prompt
from user_aware_dynamic import run_user_aware_dynamic_prompt


USE_CASES = {
    "user-agnostic": "User agnostic baseline",
    "user-aware": "User aware baseline",
    "user-aware-numeric": "User aware numerical valence/arousal baseline",
    "user-agnostic-split": "User agnostic with words/essay split",
    "user-aware-split": "User aware with words/essay split",
    "user-agnostic-dynamic": "User agnostic dynamic",
    "user-aware-dynamic": "User aware dynamic",
}


def parse_args() -> argparse.Namespace:
    """Parse supported CLI options for all experiment variants."""
    parser = argparse.ArgumentParser(description="Run experiment variants.")
    parser.add_argument("--use-case", choices=USE_CASES.keys(), required=True)
    parser.add_argument("--provider", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--model", default=configs.MODEL_GPT_OSS_120B)
    parser.add_argument("--dataset", choices=["train", "test"], default="train")
    parser.add_argument("--output", help="Output JSON path. Defaults to outputs/<use-case>.json")
    parser.add_argument("--buckets", type=int, default=70, help="Bucket count for user-agnostic runs.")
    parser.add_argument("--essay-buckets", type=int, default=80, help="Essay bucket count for split runs.")
    parser.add_argument("--word-buckets", type=int, default=50, help="Word bucket count for split runs.")
    parser.add_argument("--shots", type=int, default=15, help="Number of in-context examples for user-aware logic.")
    parser.add_argument("--n-pred", type=int, default=5, help="Chunk size for dynamic prediction.")
    parser.add_argument("--words-limit", type=int, default=70)
    parser.add_argument("--prompt-type", choices=["emotion", "valence", "arousal", "val_aro"], default="val_aro", help="Output mode for user-aware-numeric.")
    parser.add_argument("--shuffled", action="store_true", default=False, help="Shuffle user-agnostic buckets instead of preserving chronology.")
    parser.add_argument("--openai-api-key", default=os.getenv(configs.OPENAI_API_KEY_ENV))
    parser.add_argument("--ollama-base-url", default=configs.OLLAMA_DEFAULT_BASE_URL)
    return parser.parse_args()


def default_output_path(use_case: str) -> Path:
    """Return the default output path for a selected use case."""
    configs.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return configs.RESULTS_DIR / f"{use_case}.json"


def main() -> None:
    """Run the selected use case and save the produced predictions."""
    args = parse_args()
    if args.use_case != "user-aware-numeric" and args.prompt_type != "val_aro":
        raise ValueError("--prompt-type is only intended for --use-case user-aware-numeric.")
    if args.use_case == "user-aware-numeric" and args.prompt_type == "emotion":
        raise ValueError("Use --prompt-type valence, arousal, or val_aro with --use-case user-aware-numeric.")

    test_data = args.dataset == "test"
    output_path = Path(args.output) if args.output else default_output_path(args.use_case)

    if args.use_case == "user-agnostic":
        results, failures = run_user_agnostic_prompt(
            provider=args.provider,
            model_name=args.model,
            test_data=test_data,
            shuffled=args.shuffled,
            num_of_buckets=args.buckets,
            openai_api_key=args.openai_api_key,
            ollama_base_url=args.ollama_base_url,
        )
    elif args.use_case == "user-aware":
        results, failures = run_user_aware_prompt_static(
            provider=args.provider,
            model_name=args.model,
            prompt_type="emotion",
            split_words_essays=False,
            n_shot=args.shots,
            openai_api_key=args.openai_api_key,
            ollama_base_url=args.ollama_base_url,
        )
    elif args.use_case == "user-aware-numeric":
        results, failures = run_user_aware_prompt_static(
            provider=args.provider,
            model_name=args.model,
            prompt_type=args.prompt_type,
            split_words_essays=False,
            n_shot=args.shots,
            openai_api_key=args.openai_api_key,
            ollama_base_url=args.ollama_base_url,
        )
    elif args.use_case == "user-agnostic-split":
        results, failures = run_uag_prompt_essays_and_words(
            provider=args.provider,
            model_name=args.model,
            n_bucket_essays=args.essay_buckets,
            n_bucket_words=args.word_buckets,
            words_limit=args.words_limit,
            openai_api_key=args.openai_api_key,
            ollama_base_url=args.ollama_base_url,
        )
    elif args.use_case == "user-aware-split":
        results, failures = run_user_aware_prompt_static(
            provider=args.provider,
            model_name=args.model,
            prompt_type="emotion",
            split_words_essays=True,
            n_shot=args.shots,
            openai_api_key=args.openai_api_key,
            ollama_base_url=args.ollama_base_url,
        )
    elif args.use_case == "user-agnostic-dynamic":
        results, failures = run_dynamic_prompt(
            provider=args.provider,
            model_name=args.model,
            full_text=True,
            n_pred=args.n_pred,
            openai_api_key=args.openai_api_key,
            ollama_base_url=args.ollama_base_url,
        )
    else:
        results, failures = run_user_aware_dynamic_prompt(
            provider=args.provider,
            model_name=args.model,
            n_pred=args.n_pred,
            openai_api_key=args.openai_api_key,
            ollama_base_url=args.ollama_base_url,
        )

    saved_path, failure_path = save_results(output_path, results, failures)
    print(f"Saved {len(results)} predictions to {saved_path}")
    if failure_path is not None:
        print(f"Saved {len(failures)} unparsable responses to {failure_path}")


if __name__ == "__main__":
    main()
