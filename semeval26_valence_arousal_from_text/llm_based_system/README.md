# LLM-based System

These system implements solution for Subtask 1.

## Structure

- `main.py`: CLI entry point for all supported runs
- `configs.py`: shared paths, label mappings, and prompt templates
- `ask_ollama.py`: Ollama client wrapper
- `ask_openai.py`: OpenAI client wrapper
- `words_classification.py`: data loading and words-versus-essays splitting
- `help_functions.py`: shared bucketing helpers
- `user_agnostic_baseline.py`: user-agnostic baseline runner
- `user_agnostic_words_and_essays.py`: user-agnostic split runner
- `user_aware_baseline.py`: user-aware runner for emotions and numerical VA outputs
- `user_dynamic.py`: dynamic prediction runner
- `user_aware_dynamic.py`: user-aware dynamic wrapper
- `data_checks.py`: validation and repair helpers for saved outputs

## Supported use cases

The CLI supports the following experiment modes:

1. `user-agnostic`
   Predict emotions without user-specific history.
2. `user-aware`
   Predict emotions using user-specific labeled history.
3. `user-aware-numeric`
   Predict numerical `valence`, `arousal`, or joint `val_aro` outputs using user-specific history.
4. `user-agnostic-split`
   Predict word lists and essays separately, then merge the outputs.
5. `user-aware-split`
   User-aware prediction with word-list and essay histories split apart.
6. `user-agnostic-dynamic`
   Dynamic prediction using rolling history updates with generic seed history.
7. `user-aware-dynamic`
   Dynamic prediction using rolling user-specific history.

## Basic usage

```bash
python main.py --use-case user-agnostic
python main.py --use-case user-aware
python main.py --use-case user-agnostic-split
python main.py --use-case user-aware-split
python main.py --use-case user-agnostic-dynamic
python main.py --use-case user-aware-dynamic
```

Numerical user-aware prediction:

```bash
python main.py --use-case user-aware-numeric --prompt-type valence
python main.py --use-case user-aware-numeric --prompt-type arousal
python main.py --use-case user-aware-numeric --prompt-type val_aro
```

Using OpenAI instead of Ollama:

```bash
python main.py --use-case user-aware --provider openai --model gpt-5
```

Changing output location:

```bash
python main.py --use-case user-agnostic --output outputs/custom_run.json
```

## Configuration

Environment variables:

- `OPENAI_API_KEY`
- `OLLAMA_BASE_URL`

Common CLI options:

- `--provider ollama|openai`
- `--model MODEL_NAME`
- `--dataset train|test`
- `--output PATH`
- `--buckets N`
- `--essay-buckets N`
- `--word-buckets N`
- `--shots N`
- `--n-pred N`
- `--prompt-type emotion|valence|arousal|val_aro`

## Input data assumptions

This package assumes the same local data layout as the surrounding repository:

- training CSV for the original SemEval data
- test CSV for inference-only runs
- enriched training CSV with an `emotion` column for user-aware and dynamic experiments

The user-aware and dynamic flows depend on data enriched with emotion labels, because they need labeled history examples.

## Output format

Outputs are written to `outputs/` by default.

Typical output formats:

- emotion runs: `{text_id: "Emotion Label"}`
- valence runs: `{text_id: integer}`
- arousal runs: `{text_id: integer}`
- joint numerical runs: `{text_id: {"valence": integer, "arousal": integer}}`

If a model response cannot be parsed, the raw response is saved in a sibling `_BAD.txt` file for later inspection.

## Installation

Minimal package requirements for this folder:

```bash
pip install -r requirements.txt
```
