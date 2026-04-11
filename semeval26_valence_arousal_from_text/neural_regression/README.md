# Neural Regression

##  Structure

- `main.py`: core model code plus CLI for training and inference
- `config.py`: default file paths, model names, and training/inference defaults

## What the model does

The model:

1. groups examples by user
2. builds history windows
3. encodes text with a transformer model when text features are enabled
4. combines text, current valence/arousal values, recent momentum, and user embeddings
5. predicts the next change in:
   - `valence`
   - `arousal`
   - or `both`

## CLI usage

```bash
python neural_regression/main.py --mode full
```

### Train only

```bash
python neural_regression/main.py --mode train --predict-target both
```

### Infer only

```bash
python neural_regression/main.py --mode infer --predict-target valence --checkpoint-path neural_regression/best_model.pth
```

```bash
python neural_regression/main.py --mode infer --predict-target arousal --checkpoint-path neural_regression/best_model.pth
```

### Train and infer in one run

```bash
python neural_regression/main.py --mode full --predict-target both
```

## Important arguments

- `--mode train|infer|full`
- `--predict-target valence|arousal|both`
- `--train-path PATH`
- `--test-path PATH`
- `--checkpoint-path PATH`
- `--output-path PATH`
- `--model-name MODEL_ID`
- `--window-size N`
- `--batch-size N`
- `--num-epochs N`
- `--hidden-dim N`
- `--user-emb-dim N`
- `--lr FLOAT`
- `--use-text`
- `--use-words`
- `--train-dataset`

To inspect all options:

```bash
python neural_regression/main.py --help
```

## Default paths

Defaults are defined in `config.py`. They currently point to the existing project files.
You need to pass your custom paths.

## Output format

Inference writes a CSV with:

- `user_id`
- `text_id_ordered`
- `pred_state_change_valence`
- `pred_state_change_arousal`

For single-target runs, the non-predicted target is filled with `0.0` so the output schema stays consistent.
