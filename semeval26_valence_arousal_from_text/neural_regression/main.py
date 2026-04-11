"""CLI entry point and core training/inference logic for neural regression."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import config


class StateChangeDataset(Dataset):
    """Training dataset that creates sliding-window state change instances per user."""

    def __init__(
        self,
        csv_path: str | Path,
        tokenizer,
        window_size: int = 5,
        max_text_length: int = config.DEFAULT_MAX_TEXT_LENGTH,
        predict_target: str = "valence",
        use_words: bool = False,
        train_dataset: bool = True,
    ) -> None:
        """Load a padded training CSV and construct history-to-next-step examples."""
        if predict_target not in {"valence", "arousal", "both"}:
            raise ValueError("predict_target must be 'valence', 'arousal', or 'both'")

        self.predict_target = predict_target
        self.df = pd.read_csv(csv_path)
        if use_words:
            self.df = self.df.drop(columns=["text"])
            self.df = self.df.rename(columns={"vector_10_binary_llm_text2": "text"})

        drop_cols = [
            "text_id",
            "timestamp",
            "collection_phase",
            "vector_10_soft",
            "state_change_val",
            "state_change_aro",
            "train",
            "is_words",
        ]
        existing_cols = [col for col in drop_cols if col in self.df.columns]
        self.df = self.df.drop(columns=existing_cols)
        self.df = self.df.loc[:, ~self.df.columns.str.startswith("vector")]

        self.tokenizer = tokenizer
        self.window_size = window_size
        self.max_text_length = max_text_length
        self.user_sequences: dict[int, pd.DataFrame] = {}
        for user_id, group in self.df.groupby("user_id"):
            ordered_group = group.sort_values("text_id_ordered")
            if train_dataset:
                ordered_group = ordered_group.iloc[:-1]
            self.user_sequences[int(user_id)] = ordered_group.reset_index(drop=True)

        self.instances: list[tuple[int, int]] = []
        seen_users: list[int] = []
        for user_id, sequence in self.user_sequences.items():
            if len(sequence) < 2:
                continue
            min_history = min(window_size - 1, len(sequence) - 1)
            for time_index in range(min_history, len(sequence) - 1):
                self.instances.append((user_id, time_index))
                seen_users.append(user_id)

        unique_users = sorted(set(seen_users))
        self.user2idx = {user_id: index for index, user_id in enumerate(unique_users)}
        self.idx2user = {index: user_id for user_id, index in self.user2idx.items()}
        print(f"Dataset: {len(self.instances)} instances from {len(self.user2idx)} users")

    def __len__(self) -> int:
        """Return the number of training instances."""
        return len(self.instances)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | list[list[int]]]:
        """Return one windowed training example and its target delta."""
        user_id, time_index = self.instances[idx]
        sequence = self.user_sequences[user_id]

        start = time_index - self.window_size + 1
        end = time_index + 1
        window = sequence.iloc[start:end]

        texts = window["text"].tolist()
        tokenized = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors=None,
        )
        va_values = torch.tensor(window[["valence", "arousal"]].values, dtype=torch.float)

        valence_now = sequence.iloc[time_index]["valence"]
        arousal_now = sequence.iloc[time_index]["arousal"]
        valence_next = sequence.iloc[time_index + 1]["valence"]
        arousal_next = sequence.iloc[time_index + 1]["arousal"]

        if self.predict_target == "valence":
            delta = torch.tensor([valence_next - valence_now], dtype=torch.float)
        elif self.predict_target == "arousal":
            delta = torch.tensor([arousal_next - arousal_now], dtype=torch.float)
        else:
            delta = torch.tensor([valence_next - valence_now, arousal_next - arousal_now], dtype=torch.float)

        return {
            "user_id": self.user2idx[user_id],
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "va_values": va_values,
            "delta": delta,
        }


class StateChangeInferenceDataset(Dataset):
    """Inference dataset with one sample per user based on available history."""

    def __init__(
        self,
        csv_path: str | Path,
        tokenizer,
        window_size: int,
        user2idx: dict[int, int],
        unknown_id: int,
        max_text_length: int = config.DEFAULT_MAX_TEXT_LENGTH,
        predict_target: str = "valence",
    ) -> None:
        """Load an inference CSV and prepare one history window per target user."""
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.user2idx = user2idx
        self.unknown_id = unknown_id
        self.max_text_length = max_text_length
        self.predict_target = predict_target
        self.samples: list[dict[str, object]] = []

        for user_id, user_df in self.df.groupby("user_id"):
            ordered_df = user_df.sort_values("text_id_ordered")
            user_idx = self.user2idx.get(int(user_id), self.unknown_id)
            target_row = ordered_df.iloc[-1]
            target_index = len(ordered_df) - 1
            start_index = max(0, target_index - window_size)
            history_df = ordered_df.iloc[start_index:target_index]
            if len(history_df) == 0:
                history_df = ordered_df.iloc[target_index : target_index + 1]

            texts = history_df["text"].tolist()
            va_values = history_df[["valence", "arousal"]].values.tolist()

            input_ids = []
            for text in texts:
                encoded = self.tokenizer(
                    text,
                    truncation=True,
                    padding=False,
                    max_length=self.max_text_length,
                    return_attention_mask=False,
                )
                input_ids.append(encoded["input_ids"])

            dummy_delta = torch.tensor([0.0, 0.0], dtype=torch.float) if predict_target == "both" else torch.tensor([0.0], dtype=torch.float)
            self.samples.append(
                {
                    "user_id": user_idx,
                    "original_user_id": int(user_id),
                    "input_ids": input_ids,
                    "va_values": torch.tensor(va_values, dtype=torch.float),
                    "text_id_ordered": int(target_row["text_id_ordered"]),
                    "delta": dummy_delta,
                }
            )

        print(f"Inference dataset: {len(self.samples)} samples")

    def __len__(self) -> int:
        """Return the number of inference samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        """Return one prepared inference sample."""
        return self.samples[idx]


def state_change_collate_fn(batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
    """Pad nested token sequences and numeric histories into a single batch."""
    user_id = torch.tensor([item["user_id"] for item in batch], dtype=torch.long)
    va_sequences = [item["va_values"] for item in batch]
    va_padded = pad_sequence(va_sequences, batch_first=True, padding_value=0.0)

    nested_input_ids = []
    nested_attention_masks = []
    for item in batch:
        instance_sequences = []
        instance_masks = []
        for sequence in item["input_ids"]:
            tensor = torch.tensor(sequence, dtype=torch.long)
            instance_sequences.append(tensor)
            instance_masks.append(torch.ones_like(tensor))
        nested_input_ids.append(instance_sequences)
        nested_attention_masks.append(instance_masks)

    padded_input_instances = []
    padded_mask_instances = []
    for sequences, masks in zip(nested_input_ids, nested_attention_masks):
        padded_input_instances.append(pad_sequence(sequences, batch_first=True, padding_value=0))
        padded_mask_instances.append(pad_sequence(masks, batch_first=True, padding_value=0))

    max_num_sequences = max(item.size(0) for item in padded_input_instances)
    max_sequence_length = max(item.size(1) for item in padded_input_instances)
    batch_size = len(batch)

    input_ids_batch = torch.zeros(batch_size, max_num_sequences, max_sequence_length, dtype=torch.long)
    attention_mask_batch = torch.zeros(batch_size, max_num_sequences, max_sequence_length, dtype=torch.long)
    for index, (input_tensor, mask_tensor) in enumerate(zip(padded_input_instances, padded_mask_instances)):
        input_ids_batch[index, : input_tensor.size(0), : input_tensor.size(1)] = input_tensor
        attention_mask_batch[index, : mask_tensor.size(0), : mask_tensor.size(1)] = mask_tensor

    delta = torch.stack([item["delta"] for item in batch])
    return {
        "user_id": user_id,
        "input_ids": input_ids_batch,
        "attention_mask": attention_mask_batch,
        "va_values": va_padded,
        "delta": delta,
    }


class TextEncoderWrapper(nn.Module):
    """Small wrapper around a transformer text encoder with mean pooling."""

    def __init__(self, encoder: AutoModel) -> None:
        """Store the frozen transformer encoder."""
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode texts and return mean pooled last-hidden-state embeddings."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state.mean(dim=1)


class SimpleStateChangeModel(nn.Module):
    """Regression model that predicts the next valence/arousal change from recent context."""

    def __init__(
        self,
        num_users: int,
        text_emb_dim: int,
        hidden_dim: int = 64,
        user_emb_dim: int = 16,
        predict_target: str = "valence",
        use_text: bool = True,
    ) -> None:
        """Initialize the regression network and user embedding table."""
        super().__init__()
        self.predict_target = predict_target
        self.use_text = use_text
        self.user_embedding = nn.Embedding(num_users + 1, user_emb_dim)
        self.unknown_user_id = num_users
        input_dim = (text_emb_dim if use_text else 0) + 2 + 2
        output_dim = 2 if predict_target == "both" else 1
        self.regressor = nn.Sequential(
            nn.Linear(input_dim + user_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, text_embeddings: torch.Tensor, va_values: torch.Tensor, user_id: torch.Tensor) -> torch.Tensor:
        """Predict the next delta from the latest state, momentum, and user embedding."""
        user_id = torch.clamp(user_id, 0, self.unknown_user_id)
        user_emb = self.user_embedding(user_id)
        current_va = va_values[:, -1, :]
        if va_values.size(1) > 1:
            previous_delta = (va_values[:, -1:, :] - va_values[:, -2:-1, :]).squeeze(1)
        else:
            previous_delta = torch.zeros_like(current_va)

        if self.use_text:
            current_text = text_embeddings[:, -1, :]
            features = torch.cat([current_text, current_va, previous_delta, user_emb], dim=-1)
        else:
            features = torch.cat([current_va, previous_delta, user_emb], dim=-1)
        return self.regressor(features)


def _create_text_encoder(model_name: str, device: torch.device) -> tuple[AutoTokenizer, TextEncoderWrapper]:
    """Load the tokenizer and frozen transformer encoder used for text features."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder_model = AutoModel.from_pretrained(model_name)
    encoder_model.eval()
    for parameter in encoder_model.parameters():
        parameter.requires_grad = False
    return tokenizer, TextEncoderWrapper(encoder_model).to(device)


def train_model(
    train_path: str | Path,
    checkpoint_path: str | Path,
    window_size: int = config.DEFAULT_WINDOW_SIZE,
    batch_size: int = config.DEFAULT_BATCH_SIZE,
    num_epochs: int = config.DEFAULT_NUM_EPOCHS,
    hidden_dim: int = config.DEFAULT_HIDDEN_DIM,
    user_emb_dim: int = config.DEFAULT_USER_EMB_DIM,
    lr: float = config.DEFAULT_LR,
    predict_target: str = config.DEFAULT_PREDICT_TARGET,
    use_text: bool = config.DEFAULT_USE_TEXT,
    model_name: str = config.DEFAULT_MODEL_NAME,
    use_words: bool = config.DEFAULT_USE_WORDS,
    train_dataset: bool = config.DEFAULT_TRAIN_DATASET_FLAG,
) -> dict[str, object]:
    """Train the regression model and save the best checkpoint to disk."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Prediction target: {predict_target}")
    print(f"Window size: {window_size}")
    print(f"Using text: {use_text}")

    tokenizer, text_encoder = _create_text_encoder(model_name, device)
    dataset = StateChangeDataset(
        csv_path=train_path,
        tokenizer=tokenizer,
        window_size=window_size,
        predict_target=predict_target,
        use_words=use_words,
        train_dataset=train_dataset,
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=state_change_collate_fn)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=state_change_collate_fn)

    model = SimpleStateChangeModel(
        num_users=len(dataset.user2idx),
        text_emb_dim=768,
        hidden_dim=hidden_dim,
        user_emb_dim=user_emb_dim,
        predict_target=predict_target,
        use_text=use_text,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    best_val_loss = float("inf")
    best_checkpoint: dict[str, object] | None = None
    patience_counter = 0
    max_patience = 5

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        for batch in train_bar:
            user_id = batch["user_id"].to(device)
            va_values = batch["va_values"].to(device)
            delta_true = batch["delta"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_size_actual, steps, sequence_length = input_ids.shape

            if use_text:
                with torch.no_grad():
                    text_embeddings = text_encoder(
                        input_ids.view(batch_size_actual * steps, sequence_length),
                        attention_mask.view(batch_size_actual * steps, sequence_length),
                    ).view(batch_size_actual, steps, -1)
            else:
                text_embeddings = torch.zeros(batch_size_actual, steps, 1, device=device)

            delta_pred = model(text_embeddings, va_values, user_id)
            loss = criterion(delta_pred, delta_true)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * batch_size_actual
            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_subset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]")
            for batch in val_bar:
                user_id = batch["user_id"].to(device)
                va_values = batch["va_values"].to(device)
                delta_true = batch["delta"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                batch_size_actual, steps, sequence_length = input_ids.shape

                if use_text:
                    text_embeddings = text_encoder(
                        input_ids.view(batch_size_actual * steps, sequence_length),
                        attention_mask.view(batch_size_actual * steps, sequence_length),
                    ).view(batch_size_actual, steps, -1)
                else:
                    text_embeddings = torch.zeros(batch_size_actual, steps, 1, device=device)

                delta_pred = model(text_embeddings, va_values, user_id)
                loss = criterion(delta_pred, delta_true)
                val_loss += loss.item() * batch_size_actual
                val_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        val_loss /= len(val_subset)
        print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_checkpoint = {
                "model_state_dict": model.state_dict(),
                "user2idx": dataset.user2idx,
                "num_users": len(dataset.user2idx),
                "window_size": window_size,
                "hidden_dim": hidden_dim,
                "user_emb_dim": user_emb_dim,
                "predict_target": predict_target,
                "use_text": use_text,
                "val_loss": val_loss,
                "model_name": model_name,
            }
            torch.save(best_checkpoint, checkpoint_path)
            print(f"Saved best model to {checkpoint_path} (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    if best_checkpoint is None:
        raise RuntimeError("Training finished without producing a checkpoint.")
    return best_checkpoint


def run_inference(
    test_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    batch_size: int = config.DEFAULT_BATCH_SIZE,
    model_name: str | None = None,
) -> pd.DataFrame:
    """Load a checkpoint, run inference, and save predictions as a CSV file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    user2idx = checkpoint["user2idx"]
    num_users = checkpoint["num_users"]
    window_size = checkpoint["window_size"]
    hidden_dim = checkpoint["hidden_dim"]
    user_emb_dim = checkpoint["user_emb_dim"]
    predict_target = checkpoint.get("predict_target", "valence")
    use_text = checkpoint.get("use_text", True)
    encoder_name = model_name or checkpoint.get("model_name", config.DEFAULT_MODEL_NAME)

    print(f"Loaded checkpoint with {num_users} training users")
    print(f"Prediction target: {predict_target}")
    print(f"Using text: {use_text}")

    tokenizer, text_encoder = _create_text_encoder(encoder_name, device)
    model = SimpleStateChangeModel(
        num_users=num_users,
        text_emb_dim=768,
        hidden_dim=hidden_dim,
        user_emb_dim=user_emb_dim,
        predict_target=predict_target,
        use_text=use_text,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    inference_dataset = StateChangeInferenceDataset(
        csv_path=test_path,
        tokenizer=tokenizer,
        window_size=window_size,
        user2idx=user2idx,
        unknown_id=num_users,
        predict_target=predict_target,
    )
    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, collate_fn=state_change_collate_fn)

    prediction_rows: list[dict[str, float | int]] = []
    sample_cursor = 0
    with torch.no_grad():
        for batch in tqdm(inference_loader, desc="Inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            va_values = batch["va_values"].to(device)
            user_id = batch["user_id"].to(device)
            batch_size_actual, steps, sequence_length = input_ids.shape

            if use_text:
                text_embeddings = text_encoder(
                    input_ids.view(batch_size_actual * steps, sequence_length),
                    attention_mask.view(batch_size_actual * steps, sequence_length),
                ).view(batch_size_actual, steps, -1)
            else:
                text_embeddings = torch.zeros(batch_size_actual, steps, 1, device=device)

            delta_pred = model(text_embeddings=text_embeddings, va_values=va_values, user_id=user_id).cpu()
            for row_index in range(batch_size_actual):
                sample = inference_dataset.samples[sample_cursor]
                row = {
                    "user_id": int(sample["original_user_id"]),
                    "text_id_ordered": int(sample["text_id_ordered"]),
                }
                if predict_target == "both":
                    row["pred_state_change_valence"] = float(delta_pred[row_index][0].item())
                    row["pred_state_change_arousal"] = float(delta_pred[row_index][1].item())
                elif predict_target == "arousal":
                    row["pred_state_change_valence"] = 0.0
                    row["pred_state_change_arousal"] = float(delta_pred[row_index].item())
                else:
                    row["pred_state_change_valence"] = float(delta_pred[row_index].item())
                    row["pred_state_change_arousal"] = 0.0
                prediction_rows.append(row)
                sample_cursor += 1

    prediction_df = pd.DataFrame(prediction_rows).sort_values(["user_id", "text_id_ordered"]).reset_index(drop=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    print(prediction_df.head(10))
    return prediction_df


def run_train_and_inference(args: argparse.Namespace) -> None:
    """Run training followed immediately by inference using the new checkpoint."""
    train_model(
        train_path=args.train_path,
        checkpoint_path=args.checkpoint_path,
        window_size=args.window_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        hidden_dim=args.hidden_dim,
        user_emb_dim=args.user_emb_dim,
        lr=args.lr,
        predict_target=args.predict_target,
        use_text=args.use_text,
        model_name=args.model_name,
        use_words=args.use_words,
        train_dataset=args.train_dataset,
    )
    run_inference(
        test_path=args.test_path,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        model_name=args.model_name,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training, inference, or both."""
    parser = argparse.ArgumentParser(description="Train and run the neural regression state-change model.")
    parser.add_argument("--mode", choices=["train", "infer", "full"], default="full")
    parser.add_argument("--predict-target", choices=["valence", "arousal", "both"], default=config.DEFAULT_PREDICT_TARGET)
    parser.add_argument("--train-path", default=str(config.DEFAULT_TRAIN_PATH))
    parser.add_argument("--test-path", default=str(config.DEFAULT_TEST_PATH))
    parser.add_argument("--checkpoint-path", default=str(config.PACKAGE_DIR / config.DEFAULT_CHECKPOINT_NAME))
    parser.add_argument("--output-path")
    parser.add_argument("--model-name", default=config.DEFAULT_MODEL_NAME)
    parser.add_argument("--window-size", type=int, default=config.DEFAULT_WINDOW_SIZE)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-epochs", type=int, default=config.DEFAULT_NUM_EPOCHS)
    parser.add_argument("--hidden-dim", type=int, default=config.DEFAULT_HIDDEN_DIM)
    parser.add_argument("--user-emb-dim", type=int, default=config.DEFAULT_USER_EMB_DIM)
    parser.add_argument("--lr", type=float, default=config.DEFAULT_LR)
    parser.add_argument("--use-text", action="store_true")
    parser.add_argument("--use-words", action="store_true")
    parser.add_argument("--train-dataset", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Execute the selected CLI mode."""
    args = parse_args()
    if not args.output_path:
        args.output_path = str(
            config.build_default_output_path(
                predict_target=args.predict_target,
                window_size=args.window_size,
                model_name=args.model_name,
                user_emb_dim=args.user_emb_dim,
                use_text=args.use_text,
                use_words=args.use_words,
                action=args.mode,
            )
        )

    if args.mode == "train":
        train_model(
            train_path=args.train_path,
            checkpoint_path=args.checkpoint_path,
            window_size=args.window_size,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            hidden_dim=args.hidden_dim,
            user_emb_dim=args.user_emb_dim,
            lr=args.lr,
            predict_target=args.predict_target,
            use_text=args.use_text,
            model_name=args.model_name,
            use_words=args.use_words,
            train_dataset=args.train_dataset,
        )
    elif args.mode == "infer":
        run_inference(
            test_path=args.test_path,
            checkpoint_path=args.checkpoint_path,
            output_path=args.output_path,
            batch_size=args.batch_size,
            model_name=args.model_name,
        )
    else:
        run_train_and_inference(args)


if __name__ == "__main__":
    main()
