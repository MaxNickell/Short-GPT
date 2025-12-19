"""Supervised pretraining trainer for ShortGPT."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict

from src.config import PretrainConfig
from src.tokenizer import ShortGPTTokenizer
from .logger import log_metrics


class PretrainTrainer:
    """
    Trainer for supervised pretraining.
    Optimizes cross-entropy loss only on path tokens.
    Uses step-based evaluation and early stopping.
    """

    def __init__(
        self,
        model: nn.Module,
        config: PretrainConfig,
        tokenizer: ShortGPTTokenizer,
        device: torch.device,
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = device

    def _compute_loss_and_acc(self, batch: Dict[str, torch.Tensor]):
        """Compute path-only cross-entropy loss and accuracy."""
        input_ids = batch["input_ids"]
        path_mask = batch["path_token_mask"]

        logits = self.model(input_ids)

        # Shift for next-token prediction
        logits = logits[:, :-1, :]
        labels = input_ids[:, 1:]
        path_mask = path_mask[:, 1:]

        B, T, V = logits.size()
        logits_flat = logits.reshape(B * T, V)
        labels_flat = labels.reshape(B * T)
        mask_flat = path_mask.reshape(B * T).float()

        # Masked cross-entropy
        loss_per_token = nn.functional.cross_entropy(logits_flat, labels_flat, reduction="none")
        num_tokens = mask_flat.sum()

        if num_tokens == 0:
            return torch.tensor(0.0, device=self.device), 0.0, 0.0

        loss = (loss_per_token * mask_flat).sum() / num_tokens

        # Accuracy on path tokens
        preds = logits_flat.argmax(dim=-1)
        correct = ((preds == labels_flat) & mask_flat.bool()).sum()

        return loss, correct.item(), num_tokens.item()

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Run pretraining with step-based evaluation and early stopping."""
        self.model.to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        best_val_loss = float("inf")
        patience_counter = 0
        global_step = 0

        # Running stats for current eval window (reset every eval_every steps)
        window_loss, window_correct, window_tokens = 0.0, 0.0, 0.0

        for epoch in range(1, self.config.num_epochs + 1):
            self.model.train()

            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                optimizer.zero_grad()
                loss, correct, tokens = self._compute_loss_and_acc(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                optimizer.step()

                global_step += 1
                window_loss += loss.item() * tokens
                window_correct += correct
                window_tokens += tokens

                # Print progress for monitoring (but don't log to file)
                if global_step % self.config.log_every == 0:
                    avg_loss = window_loss / window_tokens
                    avg_acc = window_correct / window_tokens
                    print(f"Step {global_step}: loss={avg_loss:.4f}, acc={avg_acc:.2%}")

                # Evaluate and log both train/val at same granularity
                if global_step % self.config.eval_every == 0:
                    # Compute train metrics for this window
                    train_loss = window_loss / window_tokens
                    train_acc = window_correct / window_tokens

                    # Compute val metrics on full val set
                    val_loss, val_acc = self._evaluate(val_loader)

                    print(f"Step {global_step} [EVAL]: "
                          f"train_loss={train_loss:.4f}, train_acc={train_acc:.2%}, "
                          f"val_loss={val_loss:.4f}, val_acc={val_acc:.2%}")

                    # Log both together
                    log_metrics(
                        self.config.log_path,
                        step=global_step,
                        epoch=epoch,
                        train_loss=train_loss,
                        train_acc=train_acc,
                        val_loss=val_loss,
                        val_acc=val_acc,
                    )

                    # Reset window stats
                    window_loss, window_correct, window_tokens = 0.0, 0.0, 0.0

                    # Early stopping check
                    if val_loss < best_val_loss - self.config.min_delta:
                        best_val_loss = val_loss
                        patience_counter = 0
                        self._save(self.config.save_path)
                    else:
                        patience_counter += 1
                        print(f"No improvement. Patience: {patience_counter}/{self.config.patience}")

                        if patience_counter >= self.config.patience:
                            print(f"Early stopping at step {global_step}")
                            return

                    self.model.train()

            print(f"Epoch {epoch} complete")

    def _evaluate(self, loader: DataLoader):
        """Evaluate on a dataset."""
        self.model.eval()
        total_loss, total_correct, total_tokens = 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss, correct, tokens = self._compute_loss_and_acc(batch)
                total_loss += loss.item() * tokens
                total_correct += correct
                total_tokens += tokens

        return total_loss / total_tokens, total_correct / total_tokens

    def _save(self, path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Saved to {path}")
