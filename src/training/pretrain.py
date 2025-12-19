"""Supervised pretraining trainer for ShortGPT."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict

from src.config import PretrainConfig
from src.tokenizer import ShortGPTTokenizer


class PretrainTrainer:
    """
    Trainer for supervised pretraining.
    Optimizes cross-entropy loss only on path tokens.
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

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, list]:
        """Run pretraining loop."""
        self.model.to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.config.num_epochs + 1):
            # Train
            self.model.train()
            train_loss, train_correct, train_tokens = 0.0, 0.0, 0.0

            for batch_idx, batch in enumerate(train_loader, 1):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                optimizer.zero_grad()
                loss, correct, tokens = self._compute_loss_and_acc(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                optimizer.step()

                train_loss += loss.item() * tokens
                train_correct += correct
                train_tokens += tokens

                if batch_idx % self.config.log_every == 0:
                    avg_loss = train_loss / train_tokens
                    avg_acc = train_correct / train_tokens
                    print(f"Epoch {epoch} batch {batch_idx}: loss={avg_loss:.4f}, acc={avg_acc:.2%}")

            # Validate
            val_loss, val_acc = self._evaluate(val_loader)
            train_loss /= train_tokens
            train_acc = train_correct / train_tokens

            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.2%}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.2%}")

            # Early stopping
            if val_loss < best_val_loss - self.config.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                self._save(self.config.save_path)
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

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
