"""REINFORCE-style RL trainer for ShortGPT."""

import os
import random
from typing import Dict, List

import torch
import torch.nn as nn

from src.config import RLConfig
from src.tokenizer import ShortGPTTokenizer
from src.rl.reward import compute_path_reward


class RLTrainer:
    """
    REINFORCE-style RL finetuning trainer.
    Uses policy gradient to optimize for shortest path generation.
    """

    def __init__(
        self,
        model: nn.Module,
        config: RLConfig,
        tokenizer: ShortGPTTokenizer,
        device: torch.device,
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = device

    def _sample_path(self, row: dict) -> tuple[str, torch.Tensor]:
        """Sample a path and return (generated_string, log_probs)."""
        prompt = (
            row["graph_repr"]
            + "<ORIGIN>" + str(row["origin"])
            + "<DEST>" + str(row["destination"])
            + "<START_PATH>"
        )

        input_ids = torch.tensor(
            [self.tokenizer.encode_string(prompt)],
            dtype=torch.long,
            device=self.device,
        )

        log_probs = []

        for _ in range(self.config.max_new_tokens):
            logits = self.model(input_ids)[:, -1, :]

            if self.config.temperature != 1.0:
                logits = logits / self.config.temperature

            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            next_id = dist.sample()
            log_probs.append(dist.log_prob(next_id))

            input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)

            if self.tokenizer.decode([next_id.item()])[0] == "<END_PATH>":
                break
            if input_ids.size(1) >= self.model.config.max_seq_len:
                break

        generated = "".join(self.tokenizer.decode(input_ids[0].tolist()))

        if not log_probs:
            return generated, torch.zeros(1, device=self.device, requires_grad=True)

        return generated, torch.stack(log_probs)

    def train(self, train_rows: List[dict]) -> Dict[str, list]:
        """Run RL finetuning."""
        self.model.to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        n = len(train_rows)

        for epoch in range(1, self.config.num_epochs + 1):
            print(f"=== RL Epoch {epoch}/{self.config.num_epochs} ===")
            epoch_rewards = []

            for step in range(1, self.config.steps_per_epoch + 1):
                self.model.train()

                # Sample batch
                batch_rows = [train_rows[random.randint(0, n - 1)] for _ in range(self.config.batch_size)]

                log_prob_sums = []
                rewards = []

                for row in batch_rows:
                    generated, log_probs = self._sample_path(row)
                    reward = compute_path_reward(row, generated, self.tokenizer)
                    rewards.append(reward)
                    log_prob_sums.append(log_probs.sum())

                rewards_t = torch.tensor(rewards, device=self.device)
                log_prob_sums_t = torch.stack(log_prob_sums)

                # REINFORCE with baseline
                if self.config.use_baseline:
                    advantages = rewards_t - rewards_t.mean()
                else:
                    advantages = rewards_t

                loss = -(advantages.detach() * log_prob_sums_t).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                optimizer.step()

                avg_reward = rewards_t.mean().item()
                epoch_rewards.append(avg_reward)

                if step % self.config.log_every == 0:
                    print(f"Step {step}: avg_reward={avg_reward:.4f}, loss={loss.item():.4f}")

            print(f"Epoch {epoch} mean reward: {sum(epoch_rewards)/len(epoch_rewards):.4f}")
            self._save(self.config.save_path)

    def _save(self, path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Saved to {path}")
