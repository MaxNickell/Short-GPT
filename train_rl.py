#!/usr/bin/env python3
"""RL finetune ShortGPT for shortest path generation."""

import argparse
import torch

from src.config import ShortGPTConfig, DataConfig, RLConfig
from src.tokenizer import ShortGPTTokenizer
from src.model import ShortGPT
from src.training import RLTrainer
from data import get_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to JSONL data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load configs
    model_config = ShortGPTConfig()
    data_config = DataConfig()
    rl_config = RLConfig()

    # Load pretrained model
    tokenizer = ShortGPTTokenizer()
    model = ShortGPT.from_pretrained(args.checkpoint, model_config)
    print(f"Loaded model from {args.checkpoint}")

    # Load data with SAME splits as pretraining
    train_rows, _, _ = get_splits(args.data, data_config)
    print(f"Training on {len(train_rows)} examples")

    # Train
    trainer = RLTrainer(model, rl_config, tokenizer, device)
    trainer.train(train_rows)

    print(f"Done. Checkpoint saved to {rl_config.save_path}")


if __name__ == "__main__":
    main()
