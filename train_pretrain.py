#!/usr/bin/env python3
"""Pretrain ShortGPT on shortest path data."""

import argparse
import torch
from torch.utils.data import DataLoader

from src.config import ShortGPTConfig, DataConfig, PretrainConfig
from src.tokenizer import ShortGPTTokenizer
from src.model import ShortGPT
from src.training import PretrainTrainer
from data import GraphPathDataset, GraphPathCollator, get_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to JSONL data")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load configs
    model_config = ShortGPTConfig()
    data_config = DataConfig()
    train_config = PretrainConfig()

    # Setup
    tokenizer = ShortGPTTokenizer()
    model = ShortGPT(model_config)
    print(f"Model: {model.get_num_params():,} parameters")

    # Load data with consistent splits
    train_rows, val_rows, test_rows = get_splits(args.data, data_config)
    print(f"Train: {len(train_rows)}, Val: {len(val_rows)}, Test: {len(test_rows)}")

    train_ds = GraphPathDataset(train_rows, tokenizer, model_config.max_seq_len)
    val_ds = GraphPathDataset(val_rows, tokenizer, model_config.max_seq_len)

    collator = GraphPathCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, train_config.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, train_config.batch_size, shuffle=False, collate_fn=collator)

    # Train
    trainer = PretrainTrainer(model, train_config, tokenizer, device)
    trainer.train(train_loader, val_loader)

    print(f"Done. Checkpoint saved to {train_config.save_path}")


if __name__ == "__main__":
    main()
