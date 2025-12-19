from dataclasses import dataclass


@dataclass
class ShortGPTConfig:
    """Model architecture configuration."""
    vocab_size: int = 24
    d_model: int = 256
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 1024
    max_seq_len: int = 640
    rope_base: int = 10000
    dropout: float = 0.1


@dataclass
class DataConfig:
    """Data splitting configuration. Used by all training phases."""
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 42


@dataclass
class PretrainConfig:
    """Pretraining hyperparameters."""
    num_epochs: int = 20
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Step-based evaluation and early stopping
    eval_every: int = 5000      # Evaluate every N steps
    patience: int = 3           # Stop after N evals without improvement
    min_delta: float = 0.001    # Minimum improvement to reset patience

    log_every: int = 500
    save_path: str = "checkpoints/pretrained.pt"
    log_path: str = "logs/pretrain.jsonl"


@dataclass
class RLConfig:
    """RL finetuning hyperparameters."""
    num_epochs: int = 1
    steps_per_epoch: int = 1000
    batch_size: int = 32
    lr: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_new_tokens: int = 64
    temperature: float = 1.0
    use_baseline: bool = True
    log_every: int = 100
    save_path: str = "checkpoints/rl_finetuned.pt"
    log_path: str = "logs/rl.jsonl"
