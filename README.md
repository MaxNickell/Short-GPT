# ShortGPT

A decoder-only transformer for shortest path prediction in graphs. This project investigates whether reinforcement learning (algorithmic alignment) improves performance over supervised pretraining.

## Task

Given a graph serialized as a string and an origin-destination pair, predict the shortest path.

**Input format:**
```
<EDGE>1<BD>2<EDGE>2<BD>3<EDGE>1<BD>3<ORIGIN>1<DEST>3<START_PATH>
```

**Output:**
```
1<TO>3<END_PATH>
```

## Training

### Phase 1: Supervised Pretraining

Cross-entropy loss computed **only on path tokens** (not the graph representation):

$$\mathcal{L}_{\text{pretrain}} = -\frac{1}{|P|} \sum_{t \in P} \log p_\theta(x_t | x_{<t})$$

where $P$ is the set of path token positions (from `<START_PATH>` to `<END_PATH>`).

### Phase 2: RL Finetuning

REINFORCE with baseline for variance reduction:

$$\mathcal{L}_{\text{RL}} = -\frac{1}{B} \sum_{i=1}^{B} (R_i - \bar{R}) \sum_{t} \log \pi_\theta(a_t^{(i)} | s_t^{(i)})$$

where:
- $R_i \in [0, 1]$ is the reward for trajectory $i$
- $\bar{R} = \frac{1}{B}\sum_i R_i$ is the batch baseline
- $\pi_\theta$ is the policy (model)

**Reward function:**
- $R = 0$ if path is invalid (wrong structure, invalid edges, wrong endpoints)
- $R = \max(1 - \frac{L - L^*}{L^*}, 0)$ if path is valid, where $L$ is generated path length and $L^*$ is optimal

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Exact Match** | Generated path exactly matches ground truth |
| **Valid Path** | Path has correct structure and uses valid graph edges |
| **Optimal Rate** | Path achieves optimal length ($R = 1.0$) |
| **Avg Reward** | Mean reward across test set |

## Usage

```bash
# Pretrain
python train_pretrain.py --data data/processed/merged.jsonl

# RL finetune
python train_rl.py --data data/processed/merged.jsonl --checkpoint checkpoints/pretrained.pt

# Evaluate
python evaluate.py --data data/processed/merged.jsonl \
    --checkpoint checkpoints/pretrained.pt \
    --compare checkpoints/rl_finetuned.pt
```

## Configuration

Edit `src/config.py` to modify hyperparameters:

- `ShortGPTConfig`: model architecture (d_model, n_layers, n_heads, etc.)
- `DataConfig`: train/val/test split fractions and seed
- `PretrainConfig`: pretraining hyperparameters (lr, batch_size, patience, etc.)
- `RLConfig`: RL hyperparameters (lr, temperature, steps_per_epoch, etc.)

## Project Structure

```
├── train_pretrain.py      # Pretraining entry point
├── train_rl.py            # RL finetuning entry point
├── evaluate.py            # Evaluation script
├── src/
│   ├── config.py          # All configurations
│   ├── tokenizer.py       # Fixed vocabulary tokenizer
│   ├── model/             # Transformer architecture
│   ├── training/          # Trainer classes
│   └── rl/reward.py       # Reward function
├── data/
│   ├── dataset.py         # PyTorch dataset
│   └── splits.py          # Consistent train/val/test splitting
└── scripts/graph_generation/  # Dataset generation utilities
```
