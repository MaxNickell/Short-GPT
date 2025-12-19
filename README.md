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

$$\mathcal{L}_{\text{pretrain}} = -\frac{1}{|P|} \sum_{t \in P} \log P_\theta(x_t | x_{<t})$$

where $P$ is the set of path token positions (from `<START_PATH>` to `<END_PATH>`).

### Phase 2: RL Finetuning (REINFORCE)

**Objective:** Maximize expected reward over generated trajectories:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

**Policy Gradient (with baseline):**

$$\nabla_\theta J = \mathbb{E}_{\tau \sim \pi_\theta}\left[A(\tau) \cdot \nabla_\theta \log \pi_\theta(\tau)\right]$$

**Trajectory Log-Probability:** For autoregressive generation:

$$\log \pi_\theta(\tau) = \sum_{t=1}^{T} \log P_\theta(a_t | a_{1:t-1})$$

**Loss Function:**

$$\mathcal{L}_{\text{RL}} = -\frac{1}{N} \sum_{i=1}^{N} A_i \cdot \sum_{t=1}^{T_i} \log P_\theta(a_t^{(i)} | a_{<t}^{(i)})$$

**Leave-One-Out Baseline:** To reduce variance without introducing bias, we use a leave-one-out baseline where each sample's baseline excludes its own reward:

$$b_i = \frac{1}{N-1} \sum_{j \neq i} R_j$$

$$A_i = R_i - b_i$$

### Dense Reward Function

Three cases with distinct reward ranges:

**Case 1 — Invalid Structure** ($R = -1$): Output does not follow `<START_PATH>node<TO>...<TO>node<END_PATH>` format.

**Case 2 — Valid Structure, Invalid Path** ($R \in [-1, 0)$): Correct format but path uses non-existent edges or wrong endpoints.

$$R = -0.5 + \frac{|\text{valid edges}|}{|\text{total edges}|} \cdot 0.5 - 0.25 \cdot \mathbb{1}[\text{wrong origin}] - 0.25 \cdot \mathbb{1}[\text{wrong dest}]$$

**Case 3 — Valid Path** ($R \in (1, 2]$): All edges exist and path connects origin to destination.

$$R = 1 + \frac{L^*}{L}$$

where $L^*$ is the optimal path length and $L$ is the generated path length.

| Case | Reward Range | Gradient Effect |
|------|--------------|-----------------|
| Invalid structure | $-1$ | Strong decrease |
| Invalid path | $[-1, 0)$ | Moderate decrease |
| Valid path | $(1, 2]$ | Increase (optimal gets strongest) |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Exact Match** | Generated path exactly matches ground truth |
| **Valid Path** | Path uses only valid graph edges |
| **Optimal Rate** | Path achieves optimal length ($R = 2$) |
| **Avg Reward** | Mean reward across test set |

## Usage

```bash
# Pretrain
python train_pretrain.py --data data/processed/merged_final.jsonl

# RL finetune
python train_rl.py --data data/processed/merged_final.jsonl --checkpoint checkpoints/pretrained.pt

# Evaluate (compare pretrained vs RL)
python evaluate.py --data data/processed/merged_final.jsonl \
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
└── scripts/               # Dataset generation and testing utilities
```
