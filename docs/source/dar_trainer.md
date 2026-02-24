# DAR Trainer

[![model badge](https://img.shields.io/badge/All_models-DAR-blue)](https://huggingface.co/models?other=dar,trl)

## Overview

TRL supports the DAR Trainer for alignment with scalar rewards, based on [Direct Advantage Regression: Aligning LLMs with Online AI Reward](https://huggingface.co/papers/2504.14177).

DAR is an online, RL-free method that performs weighted supervised fine-tuning over sampled completions. For each prompt, DAR:

1. Samples \\(K\\) completions.
2. Computes scalar rewards for each completion (from a reward model, reward function, or offline reward labels).
3. Builds a clipped importance weight from:
   - normalized advantage (reward-centered),
   - regularization toward a reference distribution and a behavior/sampling distribution.
4. Minimizes weighted negative log-likelihood.

In TRL, DAR can run in:

- **Online mode** with prompt-only data (sampling completions during training).
- **Offline mode** with precomputed completions and rewards/behavior log-probs.

## Quick start

This example trains DAR online on prompt-only data, using a reward model:

```python
from datasets import load_dataset
from trl import DARConfig, DARTrainer

train_dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train")

training_args = DARConfig(
    output_dir="Qwen2-0.5B-DAR",
    learning_rate=5e-7,
    dar_k=2,
    num_train_epochs=1,
)

trainer = DARTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    args=training_args,
    train_dataset=train_dataset,
    reward_model="Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback",
)
trainer.train()
```

Run:

```bash
accelerate launch train_dar.py
```

## Expected dataset type

DAR supports multiple formats:

- **Prompt-only** ([prompt-only dataset](dataset_formats#prompt-only)): online sampling + reward scoring.
- **Preference** ([preference dataset](dataset_formats#preference)): trainer automatically maps `chosen/rejected` pairs to two completions with rewards `[1.0, 0.0]`.
- **Prompt-completion-reward**: rows containing `prompt`, `completion`, and `reward`/`rewards`.

For fully offline DAR, you can also pass precomputed `behavior_logps`.

## Example script

A reference script is available at [`trl/scripts/dar.py`](https://github.com/huggingface/trl/blob/main/trl/scripts/dar.py).

Example command:

```bash
accelerate launch trl/scripts/dar.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/ultrafeedback-prompt \
    --reward_model_name_or_path Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback \
    --output_dir Qwen2-0.5B-DAR
```

## Logged metrics

DAR logs the following core metrics during train/eval:

- `dar/rewards_mean`, `dar/rewards_std`
- `dar/advantages_mean`, `dar/advantages_std`
- `dar/weights_mean`, `dar/weights_max`
- `dar/logps_policy`, `dar/logps_ref`, `dar/logps_behavior`
- `dar/loss`

## DARTrainer

[[autodoc]] DARTrainer
    - train
    - save_model
    - push_to_hub

## DARConfig

[[autodoc]] DARConfig

## DataCollatorForDAR

[[autodoc]] trainer.dar_trainer.DataCollatorForDAR
