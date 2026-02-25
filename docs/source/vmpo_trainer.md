# VMPO Trainer

[![model badge](https://img.shields.io/badge/All_models-VMPO-blue)](https://huggingface.co/models?other=vmpo,trl)

## Overview

TRL supports the VMPO Trainer, inspired by [V-MPO: On-Policy Maximum a Posteriori Policy Optimization for Discrete and Continuous Control](https://huggingface.co/papers/1909.12238), adapted for language-model post-training.

VMPO in TRL reuses the DAR prompt/completion/reward pipeline but changes the objective:

1. **E-step**: build a non-parametric target distribution over high-advantage samples, controlled by a dual variable \\(\eta\\).
2. **M-step**: fit the policy to that target under a KL trust region controlled by a dual variable \\(\alpha\\).

It supports multiple KL estimators:

- `ref`
- `behavior`
- `old_policy`
- `old_policy_ref`

`old_policy_ref` combines old-policy trust-region control with optional reference anchoring.

## Quick start

```python
from datasets import load_dataset
from trl import VMPOConfig, VMPOTrainer

train_dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train")

training_args = VMPOConfig(
    output_dir="Qwen2-0.5B-VMPO",
    learning_rate=5e-7,
    vmpo_k=2,
    vmpo_kl_estimator="old_policy_ref",
    vmpo_old_policy_sync_steps=16,
    num_train_epochs=1,
)

trainer = VMPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    args=training_args,
    train_dataset=train_dataset,
    reward_model="Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback",
)
trainer.train()
```

Run:

```bash
accelerate launch train_vmpo.py
```

## Expected dataset type

VMPO supports the same dataset modes as DAR:

- **Prompt-only** ([prompt-only dataset](dataset_formats#prompt-only)): online completion generation.
- **Preference** ([preference dataset](dataset_formats#preference)): auto-converted to offline completion pairs.
- **Prompt-completion-reward** with optional `behavior_logps`.

## Example script

A reference script is available at [`trl/scripts/vmpo.py`](https://github.com/huggingface/trl/blob/main/trl/scripts/vmpo.py).

Example command:

```bash
accelerate launch trl/scripts/vmpo.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/ultrafeedback-prompt \
    --reward_model_name_or_path Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback \
    --vmpo_k 2 \
    --vmpo_kl_estimator old_policy_ref \
    --vmpo_old_policy_sync_steps 16 \
    --output_dir Qwen2-0.5B-VMPO
```

## Logged metrics

VMPO logs the following core metrics during train/eval:

- `vmpo/rewards_mean`, `vmpo/rewards_std`
- `vmpo/advantages_mean`, `vmpo/advantages_std`
- `vmpo/top_advantages_mean`, `vmpo/top_advantages_std`
- `vmpo/weights_entropy`, `vmpo/selected_fraction`
- `vmpo/eta`, `vmpo/eta_loss`
- `vmpo/alpha`, `vmpo/kl_mean`
- `vmpo/logps_policy`, `vmpo/logps_ref`, `vmpo/logps_old`
- `vmpo/policy_loss`, `vmpo/kl_loss`, `vmpo/ref_anchor_loss`
- `vmpo/reward_ema_baseline`, `vmpo/near_zero_kl_streak`
- `vmpo/loss`

## Generated API HTML quick links

- [`VMPOTrainer`](#vmpotrainer)
- [`VMPOConfig`](#vmpoconfig)

## VMPOTrainer

[[autodoc]] VMPOTrainer
    - train
    - save_model
    - push_to_hub

## VMPOConfig

[[autodoc]] VMPOConfig
