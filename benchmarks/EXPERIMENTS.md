# LLM Experiments: V-MPO and DAR via TRL

All scripts live in `benchmarks/` and use `accelerate launch` with the `accelerate_4gpu_fp16.yaml` config. Every script accepts `CUDA_VISIBLE_DEVICES` and `NUM_PROCESSES` overrides so you can run on 1, 2, or 4 GPUs.

---

## 1. Head-to-Head: VMPO vs DAR vs DPO

**Purpose:** Establish whether EM-based methods (VMPO, DAR) outperform the DPO baseline on the same data and compute budget. This is the core comparison for the GEMPI paper's LLM section.

**Script:** `run_vmpo_vs_dar_vs_dpo.sh`

Runs three experiments sequentially:
1. DPO on UltraFeedback binarized (pairwise preferences)
2. DAR on UltraFeedback prompts (online generation, k=2, reward model)
3. VMPO on UltraFeedback prompts (online generation, k=2, reward model)

**What to measure:**
- Mean reward at convergence (from reward model)
- Win rate vs base model (GPT-4 judge or ArmoRM head-to-head)
- KL divergence from reference (policy drift)
- Wall-clock time per training step

**Expected outcome:** VMPO and DAR should match or exceed DPO reward while maintaining lower KL from reference, due to the explicit trust-region / regularisation mechanisms.

| Script | Method | Dataset | k | Reward |
|--------|--------|---------|---|--------|
| `run_dpo_ultrafeedback_binarized_2gpu.sh` | DPO | UltraFeedback binarized | N/A | implicit (preference) |
| `run_dar_ultrafeedback_prompt_2gpu.sh` | DAR | UltraFeedback prompts | 2 | ArmoRM |
| `run_vmpo_ultrafeedback_prompt_2gpu.sh` | VMPO | UltraFeedback prompts | 2 | ArmoRM |

---

## 2. Reasoning Task: VMPO on GSM8K

**Purpose:** Test whether EM-based optimisation improves reasoning accuracy when using a verifiable (exact-match) reward signal, not a learned reward model.

**Script:** `run_vmpo_reasoning_gsm8k.sh`

**Setup:**
- Dataset: GSM8K (math word problems)
- Reward: `reasoning_em_reward` (exact-match with format bonuses/penalties)
- k=4 completions per prompt, top-50% selection
- advantage baseline: per_prompt (crucial for reasoning, avoids easy-prompt bias)

**What to measure:**
- Accuracy on GSM8K test set (pass@1)
- Format compliance (% of outputs with proper `<think>...</think>` tags)
- Reward distribution over training (should shift right)

**Variants:** Also available as `run_vmpo_reasoning_prompt_2gpu.sh` / `_4gpu.sh` for the generic reasoning entry point.

---

## 3. Ablation: Number of Completions (k)

**Purpose:** Measure how the number of sampled completions per prompt affects reward quality and compute cost. More samples give better advantage estimates but cost more generation time.

**Script:** `run_vmpo_ablation_k.sh`

**Sweep:** k in {1, 2, 4, 8}

| k | Batch size | Grad accum | Effective batch |
|---|-----------|------------|-----------------|
| 1 | 2 | 4 | 8 |
| 2 | 2 | 4 | 8 |
| 4 | 1 | 8 | 8 |
| 8 | 1 | 4 | 4 |

**What to measure:**
- Final reward (does more k help?)
- Training throughput (samples/sec)
- Advantage variance (`vmpo/advantages_std`) -- should decrease with higher k
- Weight entropy (`vmpo/weights_entropy`) -- should increase with higher k (more uniform E-step)

**Expected outcome:** k=4 is the sweet spot. k=1 degrades to filtered SFT (no per-prompt advantage centering possible). k=8 gives diminishing returns at 4x the generation cost.

---

## 4. Ablation: KL Estimator

**Purpose:** Test which KL divergence estimator provides the most stable training signal. The GEMPI paper proposes `old_policy_ref` (dual-anchor KL) as superior. This ablation validates that claim.

**Script:** `run_vmpo_ablation_kl_estimator.sh`

**Sweep:** {ref, behavior, old_policy, old_policy_ref}

| Estimator | KL formula | Trust region anchor |
|-----------|-----------|-------------------|
| `ref` | KL(policy \|\| ref) | frozen reference model |
| `behavior` | KL(behavior \|\| policy) | current batch log-probs |
| `old_policy` | KL(old_policy \|\| policy) | periodic snapshot |
| `old_policy_ref` | KL(old_policy \|\| policy) + ref anchor | snapshot + frozen ref |

**What to measure:**
- `vmpo/kl_mean` trajectory (should stay bounded, not collapse to 0)
- `vmpo/near_zero_kl_streak` (should stay 0 for stable estimators)
- `vmpo/alpha` trajectory (should converge, not oscillate)
- Final reward (which estimator yields best reward?)

**Expected outcome:** `ref` will show KL collapse after many steps (policy drifts, ref becomes irrelevant). `old_policy_ref` should be most stable because the old-policy anchor tracks the moving policy while the ref anchor prevents unbounded drift.

---

## 5. Ablation: Top-k Fraction (E-step selection pressure)

**Purpose:** The top-k fraction controls how aggressively the E-step filters samples. At 1.0 it reduces to soft advantage-weighted regression (all samples used). At 0.25 only the top quartile is used (aggressive filtering).

**Script:** `run_vmpo_ablation_topk.sh`

**Sweep:** fraction in {0.25, 0.5, 0.75, 1.0} with k=4

**What to measure:**
- `vmpo/selected_fraction` (should match the setting)
- `vmpo/weights_entropy` (lower at aggressive fractions)
- `vmpo/top_advantages_mean` (higher at aggressive fractions)
- Final reward vs training stability (reward curves)

**Expected outcome:** 0.5 is a good default. 0.25 may overfit to lucky samples. 1.0 reduces to AWR-like behavior (loses the VMPO "hard filter" benefit).

---

## 6. Ablation: DAR Alpha (Reference Regularisation)

**Purpose:** alpha controls how much the importance weights are pulled toward the reference policy. At alpha=0, DAR reduces to pure advantage-weighted regression. At high alpha, training becomes conservative (slow improvement but stable).

**Script:** `run_dar_ablation_alpha.sh`

**Sweep:** alpha in {0.0, 0.01, 0.1, 0.5, 1.0}

**What to measure:**
- `dar/weights_clip_fraction` (higher at low alpha -- weights blow up)
- `dar/kl_from_ref` (should grow faster at low alpha)
- `dar/log_reg_weight_mean` (should increase with alpha)
- Final reward vs KL trade-off curve

**Expected outcome:** alpha=0.0 gives highest reward but also highest KL (most policy drift). alpha=0.1 is a good Pareto point. alpha=1.0 is too conservative, reward barely improves over SFT.

---

## 7. DAR Paper Configuration

**Purpose:** Reproduce the exact hyperparameters from the GEMPI paper's DAR experiments.

**Scripts:**
- `run_dar_ultrafeedback_prompt_paper_2gpu.sh` (2-GPU)
- `run_dar_ultrafeedback_prompt_paper_4gpu.sh` (4-GPU)

**Key differences from default DAR:**
- k=4 (paper used more samples)
- alpha=0.5 (higher reference regularisation)
- per_device_batch_size=1 with gradient_accumulation=8

---

## Running Experiments

### Prerequisites

```bash
pip install -e ".[dev]"
pip install wandb accelerate
wandb login
```

### Single experiment

```bash
# 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 bash benchmarks/run_vmpo_ultrafeedback_prompt_2gpu.sh

# 4 GPUs
bash benchmarks/run_vmpo_ultrafeedback_prompt_4gpu.sh

# Custom model
MODEL=Qwen/Qwen2.5-1.5B-Instruct bash benchmarks/run_vmpo_ultrafeedback_prompt_4gpu.sh
```

### Full comparison

```bash
bash benchmarks/run_vmpo_vs_dar_vs_dpo.sh
```

### Ablation sweeps

```bash
# Run one at a time (each does a sequential sweep)
bash benchmarks/run_vmpo_ablation_k.sh
bash benchmarks/run_vmpo_ablation_kl_estimator.sh
bash benchmarks/run_vmpo_ablation_topk.sh
bash benchmarks/run_dar_ablation_alpha.sh
```

---

## Metrics to Track (WandB)

All experiments log to WandB. Key metrics to compare across runs:

### Reward Quality
- `vmpo/rewards_mean` or `dar/rewards_mean` -- primary signal
- `vmpo/rewards_std` -- reward variance (lower = more consistent)

### Training Stability
- `vmpo/kl_mean` or `dar/kl_from_ref` -- policy drift
- `vmpo/alpha`, `vmpo/eta` -- dual variable convergence
- `vmpo/near_zero_kl_streak` -- KL collapse indicator
- `dar/weights_clip_fraction` -- importance weight explosion indicator

### E-step / M-step Health
- `vmpo/weights_entropy` -- E-step distribution concentration
- `vmpo/eta_loss` -- dual feasibility
- `vmpo/top_advantages_mean` -- signal strength
- `dar/log_reg_weight_mean`, `dar/log_adv_weight_mean` -- weight decomposition

### Generation Quality
- `vmpo/completion_length_mean` -- output length (watch for reward hacking via short/long outputs)

---

## File Index

| File | Description |
|------|-------------|
| `run_dpo_ultrafeedback_binarized_2gpu.sh` | DPO baseline (2-GPU) |
| `run_vmpo_ultrafeedback_prompt_2gpu.sh` | VMPO on UltraFeedback (2-GPU wrapper) |
| `run_vmpo_ultrafeedback_prompt_4gpu.sh` | VMPO on UltraFeedback (main script) |
| `run_vmpo_reasoning_prompt_2gpu.sh` | VMPO reasoning (2-GPU wrapper) |
| `run_vmpo_reasoning_prompt_4gpu.sh` | VMPO reasoning (main script) |
| `run_vmpo_reasoning_gsm8k.sh` | VMPO on GSM8K (standalone) |
| `run_dar_ultrafeedback_prompt_2gpu.sh` | DAR on UltraFeedback (2-GPU wrapper) |
| `run_dar_ultrafeedback_prompt_4gpu.sh` | DAR on UltraFeedback (main script) |
| `run_dar_ultrafeedback_prompt_paper_2gpu.sh` | DAR paper config (2-GPU wrapper) |
| `run_dar_ultrafeedback_prompt_paper_4gpu.sh` | DAR paper config (main script) |
| `run_vmpo_vs_dar_vs_dpo.sh` | Head-to-head comparison |
| `run_vmpo_ablation_k.sh` | Ablation: completions per prompt |
| `run_vmpo_ablation_kl_estimator.sh` | Ablation: KL estimator type |
| `run_vmpo_ablation_topk.sh` | Ablation: E-step selection pressure |
| `run_dar_ablation_alpha.sh` | Ablation: reference regularisation |
| `rewards/reasoning_reward.py` | Exact-match reasoning reward function |
| `accelerate_4gpu_fp16.yaml` | Accelerate multi-GPU config |
