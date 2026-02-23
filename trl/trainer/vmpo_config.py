# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from .dar_config import DARConfig


@dataclass
class VMPOConfig(DARConfig):
    r"""
    Configuration class for [`VMPOTrainer`].

    VMPO (Variational Maximum a Posteriori Policy Optimization) performs policy improvement in two
    complementary steps:
    1. E-step: fit a non-parametric target distribution over high-advantage samples.
    2. M-step: update the policy toward that target while enforcing a KL trust region.

    This config reuses DAR data/reward-generation settings and adds VMPO-specific dual parameters.
    """

    vmpo_k: int = field(
        default=2,
        metadata={"help": "Number of sampled completions per prompt when running online VMPO."},
    )
    vmpo_topk_fraction: float = field(
        default=0.5,
        metadata={"help": "Fraction of highest-advantage samples used in the VMPO E-step. Must be in (0, 1]."},
    )
    vmpo_eps_eta: float = field(
        default=0.1,
        metadata={"help": "Constraint level for the temperature dual objective (eta)."},
    )
    vmpo_eps_alpha: float = field(
        default=0.1,
        metadata={"help": "Target KL trust-region level for the alpha dual update."},
    )
    vmpo_eta_init: float = field(
        default=1.0,
        metadata={"help": "Initial value of the VMPO temperature dual variable eta."},
    )
    vmpo_alpha_init: float = field(
        default=1.0,
        metadata={"help": "Initial value of the VMPO KL dual variable alpha."},
    )
    vmpo_eta_lr: float = field(
        default=1e-2,
        metadata={"help": "Step size used for eta dual updates."},
    )
    vmpo_alpha_lr: float = field(
        default=1e-2,
        metadata={"help": "Step size used for alpha dual updates."},
    )
    vmpo_min_eta: float = field(
        default=1e-8,
        metadata={"help": "Lower bound used to project eta after updates."},
    )
    vmpo_min_alpha: float = field(
        default=1e-8,
        metadata={"help": "Lower bound used to project alpha after updates."},
    )
    vmpo_kl_estimator: str = field(
        default="old_policy_ref",
        metadata={
            "help": "KL estimator used in the M-step.",
            "choices": ["ref", "behavior", "old_policy", "old_policy_ref"],
        },
    )
    vmpo_old_policy_sync_steps: int = field(
        default=16,
        metadata={"help": "How often (in optimizer steps) to refresh the old-policy snapshot."},
    )
    vmpo_ref_anchor_coef: float = field(
        default=0.1,
        metadata={"help": "Coefficient of the reference-anchor term used with `vmpo_kl_estimator='old_policy_ref'`."},
    )
    vmpo_advantage_baseline: str = field(
        default="ema",
        metadata={
            "help": "How to construct the baseline for reward advantages.",
            "choices": ["per_prompt", "batch", "ema"],
        },
    )
    vmpo_reward_ema_decay: float = field(
        default=0.98,
        metadata={"help": "EMA decay used when `vmpo_advantage_baseline='ema'`."},
    )
    vmpo_max_eta: float = field(
        default=100.0,
        metadata={"help": "Upper clamp for eta dual variable to avoid instability."},
    )
    vmpo_max_alpha: float = field(
        default=100.0,
        metadata={"help": "Upper clamp for alpha dual variable to avoid instability."},
    )
    vmpo_kl_zero_tol: float = field(
        default=1e-6,
        metadata={"help": "Threshold below which KL is considered effectively zero."},
    )
    vmpo_kl_warning_patience: int = field(
        default=20,
        metadata={"help": "Number of consecutive near-zero KL train steps before emitting a warning."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.vmpo_k <= 0:
            raise ValueError("`vmpo_k` must be >= 1.")
        if self.vmpo_topk_fraction <= 0 or self.vmpo_topk_fraction > 1:
            raise ValueError("`vmpo_topk_fraction` must be in `(0, 1]`.")
        if self.vmpo_eta_init <= 0:
            raise ValueError("`vmpo_eta_init` must be > 0.")
        if self.vmpo_alpha_init <= 0:
            raise ValueError("`vmpo_alpha_init` must be > 0.")
        if self.vmpo_eta_lr <= 0:
            raise ValueError("`vmpo_eta_lr` must be > 0.")
        if self.vmpo_alpha_lr <= 0:
            raise ValueError("`vmpo_alpha_lr` must be > 0.")
        if self.vmpo_min_eta <= 0:
            raise ValueError("`vmpo_min_eta` must be > 0.")
        if self.vmpo_min_alpha <= 0:
            raise ValueError("`vmpo_min_alpha` must be > 0.")
        if self.vmpo_kl_estimator not in {"ref", "behavior", "old_policy", "old_policy_ref"}:
            raise ValueError("`vmpo_kl_estimator` must be one of: `ref`, `behavior`, `old_policy`, `old_policy_ref`.")
        if self.vmpo_old_policy_sync_steps <= 0:
            raise ValueError("`vmpo_old_policy_sync_steps` must be >= 1.")
        if self.vmpo_ref_anchor_coef < 0:
            raise ValueError("`vmpo_ref_anchor_coef` must be >= 0.")
        if self.vmpo_advantage_baseline not in {"per_prompt", "batch", "ema"}:
            raise ValueError("`vmpo_advantage_baseline` must be one of: `per_prompt`, `batch`, `ema`.")
        if self.vmpo_reward_ema_decay < 0 or self.vmpo_reward_ema_decay >= 1:
            raise ValueError("`vmpo_reward_ema_decay` must be in `[0, 1)`.")
        if self.vmpo_max_eta <= 0:
            raise ValueError("`vmpo_max_eta` must be > 0.")
        if self.vmpo_max_alpha <= 0:
            raise ValueError("`vmpo_max_alpha` must be > 0.")
        if self.vmpo_max_eta < self.vmpo_min_eta:
            raise ValueError("`vmpo_max_eta` must be >= `vmpo_min_eta`.")
        if self.vmpo_max_alpha < self.vmpo_min_alpha:
            raise ValueError("`vmpo_max_alpha` must be >= `vmpo_min_alpha`.")
        if self.vmpo_kl_zero_tol < 0:
            raise ValueError("`vmpo_kl_zero_tol` must be >= 0.")
        if self.vmpo_kl_warning_patience <= 0:
            raise ValueError("`vmpo_kl_warning_patience` must be >= 1.")
