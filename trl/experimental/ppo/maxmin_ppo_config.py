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

from .ppo_config import PPOConfig


@dataclass
class MaxMinPPOConfig(PPOConfig):
    r"""
    Configuration class for the MaxMin PPO Trainer, implementing the MaxMin-RLHF algorithm
    from "MaxMin-RLHF: Alignment with Diverse Human Preferences" (Chakraborty et al., 2024).

    Extends [`PPOConfig`] with parameters for multiple reward models (one per user subpopulation).
    At each PPO step, the policy is updated to maximize the reward of the worst-off subpopulation
    (the one with the minimum expected reward), following the Egalitarian principle from social
    choice theory.

    Parameters:
        reward_model_paths (`list[str]`, *optional*, defaults to `None`):
            List of paths to reward models, one per user subpopulation. Each reward model is an
            `AutoModelForSequenceClassification` checkpoint. If `None`, falls back to the single
            `reward_model_path` from [`PPOConfig`].
        maxmin_strategy (`str`, *optional*, defaults to `"min"`):
            Strategy for aggregating rewards across subpopulations. `"min"` selects the minimum
            reward (MaxMin / Egalitarian), `"mean"` averages across reward models (standard
            multi-reward), `"softmin"` uses a soft-minimum via negative temperature softmax.
        softmin_temperature (`float`, *optional*, defaults to `0.1`):
            Temperature for the softmin aggregation. Lower values approximate hard min more closely.
            Only used when `maxmin_strategy="softmin"`.
    """

    reward_model_paths: list[str] | None = field(
        default=None,
        metadata={
            "help": "List of paths to reward models, one per user subpopulation. "
            "If None, falls back to the single reward_model_path."
        },
    )
    maxmin_strategy: str = field(
        default="min",
        metadata={
            "help": "Strategy for aggregating rewards: 'min' (MaxMin/Egalitarian), "
            "'mean' (average), or 'softmin' (differentiable approximation to min)."
        },
    )
    softmin_temperature: float = field(
        default=0.1,
        metadata={"help": "Temperature for softmin aggregation. Lower = closer to hard min."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.maxmin_strategy not in {"min", "mean", "softmin"}:
            raise ValueError(
                f"maxmin_strategy must be 'min', 'mean', or 'softmin', got '{self.maxmin_strategy}'"
            )
        if self.reward_model_paths is None:
            self.reward_model_paths = [self.reward_model_path]
