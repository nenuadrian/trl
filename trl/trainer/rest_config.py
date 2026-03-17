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

from .sft_config import SFTConfig


@dataclass
class RESTConfig(SFTConfig):
    r"""
    Configuration class for [`RESTTrainer`].

    ReST_EM (Reinforced Self-Training with Expectation-Maximization) is an iterative self-training algorithm that
    alternates between generating completions (E-step) and fine-tuning on filtered/reward-weighted completions
    (M-step).

    Extends [`SFTConfig`] with parameters specific to the ReST_EM pipeline: generation, reward scoring, filtering,
    and iteration control.
    """

    # EM iteration parameters
    num_iterations: int = field(
        default=3,
        metadata={"help": "Number of EM (Generate + Improve) iterations."},
    )
    reset_model_each_iteration: bool = field(
        default=True,
        metadata={
            "help": "Whether to reset the model to the base checkpoint before each M-step. "
            "True corresponds to ReST_EM (Singh et al., 2024); False corresponds to original ReST (Gulcehre et al., 2023)."
        },
    )

    # E-step generation parameters
    num_samples_per_prompt: int = field(
        default=32,
        metadata={"help": "Number of completions to generate per prompt during the E-step."},
    )
    generation_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size for generation during the E-step (number of prompts per batch)."},
    )
    max_new_tokens: int = field(
        default=512,
        metadata={"help": "Maximum number of new tokens to generate per completion."},
    )
    generation_temperature: float = field(
        default=0.7,
        metadata={"help": "Sampling temperature for E-step generation."},
    )
    generation_top_k: int = field(
        default=40,
        metadata={"help": "Top-k sampling value for E-step generation. Set to 0 to disable."},
    )
    generation_top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p (nucleus) sampling value for E-step generation."},
    )

    # Reward and filtering parameters
    reward_threshold: float = field(
        default=0.5,
        metadata={"help": "Minimum reward score for a completion to be kept during filtering."},
    )
    max_solutions_per_problem: int | None = field(
        default=10,
        metadata={
            "help": "Maximum number of filtered completions to keep per prompt. "
            "Prevents dataset imbalance where easier problems dominate. Set to None to disable."
        },
    )
    reward_weighted_loss: bool = field(
        default=False,
        metadata={
            "help": "If True, weight the NLL loss by the reward: J(theta) = E[r(x,y) * (-log p(y|x))]. "
            "If False, use standard NLL on filtered completions (binary filtering)."
        },
    )
    reward_model_path: str | None = field(
        default=None,
        metadata={
            "help": "Path or Hub ID of a reward model (AutoModelForSequenceClassification) for scoring completions. "
            "Provide either this or a `reward_fn` to the trainer, not both."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        if self.num_iterations < 1:
            raise ValueError("`num_iterations` must be >= 1.")
        if self.num_samples_per_prompt < 1:
            raise ValueError("`num_samples_per_prompt` must be >= 1.")
        if self.generation_batch_size < 1:
            raise ValueError("`generation_batch_size` must be >= 1.")
        if self.max_new_tokens < 1:
            raise ValueError("`max_new_tokens` must be >= 1.")
        if self.max_solutions_per_problem is not None and self.max_solutions_per_problem < 1:
            raise ValueError("`max_solutions_per_problem` must be >= 1 or None.")
