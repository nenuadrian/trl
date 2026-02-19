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

from .dpo_config import DPOConfig


@dataclass
class DARConfig(DPOConfig):
    r"""
    Configuration class for [`DARTrainer`].

    DAR turns preference optimization into reward-weighted SFT:
    completions are sampled per prompt, scored with scalar rewards, and weighted by both reward advantage and
    reference regularization before computing a weighted NLL objective.
    """

    alpha: float = field(
        default=0.1,
        metadata={"help": "Reference-regularization coefficient α in DAR."},
    )
    dar_k: int = field(
        default=2,
        metadata={"help": "Number of sampled completions per prompt (K)."},
    )
    dar_wclip: float = field(
        default=50.0,
        metadata={"help": "Maximum importance weight value used to clip DAR weights."},
    )
    adv_norm: str = field(
        default="per_prompt",
        metadata={
            "help": "Advantage normalization mode.",
            "choices": ["per_prompt", "batch", "none"],
        },
    )
    max_new_tokens: int = field(
        default=128,
        metadata={"help": "Maximum number of new tokens generated per sampled completion."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature for online DAR completion generation."},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p sampling value used during completion generation."},
    )
    top_k: int = field(
        default=0,
        metadata={"help": "Top-k sampling value used during completion generation."},
    )
    min_p: float | None = field(
        default=None,
        metadata={"help": "Optional min-p sampling value used during completion generation."},
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "Repetition penalty used during completion generation."},
    )
    reward_model_path: str | None = field(
        default=None,
        metadata={
            "help": "Optional reward model path/id. Use this when rewards are not provided in the dataset and no "
            "custom reward function is supplied to the trainer."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        if self.alpha < 0:
            raise ValueError("`alpha` must be >= 0.")
        if self.beta <= 0:
            raise ValueError("`beta` must be > 0 for DAR.")
        if self.alpha + self.beta <= 0:
            raise ValueError("`alpha + beta` must be > 0 for DAR.")
        if self.dar_k <= 0:
            raise ValueError("`dar_k` must be >= 1.")
        if self.dar_wclip <= 0:
            raise ValueError("`dar_wclip` must be > 0.")
