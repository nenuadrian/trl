import gc
import math
import os
import textwrap
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator, logging
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import (
    CallbackHandler,
    ExportableState,
    PrinterCallback,
    ProgressCallback,
)
from transformers import TrainingArguments
from transformers.utils import ModelOutput, is_peft_available, is_rich_available

from ...models.utils import (
    create_reference_model,
    peft_module_casting_to_bf16,
    unwrap_model_for_generation,
)
from ...trainer.base_trainer import BaseTrainer
from ...trainer.utils import (
    disable_dropout_in_model,
    empty_cache,
    log_table_to_comet_experiment,
    pad,
    selective_log_softmax,
)
from ..utils import first_true_indices, get_reward


@dataclass
class TokenWeightedVMPOTrainerConfig(TrainingArguments):
    r"""
    Configuration class for the [`experimental.token_weighted_vmpo_trainer.TokenWeightedVMPOTrainer`].
    """

    eps_eta: float = field(
        default=2.0,
        metadata={"help": "Dual variable target εη for the V-MPO temperature loss."},
    )
    eta_init: float = field(
        default=1.0,
        metadata={"help": "Initial value for the V-MPO temperature η."},
    )
    eta_min: float = field(
        default=1e-4,
        metadata={"help": "Minimum value added after softplus to keep η positive."},
    )
    eta_inner_steps: int = field(
        default=10,
        metadata={
            "help": "Number of inner optimization steps for the temperature dual variable η."
        },
    )
    eps_alpha: float = field(
        default=0.01,
        metadata={"help": "KL trust region target εα for V-MPO dual α."},
    )
    alpha_init: float = field(
        default=1.0,
        metadata={"help": "Initial value for the V-MPO KL dual α."},
    )
    alpha_min: float = field(
        default=1e-4,
        metadata={"help": "Minimum value added after softplus to keep α positive."},
    )
    psi_topk_ratio: float | None = field(
        default=0.2,
        metadata={
            "help": "Optional top-k truncation ratio for token-level ψ support (e.g., 0.2 keeps top 20% tokens by advantage). "
            "Set to None to disable."
        },
    )
    psi_ess_min_tokens: int = field(
        default=256,
        metadata={
            "help": "Enable entropy regularization when token-level ψ ESS drops below this threshold (in tokens)."
        },
    )
    entropy_beta: float = field(
        default=0.0,
        metadata={
            "help": "Entropy regularization coefficient β (only applied when ψ ESS < psi_ess_min_tokens). Set 0 to disable."
        },
    )
    reward_spread_k: int = field(
        default=1,
        metadata={
            "help": "Diffuse the terminal scalar reward over the last K response tokens (K=1 keeps old behavior)."
        },
    )
    eta_lr_mult: float = field(
        default=0.1,
        metadata={
            "help": "Learning-rate multiplier for η dual optimizer relative to policy learning rate."
        },
    )
    alpha_lr_mult: float = field(
        default=1.0,
        metadata={
            "help": "Learning-rate multiplier for α dual optimizer relative to policy learning rate."
        },
    )
    advantage_clip: float | None = field(
        default=10.0,
        metadata={
            "help": "Clamp centered advantages to [-advantage_clip, advantage_clip] before exponentiation in ψ/η "
            "updates (numerical hygiene). Set to None to disable."
        },
    )

    # Parameters whose default values are overridden from TrainingArguments
    logging_steps: float = field(
        default=10,
        metadata={
            "help": "Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, "
            "will be interpreted as ratio of total training steps."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    bf16: bool | None = field(
        default=None,
        metadata={
            "help": "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA "
            "architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if "
            "`fp16` is not set."
        },
    )
    # Transformers 4.57.0 introduced a bug that caused the dtype of `lr_scheduler_kwargs` to be unparsable. This issue
    # was fixed in https://github.com/huggingface/transformers/pull/41322 and released in 4.57.5. We add a temporary
    # workaround here, which can be removed once we drop support for versions older than 4.57.5.
    lr_scheduler_kwargs: dict | str | None = field(
        default=None,
        metadata={
            "help": "Additional parameters for the lr_scheduler, such as {'num_cycles': 1} for cosine with hard "
            "restarts."
        },
    )

    run_name: str | None = field(
        default=None,
        metadata={"help": "Name of the run."},
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    num_mini_batches: int = field(
        default=1,
        metadata={"help": "Number of minibatches to split a batch into."},
    )
    m_steps: int = field(
        default=2,
        metadata={
            "help": "Number of policy/value gradient steps (M-step iterations) per rollout with ψ fixed."
        },
    )
    total_episodes: int | None = field(
        default=None,
        metadata={"help": "Total number of episodes in the dataset."},
    )
    local_rollout_forward_batch_size: int = field(
        default=64,
        metadata={"help": "Per rank no grad forward pass in the rollout phase."},
    )
    num_sample_generations: int = field(
        default=10,
        metadata={
            "help": "Number of debugging samples generations (i.e., `generate_completions` calls) throughout training."
        },
    )
    response_length: int = field(
        default=53,
        metadata={"help": "Length of the response."},
    )
    stop_token: Literal["eos"] | None = field(
        default=None,
        metadata={
            "help": "Specifies the stop token to use for text generation. This parameter is mutually exclusive with "
            "`stop_token_id`."
        },
    )
    stop_token_id: int | None = field(
        default=None,
        metadata={
            "help": "Specifies the ID of the stop token to use for text generation. If `None`, no stop token ID is "
            "applied, unless `stop_token` is specified. This parameter is mutually exclusive with `stop_token`."
        },
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "Sampling temperature."},
    )
    missing_eos_penalty: float | None = field(
        default=None,
        metadata={
            "help": "Penalty applied to the score when the model fails to generate an EOS token. This is useful to "
            "encourage to generate completions shorter than the maximum length (`max_new_tokens`). The penalty must be "
            "a positive value."
        },
    )
    eos_bonus: float = field(
        default=0.1,
        metadata={
            "help": "Optional positive shaping reward added at the EOS position."
        },
    )
    sft_model_path: str = field(
        default="EleutherAI/pythia-160m",
        metadata={"help": "Path to the SFT model."},
    )
    world_size: int | None = field(
        default=None,
        metadata={"help": "Number of processes (GPUs) to use for the training."},
    )
    num_total_batches: int | None = field(
        default=None,
        metadata={"help": "Number of total batches to train."},
    )
    micro_batch_size: int | None = field(
        default=None,
        metadata={
            "help": "Micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)."
        },
    )
    local_batch_size: int | None = field(
        default=None,
        metadata={
            "help": "Batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)."
        },
    )
    batch_size: int | None = field(
        default=None,
        metadata={
            "help": "Batch size across devices (HF's `per_device_train_batch_size` * `world_size` * "
            "`gradient_accumulation_steps`)."
        },
    )
    local_mini_batch_size: int | None = field(
        default=None,
        metadata={"help": "Mini batch size per GPU."},
    )
    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the model to the Hub after training."},
    )
    exp_name: str = field(
        default=os.path.basename(__file__)[:-3],
        metadata={"help": "Name of this experiment."},
    )
    reward_model_path: str = field(
        default="EleutherAI/pythia-160m",
        metadata={"help": "Path to the reward model."},
    )
    model_adapter_name: str | None = field(
        default=None,
        metadata={
            "help": "Name of the train target PEFT adapter, when using LoRA with multiple adapters."
        },
    )
    ref_adapter_name: str | None = field(
        default=None,
        metadata={
            "help": "Name of the reference PEFT adapter, when using LoRA with multiple adapters."
        },
    )
    kl_estimator: Literal["k1", "k3"] = field(
        default="k1",
        metadata={
            "help": "Which estimator for KL-Divergence to use from Approximating KL "
            "Divergence "
            "(http://joschu.net/blog/kl-approx.html). Defaults to 'k1', a straightforward, unbiased estimator. Can be "
            "set to 'k3', an unbiased estimator with lower variance which 'appears to be a strictly better "
            "estimator'. Cannot be set to 'k2', as it is used for logging purposes."
        },
    )
    vf_coef: float = field(
        default=0.1,
        metadata={"help": "Value function coefficient."},
    )
    value_clip: float = field(
        default=0.2,
        metadata={"help": "Clip range for value targets (±value_clip)."},
    )
    gamma: float = field(
        default=0.99,
        metadata={"help": "Discount factor."},
    )
    lam: float = field(
        default=0.8,
        metadata={"help": "Lambda value for GAE."},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation."
        },
    )

    def __post_init__(self):
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16
        if self.missing_eos_penalty is not None and self.missing_eos_penalty <= 0:
            raise ValueError(
                "`missing_eos_penalty` must be > 0 to enable EOS penalization."
            )
        if self.eos_bonus < 0:
            raise ValueError("`eos_bonus` must be non-negative.")
        if self.psi_topk_ratio is not None and not (0.0 < self.psi_topk_ratio <= 1.0):
            raise ValueError("`psi_topk_ratio` must be in (0, 1] or None.")
        if self.psi_ess_min_tokens < 1:
            raise ValueError("`psi_ess_min_tokens` must be >= 1.")
        if self.entropy_beta < 0:
            raise ValueError("`entropy_beta` must be >= 0.")
        if self.reward_spread_k < 1:
            raise ValueError("`reward_spread_k` must be >= 1.")
        if self.eta_lr_mult < 0:
            raise ValueError("`eta_lr_mult` must be >= 0.")
        if self.alpha_lr_mult < 0:
            raise ValueError("`alpha_lr_mult` must be >= 0.")
        if self.advantage_clip is not None and self.advantage_clip <= 0:
            raise ValueError("`advantage_clip` must be > 0 or None.")
        super().__post_init__()


if is_rich_available():
    from rich.console import Console
    from rich.table import Table


logger = logging.get_logger(__name__)

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model


INVALID_LOGPROB = 0.0


def inv_softplus(x: torch.Tensor) -> torch.Tensor:
    # assumes x > 0; stable inverse of softplus
    return x + torch.log1p(-torch.exp(-x))


def generate(
    lm_backbone: torch.nn.Module,
    queries: torch.Tensor,
    pad_token_id: int,
    generation_kwargs: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates sequences from the language model backbone in a way that does not affect padding tokens.

    Args:
        lm_backbone (`torch.nn.Module`):
            The language model backbone used for generation.
        queries (`torch.Tensor`):
            The tensor containing the input queries.
        pad_token_id (`int`):
            The token ID representing the pad token.
        generation_kwargs (`dict`):
            The generation parameters.

    Returns:
        tuple:
            - `generated_sequences` (`torch.Tensor`):
                The concatenated tensor of input queries and generated sequences.
            - `logits` (`torch.Tensor`):
                The logits output from the generation process.
    """
    context_length = queries.shape[1]
    attention_mask = queries != pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)

    # NOTE: Pass generation parameters explicitly (no generation_config) to avoid
    # "Please pass either a generation_config object OR all generation parameters explicitly, but not both."
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict_in_generate=True,
        output_scores=True,
        **generation_kwargs,
    )
    logits = torch.stack(output.scores, 1)
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1), logits


@torch.no_grad()
def batch_generation(
    model: torch.nn.Module,
    queries: torch.Tensor,
    local_rollout_forward_batch_size: int,
    pad_token_id: int,
    generation_kwargs: dict,
):
    """Generate responses in micro-batches to control memory, then pad and reshape to the original batch size."""
    query_responses = []
    logitss = []
    batch_size = queries.shape[0]
    for i in range(0, batch_size, local_rollout_forward_batch_size):
        # Slice a micro-batch to stay within memory limits during generation
        query = queries[i : i + local_rollout_forward_batch_size]
        query_response, logits = generate(
            model,
            query,
            pad_token_id,
            generation_kwargs,
        )
        query_responses.append(query_response)
        logitss.append(logits)

    # Pad variable-length generations so they can be concatenated back to a full batch tensor
    padded_query_responses = pad(
        query_responses, padding_value=pad_token_id, padding_side="right"
    )
    padded_logitss = pad(logitss, padding_value=0, padding_side="right")

    # Reshape the padded tensors to match the original batch ordering and size
    padded_query_responses = padded_query_responses.view(
        -1, padded_query_responses.shape[-1]
    )[:batch_size]
    padded_logitss = padded_logitss.view(-1, *padded_logitss.shape[2:])[:batch_size]

    return padded_query_responses, padded_logitss


def exact_div(a, b, custom_error_message=""):
    q = a // b
    if a != q * b:
        raise ValueError(
            f"{custom_error_message}, inexact division: {a} / {b} = {a / b}"
        )
    return q


def print_rich_table(df: pd.DataFrame) -> None:
    if not is_rich_available():
        raise ImportError(
            "The function `print_rich_table` requires the `rich` library. Please install it with `pip install rich`."
        )
    console = Console()
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.print(table)


def truncate_response(
    stop_token_id: int, pad_token_id: int, responses: torch.Tensor
) -> torch.Tensor:
    """
    Truncates the responses at the first occurrence of the stop token, filling the rest with pad tokens.

    Args:
        stop_token_id (`int`):
            The token ID representing the stop token where truncation occurs.
        pad_token_id (`int`):
            The token ID representing the pad token used to fill the truncated responses.
        responses (`torch.Tensor`):
            The tensor containing the responses to be truncated.

    Returns:
        `torch.Tensor`:
            The truncated responses tensor with pad tokens filled after the stop token.
    """
    trunc_idxs = first_true_indices(responses == stop_token_id).unsqueeze(-1)
    new_size = [1] * (len(responses.size()) - 1) + [responses.shape[1]]
    idxs = torch.arange(responses.shape[1], device=responses.device).view(*new_size)
    postprocessed_responses = torch.masked_fill(
        responses, idxs > trunc_idxs, pad_token_id
    )
    return postprocessed_responses


def forward(
    model: torch.nn.Module,
    query_responses: torch.Tensor,
    pad_token_id: int,
) -> ModelOutput:
    """
    Performs a forward pass through the model with the given query responses and pad token ID.

    Args:
        model (`torch.nn.Module`):
            The model to perform the forward pass.
        query_responses (`torch.Tensor`):
            The tensor containing the query responses.
        pad_token_id (`int`):
            The token ID representing the pad token.

    Returns:
        `ModelOutput`:
            The output of the model, including hidden states.
    """
    attention_mask = query_responses != pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )


@dataclass
class OnlineTrainerState(TrainerState):
    """
    Training state for online/on-policy trainers.

    Extends [`~transformers.TrainerState`] with an `episode` counter to track the current rollout/episode.

    Args:
        episode (`int`, defaults to 0): Zero-based episode index.
    """

    episode: int = 0


def masked_mean(
    values: torch.Tensor, mask: torch.Tensor, axis: bool | None = None
) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(
    values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True
) -> torch.Tensor:
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    mask_sum = mask.sum()
    if mask_sum == 0:
        raise ValueError(
            "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
            "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
        )
    if unbiased and mask_sum > 1:
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(
    values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True
) -> torch.Tensor:
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, value_model) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model
        self.critic_backbone = getattr(value_model, value_model.base_model_prefix)
        self.is_gradient_checkpointing = policy.is_gradient_checkpointing

    def forward(self, **kwargs):
        output = self.critic_backbone(**kwargs)
        logits = self.value_model.score(output.hidden_states[-1])
        return self.policy(**kwargs), logits


class TokenWeightedVMPOTrainer(BaseTrainer):
    """Trainer for TokenWeightedVMPOTrainer.

    Args:
        args ([`experimental.token_weighted_vmpo_trainer.TokenWeightedVMPOTrainerConfig`]):
            Training arguments.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.BaseImageProcessor`], [`~transformers.FeatureExtractionMixin`] or [`~transformers.ProcessorMixin`]):
            Class to process the data.
        model (`torch.nn.Module`):
            Model to be trained. This is the policy model.
        ref_model (`torch.nn.Module`, *optional*):
            Reference model used to compute the KL divergence. If `None`, a copy of the policy model is created.
        reward_model (`torch.nn.Module`):
            Reward model used to compute the rewards.
        train_dataset ([`~datasets.Dataset`]):
            Dataset for training.
        value_model (`torch.nn.Module`):
            Value model used to predict the value of a state.
        data_collator ([`~transformers.DataCollatorWithPadding`], *optional*):
            Data collator to batch and pad samples from the dataset. If `None`, a default data collator is created
            using the `processing_class`.
        eval_dataset ([`~datasets.Dataset`] or `dict` of [`~datasets.Dataset`], *optional*):
            Dataset for evaluation.
        optimizers (`tuple` of `torch.optim.Optimizer` and `torch.optim.lr_scheduler.LambdaLR`, *optional*, defaults to `(None, None)`):
            Tuple containing the optimizer and the learning rate scheduler to use for training. If `None`, the
            optimizer and the learning rate scheduler are created using the
            [`~transformers.Trainer.create_optimizer_and_scheduler`] method.
        callbacks (`list` of [`~transformers.TrainerCallback`], *optional*):
            Callbacks to use during training.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration to use PEFT for training. If `None`, PEFT is not used. If provided, the policy `model`
            will be wrapped with the specified PEFT adapter.
    """

    _tag_names = ["trl", "token_weighted_vmpo_trainer"]
    _name = "TokenWeightedVMPOTrainer"
    _paper = {
        "title": "V-MPO: On-Policy Maximum a Posteriori Policy Optimization for Discrete and Continuous Control",
        "id": "song2019token_weighted_vmpo_traineronpolicymaximumposteriori",
        "citation": textwrap.dedent(
            """\
            @misc{song2019token_weighted_vmpo_traineronpolicymaximumposteriori,
                title={V-MPO: On-Policy Maximum a Posteriori Policy Optimization for Discrete and Continuous Control}, 
                author={H. Francis Song and Abbas Abdolmaleki and Jost Tobias Springenberg and Aidan Clark and Hubert Soyer and Jack W. Rae and Seb Noury and Arun Ahuja and Siqi Liu and Dhruva Tirumala and Nicolas Heess and Dan Belov and Martin Riedmiller and Matthew M. Botvinick},
                year={2019},
                eprint={1909.12238},
                archivePrefix={arXiv},
                primaryClass={cs.AI},
                url={https://arxiv.org/abs/1909.12238}, 
            }"""
        ),
    }

    def __init__(
        self,
        args: TokenWeightedVMPOTrainerConfig,
        processing_class: (
            PreTrainedTokenizerBase
            | BaseImageProcessor
            | FeatureExtractionMixin
            | ProcessorMixin
        ),
        model: nn.Module,
        ref_model: nn.Module | None,
        reward_model: nn.Module,
        train_dataset: Dataset,
        value_model: nn.Module,
        data_collator: DataCollatorWithPadding | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        # less commonly used
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        callbacks: list[TrainerCallback] | None = None,
        peft_config: "PeftConfig | None" = None,
    ) -> None:
        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, you must make a copy of it, or `None` if you use peft."
            )

        self.args = args
        self.processing_class = processing_class
        self.policy_model = model
        self.is_deepspeed_enabled = False

        # Define the collator if not provided
        if data_collator is None:
            data_collator = DataCollatorWithPadding(self.processing_class)

        # Force a single EOS definition based on the tokenizer
        self.stop_token_id = processing_class.eos_token_id
        self.policy_model.generation_config.eos_token_id = self.stop_token_id

        # Check that the kl estimator is valid
        if self.args.kl_estimator not in {"k1", "k3"}:
            raise ValueError(
                "kl_estimator must be either 'k1' (straightforward, unbiased) or 'k3' (lower variance, unbiased, "
                "appears to be a strictly better estimator). See "
                "[Approximating KL Divergence](http://joschu.net/blog/kl-approx.html) for details."
            )

        # peft support
        if not is_peft_available() and peft_config is not None:
            raise ImportError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            if isinstance(self.policy_model, PeftModel):
                raise ValueError(
                    "You passed a `PeftModel` instance together with a `peft_config` to the trainer. Please first "
                    "merge and unload the existing adapter, save the resulting base model, and then pass that base "
                    "model along with the new `peft_config` to the trainer."
                )

            # get peft model with the given config
            self.policy_model = get_peft_model(self.policy_model, peft_config)
            if args.bf16 and getattr(self.policy_model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(self.policy_model)

        self.is_peft_model = is_peft_available() and isinstance(
            self.policy_model, PeftModel
        )
        self.model_adapter_name = args.model_adapter_name
        self.ref_adapter_name = args.ref_adapter_name

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model:
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(self.policy_model)

        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.value_model = value_model
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47

        #########
        # calculate various batch sizes
        #########
        if (
            args.total_episodes is None
        ):  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.local_mini_batch_size = exact_div(
            args.local_batch_size,
            args.num_mini_batches,
            "`local_batch_size` must be a multiple of `num_mini_batches`",
        )
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(
            time_tensor, 0
        ).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(
                1, args.num_total_batches // args.num_sample_generations
            )
        self.local_dataloader_batch_size = args.local_batch_size

        #########
        # setup model, optimizer, and others
        #########
        for module in [
            self.policy_model,
            self.ref_model,
            self.value_model,
            self.reward_model,
        ]:
            if module is not None:
                disable_dropout_in_model(module)
        self.model = PolicyAndValueWrapper(self.policy_model, self.value_model)
        self.model.config = self.policy_model.config  # needed for pushing to hub
        self.model.eta_raw = nn.Parameter(
            inv_softplus(torch.tensor(args.eta_init, device=self.accelerator.device))
        )
        self.model.alpha_raw = nn.Parameter(
            inv_softplus(torch.tensor(args.alpha_init, device=self.accelerator.device))
        )
        self.create_optimizer_and_scheduler(num_training_steps=args.num_total_batches)
        dual_param_ids = {id(self.model.eta_raw), id(self.model.alpha_raw)}
        for group in self.optimizer.param_groups:
            group["params"] = [
                p for p in group["params"] if id(p) not in dual_param_ids
            ]
        self.optimizer.param_groups = [
            g for g in self.optimizer.param_groups if len(g["params"]) > 0
        ]
        policy_lr = float(self.args.learning_rate)
        eta_lr = policy_lr * float(self.args.eta_lr_mult)
        alpha_lr = policy_lr * float(self.args.alpha_lr_mult)

        self.eta_optimizer = torch.optim.Adam(
            [self.model.eta_raw],
            lr=eta_lr,
            weight_decay=0.0,
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.model.alpha_raw],
            lr=alpha_lr,
            weight_decay=0.0,
        )

        #########
        # trainer specifics
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(
            self.args.report_to
        )
        self.callbacks = (
            default_callbacks if callbacks is None else default_callbacks + callbacks
        )
        self.callback_handler = CallbackHandler(
            self.callbacks,
            self.model,
            self.processing_class,
            self.optimizer,
            self.lr_scheduler,
        )
        self.add_callback(PrinterCallback)

        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb
                for cb in self.callback_handler.callbacks + [self.control]
                if isinstance(cb, ExportableState)
            ],
        )
        self.current_flos = 0
        self.hp_search_backend = None
        self.is_fsdp_enabled = (
            getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        )
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        #########
        # setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(
            self.model, self.optimizer, self.dataloader
        )
        torch.manual_seed(self.local_seed)  # reset the local seed again

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if self.ref_model is None:
            if not self.is_peft_model:
                raise ValueError("No reference model and model is not a Peft model.")
        else:
            self.ref_model = self.ref_model.to(self.accelerator.device)
        self.reward_model = self.reward_model.to(self.accelerator.device)

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with (
            self.accelerator.unwrap_model(self.model.policy).disable_adapter()
            if self.is_peft_model and not self.ref_adapter_name
            else nullcontext()
        ):
            if self.ref_adapter_name:
                self.model.policy.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.policy.set_adapter(self.model_adapter_name or "default")

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        backup_model = self.model
        self.model = self.model.policy  # save only the policy

        super().save_model(output_dir, _internal_call)

        self.model = backup_model

    def collect_rollouts(
        self,
        generation_kwargs: dict,
        accelerator: Accelerator,
        iter_dataloader,
        model,
        ref_policy,
        reward_model,
        processing_class,
        generation_config,
        eos_id,
        device,
    ):
        data = next(iter_dataloader)
        with torch.no_grad():
            queries = data["input_ids"].to(device)
            context_length = queries.shape[1]
            responses = []
            postprocessed_responses = []
            contain_eos_token_raw = []
            logprobs = []
            ref_logprobs = []
            scores = []
            sequence_lengths = []
            values = []
            rollout_logits = []
            with unwrap_model_for_generation(
                self.model,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                generation_kwargs=generation_kwargs,  # Override model.generation_config with generation_kwargs to fix transformers#42762
            ) as unwrapped_model:
                query_responses, logitss = batch_generation(
                    unwrapped_model.policy,
                    queries,
                    self.args.local_rollout_forward_batch_size,
                    processing_class.pad_token_id,
                    generation_kwargs,
                )

            for i in range(
                0, queries.shape[0], self.args.local_rollout_forward_batch_size
            ):
                query = queries[i : i + self.args.local_rollout_forward_batch_size]
                query_response = query_responses[
                    i : i + self.args.local_rollout_forward_batch_size
                ]
                response = query_response[:, context_length:]
                contain_eos_token_raw.append((response == eos_id).any(dim=1))
                logits = logitss[i : i + self.args.local_rollout_forward_batch_size]
                logprob = selective_log_softmax(logits, response)
                rollout_logits.append(logits)
                del logits
                empty_cache()

                if ref_policy is None:
                    with self.null_ref_context():
                        ref_output = forward(
                            model.policy,
                            query_response,
                            processing_class.pad_token_id,
                        )
                else:
                    ref_output = forward(
                        ref_policy, query_response, processing_class.pad_token_id
                    )
                ref_logits = ref_output.logits[:, context_length - 1 : -1]
                ref_logprob = selective_log_softmax(ref_logits, response)
                del ref_output, ref_logits
                empty_cache()

                # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                postprocessed_response = response
                if (
                    self.stop_token_id is not None
                ):  # handle the edge case when stop_token_id exists but is 0
                    postprocessed_response = truncate_response(
                        self.stop_token_id, processing_class.pad_token_id, response
                    )

                # Response Processing 2. run reward model on the truncated responses
                postprocessed_query_response = torch.cat(
                    (query, postprocessed_response), 1
                )
                pad_mask = postprocessed_response == processing_class.pad_token_id
                has_pad = pad_mask.any(dim=1)
                pad_idx = first_true_indices(pad_mask)

                sequence_length = torch.where(
                    has_pad,
                    pad_idx - 1,
                    torch.full_like(pad_idx, postprocessed_response.shape[1] - 1),
                )
                unwrapped_value_model = accelerator.unwrap_model(model).value_model
                full_value, _, _ = get_reward(
                    unwrapped_value_model,
                    query_response,
                    processing_class.pad_token_id,
                    context_length,
                )
                value = full_value[:, context_length - 1 : -1].squeeze(-1)
                _, score, _ = get_reward(
                    reward_model,
                    postprocessed_query_response,
                    processing_class.pad_token_id,
                    context_length,
                )

                responses.append(response)
                postprocessed_responses.append(postprocessed_response)
                logprobs.append(logprob)
                ref_logprobs.append(ref_logprob)
                sequence_lengths.append(sequence_length)
                scores.append(score)
                values.append(value)
            responses = torch.cat(responses, 0)
            postprocessed_responses = torch.cat(postprocessed_responses, 0)
            contain_eos_token_raw = torch.cat(contain_eos_token_raw, 0)
            logprobs = torch.cat(logprobs, 0)
            ref_logprobs = torch.cat(ref_logprobs, 0)
            sequence_lengths = torch.cat(sequence_lengths, 0)
            scores = torch.cat(scores, 0)
            values = torch.cat(values, 0)
            rollout_logits = torch.cat(rollout_logits, 0)
            del (logprob, ref_logprob, full_value, value, score, unwrapped_model)
            empty_cache()
            gc.collect()

            # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
            # Completions not passing that filter will receive a lower score.
            contain_eos_token_post = (postprocessed_responses == eos_id).any(dim=1)
            if self.args.missing_eos_penalty is not None:
                scores[~contain_eos_token_post] -= self.args.missing_eos_penalty

            # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
            response_idxs = torch.arange(
                responses.shape[1], device=responses.device
            ).repeat(responses.shape[0], 1)
            padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
            logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
            ref_logprobs = torch.masked_fill(
                ref_logprobs, padding_mask, INVALID_LOGPROB
            )

            # --- rollout diagnostics (sampling-time) ---
            valid_token_mask = ~padding_mask

            # Mean token logprob under the rollout policy (sampling proxy; NOT optimization entropy)
            rollout_token_logprobs_mean = logprobs.sum() / (
                valid_token_mask.sum() + 1e-8
            )

            # True token entropy of the rollout distribution from rollout logits (still sampling-time)
            with torch.no_grad():
                roll_logp = F.log_softmax(rollout_logits, dim=-1)
                roll_p = roll_logp.exp()
                roll_tok_entropy = -(roll_p * roll_logp).sum(dim=-1)  # [B, T]
                rollout_entropy_mean = (
                    roll_tok_entropy[valid_token_mask].mean()
                    if valid_token_mask.any()
                    else torch.zeros((), device=rollout_logits.device)
                )
            # --- end rollout diagnostics ---

            # --- entropy diagnostic: mean token logprob over valid (non-pad) response tokens ---
            valid_token_mask = ~padding_mask
            token_logprobs_mean = logprobs.sum() / (valid_token_mask.sum() + 1e-8)
            # --- end entropy diagnostic ---

            # rollout logits alignment check
            old_lp = selective_log_softmax(rollout_logits, responses)
            diff = (old_lp - logprobs).abs()
            diff = diff.masked_fill(padding_mask, 0)
            mean_diff = diff.sum() / ((~padding_mask).sum() + 1e-8)
            assert mean_diff < 1e-4, "rollout_logits misaligned with stored logprobs"

            sequence_lengths_p1 = sequence_lengths + 1
            padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
            values = torch.masked_fill(values, padding_mask_p1, 0)

            # 4. compute rewards
            # Formula used by http://joschu.net/blog/kl-approx.html for the k1 and k3 estimators
            logr = ref_logprobs - logprobs
            kl = -logr if self.args.kl_estimator == "k1" else (logr.exp() - 1) - logr
            # trust region KL is handled by α; no PPO-style shaping
            rewards = torch.zeros_like(kl)
            actual_start = torch.arange(rewards.size(0), device=rewards.device)
            actual_end = torch.where(
                sequence_lengths_p1 < rewards.size(1),
                sequence_lengths_p1,
                sequence_lengths,
            )

            # Diffuse terminal reward over the final K tokens (avoids a single-token advantage spike)
            K = int(self.args.reward_spread_k)
            if K == 1:
                rewards[actual_start, actual_end] += scores
            else:
                # spread over as many tail tokens as exist: k_eff = min(K, actual_end+1)
                k_eff = torch.minimum(
                    torch.full_like(actual_end, K),
                    actual_end + 1,
                ).to(dtype=scores.dtype)
                score_per = scores / k_eff
                for offset in range(K):
                    m = actual_end >= offset
                    if m.any():
                        idx = actual_end[m] - offset
                        rewards[actual_start[m], idx] += score_per[m]

            # logging masks/summaries
            reward_valid_mask = ~padding_mask_p1
            reward_seq = rewards.masked_fill(padding_mask_p1, 0).sum(1)
            reward_tok = reward_seq.sum() / (reward_valid_mask.sum() + 1e-8)

            # 6. compute advantages and returns
            lastgaelam = 0
            advantages_reversed = []
            gen_length = responses.shape[1]
            for t in reversed(range(gen_length)):
                nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                delta = rewards[:, t] + self.args.gamma * nextvalues - values[:, t]
                not_done = ~padding_mask_p1[:, t]
                lastgaelam = (
                    delta + self.args.gamma * self.args.lam * lastgaelam * not_done
                )
                advantages_reversed.append(lastgaelam)
            raw_advantages = torch.stack(advantages_reversed[::-1], axis=1)
            returns = raw_advantages + values
            empty_cache()
        return (
            padding_mask_p1,
            raw_advantages,
            queries,
            responses,
            context_length,
            logprobs,
            scores,
            values,
            returns,
            reward_seq,
            reward_tok,
            contain_eos_token_raw,
            contain_eos_token_post,
            rollout_logits,
            rollout_token_logprobs_mean,
            rollout_entropy_mean,
        )

    def _m_step(
        self,
        *,
        psi_global: torch.Tensor,
        apply_entropy_reg: bool,
        queries: torch.Tensor,
        responses: torch.Tensor,
        padding_mask_p1: torch.Tensor,
        old_logprobs_for_kl: torch.Tensor,  # recomputed per M-step pass
        values: torch.Tensor,
        returns: torch.Tensor,
        context_length: int,
        pg_losses: list,
        vf_losses: list,
        entropy_stats: list,
        kl_weighted_num_accum: torch.Tensor,
        kl_weighted_den_accum: torch.Tensor,
        kl_weighted_max_tok_accum: torch.Tensor,  # NEW: track local spikes
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list, list, list, list]:
        kl_vals_last_mstep: list[torch.Tensor] = []
        b_inds = np.random.permutation(self.args.local_batch_size)
        for start in range(0, len(b_inds), self.args.local_mini_batch_size):
            mb_inds = b_inds[start : start + self.args.local_mini_batch_size]
            mb_query = queries[mb_inds]
            mb_response = responses[mb_inds]
            mb_query_response = torch.cat((mb_query, mb_response), 1)
            psi_mb = psi_global[mb_inds]
            mask_valid_mb = ~padding_mask_p1[mb_inds]

            policy_outputs = forward(
                self.model.policy,
                mb_query_response,
                self.processing_class.pad_token_id,
            )
            new_logits = policy_outputs.logits[:, context_length - 1 : -1]
            new_logprobs = selective_log_softmax(new_logits, mb_response)

            # True optimization-time policy entropy diagnostic (current policy on current minibatch)
            with torch.no_grad():
                logp_all = F.log_softmax(new_logits, dim=-1)
                p_all = logp_all.exp()
                token_entropy = -(p_all * logp_all).sum(dim=-1)  # [mb, T]
                entropy_mb = (
                    token_entropy[mask_valid_mb].mean()
                    if mask_valid_mb.any()
                    else torch.zeros((), device=new_logits.device)
                )
                entropy_stats.append(entropy_mb.detach())

            psi_mass_mb = psi_mb.sum()
            if psi_mass_mb > 0:
                policy_loss = -((psi_mb * new_logprobs).sum() / (psi_mass_mb + 1e-8))
            else:
                policy_loss = torch.zeros((), device=self.accelerator.device)

            value_full, _, _ = get_reward(
                self.accelerator.unwrap_model(self.model).value_model,
                mb_query_response,
                self.processing_class.pad_token_id,
                context_length,
            )
            value_pred = value_full[:, context_length - 1 : -1].squeeze(-1)
            value_mask = mask_valid_mb
            with torch.no_grad():
                value_old_mb = values[mb_inds]
                returns_mb = returns[mb_inds]
                delta = (returns_mb - value_old_mb).clamp(
                    -self.args.value_clip, self.args.value_clip
                )
                value_target = value_old_mb + delta

            # ψ-weighted critic loss (EM-consistent): E_ψ[(V - target)^2]
            weight = psi_mb * value_mask.to(dtype=psi_mb.dtype)
            denom = weight.sum().clamp_min(1e-8)
            if denom > 0:
                value_loss = (((value_pred - value_target) ** 2) * weight).sum() / denom
            else:
                value_loss = torch.zeros((), device=self.accelerator.device)

            # expected KL under token-level ψ (use per-pass old_logprobs, not rollout logprobs)
            kl_terms_token = (
                old_logprobs_for_kl[mb_inds] - new_logprobs
            ) * mask_valid_mb
            kl_num_mb = (psi_mb * kl_terms_token).sum()
            kl_den_mb = psi_mass_mb.clamp_min(1e-8)
            kl_mean_mb = kl_num_mb / kl_den_mb

            # NEW: max ψ-weighted token KL (watch for local KL spikes)
            with torch.no_grad():
                kl_weighted_tok = (psi_mb * kl_terms_token).masked_fill(
                    ~mask_valid_mb, float("-inf")
                )
                kl_weighted_max_mb = (
                    kl_weighted_tok.max()
                    if mask_valid_mb.any()
                    else torch.tensor(float("-inf"), device=self.accelerator.device)
                )

            alpha = F.softplus(self.model.alpha_raw) + self.args.alpha_min
            total_loss = (
                policy_loss
                + alpha.detach() * kl_mean_mb
                + (self.args.vf_coef / self.args.m_steps) * value_loss
            )

            # Conditional entropy regularization (only if ESS is low for this rollout)
            if apply_entropy_reg and self.args.entropy_beta > 0.0:
                # reuse the already-computed token_entropy/logp_all/p_all shape logic
                if mask_valid_mb.any():
                    total_loss = total_loss + (
                        -self.args.entropy_beta * token_entropy[mask_valid_mb].mean()
                    )

            self.accelerator.backward(total_loss)
            self.optimizer.step()
            self.optimizer.zero_grad()

            kl_weighted_num_accum += kl_num_mb.detach()
            kl_weighted_den_accum += kl_den_mb.detach()
            kl_weighted_max_tok_accum = torch.maximum(
                kl_weighted_max_tok_accum, kl_weighted_max_mb.detach()
            )
            pg_losses.append(policy_loss.detach())
            vf_losses.append(value_loss.detach())
            kl_vals_last_mstep.append(kl_mean_mb.detach())

        # α dual update: enforce E_ψ[KL] <= eps_alpha
        self.alpha_optimizer.zero_grad()
        alpha = F.softplus(self.model.alpha_raw) + self.args.alpha_min
        kl_mean = kl_weighted_num_accum / (kl_weighted_den_accum.clamp_min(1e-8))
        l_alpha = -alpha * (kl_mean - self.args.eps_alpha)
        l_alpha.backward()
        self.alpha_optimizer.step()

        return (
            kl_weighted_num_accum,
            kl_weighted_den_accum,
            kl_weighted_max_tok_accum,
            pg_losses,
            vf_losses,
            kl_vals_last_mstep,
            entropy_stats,
        )

    def train(self):
        processing_class = self.processing_class
        dataloader = self.dataloader
        accelerator = self.accelerator
        device = accelerator.device
        model = self.model
        ref_policy = self.ref_model
        reward_model = self.reward_model

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_kwargs = {
            "max_new_tokens": self.args.response_length,
            "temperature": (self.args.temperature + 1e-7),
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "early_stopping": True,
            "eos_token_id": self.stop_token_id,
            "pad_token_id": processing_class.pad_token_id,
        }
        eos_id = self.stop_token_id
        assert eos_id is not None
        assert processing_class.pad_token_id is not None
        assert eos_id != processing_class.pad_token_id

        accelerator.print("===training policy===")
        stats_shape = (
            self.args.num_mini_batches,
            self.args.gradient_accumulation_steps,
        )
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = self.args.num_total_batches
        self.state.num_train_epochs = self.args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if self.args.logging_steps is not None:
            if self.args.logging_steps < 1:
                self.state.logging_steps = math.ceil(
                    self.state.max_steps * self.args.logging_steps
                )
            else:
                self.state.logging_steps = self.args.logging_steps
        if self.args.eval_steps is not None:
            if self.args.eval_steps < 1:
                self.state.eval_steps = math.ceil(
                    self.state.max_steps * self.args.eval_steps
                )
            else:
                self.state.eval_steps = self.args.eval_steps
        if self.args.save_steps is not None:
            if self.args.save_steps < 1:
                self.state.save_steps = math.ceil(
                    self.state.max_steps * self.args.save_steps
                )
            else:
                self.state.save_steps = self.args.save_steps
        self.control = self.callback_handler.on_train_begin(
            self.args, self.state, self.control
        )

        for update in range(1, self.args.num_total_batches + 1):
            approxkl_stats.zero_()
            pg_loss_stats.zero_()
            vf_loss_stats.zero_()
            vf_clipfrac_stats.zero_()
            self.state.episode += 1 * self.args.batch_size

            (
                padding_mask_p1,
                raw_advantages,
                queries,
                responses,
                context_length,
                logprobs,
                scores,
                values,
                returns,
                reward_seq,
                reward_tok,
                contain_eos_token_raw,
                contain_eos_token_post,
                rollout_logits,
                rollout_token_logprobs_mean,
                rollout_entropy_mean,
            ) = self.collect_rollouts(
                generation_kwargs,
                accelerator,
                iter_dataloader,
                model,
                ref_policy,
                reward_model,
                processing_class,
                None,  # CHANGED: no GenerationConfig needed
                eos_id,
                device,
            )

            psi_global, l_eta_full_values = self.e_step_dual_update(
                padding_mask_p1=padding_mask_p1,
                raw_advantages=raw_advantages,
                device=device,
            )
            psi_global = psi_global.detach()

            # rollout-level ESS over tokens (consistent trigger across all minibatches this rollout)
            with torch.no_grad():
                psi_tok = psi_global[psi_global > 0]
                psi_ess_tok = (
                    1.0 / (psi_tok.pow(2).sum().clamp_min(1e-8))
                    if psi_tok.numel() > 0
                    else torch.zeros((), device=device)
                )
                apply_entropy_reg = bool(
                    (self.args.entropy_beta > 0.0)
                    and (psi_ess_tok.item() < float(self.args.psi_ess_min_tokens))
                )

            kl_weighted_num_accum = torch.zeros((), device=device)
            kl_weighted_den_accum = torch.zeros((), device=device)
            kl_weighted_max_tok_accum = torch.tensor(float("-inf"), device=device)
            pg_losses, vf_losses, kl_vals_last_mstep, entropy_stats = [], [], [], []

            for _ in range(self.args.m_steps):
                # Recompute "old" logprobs under the current policy snapshot (before this pass's updates)
                with torch.no_grad():
                    query_responses = torch.cat((queries, responses), 1)
                    old_logprobs_chunks = []
                    bs = query_responses.shape[0]
                    step_bs = int(self.args.local_rollout_forward_batch_size)
                    for i in range(0, bs, step_bs):
                        qr = query_responses[i : i + step_bs]
                        out = forward(
                            self.model.policy,
                            qr,
                            self.processing_class.pad_token_id,
                        )
                        old_logits = out.logits[:, context_length - 1 : -1]
                        old_lp = selective_log_softmax(
                            old_logits, responses[i : i + step_bs]
                        )
                        old_logprobs_chunks.append(old_lp)
                    old_logprobs_for_kl = torch.cat(old_logprobs_chunks, dim=0)

                (
                    kl_weighted_num_accum,
                    kl_weighted_den_accum,
                    kl_weighted_max_tok_accum,
                    pg_losses,
                    vf_losses,
                    kl_vals_last_mstep,
                    entropy_stats,
                ) = self._m_step(
                    psi_global=psi_global,
                    apply_entropy_reg=apply_entropy_reg,
                    queries=queries,
                    responses=responses,
                    padding_mask_p1=padding_mask_p1,
                    old_logprobs_for_kl=old_logprobs_for_kl,
                    values=values,
                    returns=returns,
                    context_length=context_length,
                    pg_losses=pg_losses,
                    vf_losses=vf_losses,
                    entropy_stats=entropy_stats,
                    kl_weighted_num_accum=kl_weighted_num_accum,
                    kl_weighted_den_accum=kl_weighted_den_accum,
                    kl_weighted_max_tok_accum=kl_weighted_max_tok_accum,
                )
            policy_entropy_mean = (
                torch.stack(entropy_stats).mean()
                if len(entropy_stats) > 0
                else torch.zeros((), device=device)
            )

            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(
                    self.args, self.state, self.control
                )
            del (scores,)
            empty_cache()
            gc.collect()

            if (
                self.args.num_sample_generations > 0
                and (update - 1) % self.sample_generations_freq == 0
            ):
                self.generate_completions(sampling=True)
                empty_cache()
            del (
                responses,
                logprobs,
                values,
                contain_eos_token_post,
                contain_eos_token_raw,
                rollout_logits,
            )
            empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(
            self.args, self.state, self.control
        )
        if self.control.should_save:
            self._save_checkpoint(model, trial=None)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

    def e_step_dual_update(
        self,
        padding_mask_p1,
        raw_advantages: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, list]:
        # token-level ψ_{i,t} ∝ exp(A_{i,t} / η), only over valid tokens, with Σ ψ = 1
        mask_valid = ~padding_mask_p1  # [B, T]
        A = raw_advantages  # [B, T]
        psi_global = torch.zeros_like(A)
        l_eta_full_values: list[torch.Tensor] = []

        # Explicit (b,t) indices for valid tokens to make scatter-back robust to refactors
        valid_idx = mask_valid.nonzero(as_tuple=False)  # [N_tok, 2]

        A_valid = A[mask_valid]  # [N_tok]
        n_tok = A_valid.numel()
        if n_tok == 0:
            return psi_global, l_eta_full_values

        # Center advantages over valid tokens (reduces sensitivity to reward scale/offset drift)
        A_valid = A_valid - A_valid.mean()

        if self.args.psi_topk_ratio is None or self.args.psi_topk_ratio >= 1.0:
            mask_topk = torch.ones_like(A_valid, dtype=torch.bool)
        else:
            k = max(1, int(self.args.psi_topk_ratio * n_tok))
            threshold = torch.topk(A_valid, k).values[-1]
            mask_topk = A_valid >= threshold

        A_sel = A_valid[mask_topk]
        n_sel = A_sel.numel()
        if n_sel == 0:
            return psi_global, l_eta_full_values

        # Compensate eps_eta for top-k truncation to keep η behavior consistent as n_sel changes
        effective_eps_eta = self.args.eps_eta + math.log(n_tok / n_sel)

        # Helper: clamp only for exponentiation/log-mean-exp stability (do NOT affect top-k selection)
        def _clip_for_exp(x: torch.Tensor) -> torch.Tensor:
            clip = self.args.advantage_clip
            return x.clamp(min=-clip, max=clip) if clip is not None else x

        for _ in range(self.args.eta_inner_steps):
            eta = F.softplus(self.model.eta_raw) + self.args.eta_min

            A_sel_exp = _clip_for_exp(A_sel)
            A_max = A_sel_exp.max()
            weights = torch.exp((A_sel_exp - A_max) / (eta + 1e-8))
            Z = weights.sum().clamp_min(1e-8)

            log_mean_exp = torch.log(Z) + (A_max / (eta + 1e-8)) - math.log(n_sel)
            l_eta = eta * (effective_eps_eta + log_mean_exp)

            self.eta_optimizer.zero_grad()
            l_eta.backward()
            self.eta_optimizer.step()
            l_eta_full_values.append(l_eta.detach())

        # Construct ψ with final η (scatter back into [B, T])
        with torch.no_grad():
            eta = F.softplus(self.model.eta_raw) + self.args.eta_min

            A_sel_exp = _clip_for_exp(A_sel)
            A_max = A_sel_exp.max()
            weights = torch.exp((A_sel_exp - A_max) / (eta + 1e-8))
            Z = weights.sum().clamp_min(1e-8)
            psi_sel = weights / Z  # sums to 1 over selected tokens

            psi_flat = torch.zeros_like(A_valid)
            psi_flat[mask_topk] = psi_sel
            psi_global[valid_idx[:, 0], valid_idx[:, 1]] = psi_flat

        return psi_global, l_eta_full_values

    def generate_metrics(
        self,
        kl_weighted_num_accum: torch.Tensor,
        kl_weighted_den_accum: torch.Tensor,
        kl_weighted_max_tok_accum: torch.Tensor,  # NEW
        reward_seq: torch.Tensor,
        reward_tok: torch.Tensor,
        psi_global: torch.Tensor,
        l_eta_full_values: list,
        scores: torch.Tensor,
        approxkl_stats: torch.Tensor,
        pg_loss_stats: torch.Tensor,
        vf_loss_stats: torch.Tensor,
        vf_clipfrac_stats: torch.Tensor,
        contain_eos_token_raw: torch.Tensor,
        contain_eos_token_post: torch.Tensor,
        rollout_token_logprobs_mean: torch.Tensor,
        rollout_entropy_mean: torch.Tensor,
        policy_entropy_mean: torch.Tensor,
        device: torch.device,
    ) -> dict[str, float]:
        """Generate a dictionary of training metrics for logging."""
        with torch.no_grad():
            kl_weighted_post = kl_weighted_num_accum / (
                kl_weighted_den_accum.clamp_min(1e-8)
            )

            rlhf_reward = reward_seq.mean()
            eta_value = F.softplus(self.model.eta_raw) + self.args.eta_min
            alpha_value = F.softplus(self.model.alpha_raw) + self.args.alpha_min
            eta_raw = self.model.eta_raw.detach().cpu().item()
            alpha_raw = self.model.alpha_raw.detach().cpu().item()

            # token-level ψ diagnostics
            psi_tok = psi_global[psi_global > 0]
            if psi_tok.numel() > 0:
                psi_ess_tok = 1.0 / (psi_tok.pow(2).sum().clamp_min(1e-8))
                psi_max_tok = psi_tok.max()
            else:
                psi_ess_tok = torch.zeros((), device=device)
                psi_max_tok = torch.zeros((), device=device)

            l_eta_mean = (
                torch.stack(l_eta_full_values).mean()
                if len(l_eta_full_values) > 0
                else torch.zeros((), device=device)
            )
            eta_grad = self.model.eta_raw.grad
            eta_grad_norm = eta_grad.norm().item() if eta_grad is not None else 0.0

            metrics = {}
            metrics["objective/kl_ref_weighted"] = (
                self.accelerator.gather_for_metrics(kl_weighted_post).mean().item()
            )
            metrics["objective/kl_ref_weighted_max_tok"] = (
                self.accelerator.gather_for_metrics(kl_weighted_max_tok_accum)
                .mean()
                .item()
            )

            # keep "psi/*" keys but make them token-level
            metrics["psi/ess"] = (
                self.accelerator.gather_for_metrics(psi_ess_tok).mean().item()
            )
            metrics["psi/max"] = (
                self.accelerator.gather_for_metrics(psi_max_tok).mean().item()
            )

            metrics["objective/rlhf_reward_seq"] = (
                self.accelerator.gather_for_metrics(reward_seq).mean().item()
            )
            metrics["objective/rlhf_reward_tok"] = (
                self.accelerator.gather_for_metrics(reward_tok).mean().item()
            )
            metrics["objective/rlhf_reward"] = (
                self.accelerator.gather_for_metrics(rlhf_reward).mean().item()
            )
            metrics["debug/reward_std"] = scores.std().item()
            metrics["policy/approxkl_avg"] = (
                self.accelerator.gather_for_metrics(approxkl_stats).mean().item()
            )
            metrics["loss/policy_avg"] = (
                self.accelerator.gather_for_metrics(pg_loss_stats).mean().item()
            )
            metrics["loss/value_avg"] = (
                self.accelerator.gather_for_metrics(vf_loss_stats).mean().item()
            )
            metrics["val/clipfrac_avg"] = (
                self.accelerator.gather_for_metrics(vf_clipfrac_stats).mean().item()
            )
            metrics["dual/eta_grad_norm"] = eta_grad_norm
            metrics["dual/eta_value"] = (
                self.accelerator.gather_for_metrics(eta_value).mean().item()
            )
            metrics["dual/alpha_value"] = (
                self.accelerator.gather_for_metrics(alpha_value).mean().item()
            )
            metrics["dual/l_eta_mean"] = (
                self.accelerator.gather_for_metrics(l_eta_mean).mean().item()
            )
            metrics["dual/eta_raw"] = eta_raw
            metrics["dual/alpha_raw"] = alpha_raw
            metrics["val/num_eos_tokens_raw"] = contain_eos_token_raw.sum().item()
            metrics["val/num_eos_tokens"] = contain_eos_token_post.sum().item()
            metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
            metrics["episode"] = self.state.episode

            # rollout/sampling-time diagnostics (do not interpret as optimization entropy)
            metrics["rollout/token_logprobs_mean"] = (
                self.accelerator.gather_for_metrics(rollout_token_logprobs_mean)
                .mean()
                .item()
            )
            metrics["rollout/entropy_tok_mean"] = (
                self.accelerator.gather_for_metrics(rollout_entropy_mean).mean().item()
            )
            metrics["deprecated/policy_token_logprobs_mean"] = metrics[
                "rollout/token_logprobs_mean"
            ]

            # optimization-time diagnostic: current policy entropy on training minibatches
            metrics["policy/entropy_tok_mean"] = (
                self.accelerator.gather_for_metrics(policy_entropy_mean).mean().item()
            )

            # Optional sanity checks for truncation (won't error if disabled)
            metrics["psi/topk_ratio"] = float(self.args.psi_topk_ratio or 1.0)
            metrics["psi/num_tok_nonzero"] = float((psi_global > 0).sum().item())

            metrics["entropy/beta"] = float(self.args.entropy_beta)
            metrics["entropy/ess_min_tokens"] = float(self.args.psi_ess_min_tokens)
            metrics["entropy/active"] = float(
                (self.args.entropy_beta > 0.0)
                and (psi_ess_tok.item() < float(self.args.psi_ess_min_tokens))
            )

        return metrics

    def generate_completions(self, sampling: bool = False):
        args = self.args
        processing_class = self.processing_class
        generation_kwargs = {
            "max_new_tokens": args.response_length,
            "temperature": (args.temperature + 1e-7),
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "early_stopping": True,
            "eos_token_id": self.stop_token_id,
            "pad_token_id": processing_class.pad_token_id,
        }
        eos_id = self.stop_token_id
        assert eos_id is not None
        assert processing_class.pad_token_id is not None
        assert eos_id != processing_class.pad_token_id

        table = defaultdict(list)
        with unwrap_model_for_generation(
            self.model,
            self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation,
            generation_kwargs=generation_kwargs,
        ) as unwrapped_model:
            for batch in self.eval_dataloader:
                query = batch["input_ids"]
                with torch.no_grad():
                    context_length = query.shape[1]
                    query_response, _ = batch_generation(
                        unwrapped_model.policy,
                        query,
                        query.shape[0],
                        processing_class.pad_token_id,
                        generation_kwargs,
                    )
                    response = query_response[:, context_length:]
                    postprocessed_response = response
                    if (
                        self.stop_token_id is not None
                    ):  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            eos_id, processing_class.pad_token_id, response
                        )
                    table["query"].extend(
                        gather_object(
                            processing_class.batch_decode(
                                query, skip_special_tokens=True
                            )
                        )
                    )
                    table["model response"].extend(
                        gather_object(
                            processing_class.batch_decode(postprocessed_response)
                        )
                    )

                    postprocessed_query_response = torch.cat(
                        (query, postprocessed_response), 1
                    )
                    _, score, _ = get_reward(
                        self.reward_model,
                        postprocessed_query_response,
                        processing_class.pad_token_id,
                        context_length,
                    )
                    table["score"].extend(
                        self.accelerator.gather_for_metrics(score).float().cpu().numpy()
                    )

                if sampling:
                    break
        df = pd.DataFrame(table)

        if self.accelerator.is_main_process:
            if is_rich_available():
                print_rich_table(df.iloc[0 : 0 + 5])
            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

            if "comet_ml" in args.report_to:
                log_table_to_comet_experiment(
                    name="completions.csv",
                    table=df,
                )

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
