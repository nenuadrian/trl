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
)
import transformers
from transformers import TrainingArguments
from packaging.version import Version
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
    prepare_deepspeed,
    selective_log_softmax,
)
from ..utils import first_true_indices, get_reward


@dataclass
class VMPOConfig(TrainingArguments):
    r"""
    Configuration class for the [`experimental.vmpo.VMPOTrainer`].

    This class includes only the parameters that are specific to VMPO training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        run_name (`str`, *optional*):
            Name of the run.
        dataset_num_proc (`int`, *optional*):
            Number of processes to use for processing the dataset.
        num_mini_batches (`int`, *optional*, defaults to `1`):
            Number of minibatches to split a batch into.
        total_episodes (`int`, *optional*):
            Total number of episodes in the dataset.
        local_rollout_forward_batch_size (`int`, *optional*, defaults to `64`):
            Per rank no grad forward pass in the rollout phase.
        num_sample_generations (`int`, *optional*, defaults to `10`):
            Number of debugging samples generations (i.e., `generate_completions` calls) throughout training.
        response_length (`int`, *optional*, defaults to `53`):
            Length of the response.
        stop_token (`str`, *optional*):
            Specifies the stop token to use for text generation. This parameter is mutually exclusive with
            `stop_token_id`.

            - `None`: No stop token is applied, unless `stop_token_id` is specified.
            - `'eos'`: Uses the tokenizer's `eos_token`.

        stop_token_id (`int`, *optional*):
            Specifies the ID of the stop token to use for text generation. If `None`, no stop token ID is applied,
            unless `stop_token` is specified. This parameter is mutually exclusive with `stop_token`.
        temperature (`float`, *optional*, defaults to `0.7`):
            Sampling temperature.
        missing_eos_penalty (`float`, *optional*):
            Penalty applied to the score when the model fails to generate an EOS token. This is useful to encourage to
            generate completions shorter than the maximum length (`max_new_tokens`). The penalty must be a positive
            value.
        sft_model_path (`str`, *optional*, defaults to `"EleutherAI/pythia-160m"`):
            Path to the SFT model.
        world_size (`int`, *optional*):
            Number of processes (GPUs) to use for the training.
        num_total_batches (`int`, *optional*):
            Number of total batches to train.
        micro_batch_size (`int`, *optional*):
            Micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`).
        local_batch_size (`int`, *optional*):
            Batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`).
        batch_size (`int`, *optional*):
            Batch size across devices (HF's `per_device_train_batch_size` * `world_size` *
            `gradient_accumulation_steps`).
        local_mini_batch_size (`int`, *optional*):
            Mini batch size per GPU.
        mini_batch_size (`int`, *optional*):
            Mini batch size across GPUs.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the model to the Hub after training.
        exp_name (`str`, *optional*, defaults to `os.path.basename(__file__)[:-3]`):
            Name of this experiment.
        reward_model_path (`str`, *optional*, defaults to `"EleutherAI/pythia-160m"`):
            Path to the reward model.
        value_clip (`float`, *optional*, defaults to `0.2`):
            Clip range for value targets (±value_clip).
        beta_entropy (`float`, *optional*, defaults to `0.01`):
            Entropy regularization coefficient.
        beta_entropy_decay (`float`, *optional*, defaults to `0.0`):
            Exponential decay rate k for entropy coefficient: beta_t = beta0 * exp(-k * t). Set 0 to disable.

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
    top_frac: float = field(
        default=0.1,
        metadata={
            "help": "Fraction of valid token advantages to keep for V-MPO weighting (top fraction by advantage)."
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
    mini_batch_size: int | None = field(
        default=None,
        metadata={"help": "Mini batch size across GPUs."},
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
    num_vmpo_epochs: int = field(
        default=1,
        metadata={"help": "Number of epochs to train."},
    )
    whiten_rewards: bool = field(
        default=False,
        metadata={"help": "Whether to whiten the rewards."},
    )
    kl_coef: float = field(
        default=0.05,
        metadata={"help": "KL coefficient."},
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
    beta_entropy: float = field(
        default=0.01,
        metadata={"help": "Entropy regularization coefficient."},
    )
    beta_entropy_decay: float = field(
        default=0.0,
        metadata={
            "help": "Exponential decay rate k for entropy coefficient: beta_t = beta0 * exp(-k * t). Set 0 to disable."
        },
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

        # Transformers explicitly set use_reentrant=True in the past to silence a PyTorch warning, but the default was
        # never updated once PyTorch switched to recommending use_reentrant=False. Until that change lands upstream
        # (see https://github.com/huggingface/transformers/pull/43203) and is released (most likely in 5.0.0), we
        # default to the recommended non-reentrant behavior here, while preserving any user-provided value.
        if self.gradient_checkpointing and Version(transformers.__version__) < Version(
            "5.0.0"
        ):
            self.gradient_checkpointing_kwargs = (
                self.gradient_checkpointing_kwargs or {}
            )
            self.gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

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
    generation_config: GenerationConfig,
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
        generation_config ([`~transformers.GenerationConfig`]):
            The configuration for the generation process.

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
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # not needed: already adjusted in generations
        # https://github.com/huggingface/transformers/blob/ac33aeeeee2a7a89b89c93c2962e6feb90daef0a/src/transformers/models/gpt2/modeling_gpt2.py#L1227-L1250
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
    )
    logits = torch.stack(output.scores, 1)
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1), logits


@torch.no_grad()
def batch_generation(
    model: torch.nn.Module,
    queries: torch.Tensor,
    local_rollout_forward_batch_size: int,
    pad_token_id: int,
    generation_config: GenerationConfig,
):
    query_responses = []
    logitss = []
    batch_size = queries.shape[0]
    for i in range(0, batch_size, local_rollout_forward_batch_size):
        query = queries[i : i + local_rollout_forward_batch_size]
        query_response, logits = generate(
            model,
            query,
            pad_token_id,
            generation_config,
        )
        query_responses.append(query_response)
        logitss.append(logits)

    # padding tensors
    padded_query_responses = pad(
        query_responses, padding_value=pad_token_id, padding_side="right"
    )
    padded_logitss = pad(logitss, padding_value=0, padding_side="right")

    # reshaping
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
        output_hidden_states=False,
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
        logits = self.value_model.score(output.last_hidden_state)
        return self.policy(**kwargs), logits


class VMPOTrainer(BaseTrainer):
    """Trainer for VMPO.

    Args:
        args ([`experimental.vmpo.VMPOConfig`]):
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

    _tag_names = ["trl", "vmpo"]
    _name = "VMPO"
    _paper = {
        "title": "Fine-Tuning Language Models from Human Preferences",
        "id": "1909.08593",
        # docstyle-ignore
        "citation": textwrap.dedent(
            """\
            @article{mziegler2019fine-tuning,
                title        = {{Fine-Tuning Language Models from Human Preferences}},
                author       = {Daniel M. Ziegler and Nisan Stiennon and Jeffrey Wu and Tom B. Brown and Alec Radford and Dario Amodei and Paul F. Christiano and Geoffrey Irving},
                year         = 2019,
                eprint       = {arXiv:1909.08593}
            }"""
        ),
    }

    def __init__(
        self,
        args: VMPOConfig,
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

        # Define the collator if not provided
        if data_collator is None:
            data_collator = DataCollatorWithPadding(self.processing_class)

        # Handle stop token settings: update policy model's generation_config to use provided stop token
        if args.stop_token and args.stop_token_id:
            raise ValueError("You cannot set both `stop_token` and `stop_token_id`.")
        elif args.stop_token:
            if args.stop_token == "eos":
                self.policy_model.generation_config.eos_token_id = (
                    self.stop_token_id
                ) = processing_class.eos_token_id
            else:
                raise ValueError(
                    f"Unknown `stop_token` {args.stop_token}. Allowed values are: `'eos'` and `None` (no stop token)."
                )
        else:
            self.policy_model.generation_config.eos_token_id = self.stop_token_id = (
                args.stop_token_id
            )  # None or int

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
        args.mini_batch_size = exact_div(
            args.batch_size,
            args.num_mini_batches,
            "`batch_size` must be a multiple of `num_mini_batches`",
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size,
            args.num_mini_batches,
            "`local_batch_size` must be a multiple of `num_mini_batches`",
        )
        if args.whiten_rewards:
            assert (
                args.local_mini_batch_size >= 8
            ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
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
        eta_lr = self.args.learning_rate
        alpha_lr = self.args.learning_rate
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
        self.add_callback(
            PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK
        )
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
        self.is_deepspeed_enabled = (
            getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        )
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

        if self.is_deepspeed_enabled:
            self.reward_model = prepare_deepspeed(
                self.reward_model,
                args.per_device_train_batch_size,
                args.fp16,
                args.bf16,
            )

            if self.ref_model is None:
                if not self.is_peft_model:
                    raise ValueError(
                        "No reference model and model is not a Peft model."
                    )
            else:
                self.ref_model = prepare_deepspeed(
                    self.ref_model,
                    args.per_device_train_batch_size,
                    args.fp16,
                    args.bf16,
                )
        else:
            if self.ref_model is None:
                if not self.is_peft_model:
                    raise ValueError(
                        "No reference model and model is not a Peft model."
                    )
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

        if self.is_deepspeed_enabled:
            backup_deepspeed = self.deepspeed
            self.deepspeed = self.model

        super().save_model(output_dir, _internal_call)

        self.model = backup_model

        if self.is_deepspeed_enabled:
            self.deepspeed = backup_deepspeed

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_model
        reward_model = self.reward_model
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_kwargs = {
            "max_new_tokens": args.response_length,
            "temperature": (args.temperature + 1e-7),
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
        }
        generation_config = GenerationConfig(**generation_kwargs)
        generation_config.eos_token_id = processing_class.eos_token_id
        generation_config.pad_token_id = processing_class.pad_token_id
        assert generation_config.eos_token_id is not None
        assert generation_config.pad_token_id is not None
        assert generation_config.eos_token_id != generation_config.pad_token_id

        accelerator.print("===training policy===")
        stats_shape = (
            args.num_vmpo_epochs,
            args.num_mini_batches,
            args.gradient_accumulation_steps,
        )
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(
                    self.state.max_steps * args.logging_steps
                )
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(
                    self.state.max_steps * args.eval_steps
                )
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(
                    self.state.max_steps * args.save_steps
                )
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control
        )

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        for update in range(1, args.num_total_batches + 1):
            approxkl_stats.zero_()
            pg_loss_stats.zero_()
            vf_loss_stats.zero_()
            vf_clipfrac_stats.zero_()
            entropy_stats.zero_()
            l_eta_full_values: list[torch.Tensor] = []  # track temperature dual terms
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            with torch.no_grad():
                queries = data["input_ids"].to(device)
                context_length = queries.shape[1]
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                sequence_lengths = []
                values = []
                with unwrap_model_for_generation(
                    self.model,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                    generation_kwargs=generation_kwargs,  # Override model.generation_config with generation_kwargs to fix transformers#42762
                ) as unwrapped_model:
                    query_responses, logitss = batch_generation(
                        unwrapped_model.policy,
                        queries,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        generation_config,
                    )

                for i in range(
                    0, queries.shape[0], args.local_rollout_forward_batch_size
                ):
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[
                        i : i + args.local_rollout_forward_batch_size
                    ]
                    response = query_response[:, context_length:]
                    logits = logitss[i : i + args.local_rollout_forward_batch_size]
                    logprob = selective_log_softmax(logits, response)
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
                    sequence_length = (
                        first_true_indices(
                            postprocessed_response == processing_class.pad_token_id
                        )
                        - 1
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
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                values = torch.cat(values, 0)
                del (logprob, ref_logprob, full_value, value, score, unwrapped_model)
                empty_cache()
                gc.collect()

                # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
                # Completions not passing that filter will receive a lower score.
                contain_eos_token = (
                    postprocessed_responses == processing_class.eos_token_id
                ).any(dim=1)
                if self.args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty
                # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(
                    responses.shape[1], device=responses.device
                ).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(
                    ref_logprobs, padding_mask, INVALID_LOGPROB
                )
                sequence_lengths_p1 = sequence_lengths + 1
                padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
                values = torch.masked_fill(values, padding_mask_p1, 0)

                # 4. compute rewards
                # Formula used by http://joschu.net/blog/kl-approx.html for the k1 and k3 estimators
                logr = ref_logprobs - logprobs
                kl = -logr if args.kl_estimator == "k1" else (logr.exp() - 1) - logr
                # trust region KL is handled by α; no PPO-style shaping
                rewards = torch.zeros_like(kl)
                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                actual_end = torch.where(
                    sequence_lengths_p1 < rewards.size(1),
                    sequence_lengths_p1,
                    sequence_lengths,
                )
                rewards[actual_start, actual_end] += scores
                # logging masks/summaries
                valid_tok_mask = ~padding_mask
                valid_tok_count = valid_tok_mask.sum(1)
                entropy_seq = -(logprobs.masked_fill(padding_mask, 0)).sum(1)
                entropy_tok = entropy_seq.sum() / (valid_tok_count.sum() + 1e-8)
                reward_valid_mask = ~padding_mask_p1
                reward_seq = rewards.masked_fill(padding_mask_p1, 0).sum(1)
                reward_tok = reward_seq.sum() / (reward_valid_mask.sum() + 1e-8)

                # 5. whiten rewards
                if args.whiten_rewards:
                    rewards = masked_whiten(
                        rewards, mask=~padding_mask_p1, shift_mean=False
                    )
                    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

                # 6. compute advantages and returns
                lastgaelam = 0
                advantages_reversed = []
                gen_length = responses.shape[1]
                for t in reversed(range(gen_length)):
                    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
                raw_advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = raw_advantages + values
                advantages = raw_advantages  # keep raw scale for policy/dual alignment
                empty_cache()

            # Global E-step (compute ψ once per rollout and update η/α)
            eta = F.softplus(self.model.eta_raw) + args.eta_min
            alpha = F.softplus(self.model.alpha_raw) + args.alpha_min
            # sequence-level ψ: aggregate advantages per sequence before weighting
            mask_valid = ~padding_mask
            adv_seq = (raw_advantages.masked_fill(~mask_valid, 0)).sum(dim=1)
            adv_seq_eta = adv_seq
            psi_global = torch.zeros_like(raw_advantages)
            l_eta = torch.zeros((), device=device)
            num_seqs = adv_seq.numel()
            if num_seqs > 0:
                top_k = int(torch.ceil(torch.tensor(num_seqs * args.top_frac)).item())
                if top_k > 0:
                    threshold = adv_seq.topk(top_k).values.min()
                    mask_top_seq = adv_seq >= threshold
                    w_seq = torch.zeros_like(adv_seq)
                    max_adv_seq = adv_seq[mask_top_seq].max()
                    w_seq[mask_top_seq] = torch.exp(
                        (adv_seq[mask_top_seq] - max_adv_seq) / (eta + 1e-8)
                    )
                    w_sum = w_seq.sum()
                    psi_seq = torch.where(
                        mask_top_seq, w_seq / (w_sum + 1e-8), torch.zeros_like(w_seq)
                    )
                    psi_seq = psi_seq / psi_seq.sum().clamp_min(1e-8)
                    psi_global = psi_seq[:, None] * mask_valid
                    log_mean_exp = torch.logsumexp(
                        adv_seq_eta[mask_top_seq] / (eta + 1e-8), dim=0
                    ) - torch.log(
                        torch.tensor(float(mask_top_seq.sum()), device=device)
                    )
                    l_eta = eta * (args.eps_eta + log_mean_exp)
            self.eta_optimizer.zero_grad()
            if l_eta.requires_grad:
                l_eta.backward(retain_graph=True)
                self.eta_optimizer.step()
                l_eta_full_values.append(l_eta.detach())
            # detach ψ before policy/value updates to avoid in-place versioning issues
            psi_global = psi_global.detach()
            psi_state = psi_global.sum(dim=1)
            psi_state = psi_state / psi_state.sum().clamp_min(1e-8)
            # Do multiple epochs of V-MPO training
            kl_weighted_accum = torch.zeros((), device=device)
            kl_mb_count = 0
            for vmpo_epoch_idx in range(args.num_vmpo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(
                    0, args.local_batch_size, args.local_mini_batch_size
                ):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]

                    mb_query = queries[mini_batch_inds]
                    mb_response = responses[mini_batch_inds]
                    mb_query_response = torch.cat((mb_query, mb_response), 1)
                    psi_mb = psi_global[mini_batch_inds]
                    psi_state_mb = psi_state[mini_batch_inds]
                    mask_valid_mb = ~padding_mask[mini_batch_inds]

                    policy_outputs = forward(
                        model.policy,
                        mb_query_response,
                        processing_class.pad_token_id,
                    )
                    new_logits = policy_outputs.logits[:, context_length - 1 : -1]
                    new_logprobs = selective_log_softmax(new_logits, mb_response)

                    if psi_mb.sum() > 0:
                        policy_loss = -(
                            (psi_mb * new_logprobs).sum() / (psi_mb.sum() + 1e-8)
                        )
                    else:
                        policy_loss = torch.zeros((), device=device)

                    value_full, _, _ = get_reward(
                        self.accelerator.unwrap_model(model).value_model,
                        mb_query_response,
                        processing_class.pad_token_id,
                        context_length,
                    )
                    value_pred = value_full[:, context_length - 1 : -1].squeeze(-1)
                    value_mask = ~padding_mask_p1[mini_batch_inds]
                    with torch.no_grad():
                        value_old_mb = values[mini_batch_inds]
                        returns_mb = returns[mini_batch_inds]
                        delta = (returns_mb - value_old_mb).clamp(
                            -args.value_clip, args.value_clip
                        )
                        value_target = value_old_mb + delta
                    if value_mask.any():
                        value_loss = F.mse_loss(
                            value_pred[value_mask],
                            value_target[value_mask],
                        )
                    else:
                        value_loss = torch.zeros((), device=device)

                    entropy = torch.distributions.Categorical(
                        logits=new_logits
                    ).entropy()
                    entropy_seq = (entropy * mask_valid_mb).sum(dim=1) / (
                        mask_valid_mb.sum(dim=1) + 1e-8
                    )
                    entropy_mean = (psi_state_mb * entropy_seq).sum()
                    # VMPO needs ψ-weighted KL(old‖new): expectation under old rollout policy
                    kl_terms_mb = (
                        logprobs[mini_batch_inds] - new_logprobs
                    ) * mask_valid_mb
                    kl_state_mb = kl_terms_mb.sum(dim=1) / (
                        mask_valid_mb.sum(dim=1) + 1e-8
                    )
                    kl_weighted_mb = (psi_state_mb * kl_state_mb).sum()
                    alpha = F.softplus(self.model.alpha_raw) + args.alpha_min
                    beta_entropy = args.beta_entropy * math.exp(
                        -args.beta_entropy_decay * (update - 1)
                    )
                    total_loss = (
                        policy_loss
                        + alpha.detach() * kl_weighted_mb
                        + args.vf_coef * value_loss
                        - beta_entropy * entropy_mean
                    )
                    self.accelerator.backward(total_loss)
                    optimizer.step()
                    optimizer.zero_grad()
                    kl_weighted_accum += kl_weighted_mb.detach()
                    kl_mb_count += 1
                    self.alpha_optimizer.zero_grad()
                    l_alpha_mb = -alpha * (kl_weighted_mb.detach() - args.eps_alpha)
                    l_alpha_mb.backward()
                    self.alpha_optimizer.step()
                    pg_loss_stats[vmpo_epoch_idx, minibatch_idx] = policy_loss.detach()
                    vf_loss_stats[vmpo_epoch_idx, minibatch_idx] = value_loss.detach()
                    approxkl_stats[vmpo_epoch_idx, minibatch_idx] = (
                        kl_weighted_mb.detach()
                    )
                    entropy_stats[vmpo_epoch_idx, minibatch_idx] = entropy_mean.detach()
                    vf_clipfrac_stats[vmpo_epoch_idx, minibatch_idx] = torch.zeros(
                        (), device=device
                    )
                    minibatch_idx += 1
                    empty_cache()
            # removed delayed α update; trust region is enforced per minibatch
            with torch.no_grad():
                if kl_mb_count > 0:
                    kl_weighted_post = kl_weighted_accum / kl_mb_count
                else:
                    kl_weighted_post = torch.zeros((), device=device)
            # removed redundant post-epoch alpha update
            with torch.no_grad():
                mean_entropy = entropy_tok
                rlhf_reward = reward_seq.mean()
                eta_value = F.softplus(self.model.eta_raw) + args.eta_min
                alpha_value = F.softplus(self.model.alpha_raw) + args.alpha_min
                psi_ess = 1.0 / (psi_global**2).sum().clamp_min(1e-8)
                psi_max = psi_global.max()
                l_eta_mean = (
                    torch.stack(l_eta_full_values).mean()
                    if len(l_eta_full_values) > 0
                    else torch.zeros((), device=device)
                )
                eta_grad = self.model.eta_raw.grad
                eta_grad_norm = eta_grad.norm().item() if eta_grad is not None else 0.0
                metrics = {}
                # psi_state already computed above
                metrics["objective/kl_ref_weighted"] = (
                    self.accelerator.gather_for_metrics(kl_weighted_post).mean().item()
                )
                metrics["psi/ess"] = (
                    self.accelerator.gather_for_metrics(psi_ess).mean().item()
                )
                metrics["psi/max"] = (
                    self.accelerator.gather_for_metrics(psi_max).mean().item()
                )
                metrics["objective/entropy_seq"] = (
                    self.accelerator.gather_for_metrics(entropy_seq).mean().item()
                )
                metrics["objective/entropy_tok"] = (
                    self.accelerator.gather_for_metrics(entropy_tok).mean().item()
                )
                metrics["objective/entropy"] = (
                    self.accelerator.gather_for_metrics(mean_entropy).mean().item()
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
                metrics["policy/entropy_avg"] = (
                    self.accelerator.gather_for_metrics(entropy_stats).mean().item()
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
                metrics["val/num_eos_tokens"] = contain_eos_token.sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = (
                    self.state.episode / self.train_dataset_len
                )  # used by self.log
                self.state.global_step += 1
                self.log(metrics)

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(
                args, self.state, self.control
            )
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(
                    self.args, self.state, self.control
                )
            del (
                kl,
                mean_entropy,
                scores,
                metrics,
            )
            empty_cache()
            gc.collect()

            if (
                args.num_sample_generations > 0
                and (update - 1) % self.sample_generations_freq == 0
            ):
                self.generate_completions(sampling=True)
                empty_cache()
            del (
                query_responses,
                responses,
                postprocessed_responses,
                logprobs,
                ref_logprobs,
                values,
                sequence_lengths,
                contain_eos_token,
                sequence_lengths_p1,
                response_idxs,
                padding_mask,
                padding_mask_p1,
                rewards,
                actual_start,
                actual_end,
                advantages,
                returns,
            )
            empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control
        )
        if self.control.should_save:
            self._save_checkpoint(model, trial=None)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

    def generate_completions(self, sampling: bool = False):
        args = self.args
        processing_class = self.processing_class
        generation_kwargs = {
            "max_new_tokens": args.response_length,
            "temperature": (args.temperature + 1e-7),
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
        }
        generation_config = GenerationConfig(**generation_kwargs)
        generation_config.eos_token_id = processing_class.eos_token_id
        generation_config.pad_token_id = processing_class.pad_token_id
        assert generation_config.eos_token_id is not None
        assert generation_config.pad_token_id is not None
        assert generation_config.eos_token_id != generation_config.pad_token_id

        table = defaultdict(list)
        with unwrap_model_for_generation(
            self.model,
            self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation,
            generation_kwargs=generation_kwargs,  # Override model.generation_config with generation_kwargs to fix transformers#42762
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
                        generation_config,
                    )
                    response = query_response[:, context_length:]
                    postprocessed_response = response
                    if (
                        self.stop_token_id is not None
                    ):  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id, response
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
