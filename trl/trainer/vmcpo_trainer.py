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

import inspect
import math
import random
import textwrap
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


from dataclasses import dataclass, field
from typing import Any

import transformers
from packaging.version import Version
from transformers import TrainingArguments

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState, logging
from accelerate.utils import tqdm
from datasets import Dataset, IterableDataset
from torch import autocast
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
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


@dataclass
class DataCollatorForPreference(DataCollatorMixin):
    """
    Data collator used for preference data. Inputs are dynamically padded to the maximum length of a batch if they are
    not all of the same length.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.

    Examples:
    ```python
    >>> from trl import DataCollatorForPreference

    >>> collator = DataCollatorForPreference(pad_token_id=0)
    >>> examples = [
    ...     {"prompt_input_ids": [1, 2, 3], "chosen_input_ids": [4, 5], "rejected_input_ids": [6]},
    ...     {"prompt_input_ids": [7, 8], "chosen_input_ids": [9, 10], "rejected_input_ids": [11, 12, 13]},
    ... ]
    >>> collator(examples)
    {'prompt_input_ids': tensor([[1, 2, 3],
                                 [0, 7, 8]]),
     'prompt_attention_mask': tensor([[1, 1, 1],
                                      [0, 1, 1]]),
     'chosen_input_ids': tensor([[ 4,  5],
                                 [ 9, 10]]),
     'chosen_attention_mask': tensor([[1, 1],
                                      [1, 1]]),
     'rejected_input_ids': tensor([[ 6,  0,  0],
                                   [11, 12, 13]]),
     'rejected_attention_mask': tensor([[1, 0, 0],
                                        [1, 1, 1]])
    }
    ```
    """

    pad_token_id: int
    return_tensors: str = "pt"

    def torch_call(
        self, examples: list[list[int] | Any | dict[str, Any]]
    ) -> dict[str, Any]:
        # Convert to tensor
        prompt_input_ids = [
            torch.tensor(example["prompt_input_ids"]) for example in examples
        ]
        prompt_attention_mask = [
            torch.ones_like(input_ids) for input_ids in prompt_input_ids
        ]
        chosen_input_ids = [
            torch.tensor(example["chosen_input_ids"]) for example in examples
        ]
        chosen_attention_mask = [
            torch.ones_like(input_ids) for input_ids in chosen_input_ids
        ]
        rejected_input_ids = [
            torch.tensor(example["rejected_input_ids"]) for example in examples
        ]
        rejected_attention_mask = [
            torch.ones_like(input_ids) for input_ids in rejected_input_ids
        ]
        if "pixel_values" in examples[0]:
            pixel_values = [
                torch.tensor(example["pixel_values"]) for example in examples
            ]
        if "pixel_attention_mask" in examples[0]:
            pixel_attention_mask = [
                torch.tensor(example["pixel_attention_mask"]) for example in examples
            ]
        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            ref_chosen_logps = torch.tensor(
                [example["ref_chosen_logps"] for example in examples]
            )
            ref_rejected_logps = torch.tensor(
                [example["ref_rejected_logps"] for example in examples]
            )

        # Pad
        output = {}
        output["prompt_input_ids"] = pad(
            prompt_input_ids, padding_value=self.pad_token_id, padding_side="left"
        )
        output["prompt_attention_mask"] = pad(
            prompt_attention_mask, padding_value=0, padding_side="left"
        )
        output["chosen_input_ids"] = pad(
            chosen_input_ids, padding_value=self.pad_token_id
        )
        output["chosen_attention_mask"] = pad(chosen_attention_mask, padding_value=0)
        output["rejected_input_ids"] = pad(
            rejected_input_ids, padding_value=self.pad_token_id
        )
        output["rejected_attention_mask"] = pad(
            rejected_attention_mask, padding_value=0
        )
        if "pixel_values" in examples[0]:
            output["pixel_values"] = pad(pixel_values, padding_value=0.0)
        if "pixel_attention_mask" in examples[0]:
            output["pixel_attention_mask"] = pad(pixel_attention_mask, padding_value=0)
        if "image_sizes" in examples[0]:
            output["image_sizes"] = torch.tensor(
                [example["image_sizes"] for example in examples]
            )
        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            output["ref_chosen_logps"] = ref_chosen_logps
            output["ref_rejected_logps"] = ref_rejected_logps
        if "token_type_ids" in examples[0]:
            token_type_ids = [
                torch.tensor(example["token_type_ids"]) for example in examples
            ]
            output["token_type_ids"] = pad(
                token_type_ids, padding_value=0, padding_side="left"
            )

        return output


class VMCPOTrainer(BaseTrainer):
    """
    Trainer for Direct Preference Optimization (VMCPO) method.

    This class is a wrapper around the [`transformers.Trainer`] class and inherits all of its attributes and methods.

    Args:
        model (`str | PreTrainedModel`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keyword arguments in
              `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        ref_model ([`~transformers.PreTrainedModel`])
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation
            and loss. If no reference model is provided, the trainer will create a reference model with the same
            architecture as the model to be optimized.
        args ([`VMCPOConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        data_collator ([`~transformers.DataCollator`], *optional*):
            Function to use to form a batch from a list of elements of the processed `train_dataset` or `eval_dataset`.
            Will default to [`DataCollatorForPreference`].
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. VMCPO supports [preference](#preference) type and. The format of the samples can
            be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Dataset | IterableDataset]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.BaseImageProcessor`], [`~transformers.FeatureExtractionMixin`] or [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. If `None`, the processing class is loaded from the model's name
            with [`~transformers.AutoTokenizer.from_pretrained`].
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a [`EvalPrediction`] and return
            a dictionary string to metric values. *Note* When passing TrainingArgs with `batch_eval_metrics` set to
            `True`, your compute_metrics function must take a boolean `compute_result` argument. This will be triggered
            after the last eval batch to signal that the function needs to calculate and return the global summary
            statistics rather than accumulating the batch-level statistics.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        optimizer_cls_and_kwargs (`Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]`, *optional*):
            A tuple containing the optimizer class and keyword arguments to use. Overrides `optim` and `optim_args` in
            `args`. Incompatible with the `optimizers` argument.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by `compute_metrics`.

            Note that the labels (second parameter) will be `None` if the dataset does not have them.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "vmcpo"]
    _name = "VMCPO"
    _paper = {
        "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "id": "2305.18290",
        # docstyle-ignore
        "citation": textwrap.dedent(
            """\
            @inproceedings{rafailov2023direct,
                title        = {{Direct Preference Optimization: Your Language Model is Secretly a Reward Model}},
                author       = {Rafael Rafailov and Archit Sharma and Eric Mitchell and Christopher D. Manning and Stefano Ermon and Chelsea Finn},
                year         = 2023,
                booktitle    = {Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023},
                url          = {http://papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html},
                editor       = {Alice Oh and Tristan Naumann and Amir Globerson and Kate Saenko and Moritz Hardt and Sergey Levine},
            }"""
        ),
    }

    def __init__(
        self,
        model: str | nn.Module | PreTrainedModel,
        ref_model: PreTrainedModel | nn.Module | str | None = None,
        args: VMCPOConfig | None = None,
        data_collator: DataCollator | None = None,  # type: ignore
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: (
            Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None
        ) = None,
        processing_class: (
            PreTrainedTokenizerBase
            | BaseImageProcessor
            | FeatureExtractionMixin
            | ProcessorMixin
            | None
        ) = None,
        compute_metrics: Callable[[EvalLoopOutput], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[
            torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None
        ] = (None, None),
        optimizer_cls_and_kwargs: (
            tuple[type[torch.optim.Optimizer], dict[str, Any]] | None
        ) = None,
        preprocess_logits_for_metrics: (
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
        ) = None,
        peft_config: "PeftConfig | None" = None,
    ):
        # Args
        if args is None:
            model_name = (
                model if isinstance(model, str) else get_config_model_id(model.config)
            )
            model_name = model_name.split("/")[-1]
            args = VMCPOConfig(f"{model_name}-VMCPO")

        # IterableDataset requires dispatch_batches=False because Accelerate's dispatch mode may try to concatenate
        # batches from multiple processes, leading to mismatch errors.
        if isinstance(train_dataset, IterableDataset):
            if args.accelerator_config.dispatch_batches is True:
                logger.warning(
                    "You are using an `IterableDataset` for training with `dispatch_batches=True`. `dispatch_batches` "
                    "is forced to `False` when using an `IterableDataset`. To remove this warning, unset "
                    "`dispatch_batches` in `VMCPOConfig` or set it to `False`."
                )
            args.accelerator_config.dispatch_batches = False

        # Model and reference model
        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            # Distributed training requires device_map=None ("auto" fails)
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            model = create_model_from_path(model, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `VMCPOConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )
        model_id = get_config_model_id(model.config)
        if isinstance(ref_model, str):
            model_init_kwargs = args.ref_model_init_kwargs or {}
            # Distributed training requires device_map=None ("auto" fails)
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            ref_model = create_model_from_path(ref_model, **model_init_kwargs)
        else:
            if args.ref_model_init_kwargs is not None:
                logger.warning(
                    "You passed `ref_model_init_kwargs` to the `VMCPOConfig`, but your model is already instantiated. "
                    "The `ref_model_init_kwargs` will be ignored."
                )
        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, you can simply omit the `ref_model` argument and it will be created for you."
            )

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model_id)

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
            self._is_vlm = True
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
            self._is_vlm = False
        else:
            raise TypeError(
                "The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`"
            )

        # Get the pad token: if not provided, use the one from the processing class or the eos token
        # if the processing class does not have a pad token.
        pad_token = args.pad_token or tokenizer.pad_token or tokenizer.eos_token
        self.pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
        if self.pad_token_id is None:
            raise ValueError(
                f"The specified `pad_token` ('{pad_token}') is not found in the vocabulary of the given "
                f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `pad_token` exists "
                "in the vocabulary before using it as a padding token."
            )

        # PEFT configuration and model wrapping
        model = self._prepare_peft_model(model, ref_model, peft_config, args)

        if args.generate_during_eval and not (
            is_wandb_available() or is_comet_available() or is_mlflow_available()
        ):
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases, MLFlow or Comet to be installed."
                " Please install `wandb`, `mlflow` or `comet-ml` to resolve."
            )

        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.is_vision_model = (
            model.config.model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.keys()
        )
        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.model_adapter_name = args.model_adapter_name
        self.ref_adapter_name = args.ref_adapter_name
        self.reference_free = args.reference_free

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or args.precompute_ref_log_probs:
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        # Disable dropout in the model and reference model
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Liger kernel
        if args.use_liger_kernel:
            raise ValueError(
                "VMCPOTrainer no longer supports `use_liger_kernel=True`. VMPO training always follows the torch implementation."
            )
        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in VMCPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys are "prompt_input_ids", "chosen_input_ids", and
        # "rejected_input_ids". As a result, the trainer issues the warning: "Could not estimate the number of tokens
        # of the input, floating-point operations will not be computed." To suppress this warning, we set the
        # "estimate_tokens" key in the model's "warnings_issued" dictionary to True. This acts as a flag to indicate
        # that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Data collator
        if data_collator is None:
            data_collator = DataCollatorForPreference(pad_token_id=self.pad_token_id)

        self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.max_length = args.max_length
        self.truncation_mode = args.truncation_mode
        self.precompute_ref_log_probs = args.precompute_ref_log_probs
        self.use_logits_to_keep = args.use_logits_to_keep

        if args.padding_free:
            if model.config._attn_implementation != "flash_attention_2":
                logger.warning(
                    "Padding-free training is enabled, but the attention implementation is not set to "
                    "'flash_attention_2'. Padding-free training flattens batches into a single sequence, and "
                    "'flash_attention_2' is the only known attention mechanism that reliably supports this. Using "
                    "other implementations may lead to unexpected behavior. To ensure compatibility, set "
                    "`attn_implementation='flash_attention_2'` in the model configuration, or verify that your "
                    "attention mechanism can handle flattened sequences."
                )
            if args.per_device_train_batch_size == 1:
                logger.warning(
                    "You are using a per_device_train_batch_size of 1 with padding-free training. Using a batch size "
                    "of 1 annihilate the benefits of padding-free training. Please consider increasing the batch size "
                    "to at least 2."
                )
        self.padding_free = args.padding_free

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        self.beta = args.beta
        self.aux_loss_enabled = getattr(model.config, "output_router_logits", False)
        self.use_weighting = args.use_weighting
        self.aux_loss_coef = getattr(model.config, "router_aux_loss_coef", 0.0)
        if self.aux_loss_enabled and self.aux_loss_coef == 0.0:
            logger.warning(
                "You set `output_router_logits` to `True` in the model config, but `router_aux_loss_coef` is set to "
                "`0.0`, meaning the auxiliary loss will not be used. Either set `router_aux_loss_coef` to a value "
                "greater than `0.0`, or set `output_router_logits` to `False` if you don't want to use the auxiliary "
                "loss.",
            )

        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.dataset_num_proc = args.dataset_num_proc

        # Dataset preparation
        train_dataset = self._prepare_dataset(
            train_dataset, processing_class, args, "train"
        )
        if eval_dataset is not None:
            if isinstance(eval_dataset, dict):
                eval_dataset = {
                    key: self._prepare_dataset(dataset, processing_class, args, key)
                    for key, dataset in eval_dataset.items()
                }
            else:
                eval_dataset = self._prepare_dataset(
                    eval_dataset, processing_class, args, "eval"
                )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        device = self.accelerator.device
        self.eta_raw = nn.Parameter(
            inv_softplus(torch.tensor(args.eta_init, device=device))
        )
        self.alpha_raw = nn.Parameter(
            inv_softplus(torch.tensor(args.alpha_init, device=device))
        )
        self.eta_optimizer = torch.optim.Adam(
            [self.eta_raw], lr=args.learning_rate, weight_decay=0.0
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.alpha_raw], lr=args.learning_rate, weight_decay=0.0
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        if self.is_deepspeed_enabled:
            if (
                self.accelerator.state.deepspeed_plugin.zero_stage == 3
                and self.precompute_ref_log_probs
            ):
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`."
                )

        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs):
                raise ValueError(
                    "No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`"
                )
            if args.sync_ref_model:
                raise ValueError(
                    "You currently cannot use `ref_model=None` with TR-VMCPO method. Please provide `ref_model`."
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )

        if args.sync_ref_model:
            if self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with TR-VMCPO method. Please set `precompute_ref_log_probs=False`."
                )

            self.add_callback(
                SyncRefModelCallback(
                    ref_model=self.ref_model, accelerator=self.accelerator
                )
            )

    def _prepare_peft_model(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        peft_config: Any,
        args: VMCPOConfig,
    ) -> PreTrainedModel:
        """Prepares a model for PEFT training."""
        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            if isinstance(model, PeftModel):
                raise ValueError(
                    "You passed a `PeftModel` instance together with a `peft_config` to the trainer. Please first "
                    "merge and unload the existing adapter, save the resulting base model, and then pass that base "
                    "model along with the new `peft_config` to the trainer."
                )

            if ref_model is not None and not args.force_use_ref_model:
                raise ValueError(
                    "You passed both a ref_model and a peft_config. For training PEFT adapters with VMCPO there is no need to pass a reference"
                    " model. Please pass `ref_model=None` in case you want to train PEFT adapters, or pass a ref_model with `force_use_ref_model=True` in VMCPOTrainer's init."
                    " if you want to use a different ref_model."
                )

            if getattr(model, "is_loaded_in_8bit", False) or getattr(
                model, "is_loaded_in_4bit", False
            ):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {
                    "use_gradient_checkpointing": args.gradient_checkpointing
                }

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = (
                        args.gradient_checkpointing_kwargs
                    )

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)

            else:
                model = self._prepare_gradient_checkpointing(model, args)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        else:
            model = self._prepare_gradient_checkpointing(model, args)

        return model

    def _prepare_gradient_checkpointing(
        self, model: PreTrainedModel, args: VMCPOConfig
    ):
        """Prepare the gradienting checkpointing for the model."""
        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        if args.gradient_checkpointing:
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad
                )

        return model

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: (
            PreTrainedTokenizerBase
            | BaseImageProcessor
            | FeatureExtractionMixin
            | ProcessorMixin
        ),
        args: VMCPOConfig,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(
            dataset, Dataset
        ):  # IterableDataset does not support num_proc nor writer_batch_size
            map_kwargs["num_proc"] = args.dataset_num_proc
            map_kwargs["writer_batch_size"] = 10

        with PartialState().main_process_first():
            # Extract prompt if needed
            if isinstance(
                dataset, Dataset
            ):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Extracting prompt in {dataset_name} dataset"
            dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

            is_chat = is_conversational(next(iter(dataset)))

            # Apply the chat template if needed
            if isinstance(
                dataset, Dataset
            ):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
            dataset = dataset.map(
                maybe_apply_chat_template,
                fn_kwargs={"tokenizer": processing_class, "tools": args.tools},
                **map_kwargs,
            )

            # Tokenize the dataset
            if isinstance(
                dataset, Dataset
            ):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

            dataset = dataset.map(
                self.tokenize_row if not self.is_vision_model else self.process_row,
                remove_columns=["chosen", "rejected"],
                fn_kwargs={
                    "processing_class": processing_class,
                    "max_prompt_length": args.max_prompt_length,
                    "max_completion_length": args.max_completion_length,
                    # for enc-dec, we add the special tokens ([bos_token] + prompt + [eos_token]; completion + [eos_token])
                    "add_special_tokens": False,
                    "is_chat": is_chat,
                },
                **map_kwargs,
            )

        return dataset

    @staticmethod
    def tokenize_row(
        features: dict[str, str],
        processing_class: PreTrainedTokenizerBase,
        max_prompt_length: int | None = None,
        max_completion_length: int | None = None,
        add_special_tokens: bool = True,
        is_chat: bool = False,
    ) -> dict[str, list[int]]:
        """
        Tokenize a row of the dataset.

        Args:
            features (`dict[str, str]`):
                Row of the dataset, should contain the keys `"prompt"`, `"chosen"`, and `"rejected"`.
            processing_class ([`~transformers.PreTrainedTokenizerBase`]):
                Processing class used to process the data.
            max_prompt_length (`int` or `None`):
                Maximum length of the prompt sequence. If `None`, the prompt sequence is not truncated.
            max_completion_length (`int` or `None`):
                Maximum length of the completion sequences. If `None`, the completion sequences are not truncated.
            add_special_tokens (`bool`):
                Whether to add special tokens to the sequences. Typically used for encoder-decoder models. If `True`,
                the prompt sequence will have a bos token prepended and an eos token appended. In any case, the
                completion sequences will have an eos token appended.
            is_chat (`bool`):
                Whether the data is conversational. If `True`, the completion sequences will not have an eos token
                appended.

        Returns:
            `dict[str, list[int]]`:
                Tokenized sequences with the keys `"prompt_input_ids"`, `"chosen_input_ids"`, and
                `"rejected_input_ids".

        Example:
        ```python
        >>> from transformers import GPT2Tokenizer

        >>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        >>> features = {"prompt": "The sky is", "chosen": " blue", "rejected": " green"}
        >>> VMCPOTrainer.tokenize_row(
        ...     features, tokenizer, max_prompt_length=3, max_completion_length=3, add_special_tokens=False
        ... )
        {'prompt_input_ids': [464, 6766, 318], 'chosen_input_ids': [4171, 50256], 'rejected_input_ids': [4077, 50256]}
        ```
        """
        tokenizer = processing_class  # the processing class is a tokenizer
        prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)[
            "input_ids"
        ]
        chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)[
            "input_ids"
        ]
        rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)[
            "input_ids"
        ]

        # Add special tokens (typically for encoder-decoder models)
        if add_special_tokens:
            if tokenizer.bos_token_id is not None:
                prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
            if tokenizer.eos_token_id is not None:
                prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
        # For conversational data, the chat template already includes proper EOS tokens
        if not is_chat:
            chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
            rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

        # Truncate prompt and completion sequences
        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            chosen_input_ids = chosen_input_ids[:max_completion_length]
            rejected_input_ids = rejected_input_ids[:max_completion_length]

        return {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }

    @staticmethod
    def process_row(
        features: dict[str, str],
        processing_class: PreTrainedTokenizerBase,
        max_prompt_length: int | None = None,
        max_completion_length: int | None = None,
        add_special_tokens: bool = True,
        is_chat: bool = False,
    ) -> dict[str, list[int]]:
        """
        Same as `tokenize_row` but for vision models. Please refer to `tokenize_row` for more information.
        """
        processor, tokenizer = (
            processing_class,
            processing_class.tokenizer,
        )  # the processing class is a processor
        processed_features = processor(
            images=features["images"], text=features["prompt"], add_special_tokens=False
        )

        prompt_input_ids = processed_features["input_ids"][0]
        pixel_values = processed_features["pixel_values"][0]
        chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)[
            "input_ids"
        ]
        rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)[
            "input_ids"
        ]

        # Add special tokens (typically for encoder-decoder models)
        if add_special_tokens:
            if tokenizer.bos_token_id is not None:
                prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
            if tokenizer.eos_token_id is not None:
                prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
        if not is_chat:
            chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
            rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

        # Truncate prompt and completion sequences
        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            chosen_input_ids = chosen_input_ids[:max_completion_length]
            rejected_input_ids = rejected_input_ids[:max_completion_length]

        output = {
            "prompt_input_ids": prompt_input_ids,
            "pixel_values": pixel_values,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }

        if "pixel_attention_mask" in processed_features:
            output["pixel_attention_mask"] = processed_features["pixel_attention_mask"][
                0
            ]
        if "image_sizes" in processed_features:
            output["image_sizes"] = processed_features["image_sizes"][0]
        if "token_type_ids" in processed_features:
            output["token_type_ids"] = processed_features["token_type_ids"][0]

        return output

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In VMCPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by `DataCollatorForPreference`, hence the override.
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt_input_ids",
                "chosen_input_ids",
                "rejected_input_ids",
                "image_sizes",
                "token_type_ids",
                "ref_chosen_logps",
                "ref_rejected_logps",
            ]

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            batch_size = (
                self.args.precompute_ref_batch_size
                or self.args.per_device_train_batch_size
            )
            dataloader_params = {
                "batch_size": batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(
                DataLoader(self.train_dataset, **dataloader_params)
            )

            ref_chosen_logps = []
            ref_rejected_logps = []
            for padded_batch in tqdm(
                iterable=data_loader, desc="Train dataset reference log probs"
            ):
                ref_chosen_logp, ref_rejected_logp = self.compute_ref_log_probs(
                    padded_batch
                )
                ref_chosen_logp, ref_rejected_logp = (
                    self.accelerator.gather_for_metrics(
                        (ref_chosen_logp, ref_rejected_logp)
                    )
                )
                ref_chosen_logps.append(ref_chosen_logp.cpu())
                ref_rejected_logps.append(ref_rejected_logp.cpu())

                # Unnecessary cache clearing to avoid OOM
                empty_cache()
                self.accelerator.free_memory()

            all_ref_chosen_logps = torch.cat(ref_chosen_logps).float().numpy()
            all_ref_rejected_logps = torch.cat(ref_rejected_logps).float().numpy()

            self.train_dataset = self.train_dataset.add_column(
                name="ref_chosen_logps", column=all_ref_chosen_logps
            )
            self.train_dataset = self.train_dataset.add_column(
                name="ref_rejected_logps", column=all_ref_rejected_logps
            )

            self._precomputed_train_ref_log_probs = True

        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_eval_dataloader to precompute `ref_log_probs`.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
            batch_size = (
                self.args.precompute_ref_batch_size
                or self.args.per_device_eval_batch_size
            )
            dataloader_params = {
                "batch_size": batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(
                DataLoader(eval_dataset, **dataloader_params)
            )

            ref_chosen_logps = []
            ref_rejected_logps = []
            for padded_batch in tqdm(
                iterable=data_loader, desc="Eval dataset reference log probs"
            ):
                ref_chosen_logp, ref_rejected_logp = self.compute_ref_log_probs(
                    padded_batch
                )
                ref_chosen_logp, ref_rejected_logp = (
                    self.accelerator.gather_for_metrics(
                        (ref_chosen_logp, ref_rejected_logp)
                    )
                )
                ref_chosen_logps.append(ref_chosen_logp.cpu())
                ref_rejected_logps.append(ref_rejected_logp.cpu())

            all_ref_chosen_logps = torch.cat(ref_chosen_logps).float().numpy()
            all_ref_rejected_logps = torch.cat(ref_rejected_logps).float().numpy()

            eval_dataset = eval_dataset.add_column(
                name="ref_chosen_logps", column=all_ref_chosen_logps
            )
            eval_dataset = eval_dataset.add_column(
                name="ref_rejected_logps", column=all_ref_rejected_logps
            )

            # Save calculated ref_chosen_logps and ref_rejected_logps to the eval_dataset for subsequent runs
            if self.eval_dataset is not None:
                self.eval_dataset = eval_dataset
            self._precomputed_eval_ref_log_probs = True

        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with (
            self.accelerator.unwrap_model(self.model).disable_adapter()
            if self.is_peft_model and not self.ref_adapter_name
            else nullcontext()
        ):
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")

    def compute_ref_log_probs(
        self, batch: dict[str, torch.LongTensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes log probabilities of the reference model for a single padded batch of a VMCPO specific dataset."""
        compte_ref_context_manager = (
            autocast(self.accelerator.device.type)
            if self._peft_has_been_casted_to_bf16
            else nullcontext()
        )
        with torch.no_grad(), compte_ref_context_manager:
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_model_output = self.concatenated_forward(
                        self.model, batch, is_ref_model=True
                    )
            else:
                ref_model_output = self.concatenated_forward(
                    self.ref_model, batch, is_ref_model=True
                )
        return ref_model_output["chosen_logps"], ref_model_output["rejected_logps"]

    @staticmethod
    def concatenated_inputs(
        batch: dict[str, list | torch.LongTensor], padding_value: int
    ) -> dict[str, torch.LongTensor]:
        """
        Concatenate the `chosen` and `rejected` inputs from the batch into a single tensor for both the prompt and
        completion sequences.

        Args:
            batch (`dict[str, list | torch.LongTensor]`):
                A batch of input data. The batch must contain the following keys:

                - `"prompt_input_ids"`: Tensor of shape `(batch_size, prompt_length)` representing the prompt input
                  IDs.
                - `"chosen_input_ids"`: Tensor of shape `(batch_size, chosen_length)` representing the chosen
                  completion input IDs.
                - `"rejected_input_ids"`: Tensor of shape `(batch_size, rejected_length)` representing the rejected
                  completion input IDs.
                - `"prompt_pixel_values"` (optional): Tensor for pixel values, if available.
                - `"prompt_pixel_attention_mask"` (optional): Tensor for pixel attention masks, if available.

            padding_value (`int`):
                The padding value to use for the concatenated completion sequences (`chosen_input_ids` and
                `rejected_input_ids`).

        Returns:
            `dict[str, torch.LongTensor]`: A dictionary containing:

                - `"prompt_input_ids"`: Concatenated prompt input IDs of shape `(2 * batch_size, prompt_length)`.
                - `"completion_input_ids"`: Concatenated chosen and rejected completion input IDs of shape `(2 *
                  batch_size, max_completion_length)`.
                - `"prompt_attention_mask"`: Concatenated prompt attention masks of shape `(2 * batch_size,
                  prompt_length)`.
                - `"completion_attention_mask"`: Concatenated chosen and rejected attention masks of shape `(2 *
                  batch_size, max_completion_length)`.
                - `"pixel_values"` (optional): Concatenated pixel values if `"prompt_pixel_values"` are present.
                - `"pixel_attention_mask"` (optional): Concatenated pixel attention masks if
                  `"prompt_pixel_attention_mask"` are present.

        Notes:
            The completion input IDs and attention masks are padded to the maximum completion length of the chosen or
            rejected sequences.
        """
        output = {}

        # For the prompt, the input_ids are the same for both the chosen and rejected responses
        output["prompt_input_ids"] = torch.cat(
            [batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0
        )
        output["prompt_attention_mask"] = torch.cat(
            [batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0
        )
        if "pixel_values" in batch:
            output["pixel_values"] = torch.cat(
                [batch["pixel_values"], batch["pixel_values"]], dim=0
            )

        if "pixel_attention_mask" in batch:
            output["pixel_attention_mask"] = torch.cat(
                [batch["pixel_attention_mask"], batch["pixel_attention_mask"]], dim=0
            )
        if "image_sizes" in batch:
            output["image_sizes"] = torch.cat(
                [batch["image_sizes"], batch["image_sizes"]], dim=0
            )
        if "token_type_ids" in batch:
            output["token_type_ids"] = torch.cat(
                (batch["token_type_ids"], batch["token_type_ids"])
            )

        # Concatenate the chosen and rejected completions
        max_completion_length = max(
            batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1]
        )
        output["completion_input_ids"] = torch.cat(
            (
                pad_to_length(
                    batch["chosen_input_ids"],
                    max_completion_length,
                    pad_value=padding_value,
                ),
                pad_to_length(
                    batch["rejected_input_ids"],
                    max_completion_length,
                    pad_value=padding_value,
                ),
            ),
        )
        output["completion_attention_mask"] = torch.cat(
            (
                pad_to_length(
                    batch["chosen_attention_mask"], max_completion_length, pad_value=0
                ),
                pad_to_length(
                    batch["rejected_attention_mask"], max_completion_length, pad_value=0
                ),
            ),
        )

        return output

                ref_decoder_outputs = unwrapped_ref_model.get_decoder()(
                    input_ids=decoder_input_ids,
                    attention_mask=concatenated_batch["completion_attention_mask"],
                    encoder_hidden_states=ref_encoder_outputs.last_hidden_state,
                    encoder_attention_mask=concatenated_batch["prompt_attention_mask"],
                    use_cache=False,
                )
                ref_hidden_states = ref_decoder_outputs.last_hidden_state
            elif not self.reference_free:
                with self.null_ref_context():
                    ref_encoder_outputs = unwrapped_model.get_encoder()(
                        concatenated_batch["prompt_input_ids"],
                        attention_mask=concatenated_batch["prompt_attention_mask"],
                        return_dict=True,
                    )
                    ref_decoder_outputs = unwrapped_model.get_decoder()(
                        input_ids=decoder_input_ids,
                        attention_mask=concatenated_batch["completion_attention_mask"],
                        encoder_hidden_states=ref_encoder_outputs.last_hidden_state,
                        encoder_attention_mask=concatenated_batch[
                            "prompt_attention_mask"
                        ],
                        use_cache=False,
                    )
                    ref_hidden_states = ref_decoder_outputs.last_hidden_state

            labels = concatenated_batch["completion_input_ids"]
            loss_mask = completion_attention_mask.bool()
        else:
            # For decoder-only models
            input_ids = torch.cat(
                (
                    concatenated_batch["prompt_input_ids"],
                    concatenated_batch["completion_input_ids"],
                ),
                dim=1,
            )
            attention_mask = torch.cat(
                (
                    concatenated_batch["prompt_attention_mask"],
                    concatenated_batch["completion_attention_mask"],
                ),
                dim=1,
            )
            # Mask the prompt but not the completion for the loss
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )

            # Flush and truncate
            if self.max_length is not None and self.max_length < attention_mask.size(1):
                if self.truncation_mode == "keep_start":
                    # Flush left to reduce the memory usage
                    # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
                    #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
                    attention_mask, input_ids, loss_mask = flush_left(
                        attention_mask, input_ids, loss_mask
                    )
                    attention_mask = attention_mask[:, : self.max_length]
                    input_ids = input_ids[:, : self.max_length]
                    loss_mask = loss_mask[:, : self.max_length]
                elif self.truncation_mode == "keep_end":
                    # Flush right before truncating left, then flush left
                    # [[0, 0, x, x, x, x],  ->  [[0, 0, x, x],
                    #  [0, x, x, x, 0, 0]]       [0, x, x, x]]
                    attention_mask, input_ids, loss_mask = flush_right(
                        attention_mask, input_ids, loss_mask
                    )
                    input_ids = input_ids[:, -self.max_length :]
                    attention_mask = attention_mask[:, -self.max_length :]
                    loss_mask = loss_mask[:, -self.max_length :]
                    attention_mask, input_ids, loss_mask = flush_left(
                        attention_mask, input_ids, loss_mask
                    )
                else:
                    raise ValueError(
                        f"Unknown truncation mode: '{self.truncation_mode}'. Should be one of ['keep_end', "
                        "'keep_start']."
                    )
            else:
                # Flush left to reduce the memory usage
                # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
                #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
                attention_mask, input_ids, loss_mask = flush_left(
                    attention_mask, input_ids, loss_mask
                )

            # Add logits_to_keep optimization
            if self.use_logits_to_keep:
                first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
                logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1
                model_kwargs["logits_to_keep"] = logits_to_keep

            model_kwargs["output_hidden_states"] = True

            # Add padding-free training support
            if self.padding_free:
                input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
                loss_mask = loss_mask[attention_mask.bool()].unsqueeze(0)
                position_ids = (
                    attention_mask.cumsum(1)[attention_mask.bool()].unsqueeze(0) - 1
                )
                model_kwargs["position_ids"] = position_ids
            else:
                model_kwargs["attention_mask"] = attention_mask

            # Get the base model outputs (before LM head)
            if (
                hasattr(unwrapped_model, "get_decoder")
                and unwrapped_model.get_decoder() is not None
            ):
                base_model = unwrapped_model.get_decoder()
            else:
                base_attr = getattr(
                    unwrapped_model,
                    "base_model_prefix",
                    self.args.base_model_attribute_name,
                )
                base_model = getattr(unwrapped_model, base_attr, unwrapped_model)

            outputs = base_model(
                input_ids,
                use_cache=False,
                **model_kwargs,
            )
            hidden_states = outputs.last_hidden_state[:, :-1]

            # Get reference hidden states if needed
            ref_hidden_states = None
            if not self.reference_free and self.ref_model is not None:
                unwrapped_ref_model = self.accelerator.unwrap_model(self.ref_model)
                if (
                    hasattr(unwrapped_ref_model, "get_decoder")
                    and unwrapped_ref_model.get_decoder() is not None
                ):
                    ref_base_model = unwrapped_ref_model.get_decoder()
                else:
                    ref_attr = getattr(
                        unwrapped_ref_model,
                        "base_model_prefix",
                        self.args.base_model_attribute_name,
                    )
                    ref_base_model = getattr(
                        unwrapped_ref_model, ref_attr, unwrapped_ref_model
                    )

                ref_outputs = ref_base_model(
                    input_ids,
                    use_cache=False,
                    **model_kwargs,
                )
                ref_hidden_states = ref_outputs.last_hidden_state[:, :-1]
            elif not self.reference_free:
                if (
                    hasattr(unwrapped_model, "get_decoder")
                    and unwrapped_model.get_decoder() is not None
                ):
                    ref_base_model = unwrapped_model.get_decoder()
                else:
                    ref_attr = getattr(
                        unwrapped_model,
                        "base_model_prefix",
                        self.args.base_model_attribute_name,
                    )
                    ref_base_model = getattr(unwrapped_model, ref_attr, unwrapped_model)
                with self.null_ref_context():
                    ref_outputs = ref_base_model(
                        input_ids,
                        use_cache=False,
                        **model_kwargs,
                    )
                    ref_hidden_states = ref_outputs.last_hidden_state[:, :-1]

            masked_input_ids = torch.where(
                loss_mask != 0, input_ids, self.label_pad_token_id
            )
            labels = masked_input_ids[:, 1:]  # Shift right for casual LM

    def e_step_dual_update(
        self, advantages: torch.Tensor, update: bool = True
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Compute ψ weights via the VMPO E-step using offline advantages."""
        if advantages.numel() == 0:
            return advantages.new_zeros(advantages.shape), []

        psi = torch.zeros_like(advantages)
        l_eta_values: list[torch.Tensor] = []
        steps = self.args.eta_inner_steps if update else 1
        num_items = advantages.numel()

        for _ in range(steps):
            eta = F.softplus(self.eta_raw) + self.args.eta_min
            max_adv = advantages.max()
            weights = torch.exp((advantages - max_adv) / (eta + 1e-8))
            psi = weights / weights.sum().clamp_min(1e-8)

            if not update:
                break

            log_mean_exp = torch.logsumexp(advantages / (eta + 1e-8), dim=0) - math.log(
                max(num_items, 1)
            )
            l_eta = eta * (self.args.eps_eta + log_mean_exp)
            self.eta_optimizer.zero_grad()
            l_eta.backward()
            self.eta_optimizer.step()
            l_eta_values.append(l_eta.detach())

        return psi, l_eta_values

    def _m_step(
        self, logps: torch.Tensor, ref_logps: torch.Tensor, psi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the VMPO M-step loss, returning total, policy, and KL components."""
        psi_sum = psi.sum().clamp_min(1e-8)
        if psi_sum.item() == 0:
            policy_loss = torch.zeros((), device=logps.device, dtype=logps.dtype)
        else:
            policy_loss = -((psi * logps).sum() / psi_sum)

        kl_terms = ref_logps - logps
        kl_weighted = (psi * kl_terms).sum()
        alpha = F.softplus(self.alpha_raw) + self.args.alpha_min
        total_loss = policy_loss + alpha.detach() * kl_weighted
        return total_loss, policy_loss.detach(), kl_weighted.detach()

    def alpha_dual_update(
        self, kl_weighted: torch.Tensor, update: bool = True
    ) -> torch.Tensor:
        """Update the α dual variable to track the KL trust-region target."""
        if not update:
            return torch.zeros((), device=self.alpha_raw.device)

        self.alpha_optimizer.zero_grad()
        alpha = F.softplus(self.alpha_raw) + self.args.alpha_min
        l_alpha = -alpha * (kl_weighted - self.args.eps_alpha)
        l_alpha.backward()
        self.alpha_optimizer.step()
        return l_alpha.detach()

    def concatenated_forward(
        self,
        model: nn.Module,
        batch: dict[str, list | torch.LongTensor],
        is_ref_model: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Runs the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.

        Args:
            model:
                Model to run the forward pass on.
            batch:
                Batch of input data.
            is_ref_model:
                Whether this method is being called for the reference model. If `True`, length desensitization is not
                applied.
        """
        num_examples = batch["prompt_input_ids"].shape[0]

        concatenated_batch = self.concatenated_inputs(
            batch, padding_value=self.pad_token_id
        )

        model_kwargs = {"use_cache": False}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        # Add the pixel values and attention masks for vision models
        if "pixel_values" in concatenated_batch:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
        if "pixel_attention_mask" in concatenated_batch:
            model_kwargs["pixel_attention_mask"] = concatenated_batch[
                "pixel_attention_mask"
            ]
        if "image_sizes" in concatenated_batch:
            model_kwargs["image_sizes"] = concatenated_batch["image_sizes"]

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]
        if self.is_encoder_decoder:
            labels = completion_input_ids
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,  # we need the labels for the logits to be returned
                **model_kwargs,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
        else:
            # Concatenate the prompt and completion inputs
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat(
                (prompt_attention_mask, completion_attention_mask), dim=1
            )
            if "token_type_ids" in concatenated_batch:
                prompt_token_type_ids = concatenated_batch["token_type_ids"]
                token_type_ids = pad_to_length(
                    prompt_token_type_ids, input_ids.shape[1], 0
                )
            # Mask the prompt but not the completion for the loss
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )

            # Flush and truncate
            if self.max_length is not None and self.max_length < attention_mask.size(1):
                if self.truncation_mode == "keep_start":
                    # Flush left to reduce the memory usage
                    # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
                    #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
                    if "token_type_ids" in concatenated_batch:
                        attention_mask, input_ids, loss_mask, token_type_ids = (
                            flush_left(
                                attention_mask, input_ids, loss_mask, token_type_ids
                            )
                        )
                    else:
                        attention_mask, input_ids, loss_mask = flush_left(
                            attention_mask, input_ids, loss_mask
                        )
                    attention_mask = attention_mask[:, : self.max_length]
                    input_ids = input_ids[:, : self.max_length]
                    loss_mask = loss_mask[:, : self.max_length]
                elif self.truncation_mode == "keep_end":
                    # Flush right before truncating left, then flush left
                    # [[0, 0, x, x, x, x],  ->  [[0, 0, x, x],
                    #  [0, x, x, x, 0, 0]]       [0, x, x, x]]
                    if "token_type_ids" in concatenated_batch:
                        attention_mask, input_ids, loss_mask, token_type_ids = (
                            flush_left(
                                attention_mask, input_ids, loss_mask, token_type_ids
                            )
                        )
                        token_type_ids = token_type_ids[:, -self.max_length :]
                    else:
                        attention_mask, input_ids, loss_mask = flush_right(
                            attention_mask, input_ids, loss_mask
                        )
                    input_ids = input_ids[:, -self.max_length :]
                    attention_mask = attention_mask[:, -self.max_length :]
                    loss_mask = loss_mask[:, -self.max_length :]
                    if "token_type_ids" in concatenated_batch:
                        attention_mask, input_ids, loss_mask, token_type_ids = (
                            flush_left(
                                attention_mask, input_ids, loss_mask, token_type_ids
                            )
                        )
                    else:
                        attention_mask, input_ids, loss_mask = flush_left(
                            attention_mask, input_ids, loss_mask
                        )
                else:
                    raise ValueError(
                        f"Unknown truncation mode: '{self.truncation_mode}'. Should be one of ['keep_end', "
                        "'keep_start']."
                    )
            else:
                # Flush left to reduce the memory usage
                # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
                #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
                if "token_type_ids" in concatenated_batch:
                    attention_mask, input_ids, loss_mask, token_type_ids = flush_left(
                        attention_mask, input_ids, loss_mask, token_type_ids
                    )
                else:
                    attention_mask, input_ids, loss_mask = flush_left(
                        attention_mask, input_ids, loss_mask
                    )

            if "token_type_ids" in concatenated_batch:
                model_kwargs["token_type_ids"] = token_type_ids

            if self.use_logits_to_keep:
                # Compute logits_to_keep based on loss_mask pattern:
                # [[0, 0, 0, x, x, x, x],
                #  [0, 0, 0, x, x, x, 0]]
                #         ^ start computing logits from here ([:, -(7-3+1):])
                first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
                logits_to_keep = (
                    loss_mask.shape[1] - first_compute_index
                ).item() + 1  # +1 for the first label
                model_kwargs["logits_to_keep"] = logits_to_keep

            model_kwargs["output_hidden_states"] = True

            if self.padding_free:
                # Flatten the input_ids, position_ids, and loss_mask
                # input_ids = [[a, b, c, 0], ->     input_ids = [[a, b, c, d, e, f, g]]
                #              [d, e, f, g]]     position_ids = [[0, 1, 2, 0, 1, 2, 3]]
                input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
                loss_mask = loss_mask[attention_mask.bool()].unsqueeze(0)
                position_ids = (
                    attention_mask.cumsum(1)[attention_mask.bool()].unsqueeze(0) - 1
                )
                model_kwargs["position_ids"] = position_ids
            else:
                model_kwargs["attention_mask"] = attention_mask

            outputs = model(input_ids, **model_kwargs)
            logits = outputs.logits

            # Offset the logits by one to align with the labels
            labels = torch.roll(input_ids, shifts=-1, dims=1)
            loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

            if self.use_logits_to_keep:
                # Align labels with logits
                # logits:    -,  -, [x2, x3, x4, x5, x6]
                #                     ^ --------- ^       after logits[:, :-1, :]
                # labels:   [y0, y1, y2, y3, y4, y5, y6]
                #                         ^ --------- ^   with logits_to_keep=4, [:, -4:]
                # loss_mask: [0,  0,  0,  1,  1,  1,  1]
                labels = labels[:, -logits_to_keep:]
                loss_mask = loss_mask[:, -logits_to_keep:]

        if logits.shape[:2] != labels.shape[:2]:
            # for LLaVA, the returned logits include the image tokens (placed before the text tokens)
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        # Compute the log probabilities of the labels
        labels[~loss_mask] = (
            0  # dummy token; we'll ignore the losses on these tokens later
        )
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        if self.padding_free:
            # Unflatten the per_token_logps (shape: [1, sum_seq_len] -> [batch_size, seq_len])
            batch_size, seq_len = attention_mask.shape
            per_token_logps_ = torch.zeros(
                batch_size,
                seq_len,
                device=outputs.logits.device,
                dtype=outputs.logits.dtype,
            )
            per_token_logps_[attention_mask.bool()] = per_token_logps
            per_token_logps = per_token_logps_

        all_logps = per_token_logps[:, 1:].sum(-1)

        output = {}

        if self.use_weighting:
            with torch.no_grad():
                # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(
                    2 * logprobs, dim=-1
                )  # same as sum(probs**2) in log space
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(
                    -1
                ) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(
                    torch.exp(chosen_weights + rejected_weights), max=1
                )

        if self.args.rpo_alpha is not None:
            # Only use the chosen logits for the RPO loss or SFT loss
            chosen_logits = (
                logits[:num_examples, :-1]
                if not self.is_encoder_decoder
                else logits[:num_examples]
            )
            chosen_labels = (
                labels[:num_examples, :-1]
                if not self.is_encoder_decoder
                else labels[:num_examples]
            )

            # Compute the log probabilities of the labels
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1),
                torch.flatten(chosen_labels, end_dim=1),
                ignore_index=0,
            )

        if self.args.ld_alpha is not None and not is_ref_model:
            # Compute response lengths based on loss_mask
            completion_lengths = loss_mask.sum(dim=1)

            chosen_lengths = completion_lengths[:num_examples]
            rejected_lengths = completion_lengths[num_examples:]
            public_lengths = torch.min(
                chosen_lengths, rejected_lengths
            )  # l_p in the paper
            public_lengths = torch.cat([public_lengths, public_lengths], dim=0)

            seq_len = per_token_logps.size(1)
            position_ids = torch.arange(
                seq_len, device=per_token_logps.device
            ).expand_as(per_token_logps)

            ld_mask = position_ids < public_lengths.unsqueeze(1)
            mask = position_ids < completion_lengths.unsqueeze(1)

            front_mask = (ld_mask & mask).float()
            rear_mask = (~ld_mask & mask).float()
            front_logps = (per_token_logps * front_mask).sum(dim=1)
            rear_logps = (per_token_logps * rear_mask).sum(dim=1)

            all_logps = front_logps + self.args.ld_alpha * rear_logps

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]
        output["trajectory_logps"] = torch.cat(
            (output["chosen_logps"], output["rejected_logps"]), dim=0
        )

        # Compute the mean logits
        if self.padding_free:
            # position_ids contains a sequence of range identifiers (e.g., [[0, 1, 2, 0, 1, 2, 3, ...]]).
            # There are 2*num_examples ranges in total: the first half corresponds to the chosen tokens,
            # and the second half to the rejected tokens.
            # To find the start of the rejected tokens, we look for the num_examples+1-th zero in pos_id.
            split_idx = (position_ids == 0).nonzero(as_tuple=True)[1][num_examples]
            mean_chosen_logits = logits[0, :split_idx][loss_mask[0, :split_idx]].mean()
            mean_rejected_logits = logits[0, split_idx:][
                loss_mask[0, split_idx:]
            ].mean()
        else:
            mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
            mean_rejected_logits = logits[num_examples:][
                loss_mask[num_examples:]
            ].mean()

        output["mean_chosen_logits"] = mean_chosen_logits
        output["mean_rejected_logits"] = mean_rejected_logits

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output

    def get_batch_loss_metrics(
        self,
        model: PreTrainedModel | nn.Module,
        batch: dict[str, list | torch.LongTensor],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute VMPO-style losses and metrics for paired preference data."""
        metrics: dict[str, float] = {}

        updating = train_eval == "train"
        model_output = self.concatenated_forward(model, batch)

        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

        chosen_logps = model_output["chosen_logps"]
        rejected_logps = model_output["rejected_logps"]
        chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps)
        rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        advantages = torch.cat([chosen_rewards, rejected_rewards], dim=0)
        logps = model_output["trajectory_logps"]
        ref_logps_concat = torch.cat([ref_chosen_logps, ref_rejected_logps], dim=0)

        psi, l_eta_values = self.e_step_dual_update(
            advantages.detach(), update=updating
        )
        psi = psi.detach()

        if self.use_weighting:
            weights = model_output["policy_weights"].repeat_interleave(2)
            psi = psi * weights
            psi_sum = psi.sum()
            if psi_sum.item() <= 0:
                psi = torch.full_like(psi, 1.0 / psi.numel())
            else:
                psi = psi / psi_sum

        loss_components = []
        policy_components = []
        kl_components = []
        m_steps = max(1, self.args.m_steps)
        for _ in range(m_steps):
            total_loss, policy_part, kl_part = self._m_step(
                logps, ref_logps_concat, psi
            )
            loss_components.append(total_loss)
            policy_components.append(policy_part)
            kl_components.append(kl_part)

        loss = torch.stack(loss_components).mean()
        policy_loss_value = torch.stack(policy_components).mean()
        kl_weighted = torch.stack(kl_components).mean()
        l_alpha = self.alpha_dual_update(kl_weighted.detach(), update=updating)

        if self.args.rpo_alpha is not None:
            loss = loss + self.args.rpo_alpha * model_output["nll_loss"]

        prefix = "eval_" if train_eval == "eval" else ""

        if self.use_weighting:
            # weighting already absorbed into ψ; keep metric parity by logging mean weight
            metrics[f"{prefix}policy/weight_mean"] = (
                self.accelerator.gather_for_metrics(model_output["policy_weights"])
                .mean()
                .item()
            )

        if self.aux_loss_enabled:
            loss = loss + self.aux_loss_coef * model_output["aux_loss"]

        metrics[f"{prefix}rewards/chosen"] = (
            self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        )
        metrics[f"{prefix}rewards/rejected"] = (
            self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        )
        metrics[f"{prefix}rewards/accuracies"] = (
            self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        )
        metrics[f"{prefix}rewards/margins"] = (
            self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards)
            .mean()
            .item()
        )
        metrics[f"{prefix}logps/chosen"] = (
            self.accelerator.gather_for_metrics(chosen_logps).detach().mean().item()
        )
        metrics[f"{prefix}logps/rejected"] = (
            self.accelerator.gather_for_metrics(rejected_logps).detach().mean().item()
        )
        metrics[f"{prefix}logits/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["mean_chosen_logits"])
            .detach()
            .mean()
            .item()
        )
        metrics[f"{prefix}logits/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["mean_rejected_logits"])
            .detach()
            .mean()
            .item()
        )
        metrics[f"{prefix}loss/policy"] = (
            self.accelerator.gather_for_metrics(policy_loss_value.unsqueeze(0))
            .mean()
            .item()
        )
        metrics[f"{prefix}objective/kl_ref_weighted"] = (
            self.accelerator.gather_for_metrics(kl_weighted.unsqueeze(0)).mean().item()
        )

        eta_value = F.softplus(self.eta_raw) + self.args.eta_min
        alpha_value = F.softplus(self.alpha_raw) + self.args.alpha_min
        psi_state = psi
        psi_ess = 1.0 / (psi_state.square().sum().clamp_min(1e-8))
        psi_max = psi_state.max()
        metrics[f"{prefix}psi/ess"] = (
            self.accelerator.gather_for_metrics(psi_ess.unsqueeze(0)).mean().item()
        )
        metrics[f"{prefix}psi/max"] = (
            self.accelerator.gather_for_metrics(psi_max.unsqueeze(0)).mean().item()
        )
        metrics[f"{prefix}dual/eta_value"] = (
            self.accelerator.gather_for_metrics(eta_value.detach().unsqueeze(0))
            .mean()
            .item()
        )
        metrics[f"{prefix}dual/alpha_value"] = (
            self.accelerator.gather_for_metrics(alpha_value.detach().unsqueeze(0))
            .mean()
            .item()
        )
        eta_grad = self.eta_raw.grad
        alpha_grad = self.alpha_raw.grad
        eta_grad_norm = eta_grad.norm().item() if eta_grad is not None else 0.0
        alpha_grad_norm = alpha_grad.norm().item() if alpha_grad is not None else 0.0
        metrics[f"{prefix}dual/eta_grad_norm"] = eta_grad_norm
        metrics[f"{prefix}dual/alpha_grad_norm"] = alpha_grad_norm
        metrics[f"{prefix}dual/l_eta_mean"] = (
            torch.stack(l_eta_values).mean().item() if l_eta_values else 0.0
        )
        metrics[f"{prefix}dual/l_alpha"] = l_alpha.item() if updating else 0.0
        metrics[f"{prefix}loss/total"] = (
            self.accelerator.gather_for_metrics(loss.detach().unsqueeze(0))
            .mean()
            .item()
        )

        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}nll_loss"] = (
                self.accelerator.gather_for_metrics(model_output["nll_loss"])
                .detach()
                .mean()
                .item()
            )
        if self.aux_loss_enabled:
            metrics[f"{prefix}aux_loss"] = (
                self.accelerator.gather_for_metrics(model_output["aux_loss"])
                .detach()
                .mean()
                .item()
            )

        return loss, metrics

    def compute_loss(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        compute_loss_context_manager = (
            autocast(self.accelerator.device.type)
            if self._peft_has_been_casted_to_bf16
            else nullcontext()
        )
        with compute_loss_context_manager:
            loss, metrics = self.get_batch_loss_metrics(
                model, inputs, train_eval="train"
            )

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return loss, metrics

        return loss

    def generate_from_model_and_ref(
        self, model, batch: dict[str, torch.LongTensor]
    ) -> tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = (
            autocast(self.accelerator.device.type)
            if self._peft_has_been_casted_to_bf16
            else nullcontext()
        )

        with generate_context_manager:
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.pad_token_id,
            )

            # if ref_output in batch use that otherwise use the reference model
            if "ref_output" in batch:
                ref_output = batch["ref_output"]
            else:
                if self.ref_model is None:
                    with self.null_ref_context():
                        ref_output = self.model.generate(
                            input_ids=batch["prompt_input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                            max_length=self.max_length,
                            do_sample=True,
                            pad_token_id=self.pad_token_id,
                        )
                else:
                    ref_output = self.ref_model.generate(
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.pad_token_id,
                    )

        policy_output = pad_to_length(policy_output, self.max_length, self.pad_token_id)
        policy_output_decoded = self.processing_class.batch_decode(
            policy_output, skip_special_tokens=True
        )

        ref_output = pad_to_length(ref_output, self.max_length, self.pad_token_id)
        ref_output_decoded = self.processing_class.batch_decode(
            ref_output, skip_special_tokens=True
        )

        return policy_output_decoded, ref_output_decoded

    def prediction_step(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = (
            autocast(self.accelerator.device.type)
            if self._peft_has_been_casted_to_bf16
            else nullcontext()
        )

        with torch.no_grad(), prediction_context_manager:
            loss, metrics = self.get_batch_loss_metrics(
                model, inputs, train_eval="eval"
            )

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return loss.detach(), None, None

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = [v for k, v in logits_dict.items() if k not in ignore_keys]
        logits = torch.tensor(logits, device=self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(
        self, metrics: dict[str, float], train_eval: Literal["train", "eval"] = "train"
    ) -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch. Prediction/evaluation loop, shared by
        `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(
                range(num_samples), k=self.args.eval_batch_size
            )

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            policy_output_decoded, ref_output_decoded = (
                self.generate_from_model_and_ref(self.model, random_batch)
            )

            table = pd.DataFrame(
                columns=["Prompt", "Policy", "Ref Model"],
                data=[
                    [prompt, pol[len(prompt) :], ref[len(prompt) :]]
                    for prompt, pol, ref in zip(
                        random_batch_dataset["prompt"],
                        policy_output_decoded,
                        ref_output_decoded,
                        strict=True,
                    )
                ],
            )
            if "wandb" in self.args.report_to and self.accelerator.is_main_process:
                wandb.log({"game_log": wandb.Table(data=table)})

            if "comet_ml" in self.args.report_to:
                log_table_to_comet_experiment(
                    name="game_log.csv",
                    table=table,
                )

            if "mlflow" in self.args.report_to and self.accelerator.is_main_process:
                mlflow.log_table(data=table, artifact_file="game_log.json")

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        return initial_output

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`dict[str, float]`):
                The values to log.
            start_time (`float`, *optional*):
                Start time of the training.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs, start_time)

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
