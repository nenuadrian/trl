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

import textwrap
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
from accelerate import PartialState
from datasets import Dataset, IterableDataset
from torch import autocast
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer_utils import EvalLoopOutput

from ..data_utils import is_conversational, maybe_apply_chat_template, maybe_extract_prompt
from .dar_config import DARConfig
from .dpo_trainer import DPOTrainer
from .utils import cap_exp, pad, selective_log_softmax


RewardFn = Callable[..., list[float] | torch.Tensor]


@dataclass
class DataCollatorForDAR(DataCollatorMixin):
    """
    Data collator for DAR datasets.

    It always pads prompts and keeps optional offline completion/reward fields in Python lists.
    """

    pad_token_id: int
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_input_ids = [torch.tensor(example["prompt_input_ids"], dtype=torch.long) for example in examples]
        prompt_attention_mask = [torch.ones_like(input_ids) for input_ids in prompt_input_ids]
        output = {
            "prompt_input_ids": pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left"),
            "prompt_attention_mask": pad(prompt_attention_mask, padding_value=0, padding_side="left"),
        }

        if "prompt" in examples[0]:
            output["prompt"] = [example["prompt"] for example in examples]
        if "completion_input_ids" in examples[0]:
            output["completion_input_ids"] = [example["completion_input_ids"] for example in examples]
        if "rewards" in examples[0]:
            output["rewards"] = [example["rewards"] for example in examples]
        if "behavior_logps" in examples[0]:
            output["behavior_logps"] = [example["behavior_logps"] for example in examples]

        return output


class DARTrainer(DPOTrainer):
    """
    Direct Advantage Regression (DAR) trainer.

    This trainer converts DPO-style preference optimization into reward-weighted SFT by:
    1. sampling K completions per prompt (online) or reading completions from the dataset (offline),
    2. computing scalar rewards,
    3. building DAR importance/regularization weights,
    4. minimizing weighted sequence negative log-likelihood.
    """

    _tag_names = ["trl", "dar"]
    _name = "DAR"
    _paper = {
        "title": "Direct Advantage Regression (DAR)",
        "id": "2504.14177",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @misc{dar2025,
                title        = {{Direct Advantage Regression}},
                author       = {Anonymous},
                year         = 2025,
                eprint       = {arXiv:2504.14177}
            }"""),
    }

    def __init__(
        self,
        model: str | nn.Module | PreTrainedModel,
        ref_model: PreTrainedModel | nn.Module | None = None,
        args: DARConfig | None = None,
        data_collator: DataCollator | None = None,  # type: ignore
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        reward_fn: RewardFn | None = None,
        reward_model: str | PreTrainedModel | None = None,
        reward_processing_class: PreTrainedTokenizerBase | None = None,
        compute_metrics: Callable[[EvalLoopOutput], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: Any = None,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = DARConfig(f"{model_name}-DAR")

        if args.use_liger_kernel:
            raise ValueError("`use_liger_kernel=True` is not supported in `DARTrainer`.")
        if args.precompute_ref_log_probs:
            raise ValueError("`precompute_ref_log_probs=True` is not supported in `DARTrainer`.")
        if args.max_length is None:
            raise ValueError("`max_length` must be set for `DARTrainer`.")

        if data_collator is None:
            # Temporary placeholder, updated after tokenizer is resolved in the parent class.
            data_collator = DataCollatorForDAR(pad_token_id=0)

        super().__init__(
            model=model,
            ref_model=ref_model,
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
            peft_config=peft_config,
        )

        if self.is_vision_model:
            raise ValueError("`DARTrainer` currently supports text-only models.")
        if self.is_encoder_decoder:
            raise ValueError("`DARTrainer` currently supports decoder-only causal language models.")

        # Replace temporary collator with the tokenizer-derived pad token id from the parent initialization.
        self.data_collator = DataCollatorForDAR(pad_token_id=self.pad_token_id)

        self.reward_fn = reward_fn
        if reward_model is None and args.reward_model_path is not None:
            reward_model = args.reward_model_path
        self.reward_model = reward_model

        if self.reward_fn is not None and self.reward_model is not None:
            raise ValueError("Provide either `reward_fn` or `reward_model`, not both.")

        if isinstance(self.reward_model, str):
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(self.reward_model, num_labels=1)
            if reward_processing_class is None:
                reward_processing_class = AutoTokenizer.from_pretrained(self.reward_model.config._name_or_path)
        elif isinstance(self.reward_model, PreTrainedModel):
            if reward_processing_class is None:
                reward_processing_class = AutoTokenizer.from_pretrained(self.reward_model.config._name_or_path)

        if reward_processing_class is not None and reward_processing_class.pad_token_id is None:
            reward_processing_class.pad_token = reward_processing_class.eos_token
        self.reward_processing_class = reward_processing_class

        if self.reward_model is not None:
            self.reward_model.eval()
            for param in self.reward_model.parameters():
                param.requires_grad_(False)
            self.reward_model.to(self.accelerator.device)

    @staticmethod
    def _tokenize_prompt_row(
        features: dict[str, Any],
        processing_class: PreTrainedTokenizerBase | ProcessorMixin,
        max_prompt_length: int | None = None,
    ) -> dict[str, list[int]]:
        tokenizer = processing_class.tokenizer if isinstance(processing_class, ProcessorMixin) else processing_class
        prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        return {"prompt_input_ids": prompt_input_ids}

    @staticmethod
    def _tokenize_pairwise_row_to_dar(
        features: dict[str, Any],
        processing_class: PreTrainedTokenizerBase | ProcessorMixin,
        max_prompt_length: int | None = None,
        max_completion_length: int | None = None,
        is_chat: bool = False,
    ) -> dict[str, Any]:
        tokenizer = processing_class.tokenizer if isinstance(processing_class, ProcessorMixin) else processing_class
        prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
        chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)["input_ids"]
        rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)["input_ids"]

        if not is_chat and tokenizer.eos_token_id is not None:
            chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
            rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            chosen_input_ids = chosen_input_ids[:max_completion_length]
            rejected_input_ids = rejected_input_ids[:max_completion_length]

        return {
            "prompt_input_ids": prompt_input_ids,
            "completion_input_ids": [chosen_input_ids, rejected_input_ids],
            "rewards": [1.0, 0.0],
        }

    @staticmethod
    def _tokenize_prompt_completion_reward_row(
        features: dict[str, Any],
        processing_class: PreTrainedTokenizerBase | ProcessorMixin,
        max_prompt_length: int | None = None,
        max_completion_length: int | None = None,
        is_chat: bool = False,
    ) -> dict[str, Any]:
        tokenizer = processing_class.tokenizer if isinstance(processing_class, ProcessorMixin) else processing_class
        prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]

        completions = features["completion"]
        rewards = features["rewards"] if "rewards" in features else features["reward"]
        if isinstance(completions, str):
            completions = [completions]
        if not isinstance(rewards, list):
            rewards = [rewards]
        if len(completions) != len(rewards):
            raise ValueError("`completion` and reward list lengths must match for DAR offline mode.")

        completion_input_ids = [
            tokenizer(completion, add_special_tokens=False)["input_ids"] for completion in completions
        ]
        if not is_chat and tokenizer.eos_token_id is not None:
            completion_input_ids = [ids + [tokenizer.eos_token_id] for ids in completion_input_ids]
        if max_completion_length is not None:
            completion_input_ids = [ids[:max_completion_length] for ids in completion_input_ids]

        return {
            "prompt_input_ids": prompt_input_ids,
            "completion_input_ids": completion_input_ids,
            "rewards": [float(reward) for reward in rewards],
        }

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin,
        args: DARConfig,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        if not isinstance(processing_class, (PreTrainedTokenizerBase, ProcessorMixin)):
            raise TypeError("`DARTrainer` expects a tokenizer/processor with tokenizer support.")

        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc nor writer_batch_size
            map_kwargs["num_proc"] = args.dataset_num_proc
            map_kwargs["writer_batch_size"] = 10
            column_names = set(dataset.column_names)
        else:
            sample = next(iter(dataset))
            column_names = set(sample.keys())

        with PartialState().main_process_first():
            # Keep compatibility with preference datasets where prompt is implicit in chosen/rejected.
            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Extracting prompt in {dataset_name} dataset"
            dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

            sample_for_chat = next(iter(dataset))
            is_chat = is_conversational(sample_for_chat)

            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
            dataset = dataset.map(
                maybe_apply_chat_template, fn_kwargs={"tokenizer": processing_class, "tools": args.tools}, **map_kwargs
            )

            has_pairwise = "chosen" in column_names and "rejected" in column_names
            has_completion_reward = "completion" in column_names and (
                "reward" in column_names or "rewards" in column_names
            )

            if isinstance(dataset, Dataset):
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

            if has_pairwise:
                remove_columns = [col for col in ["chosen", "rejected"] if col in column_names]
                dataset = dataset.map(
                    self._tokenize_pairwise_row_to_dar,
                    fn_kwargs={
                        "processing_class": processing_class,
                        "max_prompt_length": args.max_prompt_length,
                        "max_completion_length": args.max_completion_length,
                        "is_chat": is_chat,
                    },
                    remove_columns=remove_columns,
                    **map_kwargs,
                )
            elif has_completion_reward:
                remove_columns = [col for col in ["completion", "reward"] if col in column_names]
                dataset = dataset.map(
                    self._tokenize_prompt_completion_reward_row,
                    fn_kwargs={
                        "processing_class": processing_class,
                        "max_prompt_length": args.max_prompt_length,
                        "max_completion_length": args.max_completion_length,
                        "is_chat": is_chat,
                    },
                    remove_columns=remove_columns,
                    **map_kwargs,
                )
            else:
                dataset = dataset.map(
                    self._tokenize_prompt_row,
                    fn_kwargs={
                        "processing_class": processing_class,
                        "max_prompt_length": args.max_prompt_length,
                    },
                    **map_kwargs,
                )

        return dataset

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt",
                "prompt_input_ids",
                "prompt_attention_mask",
                "completion_input_ids",
                "rewards",
                "behavior_logps",
            ]

    def _compute_sequence_logps(
        self,
        model: PreTrainedModel | nn.Module,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        completion_input_ids: torch.Tensor,
        completion_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens_to_truncate = max(prompt_input_ids.size(1) + completion_input_ids.size(1) - self.max_length, 0)
        if num_tokens_to_truncate > 0:
            prompt_input_ids = prompt_input_ids[:, num_tokens_to_truncate:]
            prompt_attention_mask = prompt_attention_mask[:, num_tokens_to_truncate:]

        input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
        attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits[:, :-1]
        labels = input_ids[:, 1:]
        per_token_logps = selective_log_softmax(logits, labels)

        completion_mask = torch.cat(
            (torch.zeros_like(prompt_attention_mask), completion_attention_mask), dim=1
        )[:, 1:].bool()
        per_token_logps[~completion_mask] = 0
        return per_token_logps.sum(-1)

    def generate_k_responses(
        self, prompts_batch: dict[str, Any], k: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_input_ids = prompts_batch["prompt_input_ids"]
        prompt_attention_mask = prompts_batch["prompt_attention_mask"]

        generation_kwargs = {
            "do_sample": True,
            "max_new_tokens": self.args.max_new_tokens,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "top_k": self.args.top_k,
            "repetition_penalty": self.args.repetition_penalty,
            "num_return_sequences": k,
            "pad_token_id": self.pad_token_id,
        }
        if self.args.min_p is not None:
            generation_kwargs["min_p"] = self.args.min_p

        gen_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )
        with torch.no_grad(), gen_context_manager:
            generated = self.model.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                **generation_kwargs,
            )

        completion_input_ids = generated[:, prompt_input_ids.size(1) :]
        completion_attention_mask = (completion_input_ids != self.pad_token_id).long()
        prompt_input_ids_rep = prompt_input_ids.repeat_interleave(k, dim=0)
        prompt_attention_mask_rep = prompt_attention_mask.repeat_interleave(k, dim=0)
        return prompt_input_ids_rep, prompt_attention_mask_rep, completion_input_ids, completion_attention_mask

    def _prepare_offline_completions(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor | None, torch.Tensor | None]:
        prompt_input_ids = batch["prompt_input_ids"]
        device = prompt_input_ids.device

        completion_groups = batch["completion_input_ids"]
        if len(completion_groups) == 0:
            raise ValueError("`completion_input_ids` is empty in DAR offline mode.")

        k = len(completion_groups[0])
        if k == 0:
            raise ValueError("Each row in `completion_input_ids` must contain at least one completion.")
        if not all(len(group) == k for group in completion_groups):
            raise ValueError("All rows in `completion_input_ids` must contain the same number of completions.")

        flat_completions = [
            torch.tensor(completion_ids, dtype=torch.long, device=device)
            for completion_group in completion_groups
            for completion_ids in completion_group
        ]
        completion_input_ids = pad(flat_completions, padding_value=self.pad_token_id)
        completion_attention_mask = (completion_input_ids != self.pad_token_id).long()

        rewards_flat = None
        if "rewards" in batch:
            rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=device)
            if rewards.ndim == 1:
                rewards = rewards.unsqueeze(-1)
            if rewards.shape[1] != k:
                raise ValueError("`rewards` second dimension must match the number of completions per prompt.")
            rewards_flat = rewards.reshape(-1)

        behavior_logps_flat = None
        if "behavior_logps" in batch:
            behavior_logps = torch.tensor(batch["behavior_logps"], dtype=torch.float32, device=device)
            if behavior_logps.ndim == 1:
                behavior_logps = behavior_logps.unsqueeze(-1)
            if behavior_logps.shape[1] != k:
                raise ValueError("`behavior_logps` second dimension must match the number of completions per prompt.")
            behavior_logps_flat = behavior_logps.reshape(-1)

        return completion_input_ids, completion_attention_mask, k, rewards_flat, behavior_logps_flat

    def _compute_rewards(
        self, prompts: list[str], completions: list[str], completion_ids: torch.Tensor, batch: dict[str, Any]
    ) -> torch.Tensor:
        if self.reward_fn is None and self.reward_model is None:
            raise ValueError(
                "DAR requires rewards. Provide a reward source with `reward_fn`, `reward_model`, "
                "`args.reward_model_path`, or include `rewards` in the dataset."
            )

        if self.reward_fn is not None:
            completion_ids_list = [completion_ids[i].tolist() for i in range(completion_ids.shape[0])]
            try:
                rewards = self.reward_fn(
                    prompts=prompts,
                    completions=completions,
                    completion_ids=completion_ids_list,
                    batch=batch,
                )
            except TypeError:
                rewards = self.reward_fn(prompts, completions)
            rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.accelerator.device)
            if rewards.numel() != len(prompts):
                raise ValueError("`reward_fn` must return one scalar reward per (prompt, completion).")
            return rewards

        texts = [prompt + completion for prompt, completion in zip(prompts, completions, strict=True)]
        reward_inputs = self.reward_processing_class(
            text=texts,
            return_tensors="pt",
            padding=True,
            padding_side="right",
            add_special_tokens=False,
        )
        reward_inputs = {key: value.to(self.accelerator.device) for key, value in reward_inputs.items()}
        with torch.inference_mode():
            rewards = self.reward_model(**reward_inputs).logits[:, 0]
        return rewards

    def _normalize_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        if self.args.adv_norm == "none":
            return advantages
        if self.args.adv_norm == "per_prompt":
            mean = advantages.mean(dim=1, keepdim=True)
            std = advantages.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
            return (advantages - mean) / std
        if self.args.adv_norm == "batch":
            flat = advantages.reshape(-1)
            mean = flat.mean()
            std = flat.std(unbiased=False).clamp_min(1e-6)
            return ((flat - mean) / std).view_as(advantages)
        raise ValueError(f"Unknown `adv_norm`: {self.args.adv_norm}")

    def get_batch_loss_metrics(
        self,
        model: PreTrainedModel | nn.Module,
        batch: dict[str, Any],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        prompt_input_ids = batch["prompt_input_ids"]
        prompt_attention_mask = batch["prompt_attention_mask"]
        batch_size = prompt_input_ids.size(0)

        if "completion_input_ids" in batch:
            (
                completion_input_ids,
                completion_attention_mask,
                k,
                rewards_flat,
                behavior_logps_flat,
            ) = self._prepare_offline_completions(batch)
            prompt_input_ids_rep = prompt_input_ids.repeat_interleave(k, dim=0)
            prompt_attention_mask_rep = prompt_attention_mask.repeat_interleave(k, dim=0)
        else:
            k = self.args.dar_k
            (
                prompt_input_ids_rep,
                prompt_attention_mask_rep,
                completion_input_ids,
                completion_attention_mask,
            ) = self.generate_k_responses(batch, k)
            rewards_flat = None
            behavior_logps_flat = None

        policy_logps = self._compute_sequence_logps(
            model,
            prompt_input_ids_rep,
            prompt_attention_mask_rep,
            completion_input_ids,
            completion_attention_mask,
        )

        if rewards_flat is None:
            if "prompt" in batch:
                prompts = [prompt for prompt in batch["prompt"] for _ in range(k)]
            else:
                decoded_prompts = self.processing_class.batch_decode(
                    prompt_input_ids.detach().cpu(), skip_special_tokens=True
                )
                prompts = [prompt for prompt in decoded_prompts for _ in range(k)]
            completions = self.processing_class.batch_decode(
                completion_input_ids.detach().cpu(),
                skip_special_tokens=True,
            )
            rewards_flat = self._compute_rewards(prompts, completions, completion_input_ids, batch)

        if rewards_flat.numel() != batch_size * k:
            raise ValueError("DAR rewards must have shape `(batch_size * K,)`.")

        if behavior_logps_flat is None:
            behavior_logps_flat = policy_logps.detach()
        else:
            behavior_logps_flat = behavior_logps_flat.to(policy_logps.device).to(policy_logps.dtype)

        if self.reference_free:
            ref_logps = torch.zeros_like(policy_logps)
        else:
            ref_context = (
                autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
            )
            with torch.no_grad(), ref_context:
                if self.ref_model is None:
                    with self.null_ref_context():
                        ref_logps = self._compute_sequence_logps(
                            self.model,
                            prompt_input_ids_rep,
                            prompt_attention_mask_rep,
                            completion_input_ids,
                            completion_attention_mask,
                        )
                else:
                    ref_logps = self._compute_sequence_logps(
                        self.ref_model,
                        prompt_input_ids_rep,
                        prompt_attention_mask_rep,
                        completion_input_ids,
                        completion_attention_mask,
                    )

        rewards = rewards_flat.view(batch_size, k)
        centered_advantages = rewards - rewards.mean(dim=1, keepdim=True)
        normalized_advantages = self._normalize_advantages(centered_advantages)
        normalized_advantages_flat = normalized_advantages.reshape(-1)

        scale = float(self.args.alpha + self.args.beta)
        log_reg_weight = (self.args.alpha / scale) * (ref_logps - behavior_logps_flat)
        log_adv_weight = normalized_advantages_flat / scale
        weights = cap_exp(log_reg_weight + log_adv_weight)
        weights = torch.clamp(weights, max=self.args.dar_wclip)

        losses = -(weights * policy_logps)
        loss = losses.mean()

        prefix = "eval_" if train_eval == "eval" else ""
        gathered_rewards = self.accelerator.gather_for_metrics(rewards_flat.detach())
        gathered_advantages = self.accelerator.gather_for_metrics(normalized_advantages_flat.detach())
        gathered_weights = self.accelerator.gather_for_metrics(weights.detach())
        gathered_policy = self.accelerator.gather_for_metrics(policy_logps.detach())
        gathered_ref = self.accelerator.gather_for_metrics(ref_logps.detach())
        gathered_behavior = self.accelerator.gather_for_metrics(behavior_logps_flat.detach())
        gathered_losses = self.accelerator.gather_for_metrics(losses.detach())

        metrics = {
            f"{prefix}dar/rewards_mean": gathered_rewards.mean().item(),
            f"{prefix}dar/rewards_std": gathered_rewards.std(unbiased=False).item(),
            f"{prefix}dar/advantages_mean": gathered_advantages.mean().item(),
            f"{prefix}dar/advantages_std": gathered_advantages.std(unbiased=False).item(),
            f"{prefix}dar/weights_mean": gathered_weights.mean().item(),
            f"{prefix}dar/weights_max": gathered_weights.max().item(),
            f"{prefix}dar/logps_policy": gathered_policy.mean().item(),
            f"{prefix}dar/logps_ref": gathered_ref.mean().item(),
            f"{prefix}dar/logps_behavior": gathered_behavior.mean().item(),
            f"{prefix}dar/loss": gathered_losses.mean().item(),
        }

        return loss, metrics

    def prediction_step(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        prediction_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )
        with torch.no_grad(), prediction_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return loss.detach(), None, None

        logits = torch.tensor(
            [metrics["eval_dar/weights_mean"], metrics["eval_dar/rewards_mean"]],
            device=self.accelerator.device,
        )
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)
        return loss.detach(), logits, labels
