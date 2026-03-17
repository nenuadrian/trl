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

import os
import textwrap
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from accelerate import PartialState
from accelerate.logging import get_logger
from datasets import Dataset, IterableDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction

from ..data_utils import is_conversational, maybe_apply_chat_template, maybe_extract_prompt
from .base_trainer import BaseTrainer
from .rest_config import RESTConfig
from .sft_trainer import DataCollatorForLanguageModeling
from .utils import create_model_from_path, get_config_model_id, selective_log_softmax


logger = get_logger(__name__)

RewardFn = Callable[..., list[float] | torch.Tensor]


class RESTTrainer(BaseTrainer):
    r"""
    Trainer for the ReST_EM (Reinforced Self-Training with Expectation-Maximization) method.

    Implements the iterative self-training pipeline from [Beyond Human Data: Scaling Self-Training for
    Problem-Solving with Language Models](https://huggingface.co/papers/2312.06585) and
    [Reinforced Self-Training (ReST) for Language Modeling](https://huggingface.co/papers/2308.08998).

    The algorithm alternates between:
    - **E-step (Generate)**: Sample multiple completions per prompt from the current policy, score them with a
      reward function or model, and filter to keep high-quality completions.
    - **M-step (Improve)**: Fine-tune the model on the filtered completions using (optionally reward-weighted) NLL loss.

    Example:

    ```python
    from trl import RESTConfig, RESTTrainer

    def binary_reward(prompts, completions):
        return [1.0 if "correct" in c else 0.0 for c in completions]

    trainer = RESTTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        args=RESTConfig(num_iterations=2, num_samples_per_prompt=4, output_dir="rest_output"),
        train_dataset=dataset,  # prompt-only dataset
        reward_fn=binary_reward,
    )
    trainer.train()
    ```

    Args:
        model (`str` or `PreTrainedModel`):
            Model to train. Can be a model ID string or an instantiated model.
        args (`RESTConfig`, *optional*):
            Training configuration. If `None`, defaults are used.
        data_collator (`DataCollator`, *optional*):
            Data collator for the M-step SFT training.
        train_dataset (`Dataset` or `IterableDataset`):
            Prompt-only dataset. Must contain a `prompt` column.
        eval_dataset (`Dataset`, *optional*):
            Evaluation dataset.
        processing_class (`PreTrainedTokenizerBase` or `ProcessorMixin`, *optional*):
            Tokenizer or processor. Loaded from the model if not provided.
        reward_fn (`Callable`, *optional*):
            Reward function: `(prompts: list[str], completions: list[str]) -> list[float]`.
        reward_model (`str` or `PreTrainedModel`, *optional*):
            Reward model for scoring completions. Provide either this or `reward_fn`.
        reward_processing_class (`PreTrainedTokenizerBase`, *optional*):
            Tokenizer for the reward model.
        compute_metrics (`Callable`, *optional*):
            Metrics computation function for evaluation.
        callbacks (`list[TrainerCallback]`, *optional*):
            Custom callbacks.
        optimizers (`tuple`, *optional*):
            Optimizer and scheduler tuple.
        peft_config (*optional*):
            PEFT configuration for parameter-efficient fine-tuning.
    """

    _tag_names = ["trl", "rest"]
    _name = "REST"
    _paper = {
        "title": "Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models",
        "id": "2312.06585",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{singh2024restem,
                title   = {{Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models}},
                author  = {Avi Singh and John D Co-Reyes and Rishabh Agarwal and Ankesh Anand and Piyush Patil and others},
                journal = {Transactions on Machine Learning Research},
                year    = 2024,
                eprint  = {arXiv:2312.06585}
            }"""),
    }

    def __init__(
        self,
        model: str | nn.Module | PreTrainedModel,
        args: RESTConfig | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_fn: RewardFn | None = None,
        reward_model: str | PreTrainedModel | None = None,
        reward_processing_class: PreTrainedTokenizerBase | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: Any = None,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = RESTConfig(f"{model_name}-REST")

        # Store the original model path/id for resetting between iterations
        self._model_name_or_path = model if isinstance(model, str) else get_config_model_id(model.config)
        self._model_init_kwargs = args.model_init_kwargs or {}
        self._peft_config = peft_config

        # Store the original prompt-only dataset (used in E-step)
        self._prompt_dataset = train_dataset

        # Load model if string
        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            model = create_model_from_path(model, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `RESTConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(get_config_model_id(model.config))

        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError("`processing_class` must be `PreTrainedTokenizerBase` or `ProcessorMixin`.")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        self.pad_token_id = tokenizer.pad_token_id

        # Data collator - use SFT's collator for the M-step
        if data_collator is None:
            data_collator = DataCollatorForLanguageModeling(
                pad_token_id=self.pad_token_id,
                completion_only_loss=True,
            )

        # We need a dummy SFT-formatted dataset for the initial super().__init__ call.
        # The actual training dataset is generated during the E-step.
        dummy_dataset = self._create_dummy_sft_dataset(train_dataset, tokenizer, args)

        # PEFT
        if peft_config is not None:
            from peft import get_peft_model

            model = get_peft_model(model, peft_config)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=dummy_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Reward setup
        self.reward_fn = reward_fn
        if reward_model is None and args.reward_model_path is not None:
            reward_model = args.reward_model_path
        self.reward_model = reward_model

        if self.reward_fn is not None and self.reward_model is not None:
            raise ValueError("Provide either `reward_fn` or `reward_model`, not both.")
        if self.reward_fn is None and self.reward_model is None:
            raise ValueError("Either `reward_fn` or `reward_model` must be provided for RESTTrainer.")

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
            if self.reward_processing_class is None:
                raise ValueError(
                    "A reward processing class is required when using `reward_model`. "
                    "Please provide `reward_processing_class` or ensure it can be auto-loaded."
                )
            self.reward_model.config.pad_token_id = self.reward_processing_class.pad_token_id
            self.reward_model.eval()
            for param in self.reward_model.parameters():
                param.requires_grad_(False)
            self.reward_model.to(self.accelerator.device)

    def _create_dummy_sft_dataset(
        self, prompt_dataset: Dataset | IterableDataset | None, tokenizer: PreTrainedTokenizerBase, args: RESTConfig
    ) -> Dataset:
        """Create a minimal tokenized dataset so the parent Trainer can initialize."""
        if prompt_dataset is None:
            return Dataset.from_dict({"input_ids": [[0]], "labels": [[0]], "attention_mask": [[1]]})

        # Take just one example to create the structure
        sample = next(iter(prompt_dataset))
        prompt_text = sample.get("prompt", "")
        if not prompt_text and "messages" in sample:
            prompt_text = str(sample["messages"])
        if not prompt_text and "text" in sample:
            prompt_text = sample["text"]
        dummy_text = prompt_text + " dummy"
        encoded = tokenizer(dummy_text, truncation=True, max_length=args.max_length or 512, add_special_tokens=False)
        return Dataset.from_dict({
            "input_ids": [encoded["input_ids"]],
            "labels": [encoded["input_ids"]],
            "attention_mask": [encoded["attention_mask"]],
        })

    def train(self, resume_from_checkpoint=None, **kwargs):
        """
        Run the full ReST_EM training pipeline: iterate between E-step (Generate) and M-step (Improve).
        """
        args: RESTConfig = self.args

        # Save the base model checkpoint for resetting between iterations
        base_checkpoint_dir = os.path.join(args.output_dir, "_rest_base_checkpoint")
        if self.is_world_process_zero():
            os.makedirs(base_checkpoint_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(base_checkpoint_dir)

        tokenizer = (
            self.processing_class.tokenizer
            if isinstance(self.processing_class, ProcessorMixin)
            else self.processing_class
        )

        for iteration in range(args.num_iterations):
            logger.info(f"=== ReST_EM Iteration {iteration + 1}/{args.num_iterations} ===", main_process_only=True)

            # ---- E-step: Generate and filter completions ----
            logger.info("E-step: Generating completions...", main_process_only=True)
            generated_data = self._generate_completions(tokenizer)

            logger.info("E-step: Scoring completions...", main_process_only=True)
            scored_data = self._score_completions(generated_data)

            logger.info("E-step: Filtering completions...", main_process_only=True)
            filtered_data = self._filter_completions(scored_data)

            num_filtered = len(filtered_data)
            logger.info(f"E-step complete: {num_filtered} completions passed filtering.", main_process_only=True)

            if num_filtered == 0:
                logger.warning(
                    f"No completions passed filtering at iteration {iteration + 1}. "
                    "Consider lowering `reward_threshold` or increasing `num_samples_per_prompt`.",
                    main_process_only=True,
                )
                continue

            # ---- M-step: Fine-tune on filtered data ----
            logger.info("M-step: Preparing SFT dataset...", main_process_only=True)
            sft_dataset = self._prepare_sft_dataset(filtered_data, tokenizer)

            # Reset model to base checkpoint if configured (ReST_EM behavior)
            if args.reset_model_each_iteration and iteration > 0:
                logger.info("M-step: Resetting model to base checkpoint...", main_process_only=True)
                self._reset_model_to_base(base_checkpoint_dir)

            # Update the training dataset
            self.train_dataset = sft_dataset

            # Reset trainer state for new iteration
            self.state.global_step = 0
            self.state.epoch = 0

            # Re-create optimizer and scheduler for the new iteration
            self.optimizer = None
            self.lr_scheduler = None

            logger.info(f"M-step: Training on {len(sft_dataset)} examples...", main_process_only=True)
            super().train(resume_from_checkpoint=None)

            logger.info(f"=== Iteration {iteration + 1} complete ===", main_process_only=True)

        # Save the final model
        self.save_model(args.output_dir)

        # Clean up base checkpoint
        if self.is_world_process_zero():
            import shutil

            if os.path.exists(base_checkpoint_dir):
                shutil.rmtree(base_checkpoint_dir, ignore_errors=True)

        return self.state

    @torch.no_grad()
    def _generate_completions(self, tokenizer: PreTrainedTokenizerBase) -> list[dict[str, Any]]:
        """E-step: Generate `num_samples_per_prompt` completions per prompt using `num_return_sequences`."""
        args: RESTConfig = self.args
        model = self.accelerator.unwrap_model(self.model)
        model.eval()

        dataset = self._prompt_dataset
        if dataset is None:
            raise ValueError("No training dataset provided for generation.")

        # Build (prompt_text, original_sample) pairs
        prompt_texts: list[str] = []
        original_samples: list[dict[str, Any]] = []
        for sample in dataset:
            prompt_text = sample.get("prompt", "")
            if not prompt_text:
                if "messages" in sample:
                    processed = maybe_apply_chat_template(sample, tokenizer=tokenizer)
                    prompt_text = processed.get("prompt", str(sample["messages"]))
                elif "text" in sample:
                    prompt_text = sample["text"]
            if prompt_text:
                prompt_texts.append(prompt_text)
                original_samples.append(dict(sample))

        if not prompt_texts:
            raise ValueError("No prompts found in the dataset. Ensure the dataset has a 'prompt' column.")

        num_samples = args.num_samples_per_prompt
        gen_batch_size = args.generation_batch_size

        generation_kwargs = {
            "do_sample": True,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.generation_temperature,
            "top_k": args.generation_top_k if args.generation_top_k > 0 else None,
            "top_p": args.generation_top_p,
            "pad_token_id": self.pad_token_id,
            # Generate all num_samples completions for each prompt in one call.
            # Outputs are interleaved: [p0_s0, p0_s1, ..., p1_s0, p1_s1, ...]
            "num_return_sequences": num_samples,
        }

        all_results: list[dict[str, Any]] = []

        for batch_start in range(0, len(prompt_texts), gen_batch_size):
            batch_prompts = prompt_texts[batch_start : batch_start + gen_batch_size]
            batch_samples = original_samples[batch_start : batch_start + gen_batch_size]

            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=(args.max_length or 512) - args.max_new_tokens,
                add_special_tokens=False,
            )
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)
            prompt_length = input_ids.shape[1]

            # Shape: (len(batch_prompts) * num_samples, seq_len)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )

            completion_ids = generated[:, prompt_length:]
            completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

            # Deinterleave: completions[i * num_samples + j] → prompt i, sample j
            for i, (prompt, original_sample) in enumerate(zip(batch_prompts, batch_samples, strict=True)):
                extra = {k: v for k, v in original_sample.items() if k not in ("prompt", "completion", "reward")}
                for j in range(num_samples):
                    all_results.append({
                        "prompt": prompt,
                        "completion": completions[i * num_samples + j],
                        "reward": None,
                        "_prompt_idx": batch_start + i,
                        **extra,
                    })

        model.train()
        return all_results

    def _score_completions(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score all generated completions using the reward function or model."""
        prompts = [d["prompt"] for d in data]
        completions = [d["completion"] for d in data]

        if self.reward_fn is not None:
            # Forward extra fields (e.g. "answer" from GSM8K) as the `batch` kwarg so that
            # reward functions like `reasoning_em_reward` can access ground-truth answers.
            extra_keys = [k for k in data[0] if k not in ("prompt", "completion", "reward", "_prompt_idx")]
            batch = {k: [d[k] for d in data] for k in extra_keys} if extra_keys else None
            try:
                rewards = self.reward_fn(prompts=prompts, completions=completions, batch=batch)
            except TypeError:
                rewards = self.reward_fn(prompts=prompts, completions=completions)
            rewards = [float(r) for r in rewards]
        elif self.reward_model is not None:
            rewards = self._score_with_reward_model(prompts, completions)
        else:
            raise ValueError("No reward source configured.")

        for d, reward in zip(data, rewards, strict=True):
            d["reward"] = reward

        return data

    def _score_with_reward_model(self, prompts: list[str], completions: list[str]) -> list[float]:
        """Score completions using a reward model."""
        rewards = []
        batch_size = self.args.generation_batch_size

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_completions = completions[i : i + batch_size]
            texts = [p + c for p, c in zip(batch_prompts, batch_completions, strict=True)]

            inputs = self.reward_processing_class(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
            )
            inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}

            with torch.inference_mode():
                scores = self.reward_model(**inputs).logits[:, 0]
            rewards.extend(scores.cpu().tolist())

        return rewards

    def _filter_completions(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter completions by reward threshold and per-problem cap."""
        args: RESTConfig = self.args

        # Apply reward threshold
        filtered = [d for d in data if d["reward"] is not None and d["reward"] > args.reward_threshold]

        # Apply per-problem cap
        if args.max_solutions_per_problem is not None:
            per_prompt = defaultdict(list)
            for d in filtered:
                per_prompt[d["_prompt_idx"]].append(d)

            capped = []
            for prompt_idx, items in per_prompt.items():
                # Sort by reward descending, keep top-k
                items.sort(key=lambda x: x["reward"], reverse=True)
                capped.extend(items[: args.max_solutions_per_problem])
            filtered = capped

        return filtered

    def _prepare_sft_dataset(
        self, filtered_data: list[dict[str, Any]], tokenizer: PreTrainedTokenizerBase
    ) -> Dataset:
        """Convert filtered (prompt, completion, reward) tuples into a tokenized SFT dataset."""
        args: RESTConfig = self.args
        max_length = args.max_length or 1024

        all_input_ids = []
        all_labels = []
        all_attention_mask = []
        all_rewards = []

        for item in filtered_data:
            prompt_text = item["prompt"]
            completion_text = item["completion"]
            reward = item["reward"]

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]

            # Add EOS token
            if tokenizer.eos_token_id is not None and (
                not completion_ids or completion_ids[-1] != tokenizer.eos_token_id
            ):
                completion_ids = completion_ids + [tokenizer.eos_token_id]

            # Truncate if needed
            total_len = len(prompt_ids) + len(completion_ids)
            if total_len > max_length:
                # Truncate prompt from the left, completion from the right
                max_prompt = max_length - len(completion_ids)
                if max_prompt < 1:
                    completion_ids = completion_ids[: max_length - 1]
                    prompt_ids = prompt_ids[-1:]
                else:
                    prompt_ids = prompt_ids[-max_prompt:]

            input_ids = prompt_ids + completion_ids
            # Labels: -100 for prompt tokens (completion-only loss)
            labels = [-100] * len(prompt_ids) + completion_ids
            attention_mask = [1] * len(input_ids)

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_attention_mask.append(attention_mask)
            all_rewards.append(reward)

        dataset_dict = {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "attention_mask": all_attention_mask,
        }
        if args.reward_weighted_loss:
            dataset_dict["rewards"] = all_rewards

        return Dataset.from_dict(dataset_dict)

    def _reset_model_to_base(self, base_checkpoint_dir: str):
        """Reset the model weights to the base checkpoint."""
        unwrapped = self.accelerator.unwrap_model(self.model)

        from accelerate.utils import is_peft_model

        if is_peft_model(unwrapped):
            # For PEFT models, reload just the base model weights
            from peft import set_peft_model_state_dict

            base_model = type(unwrapped.get_base_model()).from_pretrained(
                base_checkpoint_dir, **self._model_init_kwargs
            )
            # Re-apply PEFT
            if self._peft_config is not None:
                from peft import get_peft_model

                base_model = get_peft_model(base_model, self._peft_config)
                state_dict = base_model.state_dict()
                unwrapped.load_state_dict(state_dict, strict=False)
        else:
            state_dict = type(unwrapped).from_pretrained(
                base_checkpoint_dir, **self._model_init_kwargs
            ).state_dict()
            unwrapped.load_state_dict(state_dict)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute NLL loss, optionally weighted by reward."""
        inputs["use_cache"] = False

        rewards = inputs.pop("rewards", None)
        outputs = model(**inputs)
        loss = outputs.loss

        # Apply reward weighting if enabled
        if rewards is not None and self.args.reward_weighted_loss:
            # Recompute loss with reward weighting
            logits = outputs.logits
            labels = inputs["labels"]

            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Per-token NLL
            per_token_logps = selective_log_softmax(shift_logits, shift_labels)
            loss_mask = shift_labels != -100
            per_token_loss = -per_token_logps * loss_mask

            # Per-sequence loss
            seq_loss = per_token_loss.sum(dim=-1) / loss_mask.sum(dim=-1).clamp(min=1)

            # Weight by reward
            reward_tensor = torch.tensor(rewards, dtype=seq_loss.dtype, device=seq_loss.device)
            weighted_loss = (reward_tensor * seq_loss).mean()
            loss = weighted_loss

        if return_outputs:
            return loss, outputs
        return loss
