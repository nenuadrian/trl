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

import math
import textwrap
import logging
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, Literal

import torch
import torch.nn as nn
from datasets import Dataset, IterableDataset
from torch import autocast
from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.trainer_utils import EvalLoopOutput

from ..models import create_reference_model
from .callbacks import SyncRefModelCallback
from .dar_trainer import DARTrainer, RewardFn
from .vmpo_config import VMPOConfig

logger = logging.getLogger(__name__)


class _SyncOldPolicyModelCallback(TrainerCallback):
    """Sync the old-policy snapshot from the trainable policy."""

    def __init__(self, old_policy_model: PreTrainedModel | nn.Module, accelerator):
        self.old_policy_model = old_policy_model
        self.accelerator = accelerator

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0:
            return
        if state.global_step % args.vmpo_old_policy_sync_steps != 0:
            return

        model = kwargs["model"]
        target_model = self.old_policy_model
        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
            target_model = self.accelerator.unwrap_model(target_model)
        SyncRefModelCallback.sync_target_model(model, target_model, 1.0)


class VMPOTrainer(DARTrainer):
    """
    Variational Maximum a Posteriori Policy Optimization (VMPO) trainer.

    This trainer reuses DAR's prompt/completion/reward pipeline and swaps the objective for
    VMPO-style updates:
    1. E-step: build a non-parametric target over top-advantage samples using an adaptive
       temperature dual variable eta.
    2. M-step: fit the policy to that target and apply an adaptive KL trust-region penalty with
       dual variable alpha.
    """

    _tag_names = ["trl", "vmpo"]
    _name = "VMPO"
    _paper = {
        "title": "V-MPO: On-Policy Maximum a Posteriori Policy Optimization for Discrete and Continuous Control",
        "id": "1909.12238",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{song2019vmpo,
                title        = {{V-MPO: On-Policy Maximum a Posteriori Policy Optimization for Discrete and Continuous Control}},
                author       = {Song, H. Francis and Abdolmaleki, Abbas and Springenberg, Jost Tobias and Clark, Aidan and Soyer, Hubert and Rae, Jack W. and others},
                year         = 2019,
                journal      = {arXiv preprint arXiv:1909.12238}
            }"""),
    }

    def __init__(
        self,
        model: str | nn.Module | PreTrainedModel,
        ref_model: PreTrainedModel | nn.Module | None = None,
        args: VMPOConfig | None = None,
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
            args = VMPOConfig(f"{model_name}-VMPO")

        super().__init__(
            model=model,
            ref_model=ref_model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_fn=reward_fn,
            reward_model=reward_model,
            reward_processing_class=reward_processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
        )

        self._vmpo_eta = float(args.vmpo_eta_init)
        self._vmpo_alpha = float(args.vmpo_alpha_init)
        self._uses_old_policy_estimator = args.vmpo_kl_estimator in {"old_policy", "old_policy_ref"}
        self.old_policy_model = None
        self._reward_ema_baseline = 0.0
        self._near_zero_kl_steps = 0

        if self._uses_old_policy_estimator:
            if args.vmpo_old_policy_sync_steps == 1:
                logger.warning(
                    "`vmpo_old_policy_sync_steps=1` with old-policy KL can collapse the trust-region signal "
                    "(KL≈0). Consider setting it to >= 4."
                )
            if self.is_deepspeed_enabled and self.accelerator.state.deepspeed_plugin.zero_stage == 3:
                raise ValueError(
                    "`vmpo_kl_estimator` in `{old_policy, old_policy_ref}` is not supported with DeepSpeed ZeRO-3."
                )

            policy_model = self.accelerator.unwrap_model(self.model)
            self.old_policy_model = create_reference_model(policy_model)
            self.old_policy_model = self.accelerator.prepare_model(self.old_policy_model, evaluation_mode=True)
            self.add_callback(
                _SyncOldPolicyModelCallback(
                    old_policy_model=self.old_policy_model,
                    accelerator=self.accelerator,
                )
            )

    @staticmethod
    def _project_positive(value: float, minimum: float, maximum: float | None = None) -> float:
        if not math.isfinite(value):
            return minimum
        projected = max(value, minimum)
        if maximum is not None:
            projected = min(projected, maximum)
        return projected

    def _eta_dual_loss(self, top_advantages: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        log_normalizer = torch.logsumexp(top_advantages / eta, dim=0) - math.log(top_advantages.numel())
        return eta * self.args.vmpo_eps_eta + eta * log_normalizer

    def _compute_e_step_weights(self, advantages_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_samples = advantages_flat.numel()
        num_selected = max(1, math.ceil(num_samples * self.args.vmpo_topk_fraction))

        top_advantages, top_indices = torch.topk(advantages_flat, k=num_selected)
        scaled_advantages = top_advantages / max(self._vmpo_eta, self.args.vmpo_min_eta)
        scaled_advantages = scaled_advantages - scaled_advantages.max()
        top_weights = torch.softmax(scaled_advantages, dim=0)

        weights = torch.zeros_like(advantages_flat)
        weights[top_indices] = top_weights
        return weights, top_advantages

    def _update_eta_dual(self, top_advantages: torch.Tensor) -> float:
        eta = torch.tensor(
            self._project_positive(self._vmpo_eta, self.args.vmpo_min_eta, self.args.vmpo_max_eta),
            dtype=top_advantages.dtype,
            device=top_advantages.device,
            requires_grad=True,
        )
        eta_loss = self._eta_dual_loss(top_advantages.detach(), eta)
        grad_eta = torch.autograd.grad(eta_loss, eta)[0]
        if torch.isfinite(grad_eta):
            eta = eta - self.args.vmpo_eta_lr * grad_eta
            self._vmpo_eta = self._project_positive(
                float(eta.detach().item()), self.args.vmpo_min_eta, self.args.vmpo_max_eta
            )
        else:
            self._vmpo_eta = self._project_positive(self._vmpo_eta, self.args.vmpo_min_eta, self.args.vmpo_max_eta)
        return float(eta_loss.detach().item())

    def _update_alpha_dual(self, kl_mean: torch.Tensor):
        kl_excess = float(kl_mean.detach().item()) - self.args.vmpo_eps_alpha
        if math.isfinite(kl_excess):
            updated_alpha = self._vmpo_alpha + self.args.vmpo_alpha_lr * kl_excess
            self._vmpo_alpha = self._project_positive(updated_alpha, self.args.vmpo_min_alpha, self.args.vmpo_max_alpha)
        else:
            self._vmpo_alpha = self._project_positive(
                self._vmpo_alpha, self.args.vmpo_min_alpha, self.args.vmpo_max_alpha
            )

    def _compute_advantages(self, rewards: torch.Tensor, train_eval: Literal["train", "eval"]) -> torch.Tensor:
        if self.args.vmpo_advantage_baseline == "per_prompt":
            centered_advantages = rewards - rewards.mean(dim=1, keepdim=True)
        elif self.args.vmpo_advantage_baseline == "batch":
            centered_advantages = rewards - rewards.mean()
        elif self.args.vmpo_advantage_baseline == "ema":
            if train_eval == "train":
                batch_mean = rewards.mean().detach().item()
                self._reward_ema_baseline = (
                    self.args.vmpo_reward_ema_decay * self._reward_ema_baseline
                    + (1.0 - self.args.vmpo_reward_ema_decay) * batch_mean
                )
            baseline = rewards.new_tensor(self._reward_ema_baseline)
            centered_advantages = rewards - baseline
        else:
            raise ValueError(f"Unknown `vmpo_advantage_baseline`: {self.args.vmpo_advantage_baseline}")

        return self._normalize_advantages(centered_advantages)

    def _generate_k_responses_from_model(
        self, generation_model: PreTrainedModel | nn.Module, prompts_batch: dict[str, Any], k: int
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
            generated = generation_model.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                **generation_kwargs,
            )

        completion_input_ids = generated[:, prompt_input_ids.size(1) :]
        completion_attention_mask = (completion_input_ids != self.pad_token_id).long()
        prompt_input_ids_rep = prompt_input_ids.repeat_interleave(k, dim=0)
        prompt_attention_mask_rep = prompt_attention_mask.repeat_interleave(k, dim=0)
        return prompt_input_ids_rep, prompt_attention_mask_rep, completion_input_ids, completion_attention_mask

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
            k = self.args.vmpo_k
            generation_model = self.model
            if train_eval == "train" and self._uses_old_policy_estimator and self.old_policy_model is not None:
                generation_model = self.old_policy_model
            (
                prompt_input_ids_rep,
                prompt_attention_mask_rep,
                completion_input_ids,
                completion_attention_mask,
            ) = self._generate_k_responses_from_model(generation_model, batch, k)
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
            raise ValueError("VMPO rewards must have shape `(batch_size * K,)`.")

        if behavior_logps_flat is not None:
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

        old_policy_logps = None
        if self._uses_old_policy_estimator:
            if self.old_policy_model is None:
                raise ValueError("`old_policy_model` is not initialized while using an old-policy KL estimator.")
            old_policy_context = (
                autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
            )
            with torch.no_grad(), old_policy_context:
                old_policy_logps = self._compute_sequence_logps(
                    self.old_policy_model,
                    prompt_input_ids_rep,
                    prompt_attention_mask_rep,
                    completion_input_ids,
                    completion_attention_mask,
                )

        rewards = rewards_flat.view(batch_size, k)
        normalized_advantages = self._compute_advantages(rewards, train_eval=train_eval)
        advantages_flat = normalized_advantages.reshape(-1)

        e_step_weights, top_advantages = self._compute_e_step_weights(advantages_flat)
        policy_loss = -(e_step_weights * policy_logps).sum()

        eta_value = torch.tensor(
            self._project_positive(self._vmpo_eta, self.args.vmpo_min_eta, self.args.vmpo_max_eta),
            dtype=top_advantages.dtype,
            device=top_advantages.device,
        )
        if train_eval == "train":
            eta_loss = self._update_eta_dual(top_advantages)
        else:
            eta_loss = float(self._eta_dual_loss(top_advantages.detach(), eta_value).detach().item())

        ref_anchor_loss = policy_logps.new_zeros(())
        if self.args.vmpo_kl_estimator == "behavior":
            if behavior_logps_flat is None:
                behavior_logps_flat = policy_logps.detach()
            kl_values = behavior_logps_flat - policy_logps
        elif self.args.vmpo_kl_estimator == "old_policy":
            kl_values = old_policy_logps - policy_logps
        elif self.args.vmpo_kl_estimator == "old_policy_ref":
            kl_values = old_policy_logps - policy_logps
            if not self.reference_free:
                ref_anchor_values = policy_logps - ref_logps
                ref_anchor_loss = self.args.vmpo_ref_anchor_coef * ref_anchor_values.mean()
        elif self.reference_free:
            kl_values = torch.zeros_like(policy_logps)
        else:
            kl_values = policy_logps - ref_logps
        kl_mean = kl_values.mean()
        kl_mean_value = float(kl_mean.detach().item())

        alpha_value = self._vmpo_alpha
        kl_loss = policy_logps.new_tensor(alpha_value) * kl_mean
        loss = policy_loss + kl_loss + ref_anchor_loss

        if train_eval == "train":
            if abs(kl_mean_value) <= self.args.vmpo_kl_zero_tol:
                self._near_zero_kl_steps += 1
            else:
                self._near_zero_kl_steps = 0

            if self._near_zero_kl_steps == self.args.vmpo_kl_warning_patience:
                logger.warning(
                    "VMPO KL signal has stayed near zero for %d consecutive train steps "
                    "(|kl_mean| <= %.2e). Consider increasing `vmpo_old_policy_sync_steps` and/or "
                    "using `vmpo_kl_estimator='old_policy_ref'`.",
                    self.args.vmpo_kl_warning_patience,
                    self.args.vmpo_kl_zero_tol,
                )

        if train_eval == "train":
            self._update_alpha_dual(kl_mean)

        prefix = "eval_" if train_eval == "eval" else ""
        gathered_rewards = self.accelerator.gather_for_metrics(rewards_flat.detach())
        gathered_advantages = self.accelerator.gather_for_metrics(advantages_flat.detach())
        gathered_top_advantages = self.accelerator.gather_for_metrics(top_advantages.detach())
        gathered_weights = self.accelerator.gather_for_metrics(e_step_weights.detach())
        gathered_policy = self.accelerator.gather_for_metrics(policy_logps.detach())
        gathered_ref = self.accelerator.gather_for_metrics(ref_logps.detach())
        gathered_old = (
            self.accelerator.gather_for_metrics(old_policy_logps.detach()) if old_policy_logps is not None else None
        )
        gathered_kl = self.accelerator.gather_for_metrics(kl_values.detach())
        gathered_policy_loss = self.accelerator.gather_for_metrics(policy_loss.detach())
        gathered_kl_loss = self.accelerator.gather_for_metrics(kl_loss.detach())
        gathered_ref_anchor_loss = self.accelerator.gather_for_metrics(ref_anchor_loss.detach())
        gathered_losses = self.accelerator.gather_for_metrics(loss.detach())

        positive_weights = gathered_weights[gathered_weights > 0]
        if positive_weights.numel() > 0:
            weight_entropy = -(positive_weights * positive_weights.clamp_min(1e-8).log()).sum().item()
        else:
            weight_entropy = 0.0

        metrics = {
            f"{prefix}vmpo/rewards_mean": gathered_rewards.mean().item(),
            f"{prefix}vmpo/rewards_std": gathered_rewards.std(unbiased=False).item(),
            f"{prefix}vmpo/advantages_mean": gathered_advantages.mean().item(),
            f"{prefix}vmpo/advantages_std": gathered_advantages.std(unbiased=False).item(),
            f"{prefix}vmpo/top_advantages_mean": gathered_top_advantages.mean().item(),
            f"{prefix}vmpo/top_advantages_std": gathered_top_advantages.std(unbiased=False).item(),
            f"{prefix}vmpo/weights_entropy": weight_entropy,
            f"{prefix}vmpo/selected_fraction": (gathered_weights > 0).float().mean().item(),
            f"{prefix}vmpo/eta": self._vmpo_eta,
            f"{prefix}vmpo/eta_loss": eta_loss,
            f"{prefix}vmpo/alpha": self._vmpo_alpha,
            f"{prefix}vmpo/kl_mean": gathered_kl.mean().item(),
            f"{prefix}vmpo/reward_ema_baseline": self._reward_ema_baseline,
            f"{prefix}vmpo/near_zero_kl_streak": float(self._near_zero_kl_steps),
            f"{prefix}vmpo/logps_policy": gathered_policy.mean().item(),
            f"{prefix}vmpo/logps_ref": gathered_ref.mean().item(),
            f"{prefix}vmpo/logps_old": gathered_old.mean().item() if gathered_old is not None else 0.0,
            f"{prefix}vmpo/policy_loss": gathered_policy_loss.mean().item(),
            f"{prefix}vmpo/kl_loss": gathered_kl_loss.mean().item(),
            f"{prefix}vmpo/ref_anchor_loss": gathered_ref_anchor_loss.mean().item(),
            f"{prefix}vmpo/loss": gathered_losses.mean().item(),
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
            [metrics["eval_vmpo/kl_mean"], metrics["eval_vmpo/rewards_mean"]],
            device=self.accelerator.device,
        )
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)
        return loss.detach(), logits, labels
