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

"""
MaxMin PPO Trainer implementing the MaxMin-RLHF algorithm from:

    "MaxMin-RLHF: Alignment with Diverse Human Preferences"
    Chakraborty et al., ICML 2024
    https://arxiv.org/abs/2402.08925

The key idea: instead of training with a single reward model, we maintain multiple reward models
(one per user subpopulation) and at each PPO step, optimize the policy against the worst-off
subpopulation (the one giving the lowest reward). This follows the Egalitarian principle from
social choice theory: maximize the minimum utility across all groups.

Algorithm (from the paper):
    1. Learn multiple reward models via EM (see em_reward_learning.py)
    2. At each PPO step:
        a. Generate responses from the policy
        b. Score responses with ALL reward models
        c. Select the reward model giving the minimum score (worst-off group)
        d. Use that reward for the PPO update
"""

import gc
import math
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import logging
from datasets import Dataset
from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.utils import is_peft_available

from ...trainer.utils import empty_cache, selective_log_softmax
from ..utils import first_true_indices, get_reward
from .maxmin_ppo_config import MaxMinPPOConfig
from .ppo_trainer import (
    INVALID_LOGPROB,
    PPOTrainer,
    batch_generation,
    forward,
    masked_mean,
    masked_whiten,
    truncate_response,
)


logger = logging.get_logger(__name__)

if is_peft_available():
    from peft import PeftConfig


class MaxMinPPOTrainer(PPOTrainer):
    """Trainer for MaxMin PPO (MaxMin-RLHF).

    Extends [`PPOTrainer`] to support multiple reward models and the MaxMin alignment objective.
    At each training step, the policy is updated to maximize the reward of the worst-off
    subpopulation, following Algorithm 1 from the paper.

    Args:
        args ([`MaxMinPPOConfig`]):
            Training arguments.
        processing_class: Tokenizer or processor.
        model (`torch.nn.Module`): Policy model.
        ref_model (`torch.nn.Module` or `None`): Reference model for KL divergence.
        reward_models (`list[torch.nn.Module]`):
            List of reward models, one per user subpopulation. These correspond to the
            reward functions r_{phi_u} learned via the EM algorithm (Algorithm 2).
        train_dataset ([`~datasets.Dataset`]): Training dataset.
        value_model (`torch.nn.Module`): Value model for advantage estimation.
        data_collator: Optional data collator.
        eval_dataset: Optional evaluation dataset.
        optimizers: Optional (optimizer, scheduler) tuple.
        callbacks: Optional list of callbacks.
        peft_config: Optional PEFT configuration.
    """

    _tag_names = ["trl", "maxmin-ppo"]
    _name = "MaxMinPPO"
    _paper = {
        "title": "MaxMin-RLHF: Alignment with Diverse Human Preferences",
        "id": "2402.08925",
        "citation": (
            "@inproceedings{chakraborty2024maxmin,\n"
            "    title        = {{MaxMin-RLHF: Alignment with Diverse Human Preferences}},\n"
            "    author       = {Souradip Chakraborty and Jiahao Qiu and Hui Yuan and Alec Koppel "
            "and Dinesh Manocha and Furong Huang and Amrit Singh Bedi and Mengdi Wang},\n"
            "    year         = 2024,\n"
            "    booktitle    = {ICML},\n"
            "    eprint       = {arXiv:2402.08925}\n"
            "}"
        ),
    }

    def __init__(
        self,
        args: MaxMinPPOConfig,
        processing_class: PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin,
        model: nn.Module,
        ref_model: nn.Module | None,
        reward_models: list[nn.Module],
        train_dataset: Dataset,
        value_model: nn.Module,
        data_collator: DataCollatorWithPadding | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: list[TrainerCallback] | None = None,
        peft_config: "PeftConfig | None" = None,
    ) -> None:
        if len(reward_models) < 1:
            raise ValueError("At least one reward model must be provided.")

        # Store all reward models before calling super().__init__
        # Pass the first reward model to the parent class as the "primary" reward model
        self.reward_models = reward_models
        self.num_reward_models = len(reward_models)

        super().__init__(
            args=args,
            processing_class=processing_class,
            model=model,
            ref_model=ref_model,
            reward_model=reward_models[0],
            train_dataset=train_dataset,
            value_model=value_model,
            data_collator=data_collator,
            eval_dataset=eval_dataset,
            optimizers=optimizers,
            callbacks=callbacks,
            peft_config=peft_config,
        )

        # Disable dropout and move additional reward models to device
        # (first one is already handled by parent via self.reward_model)
        from ...trainer.utils import disable_dropout_in_model, prepare_deepspeed

        for i in range(1, len(self.reward_models)):
            disable_dropout_in_model(self.reward_models[i])
            if self.is_deepspeed_enabled:
                self.reward_models[i] = prepare_deepspeed(
                    self.reward_models[i], args.per_device_train_batch_size, args.fp16, args.bf16
                )
            else:
                self.reward_models[i] = self.reward_models[i].to(self.accelerator.device)

        # Validate that reward models and value model have num_labels=1
        for i, rm in enumerate(self.reward_models):
            num_labels = getattr(rm.config, "num_labels", None)
            if num_labels is not None and num_labels != 1:
                logger.warning(
                    f"Reward model {i} has num_labels={num_labels} (expected 1). "
                    f"This may happen when loading a CausalLM checkpoint as AutoModelForSequenceClassification. "
                    f"Only the first label dimension will be used."
                )

        # Track which reward model was selected at each step (for logging)
        self._selected_reward_model_counts = defaultdict(int)

        logger.info(f"MaxMin PPO initialized with {self.num_reward_models} reward models, strategy='{args.maxmin_strategy}'")

    def _aggregate_scores(self, all_scores: list[torch.Tensor]) -> tuple[torch.Tensor, int]:
        """Aggregate scores from multiple reward models according to the MaxMin strategy.

        Args:
            all_scores: List of score tensors, one per reward model. Each has shape (batch_size,).

        Returns:
            Tuple of (aggregated_scores, selected_model_index).
            - aggregated_scores: shape (batch_size,)
            - selected_model_index: index of the selected reward model (-1 for non-argmin strategies)
        """
        stacked = torch.stack(all_scores, dim=0)  # (num_models, batch_size)
        strategy = self.args.maxmin_strategy

        if strategy == "min":
            # MaxMin: take the minimum score across reward models for each sample
            # Then select the reward model with the lowest mean score for logging
            mean_scores = stacked.mean(dim=1)  # (num_models,)
            selected_idx = mean_scores.argmin().item()
            aggregated = stacked.min(dim=0).values
            return aggregated, selected_idx

        elif strategy == "mean":
            aggregated = stacked.mean(dim=0)
            return aggregated, -1

        elif strategy == "softmin":
            # Softmin: differentiable approximation to min
            tau = self.args.softmin_temperature
            weights = F.softmax(-stacked / tau, dim=0)  # (num_models, batch_size)
            aggregated = (weights * stacked).sum(dim=0)
            selected_idx = weights.mean(dim=1).argmax().item()
            return aggregated, selected_idx

        else:
            raise ValueError(f"Unknown maxmin_strategy: {strategy}")

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_model
        reward_models = self.reward_models
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

        accelerator.print(f"===training policy with MaxMin-RLHF ({self.num_reward_models} reward models)===")
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        from ...models.utils import unwrap_model_for_generation

        for update in range(1, args.num_total_batches + 1):
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
                # Per-reward-model scores for MaxMin aggregation
                all_rm_scores = [[] for _ in range(self.num_reward_models)]

                with (
                    unwrap_model_for_generation(
                        self.model,
                        self.accelerator,
                        gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                        generation_kwargs=generation_kwargs,
                    ) as unwrapped_model
                ):
                    query_responses, logitss = batch_generation(
                        unwrapped_model.policy,
                        queries,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        generation_config,
                    )

                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    response = query_response[:, context_length:]
                    logits = logitss[i : i + args.local_rollout_forward_batch_size]
                    logprob = selective_log_softmax(logits, response)
                    del logits
                    empty_cache()

                    if ref_policy is None:
                        with self.null_ref_context():
                            ref_output = forward(model.policy, query_response, processing_class.pad_token_id)
                    else:
                        ref_output = forward(ref_policy, query_response, processing_class.pad_token_id)
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.temperature + 1e-7
                    ref_logprob = selective_log_softmax(ref_logits, response)
                    del ref_output, ref_logits
                    empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    if self.stop_token_id is not None:
                        postprocessed_response = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id, response
                        )

                    # Response Processing 2. run ALL reward models on the truncated responses
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    sequence_length = first_true_indices(postprocessed_response == processing_class.pad_token_id) - 1
                    unwrapped_value_model = accelerator.unwrap_model(model).value_model
                    full_value, _, _ = get_reward(
                        unwrapped_value_model, query_response, processing_class.pad_token_id, context_length
                    )
                    value = full_value[:, context_length - 1 : -1]
                    # Ensure value is 2D (batch, seq). When num_labels > 1 (e.g. score head
                    # initialized from a causal LM checkpoint), take only the first label.
                    if value.dim() == 3:
                        if value.size(-1) == 1:
                            value = value.squeeze(-1)
                        else:
                            value = value[:, :, 0]

                    # Score with each reward model
                    for rm_idx, rm in enumerate(reward_models):
                        _, rm_score, _ = get_reward(
                            rm, postprocessed_query_response, processing_class.pad_token_id, context_length
                        )
                        # Handle reward models with num_labels > 1 (e.g. CausalLM checkpoint
                        # loaded as SeqCls without proper score head reinitialization)
                        if rm_score.dim() > 1:
                            rm_score = rm_score[:, 0]
                        all_rm_scores[rm_idx].append(rm_score)

                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    values.append(value)

                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                values = torch.cat(values, 0)

                # Concatenate per-RM scores and aggregate
                all_rm_scores_cat = [torch.cat(rm_scores, 0) for rm_scores in all_rm_scores]
                scores, selected_rm_idx = self._aggregate_scores(all_rm_scores_cat)
                self._selected_reward_model_counts[selected_rm_idx] += 1

                del logprob, ref_logprob, full_value, value, unwrapped_model
                empty_cache()
                gc.collect()

                # Response Processing 3. Filter completion
                contain_eos_token = torch.any(postprocessed_responses == self.processing_class.eos_token_id, dim=-1)
                if self.args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty

                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
                sequence_lengths_p1 = sequence_lengths + 1
                padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
                values = torch.masked_fill(values, padding_mask_p1, 0)

                # 4. compute rewards
                logr = ref_logprobs - logprobs
                kl = -logr if args.kl_estimator == "k1" else (logr.exp() - 1) - logr
                non_score_reward = -args.kl_coef * kl
                rewards = non_score_reward.clone()
                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
                rewards[actual_start, actual_end] += scores

                # 5. whiten rewards
                if args.whiten_rewards:
                    rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
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
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = advantages + values
                advantages = masked_whiten(advantages, ~padding_mask)
                advantages = torch.masked_fill(advantages, padding_mask, 0)
                empty_cache()

            # Do multiple epochs of PPO training
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                            mb_advantage = advantages[micro_batch_inds]
                            mb_responses = responses[micro_batch_inds]
                            mb_query_responses = query_responses[micro_batch_inds]
                            mb_logprobs = logprobs[micro_batch_inds]
                            mb_return = returns[micro_batch_inds]
                            mb_values = values[micro_batch_inds]

                            output, vpred_temp = forward(model, mb_query_responses, processing_class.pad_token_id)
                            logits = output.logits[:, context_length - 1 : -1]
                            logits /= args.temperature + 1e-7
                            new_logprobs = selective_log_softmax(logits, mb_responses)
                            new_logprobs = torch.masked_fill(
                                new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                            )
                            vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                            vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds], 0)
                            vpredclipped = torch.clamp(
                                vpred,
                                mb_values - args.cliprange_value,
                                mb_values + args.cliprange_value,
                            )
                            vf_losses1 = torch.square(vpred - mb_return)
                            vf_losses2 = torch.square(vpredclipped - mb_return)
                            vf_loss_max = torch.max(vf_losses1, vf_losses2)
                            vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1[micro_batch_inds])
                            vf_clipfrac = masked_mean(
                                (vf_losses2 > vf_losses1).float(), ~padding_mask_p1[micro_batch_inds]
                            )
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
                            loss = pg_loss + args.vf_coef * vf_loss
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                pg_clipfrac = masked_mean(
                                    (pg_losses2 > pg_losses).float(), ~padding_mask[micro_batch_inds]
                                )
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_clipfrac
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                                vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_clipfrac
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    del (
                        output, vpred_temp, logits, new_logprobs, vpred, vpredclipped,
                        vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max,
                        pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,
                        mb_advantage, mb_values, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    empty_cache()

            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.sum(1).mean()
                rlhf_reward = mean_non_score_reward + scores.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather_for_metrics(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather_for_metrics(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = (
                    self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
                )
                metrics["objective/rlhf_reward"] = self.accelerator.gather_for_metrics(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather_for_metrics(scores.mean()).mean().item()

                # Log per-reward-model scores
                for rm_idx, rm_scores in enumerate(all_rm_scores_cat):
                    metrics[f"objective/rm_{rm_idx}_scores"] = (
                        self.accelerator.gather_for_metrics(rm_scores.mean()).mean().item()
                    )
                metrics["maxmin/selected_rm_idx"] = float(selected_rm_idx)
                metrics["maxmin/strategy"] = 0.0  # placeholder for logging

                metrics["policy/approxkl_avg"] = self.accelerator.gather_for_metrics(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather_for_metrics(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather_for_metrics(pg_loss_stats).mean().item()
                metrics["loss/value_avg"] = self.accelerator.gather_for_metrics(vf_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather_for_metrics(vf_clipfrac_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather_for_metrics(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather_for_metrics(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather_for_metrics(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == processing_class.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len
                self.state.global_step += 1
                self.log(metrics)

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores, metrics, non_score_reward
            del all_rm_scores_cat
            empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
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
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        # Log final RM selection statistics
        accelerator.print(f"MaxMin RM selection counts: {dict(self._selected_reward_model_counts)}")

    def generate_completions(self, sampling: bool = False):
        """Override to score with all reward models and handle num_labels > 1."""
        if self.eval_dataset is None:
            return
        args = self.args
        processing_class = self.processing_class
        generation_kwargs = {
            "max_new_tokens": args.response_length,
            "temperature": (0.01 + 1e-7),
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
        }
        generation_config = GenerationConfig(**generation_kwargs)

        from collections import defaultdict

        import pandas as pd
        from accelerate.utils import gather_object

        from ...models.utils import unwrap_model_for_generation

        table = defaultdict(list)
        with (
            unwrap_model_for_generation(
                self.model,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                generation_kwargs=generation_kwargs,
            ) as unwrapped_model
        ):
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
                    if self.stop_token_id is not None:
                        postprocessed_response = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id, response
                        )
                    table["query"].extend(
                        gather_object(processing_class.batch_decode(query, skip_special_tokens=True))
                    )
                    table["model response"].extend(
                        gather_object(processing_class.batch_decode(postprocessed_response))
                    )

                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    # Score with all reward models
                    for rm_idx, rm in enumerate(self.reward_models):
                        _, score, _ = get_reward(
                            rm, postprocessed_query_response, processing_class.pad_token_id, context_length
                        )
                        if score.dim() > 1:
                            score = score[:, 0]
                        table[f"score_rm{rm_idx}"].extend(
                            self.accelerator.gather_for_metrics(score).float().cpu().numpy()
                        )

                    # Aggregate for the main "score" column
                    all_scores = []
                    for rm_idx in range(self.num_reward_models):
                        all_scores.append(torch.tensor(table[f"score_rm{rm_idx}"][-query.shape[0]:]))
                    agg_score, _ = self._aggregate_scores(all_scores)
                    table["score"].extend(agg_score.float().cpu().numpy())

                if sampling:
                    break
        df = pd.DataFrame(table)

        if self.accelerator.is_main_process:
            from rich.console import Console
            from rich.table import Table as RichTable

            try:
                rich_table = RichTable(show_lines=True)
                for col in df.columns:
                    rich_table.add_column(col)
                for _, row in df.head(5).iterrows():
                    rich_table.add_row(*[str(v) for v in row])
                Console().print(rich_table)
            except Exception:
                print(df.head(5).to_string())

            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

            if "comet_ml" in args.report_to:
                from ...trainer.utils import log_table_to_comet_experiment

                log_table_to_comet_experiment(name="completions.csv", table=df)
