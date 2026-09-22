#!/usr/bin/env python3
"""
train_pmpo_lmsys.py

Online PMPO on LMSYS-Chat-1M with a small Qwen Instruct model.

Pipeline

    prompt
      -> K online rollouts from the current reference policy
      -> reward model scores all K responses
      -> top K/2 are preferred
      -> bottom K/2 are dis-preferred
      -> PMPO update
      -> reference policy refresh

The minimized objective is

    L(theta)
      = (1 / B) sum_b [
            -alpha E_{Da_b}[log pi_theta(y|x_b)]
            + (1-alpha) E_{Dr_b}[log pi_theta(y|x_b)]
            + beta K_b
        ]

where K_b is the sequence-level sum of the per-token categorical
KL terms evaluated along prefixes sampled from the reference policy.

Important implementation details

    * The policy likelihood term is divided by B.
    * The KL term is normalized by B*K.
    * Generated token IDs are preserved for policy and KL computation.
    * Decoded text is used only for reward-model scoring.
    * Reference refresh is configurable and defaults to 16 optimizer steps.
    * Gradient clipping reports the pre-clip norm and clipping scale.
    * Full-vocabulary logits are held only for one forward microbatch.
    * Held-out evaluation reports mean, standard deviation, best, and worst
      reward across fresh responses.
"""

import argparse
import copy
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

try:
    import wandb
except ImportError:
    wandb = None


def parse_args():
    p = argparse.ArgumentParser()

    # Models / data
    p.add_argument(
        "--model",
        default="Qwen/Qwen2-0.5B-Instruct",
    )
    p.add_argument(
        "--reward-model",
        default="trl-lib/Qwen2-0.5B-Reward",
    )
    p.add_argument(
        "--dataset",
        default="lmsys/lmsys-chat-1m",
    )

    # Data scale
    p.add_argument("--num-examples", type=int, default=1000)
    p.add_argument("--eval-examples", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=-1)

    # Rollouts
    p.add_argument("--num-rollouts", type=int, default=4)
    p.add_argument(
        "--generation-batch-size",
        type=int,
        default=2,
        help="Number of prompts generated at once.",
    )
    p.add_argument(
        "--positive-fraction",
        type=float,
        default=0.5,
        help="Fraction selected as preferred. K=4 and 0.5 gives 2 positive and 2 negative.",
    )

    # PMPO
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.5)

    # Optimizer
    p.add_argument("--learning-rate", type=float, default=5e-7)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    # Sequence lengths
    p.add_argument("--max-prompt-tokens", type=int, default=768)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--max-sequence-tokens", type=int, default=1024)
    p.add_argument("--max-reward-tokens", type=int, default=1024)

    # Forward / reward microbatching
    p.add_argument(
        "--forward-microbatch-size",
        type=int,
        default=8,
        help="Number of prompt+completion sequences per policy/reference forward.",
    )
    p.add_argument(
        "--reward-microbatch-size",
        type=int,
        default=8,
    )

    # Sampling
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)

    # Reference updates
    p.add_argument(
        "--ref-update-steps",
        type=int,
        default=16,
        help="Refresh reference policy after this many optimizer steps.",
    )

    # Evaluation
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument(
        "--eval-rollouts",
        type=int,
        default=4,
        help="Fresh responses per held-out prompt.",
    )

    # Logging / checkpoints
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument(
        "--output-dir",
        default="./qwen-pmpo-lmsys",
    )

    # W&B
    p.add_argument(
        "--wandb-project",
        default="pmpo-lmsys",
    )
    p.add_argument(
        "--wandb-run-name",
        default=None,
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
    )

    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_lmsys_prompts(
    dataset_name: str,
    num_examples: int,
    seed: int,
) -> Dataset:
    """
    Stream LMSYS so a prototype does not materialize the full dataset.
    """
    stream = load_dataset(
        dataset_name,
        split="train",
        streaming=True,
    )

    records = []

    for row in stream:
        if row.get("language") != "English":
            continue

        conversation = row.get("conversation")
        if not conversation:
            continue

        last_user = -1

        for i, message in enumerate(conversation):
            if message.get("role") == "user":
                last_user = i

        if last_user < 0:
            continue

        messages = conversation[: last_user + 1]

        if not messages:
            continue

        if messages[-1].get("role") != "user":
            continue

        records.append({"messages": messages})

        if len(records) >= num_examples:
            break

    if len(records) < num_examples:
        raise RuntimeError(
            f"Only collected {len(records)} usable prompts, "
            f"requested {num_examples}."
        )

    ds = Dataset.from_list(records)

    return ds.shuffle(seed=seed)


def render_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _strip_generated_special_tokens(
    token_ids: Sequence[int],
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> List[int]:
    """
    Remove trailing padding and stop at the first EOS token.
    """
    result = []

    for token_id in token_ids:
        token_id = int(token_id)

        if pad_token_id is not None and token_id == pad_token_id:
            break

        result.append(token_id)

        if eos_token_id is not None and token_id == eos_token_id:
            break

    return result


@torch.inference_mode()
def generate_rollouts(
    sampling_model,
    tokenizer,
    batch_messages,
    num_rollouts,
    generation_batch_size,
    max_prompt_tokens,
    max_new_tokens,
    temperature,
    top_p,
) -> Tuple[List[List[List[int]]], List[List[str]]]:
    """
    Generate K completions per prompt.

    Returns

        completion_ids[prompt][rollout] -> token IDs
        completion_texts[prompt][rollout] -> decoded text

    The original sampled token IDs are preserved for the PMPO update.
    """
    sampling_model.eval()

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    all_completion_ids = []
    all_completion_texts = []

    try:
        for start in range(
            0,
            len(batch_messages),
            generation_batch_size,
        ):
            group_messages = batch_messages[
                start : start + generation_batch_size
            ]

            prompts = [
                render_prompt(tokenizer, messages)
                for messages in group_messages
            ]

            inputs = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_prompt_tokens,
                return_tensors="pt",
            )

            device = next(sampling_model.parameters()).device

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            outputs = sampling_model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_rollouts,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

            prompt_width = inputs["input_ids"].shape[1]

            raw_completion_ids = outputs[
                :,
                prompt_width:,
            ].detach().cpu().tolist()

            group_completion_ids = []
            group_completion_texts = []

            for i in range(len(group_messages)):
                prompt_completion_ids = []
                prompt_completion_texts = []

                for j in range(num_rollouts):
                    flat_index = i * num_rollouts + j

                    ids = _strip_generated_special_tokens(
                        raw_completion_ids[flat_index],
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                    text = tokenizer.decode(
                        ids,
                        skip_special_tokens=True,
                    )

                    prompt_completion_ids.append(ids)
                    prompt_completion_texts.append(text)

                group_completion_ids.append(
                    prompt_completion_ids
                )
                group_completion_texts.append(
                    prompt_completion_texts
                )

            all_completion_ids.extend(group_completion_ids)
            all_completion_texts.extend(group_completion_texts)

            del inputs
            del outputs

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        tokenizer.padding_side = old_padding_side

    return (
        all_completion_ids,
        all_completion_texts,
    )


def render_reward_text(
    reward_tokenizer,
    messages,
    completion,
):
    full_messages = list(messages) + [
        {
            "role": "assistant",
            "content": completion,
        }
    ]

    try:
        return reward_tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        pieces = []

        for message in full_messages:
            pieces.append(
                f"{message['role']}: {message['content']}"
            )

        return "\n".join(pieces)


@torch.inference_mode()
def score_rollouts(
    reward_model,
    reward_tokenizer,
    batch_messages,
    completion_texts,
    max_reward_tokens,
    reward_microbatch_size,
):
    """
    Score all B*K responses.
    """
    reward_model.eval()

    texts = []

    for messages, group in zip(
        batch_messages,
        completion_texts,
    ):
        for completion in group:
            texts.append(
                render_reward_text(
                    reward_tokenizer,
                    messages,
                    completion,
                )
            )

    scores = []

    for start in range(
        0,
        len(texts),
        reward_microbatch_size,
    ):
        group = texts[
            start : start + reward_microbatch_size
        ]

        inputs = reward_tokenizer(
            group,
            padding=True,
            truncation=True,
            max_length=max_reward_tokens,
            return_tensors="pt",
            add_special_tokens=False,
        )

        device = next(reward_model.parameters()).device

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        outputs = reward_model(**inputs)

        if outputs.logits.ndim != 2:
            raise RuntimeError(
                "Unexpected reward model logits shape "
                f"{tuple(outputs.logits.shape)}"
            )

        if outputs.logits.shape[-1] < 1:
            raise RuntimeError(
                "Reward model returned an empty logits dimension."
            )

        scores.append(
            outputs.logits[:, 0].float().cpu()
        )

        del inputs
        del outputs

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rewards = torch.cat(scores)

    B = len(batch_messages)
    K = len(completion_texts[0])

    return rewards.view(B, K)


def encode_training_batch(
    tokenizer,
    batch_messages,
    completion_ids_groups,
    max_prompt_tokens,
    max_sequence_tokens,
):
    """
    Build prompt+completion sequences using the original generated IDs.

    completion_mask is 1 only on completion tokens.

    Generation and training both use the same prompt rendering and the same
    max_prompt_tokens truncation before the sequence-length safety trim.
    """
    all_ids = []
    all_masks = []

    for messages, group in zip(
        batch_messages,
        completion_ids_groups,
    ):
        prompt_text = render_prompt(
            tokenizer,
            messages,
        )

        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_tokens,
        )["input_ids"]

        for sampled_completion_ids in group:
            completion = list(
                int(x)
                for x in sampled_completion_ids
            )

            if tokenizer.eos_token_id is not None:
                if (
                    not completion
                    or completion[-1]
                    != tokenizer.eos_token_id
                ):
                    completion.append(
                        tokenizer.eos_token_id
                    )

            max_completion_length = max_sequence_tokens

            if len(completion) > max_completion_length:
                completion = completion[
                    :max_completion_length
                ]

                if (
                    tokenizer.eos_token_id is not None
                    and completion
                    and completion[-1]
                    != tokenizer.eos_token_id
                ):
                    completion[-1] = (
                        tokenizer.eos_token_id
                    )

            available_prompt = max(
                0,
                max_sequence_tokens
                - len(completion),
            )

            prompt_used = (
                prompt_ids[-available_prompt:]
                if available_prompt > 0
                else []
            )

            ids = prompt_used + completion

            mask = (
                [0] * len(prompt_used)
                + [1] * len(completion)
            )

            all_ids.append(ids)
            all_masks.append(mask)

    if not all_ids:
        raise RuntimeError(
            "Training batch produced no sequences."
        )

    max_len = max(
        len(ids)
        for ids in all_ids
    )

    pad_id = tokenizer.pad_token_id

    if pad_id is None:
        raise RuntimeError(
            "Tokenizer must have a pad token."
        )

    input_ids = torch.full(
        (len(all_ids), max_len),
        pad_id,
        dtype=torch.long,
    )

    attention_mask = torch.zeros(
        (len(all_ids), max_len),
        dtype=torch.long,
    )

    completion_mask = torch.zeros(
        (len(all_ids), max_len),
        dtype=torch.long,
    )

    for i, (ids, mask) in enumerate(
        zip(all_ids, all_masks)
    ):
        n = len(ids)
        offset = max_len - n

        input_ids[
            i,
            offset:,
        ] = torch.tensor(
            ids,
            dtype=torch.long,
        )

        attention_mask[
            i,
            offset:,
        ] = 1

        completion_mask[
            i,
            offset:,
        ] = torch.tensor(
            mask,
            dtype=torch.long,
        )

    return (
        input_ids,
        attention_mask,
        completion_mask,
    )


def compute_sequence_logp_and_kl(
    policy_logits,
    ref_logits,
    input_ids,
    completion_mask,
):
    """
    Returns

        sequence_logp
        token_mean_logp
        kl_sum
        num_completion_tokens

    sequence_logp is the sum of selected completion-token log probabilities.

    kl_sum is the sum over completion positions of the categorical
    KL(ref || policy) evaluated at each autoregressive prefix.
    """
    shift_input_ids = input_ids[:, 1:]

    shift_mask = (
        completion_mask[:, 1:]
        .float()
    )

    policy_log_probs = F.log_softmax(
        policy_logits[:, :-1],
        dim=-1,
    )

    chosen_token_logp = (
        policy_log_probs
        .gather(
            dim=-1,
            index=shift_input_ids.unsqueeze(-1),
        )
        .squeeze(-1)
    )

    sequence_logp = (
        chosen_token_logp
        * shift_mask
    ).sum(dim=-1)

    num_completion_tokens = (
        shift_mask.sum(dim=-1)
        .clamp_min(1.0)
    )

    token_mean_logp = (
        sequence_logp
        / num_completion_tokens
    )

    ref_log_probs = F.log_softmax(
        ref_logits[:, :-1],
        dim=-1,
    )

    ref_probs = ref_log_probs.exp()

    token_kl = (
        ref_probs
        * (
            ref_log_probs
            - policy_log_probs
        )
    ).sum(dim=-1)

    kl_sum = (
        token_kl
        * shift_mask
    ).sum(dim=-1)

    return (
        sequence_logp,
        token_mean_logp,
        kl_sum,
        num_completion_tokens,
    )


def rank_weights(
    rewards,
    alpha,
    positive_fraction,
):
    """
    Construct per-sample PMPO likelihood coefficients.

    For K=4 and positive_fraction=0.5

        top 2     -> -alpha / 2
        bottom 2  -> +(1-alpha) / 2
    """
    B, K = rewards.shape

    num_positive = int(
        round(K * positive_fraction)
    )

    num_positive = max(
        1,
        min(
            K - 1,
            num_positive,
        ),
    )

    num_negative = K - num_positive

    if num_positive != num_negative:
        raise ValueError(
            "This implementation assumes symmetric top/bottom groups. "
            "Use positive_fraction=0.5."
        )

    order = rewards.argsort(
        dim=1,
        descending=True,
    )

    positive_idx = order[
        :,
        :num_positive,
    ]

    negative_idx = order[
        :,
        -num_negative:,
    ]

    weights = torch.zeros_like(
        rewards,
        dtype=torch.float32,
    )

    weights.scatter_(
        1,
        positive_idx,
        -alpha / num_positive,
    )

    weights.scatter_(
        1,
        negative_idx,
        (1.0 - alpha) / num_negative,
    )

    return (
        weights,
        positive_idx,
        negative_idx,
    )


def refresh_reference(
    accelerator,
    model,
    ref_model,
):
    """
    Reference policy <- current trainable policy.
    """
    accelerator.wait_for_everyone()

    unwrapped_model = (
        accelerator.unwrap_model(model)
    )

    unwrapped_ref = (
        accelerator.unwrap_model(ref_model)
    )

    unwrapped_ref.load_state_dict(
        unwrapped_model.state_dict(),
        strict=True,
    )

    unwrapped_ref.eval()

    for parameter in unwrapped_ref.parameters():
        parameter.requires_grad_(False)

    accelerator.wait_for_everyone()


def save_checkpoint(
    accelerator,
    model,
    tokenizer,
    output_dir,
    step,
):
    accelerator.wait_for_everyone()

    path = (
        Path(output_dir)
        / f"checkpoint-{step}"
    )

    if accelerator.is_main_process:
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        unwrapped_model = (
            accelerator.unwrap_model(model)
        )

        unwrapped_model.save_pretrained(
            path,
            safe_serialization=True,
        )

        tokenizer.save_pretrained(path)

        print(
            f"Saved checkpoint {path}"
        )


@torch.inference_mode()
def evaluate(
    model,
    tokenizer,
    reward_model,
    reward_tokenizer,
    eval_dataset,
    num_rollouts,
    generation_batch_size,
    max_prompt_tokens,
    max_new_tokens,
    max_reward_tokens,
    reward_microbatch_size,
    temperature,
    top_p,
):
    """
    Generate fresh responses on held-out prompts and score them.
    """
    was_training = model.training

    messages = [
        example["messages"]
        for example in eval_dataset
    ]

    (
        _completion_ids,
        completion_texts,
    ) = generate_rollouts(
        sampling_model=model,
        tokenizer=tokenizer,
        batch_messages=messages,
        num_rollouts=num_rollouts,
        generation_batch_size=generation_batch_size,
        max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    rewards = score_rollouts(
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        batch_messages=messages,
        completion_texts=completion_texts,
        max_reward_tokens=max_reward_tokens,
        reward_microbatch_size=reward_microbatch_size,
    )

    if was_training:
        model.train()

    return {
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std(
            unbiased=False
        ).item(),
        "reward_best": rewards.max(
            dim=1
        ).values.mean().item(),
        "reward_worst": rewards.min(
            dim=1
        ).values.mean().item(),
    }


def main():
    args = parse_args()

    set_seed(args.seed)

    if args.num_rollouts < 2:
        raise ValueError(
            "--num-rollouts must be >= 2"
        )

    if args.num_rollouts % 2 != 0:
        raise ValueError(
            "--num-rollouts must be even"
        )

    if abs(args.positive_fraction - 0.5) > 1e-8:
        raise ValueError(
            "This implementation expects --positive-fraction 0.5."
        )

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1"
        )

    if args.generation_batch_size < 1:
        raise ValueError(
            "--generation-batch-size must be >= 1"
        )

    if args.forward_microbatch_size < 1:
        raise ValueError(
            "--forward-microbatch-size must be >= 1"
        )

    if args.reward_microbatch_size < 1:
        raise ValueError(
            "--reward-microbatch-size must be >= 1"
        )

    if args.max_prompt_tokens < 1:
        raise ValueError(
            "--max-prompt-tokens must be >= 1"
        )

    if args.max_new_tokens < 1:
        raise ValueError(
            "--max-new-tokens must be >= 1"
        )

    if args.max_sequence_tokens < 2:
        raise ValueError(
            "--max-sequence-tokens must be >= 2"
        )

    accelerator = Accelerator(
        mixed_precision=(
            "bf16"
            if (
                torch.cuda.is_available()
                and torch.cuda.is_bf16_supported()
            )
            else "no"
        )
    )

    device = accelerator.device

    if accelerator.is_main_process:
        print("=" * 88)
        print("Online PMPO - LMSYS Chat-1M")
        print("=" * 88)
        print(
            f"Policy               {args.model}"
        )
        print(
            f"Reward model         {args.reward_model}"
        )
        print(
            f"Train examples       {args.num_examples}"
        )
        print(
            f"Eval examples        {args.eval_examples}"
        )
        print(
            f"Prompt batch         {args.batch_size}"
        )
        print(
            f"Rollouts per prompt  {args.num_rollouts}"
        )
        print(
            f"Generation batch     {args.generation_batch_size}"
        )
        print(
            f"Forward microbatch   {args.forward_microbatch_size}"
        )
        print(
            f"Alpha                {args.alpha}"
        )
        print(
            f"Beta                 {args.beta}"
        )
        print(
            f"Learning rate        {args.learning_rate}"
        )
        print(
            f"Max grad norm        {args.max_grad_norm}"
        )
        print(
            f"Prompt tokens        {args.max_prompt_tokens}"
        )
        print(
            f"New tokens           {args.max_new_tokens}"
        )
        print(
            f"Sequence tokens      {args.max_sequence_tokens}"
        )
        print(
            f"Reference refresh    {args.ref_update_steps}"
        )
        print(
            f"Device               {device}"
        )
        print()

    # ------------------------------------------------------------------
    # Tokenizers
    # ------------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        args.model
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    reward_tokenizer = AutoTokenizer.from_pretrained(
        args.reward_model
    )

    if reward_tokenizer.pad_token_id is None:
        reward_tokenizer.pad_token = (
            reward_tokenizer.eos_token
        )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    total_needed = (
        args.num_examples
        + args.eval_examples
    )

    dataset = collect_lmsys_prompts(
        dataset_name=args.dataset,
        num_examples=total_needed,
        seed=args.seed,
    )

    train_dataset = dataset.select(
        range(args.num_examples)
    )

    eval_start = args.num_examples
    eval_end = total_needed

    eval_dataset = dataset.select(
        range(eval_start, eval_end)
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: batch,
        drop_last=False,
    )

    if accelerator.is_main_process:
        print(
            f"Train rows  {len(train_dataset)}"
        )
        print(
            f"Eval rows   {len(eval_dataset)}"
        )
        print()

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    model_dtype = (
        torch.bfloat16
        if (
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )
        else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=model_dtype,
    )

    model.config.use_cache = False

    reward_model = (
        AutoModelForSequenceClassification.from_pretrained(
            args.reward_model,
            dtype=model_dtype,
        )
    )

    reward_model.eval()

    for parameter in reward_model.parameters():
        parameter.requires_grad_(False)

    ref_model = copy.deepcopy(model)

    ref_model.eval()

    for parameter in ref_model.parameters():
        parameter.requires_grad_(False)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    (
        model,
        ref_model,
        optimizer,
        reward_model,
        dataloader,
    ) = accelerator.prepare(
        model,
        ref_model,
        optimizer,
        reward_model,
        dataloader,
    )

    model.train()
    ref_model.eval()
    reward_model.eval()

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------

    wb_run = None

    if (
        accelerator.is_main_process
        and not args.no_wandb
    ):
        if wandb is None:
            raise RuntimeError(
                "W&B requested but wandb is not installed."
            )

        run_name = args.wandb_run_name

        if run_name is None:
            run_name = (
                f"qwen05b-pmpo-k{args.num_rollouts}"
                f"-b{args.batch_size}"
            )

        wb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            tags=[
                "pmpo",
                "online",
                "lmsys",
                "qwen2-0.5b",
            ],
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    global_step = 0

    total_optimizer_steps = (
        args.max_steps
        if args.max_steps > 0
        else args.epochs
        * math.ceil(
            len(train_dataset)
            / args.batch_size
        )
    )

    model.train()

    for epoch in range(args.epochs):
        if global_step >= total_optimizer_steps:
            break

        for raw_batch in dataloader:
            if global_step >= total_optimizer_steps:
                break

            B = len(raw_batch)

            batch_messages = [
                example["messages"]
                for example in raw_batch
            ]

            # ----------------------------------------------------------
            # 1. Online sampling from current reference policy
            # ----------------------------------------------------------

            (
                completion_ids_groups,
                completion_texts,
            ) = generate_rollouts(
                sampling_model=ref_model,
                tokenizer=tokenizer,
                batch_messages=batch_messages,
                num_rollouts=args.num_rollouts,
                generation_batch_size=args.generation_batch_size,
                max_prompt_tokens=args.max_prompt_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            # ----------------------------------------------------------
            # 2. Reward evaluation
            # ----------------------------------------------------------

            rewards_cpu = score_rollouts(
                reward_model=reward_model,
                reward_tokenizer=reward_tokenizer,
                batch_messages=batch_messages,
                completion_texts=completion_texts,
                max_reward_tokens=args.max_reward_tokens,
                reward_microbatch_size=args.reward_microbatch_size,
            )

            rewards = rewards_cpu.to(
                device=device,
                dtype=torch.float32,
            )

            K = args.num_rollouts

            # ----------------------------------------------------------
            # 3. Rank responses and construct PMPO coefficients
            # ----------------------------------------------------------

            (
                sample_weights,
                positive_idx,
                negative_idx,
            ) = rank_weights(
                rewards=rewards,
                alpha=args.alpha,
                positive_fraction=args.positive_fraction,
            )

            flat_sample_weights = (
                sample_weights.reshape(-1)
            )

            # The KL contribution is averaged over all B*K responses.
            kl_coefficient = (
                args.beta
                / (B * K)
            )

            # ----------------------------------------------------------
            # 4. Tokenize prompt + completion
            # ----------------------------------------------------------

            (
                input_ids,
                attention_mask,
                completion_mask,
            ) = encode_training_batch(
                tokenizer=tokenizer,
                batch_messages=batch_messages,
                completion_ids_groups=completion_ids_groups,
                max_prompt_tokens=args.max_prompt_tokens,
                max_sequence_tokens=args.max_sequence_tokens,
            )

            N = input_ids.shape[0]

            model.train()

            optimizer.zero_grad(
                set_to_none=True
            )

            positive_logp_values = []
            negative_logp_values = []

            positive_token_logp_values = []
            negative_token_logp_values = []

            kl_sum_values = []
            kl_per_token_values = []
            token_counts = []

            # ----------------------------------------------------------
            # 5. PMPO M-step
            # ----------------------------------------------------------

            for start in range(
                0,
                N,
                args.forward_microbatch_size,
            ):
                end = min(
                    start
                    + args.forward_microbatch_size,
                    N,
                )

                ids = input_ids[
                    start:end
                ].to(
                    device,
                    non_blocking=True,
                )

                mask = attention_mask[
                    start:end
                ].to(
                    device,
                    non_blocking=True,
                )

                completion = completion_mask[
                    start:end
                ].to(
                    device,
                    non_blocking=True,
                )

                policy_out = model(
                    input_ids=ids,
                    attention_mask=mask,
                    use_cache=False,
                )

                with torch.no_grad():
                    ref_out = ref_model(
                        input_ids=ids,
                        attention_mask=mask,
                        use_cache=False,
                    )

                (
                    sequence_logp,
                    token_mean_logp,
                    kl_sum,
                    num_tokens,
                ) = compute_sequence_logp_and_kl(
                    policy_logits=policy_out.logits,
                    ref_logits=ref_out.logits,
                    input_ids=ids,
                    completion_mask=completion,
                )

                weights = flat_sample_weights[
                    start:end
                ]

                # ------------------------------------------------------
                # Correctly normalized PMPO objective
                #
                # Divide the likelihood contribution by B because the
                # PMPO objective is an average over conditioning prompts.
                # The KL contribution is already normalized by B*K.
                # ------------------------------------------------------

                likelihood_loss = (
                    weights
                    * sequence_logp
                ).sum() / B

                kl_loss = (
                    kl_coefficient
                    * kl_sum.sum()
                )

                micro_loss = (
                    likelihood_loss
                    + kl_loss
                )

                accelerator.backward(
                    micro_loss
                )

                # Diagnostics
                with torch.no_grad():
                    is_positive = (
                        weights < 0
                    )

                    is_negative = (
                        weights > 0
                    )

                    if is_positive.any():
                        positive_logp_values.append(
                            sequence_logp[
                                is_positive
                            ].detach()
                        )

                        positive_token_logp_values.append(
                            token_mean_logp[
                                is_positive
                            ].detach()
                        )

                    if is_negative.any():
                        negative_logp_values.append(
                            sequence_logp[
                                is_negative
                            ].detach()
                        )

                        negative_token_logp_values.append(
                            token_mean_logp[
                                is_negative
                            ].detach()
                        )

                    kl_sum_values.append(
                        kl_sum.detach()
                    )

                    kl_per_token_values.append(
                        (
                            kl_sum
                            / num_tokens
                        ).detach()
                    )

                    token_counts.append(
                        num_tokens.detach()
                    )

                del ids
                del mask
                del completion
                del policy_out
                del ref_out
                del sequence_logp
                del token_mean_logp
                del kl_sum
                del num_tokens
                del micro_loss

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # ----------------------------------------------------------
            # 6. Clip gradients and update
            # ----------------------------------------------------------

            grad_norm = accelerator.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )

            grad_norm_value = float(
                grad_norm
            )

            if (
                args.max_grad_norm > 0
                and math.isfinite(grad_norm_value)
            ):
                clip_scale = min(
                    1.0,
                    args.max_grad_norm
                    / max(
                        grad_norm_value,
                        1e-12,
                    ),
                )
            else:
                clip_scale = 1.0

            optimizer.step()

            global_step += 1

            # ----------------------------------------------------------
            # 7. Metrics
            # ----------------------------------------------------------

            positive_logp = torch.cat(
                positive_logp_values
            ).mean()

            negative_logp = torch.cat(
                negative_logp_values
            ).mean()

            positive_token_logp = torch.cat(
                positive_token_logp_values
            ).mean()

            negative_token_logp = torch.cat(
                negative_token_logp_values
            ).mean()

            mean_kl_sum = torch.cat(
                kl_sum_values
            ).mean()

            mean_kl_per_token = torch.cat(
                kl_per_token_values
            ).mean()

            mean_tokens = torch.cat(
                token_counts
            ).mean()

            positive_rewards = torch.gather(
                rewards,
                1,
                positive_idx,
            ).mean()

            negative_rewards = torch.gather(
                rewards,
                1,
                negative_idx,
            ).mean()

            reward_mean = rewards.mean()

            reward_std = rewards.std(
                unbiased=False
            )

            reward_margin = (
                positive_rewards
                - negative_rewards
            )

            policy_margin = (
                positive_token_logp
                - negative_token_logp
            )

            diagnostic_loss = (
                -args.alpha
                * positive_logp
                + (1.0 - args.alpha)
                * negative_logp
                + args.beta
                * mean_kl_sum
            )

            if (
                accelerator.is_main_process
                and global_step % args.log_every == 0
            ):
                print(
                    f"step {global_step:5d} | "
                    f"loss {diagnostic_loss.item():9.4f} | "
                    f"pos_logp {positive_logp.item():9.2f} | "
                    f"neg_logp {negative_logp.item():9.2f} | "
                    f"pos_tok {positive_token_logp.item():8.4f} | "
                    f"neg_tok {negative_token_logp.item():8.4f} | "
                    f"KL/token {mean_kl_per_token.item():8.5f} | "
                    f"KL/sum {mean_kl_sum.item():8.4f} | "
                    f"reward {reward_mean.item():8.3f} | "
                    f"margin {reward_margin.item():7.3f} | "
                    f"pol_margin {policy_margin.item():8.4f} | "
                    f"grad_pre {grad_norm_value:8.3f} | "
                    f"clip_scale {clip_scale:7.5f} | "
                    f"tokens {mean_tokens.item():6.1f}"
                )

            if wb_run is not None:
                wb_run.log(
                    {
                        "train/step": global_step,
                        "train/loss": diagnostic_loss.item(),
                        "train/positive_logp": positive_logp.item(),
                        "train/negative_logp": negative_logp.item(),
                        "train/positive_logp_per_token": (
                            positive_token_logp.item()
                        ),
                        "train/negative_logp_per_token": (
                            negative_token_logp.item()
                        ),
                        "train/policy_preference_margin_per_token": (
                            policy_margin.item()
                        ),
                        "train/kl_sum": mean_kl_sum.item(),
                        "train/kl_per_token": (
                            mean_kl_per_token.item()
                        ),
                        "train/grad_norm_pre_clip": (
                            grad_norm_value
                        ),
                        "train/grad_clip_scale": (
                            clip_scale
                        ),
                        "train/learning_rate": (
                            optimizer.param_groups[0]["lr"]
                        ),
                        "train/mean_completion_tokens": (
                            mean_tokens.item()
                        ),
                        "reward/mean": (
                            reward_mean.item()
                        ),
                        "reward/std": (
                            reward_std.item()
                        ),
                        "reward/positive": (
                            positive_rewards.item()
                        ),
                        "reward/negative": (
                            negative_rewards.item()
                        ),
                        "reward/margin": (
                            reward_margin.item()
                        ),
                        "rollout/num_rollouts": K,
                        "rollout/responses_per_step": B * K,
                    },
                    step=global_step,
                )

            # ----------------------------------------------------------
            # 8. Refresh reference
            # ----------------------------------------------------------

            if (
                args.ref_update_steps > 0
                and global_step
                % args.ref_update_steps
                == 0
            ):
                refresh_reference(
                    accelerator=accelerator,
                    model=model,
                    ref_model=ref_model,
                )

                if accelerator.is_main_process:
                    print(
                        f"  reference refreshed at step {global_step}"
                    )

            # ----------------------------------------------------------
            # 9. Held-out evaluation
            # ----------------------------------------------------------

            if (
                args.eval_steps > 0
                and global_step
                % args.eval_steps
                == 0
            ):
                eval_metrics = evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    reward_model=reward_model,
                    reward_tokenizer=reward_tokenizer,
                    eval_dataset=eval_dataset,
                    num_rollouts=args.eval_rollouts,
                    generation_batch_size=args.generation_batch_size,
                    max_prompt_tokens=args.max_prompt_tokens,
                    max_new_tokens=args.max_new_tokens,
                    max_reward_tokens=args.max_reward_tokens,
                    reward_microbatch_size=args.reward_microbatch_size,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

                if accelerator.is_main_process:
                    print(
                        f"  eval | "
                        f"reward {eval_metrics['reward_mean']:.4f} | "
                        f"std {eval_metrics['reward_std']:.4f} | "
                        f"best {eval_metrics['reward_best']:.4f} | "
                        f"worst {eval_metrics['reward_worst']:.4f}"
                    )

                if wb_run is not None:
                    wb_run.log(
                        {
                            "eval/reward_mean": (
                                eval_metrics["reward_mean"]
                            ),
                            "eval/reward_std": (
                                eval_metrics["reward_std"]
                            ),
                            "eval/reward_best": (
                                eval_metrics["reward_best"]
                            ),
                            "eval/reward_worst": (
                                eval_metrics["reward_worst"]
                            ),
                            "eval/step": global_step,
                        },
                        step=global_step,
                    )

                model.train()

            # ----------------------------------------------------------
            # 10. Checkpoint
            # ----------------------------------------------------------

            if (
                args.save_steps > 0
                and global_step
                % args.save_steps
                == 0
            ):
                save_checkpoint(
                    accelerator=accelerator,
                    model=model,
                    tokenizer=tokenizer,
                    output_dir=args.output_dir,
                    step=global_step,
                )

    # Final save
    save_checkpoint(
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        step=global_step,
    )

    if wb_run is not None:
        wb_run.finish()


if __name__ == "__main__":
    main()
