#!/usr/bin/env python3
"""
train_pmpo_lmsys.py

A single-file online PMPO experiment for Qwen + LMSYS-Chat-1M.

Pipeline:
    LMSYS prompt
        -> sample K responses from the current reference policy
        -> score responses with a reward model
        -> top K/2 = preferred, bottom K/2 = dis-preferred
        -> PMPO update
        -> every ref_update_steps, refresh the reference policy

This follows the core setup described in:
    Abdolmaleki et al., "Preference Optimization as Probabilistic Inference"
    ICLR 2025, arXiv:2410.04166

The paper's language experiment uses 4 generations per prompt, ranks them,
and uses the top two as preferred and bottom two as dis-preferred.

This script is intentionally standalone rather than using TRL's
OnlineDPOTrainer, because OnlineDPOTrainer is pairwise while PMPO is set-wise.
"""

import argparse
import copy
import os
from pathlib import Path

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


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model",
        default="Qwen/Qwen2-0.5B-Instruct",
        help="Policy model. Qwen2-0.5B-Instruct matches TRL's current online-DPO example.",
    )
    p.add_argument(
        "--reward-model",
        default="trl-lib/Qwen2-0.5B-Reward",
    )
    p.add_argument(
        "--dataset",
        default="lmsys/lmsys-chat-1m",
    )
    p.add_argument("--num-examples", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=-1)

    p.add_argument(
        "--num-rollouts",
        type=int,
        default=4,
        help="Number of responses sampled per prompt. PMPO paper uses 4.",
    )
    p.add_argument(
        "--positive-fraction",
        type=float,
        default=0.5,
        help="Fraction of rollouts treated as preferred. 0.5 gives top-half/bottom-half.",
    )

    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="PMPO KL coefficient. This is PMPO beta, not DPO beta.",
    )
    p.add_argument("--learning-rate", type=float, default=5e-7)
    p.add_argument("--weight-decay", type=float, default=0.0)

    p.add_argument("--max-prompt-tokens", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-sequence-tokens", type=int, default=1400)
    p.add_argument("--max-reward-tokens", type=int, default=1400)

    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)

    p.add_argument(
        "--ref-update-steps",
        type=int,
        default=100,
        help="Refresh reference/sampling policy every N optimizer steps.",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument(
        "--output-dir",
        default="./qwen-pmpo-lmsys",
    )

    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_lmsys_prompts(dataset_name: str, num_examples: int) -> Dataset:
    """
    Stream the gated LMSYS dataset so a 1M-row download is not required just
    to obtain a small prototype dataset.

    We keep the full conversation context up to the final user turn.
    The existing assistant answer after that user turn is deliberately ignored.
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

        # Basic sanity check.
        if not messages or messages[-1].get("role") != "user":
            continue

        records.append({"messages": messages})

        if len(records) >= num_examples:
            break

    if len(records) < num_examples:
        raise RuntimeError(
            f"Only collected {len(records)} usable English prompts, "
            f"requested {num_examples}."
        )

    return Dataset.from_list(records)


def render_policy_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def generate_rollouts(
    sampling_model,
    tokenizer,
    batch_messages,
    num_rollouts,
    max_prompt_tokens,
    max_new_tokens,
    temperature,
    top_p,
):
    """
    Sample K responses per prompt.

    Returned structure:
        completions[B][K]
    """
    prompts = [
        render_policy_prompt(tokenizer, messages)
        for messages in batch_messages
    ]

    was_training = sampling_model.training
    sampling_model.eval()

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_prompt_tokens,
        return_tensors="pt",
    )

    device = next(sampling_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = sampling_model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_rollouts,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # generate() appends generated tokens after the padded input width.
    prompt_width = inputs["input_ids"].shape[1]
    completion_ids = outputs[:, prompt_width:]

    decoded = tokenizer.batch_decode(
        completion_ids,
        skip_special_tokens=True,
    )

    tokenizer.padding_side = old_padding_side

    if was_training:
        sampling_model.train()

    B = len(batch_messages)
    K = num_rollouts

    return [
        decoded[i * K : (i + 1) * K]
        for i in range(B)
    ]


def reward_text(reward_tokenizer, messages, completion):
    """
    Render prompt + generated assistant response for the sequence classifier.
    """
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
        # Fallback. This keeps the script usable with reward tokenizers that
        # do not expose a compatible chat template.
        prompt_text = ""
        for m in messages:
            prompt_text += f"{m['role']}: {m['content']}\n"
        prompt_text += f"assistant: {completion}"
        return prompt_text


@torch.inference_mode()
def score_rollouts(
    reward_model,
    reward_tokenizer,
    batch_messages,
    completions,
    max_reward_tokens,
):
    texts = []

    for messages, group in zip(batch_messages, completions):
        for completion in group:
            texts.append(
                reward_text(
                    reward_tokenizer,
                    messages,
                    completion,
                )
            )

    inputs = reward_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_reward_tokens,
        return_tensors="pt",
        add_special_tokens=False,
    )

    device = next(reward_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = reward_model(**inputs)

    if outputs.logits.ndim != 2 or outputs.logits.shape[-1] < 1:
        raise RuntimeError(
            f"Unexpected reward-model logits shape: {tuple(outputs.logits.shape)}"
        )

    rewards = outputs.logits[:, 0]

    B = len(batch_messages)
    K = len(completions[0])

    return rewards.view(B, K).float()


def build_training_batch(
    tokenizer,
    batch_messages,
    completions,
    max_sequence_tokens,
):
    """
    Build padded prompt+completion sequences plus a completion-token mask.

    The mask is 1 only on generated completion tokens. Prompt tokens do not
    contribute to either the likelihood terms or the KL term.
    """
    all_ids = []
    all_masks = []

    for messages, group in zip(batch_messages, completions):
        prompt_text = render_policy_prompt(tokenizer, messages)

        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]

        for completion in group:
            completion_ids = tokenizer(
                completion,
                add_special_tokens=False,
            )["input_ids"]

            # Ensure the generated response has a terminal EOS for training.
            if (
                tokenizer.eos_token_id is not None
                and (
                    not completion_ids
                    or completion_ids[-1] != tokenizer.eos_token_id
                )
            ):
                completion_ids.append(tokenizer.eos_token_id)

            # Preserve the completion. If the total exceeds the budget, remove
            # tokens from the left side of the prompt first.
            total_len = len(prompt_ids) + len(completion_ids)
            if total_len > max_sequence_tokens:
                prompt_trim = max(
                    0,
                    total_len - max_sequence_tokens,
                )
                prompt_ids_used = prompt_ids[prompt_trim:]
            else:
                prompt_ids_used = prompt_ids

            ids = prompt_ids_used + completion_ids
            mask = [0] * len(prompt_ids_used) + [1] * len(completion_ids)

            if len(ids) > max_sequence_tokens:
                # Can only happen if the completion itself is longer than the
                # sequence budget.
                overflow = len(ids) - max_sequence_tokens
                ids = ids[overflow:]
                mask = mask[overflow:]

            all_ids.append(ids)
            all_masks.append(mask)

    max_len = max(len(x) for x in all_ids)
    pad_id = tokenizer.pad_token_id

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

    for i, (ids, mask) in enumerate(zip(all_ids, all_masks)):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :n] = 1
        completion_mask[i, :n] = torch.tensor(mask, dtype=torch.long)

    return input_ids, attention_mask, completion_mask


def sequence_logprobs(
    logits,
    input_ids,
    completion_mask,
):
    """
    Sequence log pi_theta(y | x), summed over completion tokens.
    """
    shift_logits = logits[:, :-1, :]
    shift_targets = input_ids[:, 1:]
    shift_mask = completion_mask[:, 1:].float()

    log_probs = F.log_softmax(
        shift_logits,
        dim=-1,
    )

    token_log_probs = log_probs.gather(
        dim=-1,
        index=shift_targets.unsqueeze(-1),
    ).squeeze(-1)

    return (
        token_log_probs * shift_mask
    ).sum(dim=-1)


def closed_form_token_kl(
    ref_logits,
    policy_logits,
    completion_mask,
):
    """
    Estimate KL(pi_ref || pi_theta) using the paper's closed-form
    autoregressive categorical KL.

    For each completion timestep:
        KL(p_ref || p_theta)
        = sum_v p_ref(v) [log p_ref(v) - log p_theta(v)]

    The sequence KL is the sum across completion timesteps.
    """
    shift_ref = ref_logits[:, :-1, :]
    shift_policy = policy_logits[:, :-1, :]
    shift_mask = completion_mask[:, 1:].float()

    ref_log_probs = F.log_softmax(
        shift_ref,
        dim=-1,
    )
    policy_log_probs = F.log_softmax(
        shift_policy,
        dim=-1,
    )

    ref_probs = ref_log_probs.exp()

    token_kl = (
        ref_probs
        * (ref_log_probs - policy_log_probs)
    ).sum(dim=-1)

    return (
        token_kl * shift_mask
    ).sum(dim=-1)


def compute_pmpo_loss(
    policy_logits,
    ref_logits,
    input_ids,
    completion_mask,
    rewards,
    alpha,
    beta,
    positive_fraction,
):
    """
    PMPO loss for B prompts with K rollouts per prompt.

    Minimized objective:

        -alpha E_{D+}[log pi_theta(y|x)]
        +(1-alpha) E_{D-}[log pi_theta(y|x)]
        +beta KL(pi_ref || pi_theta)

    Responses are ranked by reward. By default top half are preferred and
    bottom half are dis-preferred.
    """
    B, K = rewards.shape

    seq_logp = sequence_logprobs(
        policy_logits,
        input_ids,
        completion_mask,
    ).view(B, K)

    kl = closed_form_token_kl(
        ref_logits,
        policy_logits,
        completion_mask,
    ).view(B, K)

    num_positive = max(
        1,
        min(
            K - 1,
            int(round(K * positive_fraction)),
        ),
    )

    order = rewards.argsort(
        dim=1,
        descending=True,
    )

    positive_idx = order[:, :num_positive]
    negative_idx = order[:, -num_positive:]

    positive_logp = torch.gather(
        seq_logp,
        1,
        positive_idx,
    ).mean(dim=1)

    negative_logp = torch.gather(
        seq_logp,
        1,
        negative_idx,
    ).mean(dim=1)

    # Use the KL of all sampled contexts. Since the samples are generated
    # from the reference policy, this is the Monte Carlo state/context
    # estimator of the sequence-distribution KL used by PMPO.
    kl_term = kl.mean(dim=1)

    loss_per_prompt = (
        -alpha * positive_logp
        + (1.0 - alpha) * negative_logp
        + beta * kl_term
    )

    loss = loss_per_prompt.mean()

    with torch.no_grad():
        chosen_rewards = torch.gather(
            rewards,
            1,
            positive_idx,
        ).mean()

        rejected_rewards = torch.gather(
            rewards,
            1,
            negative_idx,
        ).mean()

        top1_rewards = rewards[:, 0:1]
        best_reward = rewards.max(dim=1).values.mean()
        worst_reward = rewards.min(dim=1).values.mean()

        reward_accuracy = (
            (
                chosen_rewards > rejected_rewards
            ).float()
        )

    metrics = {
        "loss": loss.detach(),
        "positive_logp": positive_logp.mean().detach(),
        "negative_logp": negative_logp.mean().detach(),
        "kl": kl_term.mean().detach(),
        "reward_mean": rewards.mean().detach(),
        "reward_std": rewards.std(unbiased=False).detach(),
        "reward_best": best_reward.detach(),
        "reward_worst": worst_reward.detach(),
        "positive_reward": chosen_rewards.detach(),
        "negative_reward": rejected_rewards.detach(),
        "reward_group_margin": (
            chosen_rewards - rejected_rewards
        ).detach(),
        "positive_fraction": torch.tensor(
            num_positive / K,
            device=loss.device,
        ),
        "dummy": top1_rewards.mean().detach() * 0.0,
        "group_reward_accuracy": reward_accuracy.detach(),
    }

    return loss, metrics


def update_reference(accelerator, model, ref_model):
    """
    Synchronize reference policy with the newly optimized policy.
    """
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_ref = accelerator.unwrap_model(ref_model)

    unwrapped_ref.load_state_dict(
        unwrapped_model.state_dict(),
        strict=True,
    )
    unwrapped_ref.eval()

    for p in unwrapped_ref.parameters():
        p.requires_grad_(False)


def save_model(accelerator, model, tokenizer, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator.wait_for_everyone()

    unwrapped_model = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(
            output_dir,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(output_dir)

        print(f"\nSaved model to: {output_dir}")


def main():
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(
        mixed_precision="bf16"
        if torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        else "no",
    )

    device = accelerator.device

    if accelerator.is_main_process:
        print("=" * 80)
        print("PMPO online LMSYS experiment")
        print("=" * 80)
        print(f"Policy:          {args.model}")
        print(f"Reward model:    {args.reward_model}")
        print(f"Examples:        {args.num_examples}")
        print(f"Batch size:      {args.batch_size}")
        print(f"Rollouts/prompt: {args.num_rollouts}")
        print(f"alpha:            {args.alpha}")
        print(f"beta:             {args.beta}")
        print(f"LR:               {args.learning_rate}")
        print(f"Ref update:       {args.ref_update_steps} steps")
        print(f"Device:           {device}")
        print()

    if args.num_rollouts < 2:
        raise ValueError("--num-rollouts must be >= 2")

    if args.num_rollouts % 2 != 0:
        raise ValueError(
            "--num-rollouts must be even for the default balanced PMPO split."
        )

    if not 0.0 < args.positive_fraction < 1.0:
        raise ValueError("--positive-fraction must be between 0 and 1.")

    # ------------------------------------------------------------------
    # Tokenizers
    # ------------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    reward_tokenizer = AutoTokenizer.from_pretrained(
        args.reward_model,
    )

    if reward_tokenizer.pad_token_id is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    dataset = load_lmsys_prompts(
        args.dataset,
        args.num_examples,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: batch,
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32,
    )

    # Freeze a separate reward model.
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model,
        num_labels=1,
        dtype=torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32,
    )
    reward_model.eval()

    for p in reward_model.parameters():
        p.requires_grad_(False)

    # PMPO reference/sampling model starts as the same policy.
    ref_model = copy.deepcopy(model)
    ref_model.eval()

    for p in ref_model.parameters():
        p.requires_grad_(False)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    model, ref_model, optimizer, reward_model, dataloader = accelerator.prepare(
        model,
        ref_model,
        optimizer,
        reward_model,
        dataloader,
    )

    model.train()
    ref_model.eval()
    reward_model.eval()

    global_step = 0

    max_steps = args.max_steps
    if max_steps <= 0:
        max_steps = args.epochs * len(dataloader)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    for epoch in range(args.epochs):
        if global_step >= max_steps:
            break

        for raw_batch in dataloader:
            if global_step >= max_steps:
                break

            batch_messages = [
                example["messages"]
                for example in raw_batch
            ]

            # ----------------------------------------------------------
            # 1. E-step-like sampling:
            #    sample from the CURRENT REFERENCE policy
            # ----------------------------------------------------------
            completions = generate_rollouts(
                sampling_model=ref_model,
                tokenizer=tokenizer,
                batch_messages=batch_messages,
                num_rollouts=args.num_rollouts,
                max_prompt_tokens=args.max_prompt_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            # ----------------------------------------------------------
            # 2. Evaluate online samples
            # ----------------------------------------------------------
            rewards = score_rollouts(
                reward_model=reward_model,
                reward_tokenizer=reward_tokenizer,
                batch_messages=batch_messages,
                completions=completions,
                max_reward_tokens=args.max_reward_tokens,
            )

            # ----------------------------------------------------------
            # 3. Construct PMPO training sequences
            # ----------------------------------------------------------
            input_ids, attention_mask, completion_mask = (
                build_training_batch(
                    tokenizer=tokenizer,
                    batch_messages=batch_messages,
                    completions=completions,
                    max_sequence_tokens=args.max_sequence_tokens,
                )
            )

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            completion_mask = completion_mask.to(device)
            rewards = rewards.to(device)

            # ----------------------------------------------------------
            # 4. Policy forward
            # ----------------------------------------------------------
            policy_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

            # ----------------------------------------------------------
            # 5. Reference forward
            # ----------------------------------------------------------
            with torch.no_grad():
                ref_out = ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )

            # ----------------------------------------------------------
            # 6. PMPO M-step objective
            # ----------------------------------------------------------
            loss, metrics = compute_pmpo_loss(
                policy_logits=policy_out.logits,
                ref_logits=ref_out.logits,
                input_ids=input_ids,
                completion_mask=completion_mask,
                rewards=rewards,
                alpha=args.alpha,
                beta=args.beta,
                positive_fraction=args.positive_fraction,
            )

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            # ----------------------------------------------------------
            # 7. Refresh reference policy periodically.
            # ----------------------------------------------------------
            if (
                args.ref_update_steps > 0
                and global_step % args.ref_update_steps == 0
            ):
                update_reference(
                    accelerator,
                    model,
                    ref_model,
                )
                accelerator.wait_for_everyone()

            # ----------------------------------------------------------
            # Logging
            # ----------------------------------------------------------
            if global_step % args.log_every == 0:
                gathered = accelerator.gather_for_metrics(
                    torch.stack(
                        [
                            metrics["loss"],
                            metrics["positive_logp"],
                            metrics["negative_logp"],
                            metrics["kl"],
                            metrics["reward_mean"],
                            metrics["reward_std"],
                            metrics["positive_reward"],
                            metrics["negative_reward"],
                            metrics["reward_group_margin"],
                        ]
                    )
                )

                mean_metrics = gathered.view(
                    -1,
                    9,
                ).mean(dim=0)

                if accelerator.is_main_process:
                    print(
                        f"step {global_step:5d} | "
                        f"loss {mean_metrics[0].item():8.4f} | "
                        f"KL {mean_metrics[3].item():8.4f} | "
                        f"reward {mean_metrics[4].item():8.4f} | "
                        f"pos {mean_metrics[6].item():8.4f} | "
                        f"neg {mean_metrics[7].item():8.4f} | "
                        f"margin {mean_metrics[8].item():8.4f}"
                    )

            # Free very large logits before the next generation.
            del policy_out, ref_out

    save_model(
        accelerator,
        model,
        tokenizer,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
