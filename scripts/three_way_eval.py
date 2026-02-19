#!/usr/bin/env python3
import argparse
from dataclasses import dataclass

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)


@dataclass
class ModelResult:
    name: str
    rewards: np.ndarray
    completions: list[str]


def parse_ckpt_arg(item: str) -> tuple[str, str]:
    if "=" not in item:
        raise ValueError(f"Invalid --ckpt '{item}', expected name=path")
    name, path = item.split("=", 1)
    return name.strip(), path.strip()


def batched(xs, bs):
    for i in range(0, len(xs), bs):
        yield xs[i : i + bs]


@torch.inference_mode()
def generate_completions(model, tokenizer, prompts, device, max_new_tokens, batch_size):
    out_text = []
    for batch_prompts in batched(prompts, batch_size):
        tok = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        gen = model.generate(
            **tok,
            do_sample=False,  # deterministic for fair comparison
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        comp_ids = gen[:, tok["input_ids"].shape[1] :]
        out_text.extend(tokenizer.batch_decode(comp_ids, skip_special_tokens=True))
    return out_text


@torch.inference_mode()
def score_with_reward_model(
    reward_model, reward_tokenizer, prompts, completions, device, batch_size
):
    texts = [p + c for p, c in zip(prompts, completions, strict=True)]
    scores = []
    for batch_texts in batched(texts, batch_size):
        tok = reward_tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        logits = reward_model(**tok).logits[:, 0]
        scores.extend(logits.detach().float().cpu().numpy().tolist())
    return np.array(scores, dtype=np.float32)


def pairwise_stats(a: np.ndarray, b: np.ndarray):
    diff = a - b
    wins = float((diff > 0).mean())
    ties = float((diff == 0).mean())
    margin = float(diff.mean())
    return wins, ties, margin


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        action="append",
        required=True,
        help="name=checkpoint_path (repeat 3x)",
    )
    p.add_argument("--reward_model", required=True)
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--dataset_config", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--prompt_column", default="prompt")
    p.add_argument("--max_samples", type=int, default=128)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    ckpts = [parse_ckpt_arg(x) for x in args.ckpt]
    if len(ckpts) < 2:
        raise ValueError("Provide at least 2 --ckpt entries.")

    ds = load_dataset(args.dataset_name, name=args.dataset_config, split=args.split)
    prompts = ds[args.prompt_column][: args.max_samples]
    if len(prompts) == 0:
        raise ValueError("No prompts found.")
    if not isinstance(prompts[0], str):
        raise ValueError(
            "This minimal script expects string prompts (e.g., standard_prompt_only)."
        )

    reward_tok = AutoTokenizer.from_pretrained(args.reward_model)
    if reward_tok.pad_token_id is None:
        reward_tok.pad_token = reward_tok.eos_token
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model
    ).to(device)
    reward_model.eval()
    reward_model.config.pad_token_id = reward_tok.pad_token_id

    results = []
    for name, path in ckpts:
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(device)
        model.eval()

        comps = generate_completions(
            model=model,
            tokenizer=tok,
            prompts=prompts,
            device=device,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        rewards = score_with_reward_model(
            reward_model=reward_model,
            reward_tokenizer=reward_tok,
            prompts=prompts,
            completions=comps,
            device=device,
            batch_size=args.batch_size,
        )
        results.append(ModelResult(name=name, rewards=rewards, completions=comps))

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n=== Per-model reward stats ===")
    for r in results:
        print(
            f"{r.name:>12} | mean={r.rewards.mean(): .4f} | std={r.rewards.std(): .4f} | "
            f"min={r.rewards.min(): .4f} | max={r.rewards.max(): .4f}"
        )

    print("\n=== Pairwise win rates (reward-based) ===")
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a, b = results[i], results[j]
            win, tie, margin = pairwise_stats(a.rewards, b.rewards)
            print(
                f"{a.name:>12} vs {b.name:<12} | win_rate={win:.3f} | tie_rate={tie:.3f} | mean_margin={margin:.4f}"
            )


if __name__ == "__main__":
    main()
