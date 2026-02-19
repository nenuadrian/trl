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


def _content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text is not None:
                    chunks.append(str(text))
            elif isinstance(item, str):
                chunks.append(item)
            else:
                chunks.append(str(item))
        return "".join(chunks)
    return str(content)


def prompt_to_text(prompt, tokenizer):
    if isinstance(prompt, str):
        return prompt

    if isinstance(prompt, list):
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        lines = []
        for msg in prompt:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = _content_to_text(msg.get("content", ""))
                lines.append(f"{role}: {content}" if role else content)
            else:
                lines.append(str(msg))
        return "\n".join(lines)

    if isinstance(prompt, dict):
        if isinstance(prompt.get("prompt"), str):
            return prompt["prompt"]
        if isinstance(prompt.get("messages"), list):
            return prompt_to_text(prompt["messages"], tokenizer)
        return str(prompt)

    return str(prompt)


@torch.inference_mode()
def generate_completions(model, tokenizer, prompts, device, max_new_tokens, batch_size):
    out_text = []
    for batch_prompts in batched(prompts, batch_size):
        batch_prompts = [prompt_to_text(p, tokenizer) for p in batch_prompts]
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
    prompt_texts = [prompt_to_text(p, reward_tokenizer) for p in prompts]
    texts = [p + c for p, c in zip(prompt_texts, completions, strict=True)]
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


def _bootstrap_ci_from_samples(samples: np.ndarray, ci_level: float):
    alpha = (1.0 - ci_level) / 2.0
    lo = float(np.quantile(samples, alpha))
    hi = float(np.quantile(samples, 1.0 - alpha))
    return lo, hi


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, ci_level: float, rng):
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = values[idx].mean(axis=1)
    return _bootstrap_ci_from_samples(stats, ci_level)


def bootstrap_pairwise_ci(a: np.ndarray, b: np.ndarray, n_boot: int, ci_level: float, rng):
    if len(a) != len(b):
        raise ValueError("Pairwise bootstrap requires arrays with equal length.")
    n = len(a)
    if n == 0:
        return (float("nan"), float("nan")), (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(n_boot, n))
    a_bs = a[idx]
    b_bs = b[idx]
    diff = a_bs - b_bs
    margin_stats = diff.mean(axis=1)
    win_stats = (diff > 0).mean(axis=1)
    return (
        _bootstrap_ci_from_samples(win_stats, ci_level),
        _bootstrap_ci_from_samples(margin_stats, ci_level),
    )


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
    p.add_argument("--bootstrap_iters", type=int, default=2000)
    p.add_argument("--ci_level", type=float, default=0.95)
    args = p.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if not (0 < args.ci_level < 1):
        raise ValueError("--ci_level must be between 0 and 1.")
    if args.bootstrap_iters < 0:
        raise ValueError("--bootstrap_iters must be >= 0.")

    ckpts = [parse_ckpt_arg(x) for x in args.ckpt]
    if len(ckpts) < 2:
        raise ValueError("Provide at least 2 --ckpt entries.")

    from trl import maybe_extract_prompt

    ds = load_dataset(args.dataset_name, name=args.dataset_config, split=args.split)

    if args.prompt_column in ds.column_names:
        prompts = ds[args.prompt_column][: args.max_samples]
    elif "chosen" in ds.column_names and "rejected" in ds.column_names:
        ds = ds.map(maybe_extract_prompt, desc="Extracting prompts for eval")
        prompts = ds["prompt"][: args.max_samples]
    else:
        raise ValueError(f"No prompt-like column. Columns: {ds.column_names}")

    print(f"Evaluating on {len(prompts)} prompts")

    if len(prompts) == 0:
        raise ValueError("No prompts found.")

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
        # Decoder-only generation should use left padding.
        tok.padding_side = "left"

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
        if args.bootstrap_iters > 0:
            ci_lo, ci_hi = bootstrap_mean_ci(
                r.rewards, args.bootstrap_iters, args.ci_level, rng
            )
            ci_text = f" | mean_ci=[{ci_lo: .4f}, {ci_hi: .4f}]"
        else:
            ci_text = ""
        print(
            f"{r.name:>12} | mean={r.rewards.mean(): .4f} | std={r.rewards.std(): .4f} | "
            f"min={r.rewards.min(): .4f} | max={r.rewards.max(): .4f}{ci_text}"
        )

    print("\n=== Pairwise win rates (reward-based) ===")
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a, b = results[i], results[j]
            win, tie, margin = pairwise_stats(a.rewards, b.rewards)
            if args.bootstrap_iters > 0:
                win_ci, margin_ci = bootstrap_pairwise_ci(
                    a.rewards, b.rewards, args.bootstrap_iters, args.ci_level, rng
                )
                ci_text = (
                    f" | win_ci=[{win_ci[0]:.3f}, {win_ci[1]:.3f}]"
                    f" | margin_ci=[{margin_ci[0]:.4f}, {margin_ci[1]:.4f}]"
                )
            else:
                ci_text = ""
            print(
                f"{a.name:>12} vs {b.name:<12} | win_rate={win:.3f} | tie_rate={tie:.3f} | mean_margin={margin:.4f}{ci_text}"
            )


if __name__ == "__main__":
    main()
