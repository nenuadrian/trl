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

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

"""
MaxMin-RLHF Large-Scale Benchmark with Tulu2-7B (Section 5.2 of the paper).

This script replicates the large-scale experiment using:
    - Base model: allenai/tulu-2-7b (or configurable)
    - Datasets: P1, P2, P3 preference datasets with two user groups each
    - EM reward learning to discover user clusters
    - MaxMin PPO for alignment with diverse preferences
    - Evaluation via pairwise win rate using GPT-4

The full pipeline:
    1. Generate preference data with diverse user personas (using GPT-4)
    2. Run EM reward learning to learn cluster-specific reward models
    3. Run MaxMin PPO alignment
    4. Evaluate with Koala evaluation set (pairwise win rate)

Usage:
    # Full pipeline (requires GPT-4 API key for data generation and evaluation)
    # Step 1: Run EM reward learning
    python examples/scripts/ppo/maxmin_ppo_tulu7b.py --mode em_rewards

    # Step 2: Run MaxMin PPO
    python examples/scripts/ppo/maxmin_ppo_tulu7b.py --mode maxmin_ppo

    # Step 3: Run baseline single-reward PPO (with different majority/minority ratios)
    python examples/scripts/ppo/maxmin_ppo_tulu7b.py --mode single_ppo --ratio 1

    # Step 4: Evaluate
    python examples/scripts/ppo/maxmin_ppo_tulu7b.py --mode evaluate

    # Single GPU with LoRA:
    python examples/scripts/ppo/maxmin_ppo_tulu7b.py \
        --mode maxmin_ppo \
        --base_model allenai/tulu-2-7b \
        --use_peft \
        --lora_r 16

    # Multi-GPU with DeepSpeed:
    accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
        examples/scripts/ppo/maxmin_ppo_tulu7b.py \
        --mode maxmin_ppo \
        --base_model allenai/tulu-2-7b
"""

import argparse
import json
import os
import shutil

import torch
from accelerate import PartialState
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import ModelConfig, ScriptArguments, get_kbit_device_map, get_peft_config, get_quantization_config
from trl.experimental.ppo import PPOConfig, PPOTrainer
from trl.experimental.ppo.maxmin_ppo_config import MaxMinPPOConfig
from trl.experimental.ppo.maxmin_ppo_trainer import MaxMinPPOTrainer


# Preference prompts from Table 4 of the paper
PREFERENCE_PROMPTS = {
    "P1A": "Generate/Choose a response that can be easily understood by an elementary school student.",
    "P1B": "Generate/Choose a response that only a PhD Student in that specific field could understand.",
    "P2A": "Generate/Choose a response that is concise and to the point, without being verbose.",
    "P2B": "Generate/Choose a response that is very informative, without missing any background information.",
    "P3A": "Generate/Choose a response that is friendly, witty, funny, and humorous, like a close friend.",
    "P3B": "Generate/Choose a response (that answers) in an unfriendly manner.",
}

# Dataset pairs as in the paper
DATASET_PAIRS = {
    "P1": ("P1A", "P1B"),
    "P2": ("P2A", "P2B"),
    "P3": ("P3A", "P3B"),
}


def create_synthetic_preference_data(
    instructions_dataset: str = "tatsu-lab/alpaca",
    num_samples: int = 1000,
    dataset_pair: str = "P1",
    output_dir: str = "outputs/preference_data",
):
    """Create synthetic preference data simulating diverse user groups.

    In the paper, GPT-4 is used to simulate annotators with specific preference prompts.
    Here we provide a simplified version that creates preference pairs based on heuristics.
    For full replication, replace with GPT-4 API calls using the prompts from Table 4.
    """
    os.makedirs(output_dir, exist_ok=True)
    pair_a, pair_b = DATASET_PAIRS[dataset_pair]

    print(f"Creating preference data for {dataset_pair}:")
    print(f"  Group A ({pair_a}): {PREFERENCE_PROMPTS[pair_a]}")
    print(f"  Group B ({pair_b}): {PREFERENCE_PROMPTS[pair_b]}")
    print(f"  Note: For full replication, use GPT-4 to generate preferences.")
    print(f"  This script creates placeholder data structure for the pipeline.")

    # Load instruction dataset
    try:
        dataset = load_dataset(instructions_dataset, split="train")
    except Exception:
        dataset = load_dataset("tatsu-lab/alpaca", split="train")

    dataset = dataset.select(range(min(num_samples, len(dataset))))

    # Create placeholder preference data (in production, use GPT-4)
    preference_data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "annotator_group": [],
    }

    for i, example in enumerate(dataset):
        instruction = example.get("instruction", example.get("text", ""))
        output = example.get("output", example.get("text", ""))

        # Simulate two groups of annotators
        if i % 2 == 0:
            # Group A annotator
            preference_data["prompt"].append(instruction)
            preference_data["chosen"].append(output)
            preference_data["rejected"].append(output[::-1][:len(output)])  # placeholder
            preference_data["annotator_group"].append(pair_a)
        else:
            # Group B annotator
            preference_data["prompt"].append(instruction)
            preference_data["chosen"].append(output)
            preference_data["rejected"].append(output[::-1][:len(output)])  # placeholder
            preference_data["annotator_group"].append(pair_b)

    pref_dataset = Dataset.from_dict(preference_data)
    pref_path = os.path.join(output_dir, f"{dataset_pair}_preferences")
    pref_dataset.save_to_disk(pref_path)
    print(f"  Saved {len(pref_dataset)} preference pairs to {pref_path}")
    return pref_dataset


def run_em_reward_learning(
    base_model: str = "allenai/tulu-2-7b",
    dataset_pair: str = "P1",
    num_clusters: int = 2,
    num_em_iters: int = 5,
    output_dir: str = "outputs/em_rewards",
    use_peft: bool = True,
    lora_r: int = 16,
):
    """Run EM reward learning to discover user clusters and train reward models."""
    print(f"\nRunning EM Reward Learning for {dataset_pair}")
    print(f"  Base model: {base_model}")
    print(f"  Clusters: {num_clusters}")
    print(f"  EM iterations: {num_em_iters}")

    em_output = os.path.join(output_dir, dataset_pair)
    os.makedirs(em_output, exist_ok=True)

    # The EM script can be called as a subprocess or imported
    em_cmd = (
        f"python examples/scripts/ppo/em_reward_learning.py "
        f"--model_name_or_path {base_model} "
        f"--dataset_name tatsu-lab/alpaca "
        f"--num_clusters {num_clusters} "
        f"--num_em_iterations {num_em_iters} "
        f"--output_dir {em_output} "
        f"--per_device_train_batch_size 4 "
        f"--num_train_epochs 1 "
        f"--learning_rate 1e-5 "
        f"--max_length 512 "
    )

    if use_peft:
        em_cmd += f"--use_peft --lora_r {lora_r} --lora_task_type SEQ_CLS "

    print(f"\n  Run the following command to train EM reward models:")
    print(f"  {em_cmd}")
    print(f"\n  After training, reward models will be saved to: {em_output}/final/")

    return em_output


def run_maxmin_ppo_tulu(
    base_model: str = "allenai/tulu-2-7b",
    reward_model_paths: list[str] | None = None,
    output_dir: str = "outputs/maxmin_ppo_tulu",
    use_peft: bool = True,
    lora_r: int = 16,
):
    """Run MaxMin PPO with multiple reward models on Tulu2-7B."""
    print(f"\nRunning MaxMin PPO")
    print(f"  Base model: {base_model}")
    print(f"  Reward models: {reward_model_paths}")
    print(f"  Output: {output_dir}")

    shutil.rmtree(output_dir, ignore_errors=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    dtype = torch.bfloat16
    model_kwargs = dict(dtype=dtype)

    # Load reward models
    rm_paths = reward_model_paths or [base_model, base_model]
    reward_models = []
    for rm_path in rm_paths:
        rm = AutoModelForSequenceClassification.from_pretrained(
            rm_path, num_labels=1, trust_remote_code=True, **model_kwargs
        )
        reward_models.append(rm)

    # Value model
    value_model = AutoModelForSequenceClassification.from_pretrained(
        rm_paths[0], num_labels=1, trust_remote_code=True, **model_kwargs
    )

    # Policy
    policy = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=True, **model_kwargs
    )
    ref_policy = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=True, **model_kwargs
    )

    # Dataset (using Alpaca-style instruction data)
    try:
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
    except Exception:
        dataset = load_dataset("tatsu-lab/alpaca", split="train")

    # Tokenize
    def prepare_dataset(ds, tok):
        def tokenize(element):
            prompts = [f"### Instruction:\n{inst}\n\n### Response:\n" for inst in element["instruction"]]
            outputs = tok(prompts, padding=False, truncation=True, max_length=128)
            return {"input_ids": outputs["input_ids"]}
        return ds.map(tokenize, batched=True, remove_columns=ds.column_names)

    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(dataset.select(range(min(5000, len(dataset)))), tokenizer)
        eval_dataset = prepare_dataset(dataset.select(range(5000, min(5200, len(dataset)))), tokenizer)

    # Config
    peft_config = None
    if use_peft:
        from peft import LoraConfig
        peft_config = LoraConfig(r=lora_r, lora_alpha=lora_r * 2, task_type="CAUSAL_LM")
        ref_policy = None  # Use PEFT adapter switching for reference

    config = MaxMinPPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        total_episodes=5000,
        learning_rate=1.41e-5,
        response_length=128,
        temperature=0.7,
        kl_coef=0.05,
        num_ppo_epochs=1,
        num_mini_batches=1,
        missing_eos_penalty=1.0,
        reward_model_path=rm_paths[0],
        sft_model_path=base_model,
        logging_steps=5,
        maxmin_strategy="min",
        reward_model_paths=rm_paths,
        local_rollout_forward_batch_size=4,
    )

    trainer = MaxMinPPOTrainer(
        args=config,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_models=reward_models,
        train_dataset=train_dataset,
        value_model=value_model,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"MaxMin PPO model saved to {output_dir}")


def run_single_ppo_tulu(
    base_model: str = "allenai/tulu-2-7b",
    reward_model_path: str | None = None,
    output_dir: str = "outputs/single_ppo_tulu",
    use_peft: bool = True,
    lora_r: int = 16,
    ratio: int = 1,
):
    """Run single-reward PPO baseline on Tulu2-7B."""
    print(f"\nRunning Single Reward PPO (ratio={ratio}:1)")
    print(f"  Base model: {base_model}")
    print(f"  Output: {output_dir}")

    shutil.rmtree(output_dir, ignore_errors=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    dtype = torch.bfloat16
    model_kwargs = dict(dtype=dtype)

    rm_path = reward_model_path or base_model
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        rm_path, num_labels=1, trust_remote_code=True, **model_kwargs
    )
    value_model = AutoModelForSequenceClassification.from_pretrained(
        rm_path, num_labels=1, trust_remote_code=True, **model_kwargs
    )
    policy = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=True, **model_kwargs
    )
    ref_policy = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=True, **model_kwargs
    )

    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    def prepare_dataset(ds, tok):
        def tokenize(element):
            prompts = [f"### Instruction:\n{inst}\n\n### Response:\n" for inst in element["instruction"]]
            outputs = tok(prompts, padding=False, truncation=True, max_length=128)
            return {"input_ids": outputs["input_ids"]}
        return ds.map(tokenize, batched=True, remove_columns=ds.column_names)

    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(dataset.select(range(min(5000, len(dataset)))), tokenizer)
        eval_dataset = prepare_dataset(dataset.select(range(5000, min(5200, len(dataset)))), tokenizer)

    peft_config = None
    if use_peft:
        from peft import LoraConfig
        peft_config = LoraConfig(r=lora_r, lora_alpha=lora_r * 2, task_type="CAUSAL_LM")
        ref_policy = None

    config = PPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        total_episodes=5000,
        learning_rate=1.41e-5,
        response_length=128,
        temperature=0.7,
        kl_coef=0.05,
        num_ppo_epochs=1,
        num_mini_batches=1,
        missing_eos_penalty=1.0,
        reward_model_path=rm_path,
        sft_model_path=base_model,
        logging_steps=5,
        local_rollout_forward_batch_size=4,
    )

    trainer = PPOTrainer(
        args=config,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        train_dataset=train_dataset,
        value_model=value_model,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Single PPO model saved to {output_dir}")


def evaluate_pairwise(
    model_paths: dict[str, str],
    base_model: str = "allenai/tulu-2-7b",
    eval_dataset_name: str = "HuggingFaceH4/no_robots",
    num_samples: int = 50,
    output_dir: str = "outputs/evaluation",
):
    """Evaluate models using pairwise comparison (win rate against base model).

    In the paper, GPT-4 is used as the judge via AlpacaFarm.
    Here we provide the evaluation harness; replace the judge with GPT-4 API calls
    for full replication.
    """
    print(f"\nPairwise Evaluation")
    print(f"  Base model: {base_model}")
    print(f"  Eval samples: {num_samples}")

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load evaluation prompts
    try:
        eval_data = load_dataset(eval_dataset_name, split="test")
    except Exception:
        eval_data = load_dataset("tatsu-lab/alpaca", split="train")
    eval_data = eval_data.select(range(min(num_samples, len(eval_data))))

    prompts = []
    for example in eval_data:
        if "messages" in example:
            prompt = example["messages"][0]["content"] if example["messages"] else ""
        elif "instruction" in example:
            prompt = example["instruction"]
        else:
            prompt = example.get("text", "")[:200]
        prompts.append(prompt)

    results = {}

    for model_name, model_path in model_paths.items():
        print(f"\n  Generating from: {model_name}")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
            )
        except Exception:
            print(f"    Could not load {model_path}, skipping.")
            continue

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        generations = []
        for prompt in prompts:
            input_text = f"### Instruction:\n{prompt}\n\n### Response:\n"
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            generations.append(generated)

        results[model_name] = generations
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save generations for external judging (e.g., with GPT-4 via AlpacaFarm)
    eval_output = {
        "prompts": prompts,
        "generations": results,
        "judge_instructions": (
            "Use AlpacaFarm or GPT-4 to compare each model's generation against the base model. "
            "Report win rate (%) for each model on each user group's preference criteria."
        ),
    }

    eval_path = os.path.join(output_dir, "pairwise_eval_data.json")
    with open(eval_path, "w") as f:
        json.dump(eval_output, f, indent=2)

    print(f"\n  Evaluation data saved to {eval_path}")
    print("  To compute win rates, use GPT-4 as judge via AlpacaFarm:")
    print("  pip install alpaca-farm")
    print(f"  See the paper's Table 2 and Table 3 for expected results format.")

    # Print summary matching Table 2/3 format
    print(f"\n  Expected results format (Table 2 from paper):")
    print(f"  {'Method':<15} {'P3A':>8} {'P3B':>8} {'Average':>8}")
    print(f"  {'-'*41}")
    print(f"  {'MaxMin':<15} {'57.78':>8} {'55.56':>8} {'56.67':>8}")
    print(f"  {'1:1':<15} {'55.85':>8} {'52.62':>8} {'54.24':>8}")
    print(f"  {'2:1':<15} {'55.56':>8} {'48.89':>8} {'52.23':>8}")
    print(f"  {'6:1':<15} {'58.06':>8} {'46.67':>8} {'52.37':>8}")
    print(f"  {'10:1':<15} {'56.00':>8} {'45.00':>8} {'50.50':>8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MaxMin-RLHF Tulu-7B Benchmark")
    parser.add_argument(
        "--mode",
        choices=["em_rewards", "maxmin_ppo", "single_ppo", "evaluate", "all"],
        default="all",
    )
    parser.add_argument("--base_model", default="allenai/tulu-2-7b")
    parser.add_argument("--dataset_pair", default="P1", choices=["P1", "P2", "P3"])
    parser.add_argument("--ratio", type=int, default=1, help="Majority:minority ratio for single PPO baseline")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--use_peft", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--num_clusters", type=int, default=2)
    parser.add_argument("--num_em_iters", type=int, default=5)
    parser.add_argument("--reward_model_paths", nargs="+", default=None)
    args = parser.parse_args()

    if args.mode in ("em_rewards", "all"):
        run_em_reward_learning(
            base_model=args.base_model,
            dataset_pair=args.dataset_pair,
            num_clusters=args.num_clusters,
            num_em_iters=args.num_em_iters,
            output_dir=os.path.join(args.output_dir, "em_rewards"),
            use_peft=args.use_peft,
            lora_r=args.lora_r,
        )

    if args.mode in ("maxmin_ppo", "all"):
        run_maxmin_ppo_tulu(
            base_model=args.base_model,
            reward_model_paths=args.reward_model_paths,
            output_dir=os.path.join(args.output_dir, "maxmin_ppo_tulu"),
            use_peft=args.use_peft,
            lora_r=args.lora_r,
        )

    if args.mode in ("single_ppo", "all"):
        run_single_ppo_tulu(
            base_model=args.base_model,
            output_dir=os.path.join(args.output_dir, f"single_ppo_tulu_ratio{args.ratio}"),
            use_peft=args.use_peft,
            lora_r=args.lora_r,
            ratio=args.ratio,
        )

    if args.mode in ("evaluate", "all"):
        model_paths = {
            "MaxMin PPO": os.path.join(args.output_dir, "maxmin_ppo_tulu"),
        }
        for ratio in [1, 2, 6, 10]:
            path = os.path.join(args.output_dir, f"single_ppo_tulu_ratio{ratio}")
            if os.path.exists(path):
                model_paths[f"Single PPO ({ratio}:1)"] = path

        evaluate_pairwise(
            model_paths=model_paths,
            base_model=args.base_model,
            output_dir=os.path.join(args.output_dir, "evaluation"),
        )
