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
MaxMin-RLHF PPO benchmark script with meaningfully different reward models.

This script loads the same base model as two separate reward models, but
initializes their score heads with different random seeds. This ensures:
  - rm_0_scores != rm_1_scores (the RMs disagree on what's good)
  - maxmin/selected_rm_idx varies (the min operator actually switches)
  - The MaxMin mechanism is properly exercised

For a real experiment, replace these with EM-trained reward models from
different preference subpopulations (see em_reward_learning.py).
"""

import os
import shutil

import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import ModelConfig, ScriptArguments, get_kbit_device_map, get_peft_config, get_quantization_config
from trl.experimental.ppo import MaxMinPPOConfig, MaxMinPPOTrainer


os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


def _init_score_head(model, seed):
    """Reinitialize the score head with a specific random seed.

    This ensures two reward models loaded from the same base give
    genuinely different scores, exercising the MaxMin mechanism.
    """
    if hasattr(model, "score"):
        in_f = model.score.in_features
        has_bias = model.score.bias is not None
        # Force num_labels=1
        model.score = torch.nn.Linear(in_f, 1, bias=has_bias)
        model.config.num_labels = 1
        # Reinitialize with specific seed for reproducible but different heads
        gen = torch.Generator().manual_seed(seed)
        torch.nn.init.normal_(model.score.weight, mean=0.0, std=0.02, generator=gen)
        if has_bias:
            torch.nn.init.zeros_(model.score.bias)
    return model


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, MaxMinPPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Load reward models with DIFFERENT score head seeds
    # Seed 42 and seed 137 produce different scoring functions,
    # so the MaxMin min-operator will actually switch between them.
    rm_seeds = [42, 137]
    reward_models = []
    for i, rm_path in enumerate(training_args.reward_model_paths):
        rm = AutoModelForSequenceClassification.from_pretrained(
            rm_path,
            trust_remote_code=model_args.trust_remote_code,
            num_labels=1,
            ignore_mismatched_sizes=True,
            **model_kwargs,
        )
        rm = _init_score_head(rm, seed=rm_seeds[i])
        reward_models.append(rm)
        print(f"[bench] RM {i}: score head seed={rm_seeds[i]}, "
              f"weight norm={rm.score.weight.data.norm():.4f}")

    # Verify RM heads are actually different
    w0 = reward_models[0].score.weight.data
    w1 = reward_models[1].score.weight.data
    cos_sim = torch.nn.functional.cosine_similarity(w0.flatten(), w1.flatten(), dim=0)
    print(f"[bench] RM score head cosine similarity: {cos_sim:.4f} (should be << 1.0)")

    # Value model (separate from both RMs)
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_paths[0],
        trust_remote_code=model_args.trust_remote_code,
        num_labels=1,
        ignore_mismatched_sizes=True,
        **model_kwargs,
    )
    value_model = _init_score_head(value_model, seed=999)

    # Policy model
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
        )
    else:
        ref_policy = None

    ################
    # Dataset — apply chat template so the model stays in-distribution
    ################
    dataset = load_dataset(
        script_args.dataset_name, name=script_args.dataset_config, split=script_args.dataset_train_split
    )
    eval_samples = 100
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    dataset_text_field = "prompt"

    def prepare_dataset(dataset, tokenizer):
        """Tokenize prompts wrapped in chat template for instruct models."""

        def tokenize(element):
            prompts = element[dataset_text_field]
            input_ids_list = []
            for prompt in prompts:
                # Wrap in chat template so the model generates in its native format
                messages = [{"role": "user", "content": prompt}]
                # add_generation_prompt=True appends the assistant turn prefix
                # so the model can immediately start generating a response
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                ids = tokenizer(text, padding=False, add_special_tokens=False)["input_ids"]
                input_ids_list.append(ids)
            return {"input_ids": input_ids_list}

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )

    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    # Log a sample to verify formatting
    if PartialState().is_main_process:
        sample = tokenizer.decode(train_dataset[0]["input_ids"])
        print(f"[bench] Sample tokenized prompt (first 200 chars):\n{sample[:200]}")

    ################
    # Training
    ################
    trainer = MaxMinPPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_models=reward_models,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.generate_completions()
