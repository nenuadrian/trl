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

import argparse
import importlib
import os
import sys
from dataclasses import dataclass, field

import torch
from accelerate import logging
from datasets import load_dataset

from trl import (
    DARConfig,
    DARTrainer,
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.rewards import accuracy_reward, get_soft_overlong_punishment, reasoning_accuracy_reward, think_format_reward


logger = logging.get_logger(__name__)

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


reward_funcs_registry = {
    "accuracy_reward": accuracy_reward,
    "reasoning_accuracy_reward": reasoning_accuracy_reward,
    "think_format_reward": think_format_reward,
    "get_soft_overlong_punishment": get_soft_overlong_punishment(max_completion_len=1280, soft_punish_cache=256),
}


@dataclass
class DARScriptArguments(ScriptArguments):
    """
    Script arguments for DAR training.

    Args:
        reward_model_name_or_path (`str`, *optional*):
            Reward model id/path used to score generated completions.
        reward_funcs (`list[str]`, *optional*):
            Reward functions to use. Supports built-ins or dotted import paths.
    """

    reward_model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "Reward model id of a pretrained model hosted inside a model repo on huggingface.co or local path "
            "to a directory containing model weights saved using `PreTrainedModel.save_pretrained`."
        },
    )
    reward_funcs: list[str] | None = field(
        default=None,
        metadata={
            "help": "Reward functions to use. Supported values are: `accuracy_reward`, "
            "`reasoning_accuracy_reward`, `think_format_reward`, `get_soft_overlong_punishment`, or any dotted "
            "import path (e.g., `my_lib.rewards.custom_reward`)."
        },
    )


def _load_reward_funcs(reward_func_names: list[str] | None):
    if reward_func_names is None:
        return []

    reward_funcs = []
    for func_name in reward_func_names:
        if func_name in reward_funcs_registry:
            reward_funcs.append(reward_funcs_registry[func_name])
        elif "." in func_name:
            module_path, attr_name = func_name.rsplit(".", 1)
            sys.path.insert(0, os.getcwd())
            module = importlib.import_module(module_path)
            reward_funcs.append(getattr(module, attr_name))
        else:
            raise ValueError(
                f"Could not load reward function '{func_name}'. Expected one of "
                f"{list(reward_funcs_registry.keys())} or a valid import path."
            )
    return reward_funcs


def _combine_reward_funcs(reward_funcs):
    if len(reward_funcs) == 0:
        return None
    if len(reward_funcs) == 1:
        return reward_funcs[0]

    def combined_reward_fn(prompts, completions, completion_ids=None, batch=None):
        total = None
        for reward_func in reward_funcs:
            try:
                reward = reward_func(
                    prompts=prompts,
                    completions=completions,
                    completion_ids=completion_ids,
                    batch=batch,
                )
            except TypeError:
                reward = reward_func(prompts, completions)

            reward_tensor = torch.as_tensor(reward, dtype=torch.float32)
            total = reward_tensor if total is None else total + reward_tensor
        return total

    return combined_reward_fn


def main(script_args, training_args, model_args, dataset_args):
    reward_funcs = _load_reward_funcs(script_args.reward_funcs)
    reward_fn = _combine_reward_funcs(reward_funcs)

    if script_args.reward_model_name_or_path is not None and reward_fn is not None:
        raise ValueError("Provide either `reward_model_name_or_path` or `reward_funcs`, not both.")

    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_kwargs = {
        "revision": model_args.model_revision,
        "attn_implementation": model_args.attn_implementation,
        "dtype": dtype,
    }
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config
    training_args.model_init_kwargs = model_kwargs

    # Load the dataset
    if dataset_args.datasets and script_args.dataset_name:
        logger.warning(
            "Both `datasets` and `dataset_name` are provided. The `datasets` argument will be used to load the "
            "dataset and `dataset_name` will be ignored."
        )
        dataset = get_dataset(dataset_args)
    elif dataset_args.datasets and not script_args.dataset_name:
        dataset = get_dataset(dataset_args)
    elif not dataset_args.datasets and script_args.dataset_name:
        dataset = load_dataset(
            script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
        )
    else:
        raise ValueError("Either `datasets` or `dataset_name` must be provided.")

    trainer = DARTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        reward_fn=reward_fn,
        reward_model=script_args.reward_model_name_or_path,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()
    trainer.accelerator.print("✅ Training completed.")

    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"🤗 Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}.")


def make_parser(subparsers: argparse._SubParsersAction | None = None):
    dataclass_types = (DARScriptArguments, DARConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("dar", help="Run the DAR training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args, dataset_args)
