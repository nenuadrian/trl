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
#     "trackio",
#     "kernels",
# ]
# ///

"""
EM Reward Learning for MaxMin-RLHF (Algorithm 2 from the paper).

This script implements the Expectation-Maximization algorithm to learn a mixture of reward
models from preference data containing diverse user subpopulations. The learned reward models
can then be used with the MaxMinPPOTrainer.

The algorithm alternates between:
    E-step: Assign each annotator/sample to the reward model cluster that best explains their
            preferences (hard cluster assignment).
    M-step: Retrain each reward model on its assigned cluster of preference data.

Usage:
    python examples/scripts/ppo/em_reward_learning.py \
        --model_name_or_path EleutherAI/pythia-160m \
        --dataset_name trl-lib/ultrafeedback_binarized \
        --num_clusters 2 \
        --num_em_iterations 5 \
        --output_dir em_reward_models \
        --per_device_train_batch_size 8 \
        --num_train_epochs 1 \
        --learning_rate 1e-5 \
        --max_length 512
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser

from trl import ModelConfig, RewardConfig, RewardTrainer, ScriptArguments, get_peft_config


@dataclass
class EMArguments:
    """Arguments for the EM reward learning algorithm."""

    num_clusters: int = field(
        default=2,
        metadata={"help": "Number of user subpopulation clusters (|U| in the paper)."},
    )
    num_em_iterations: int = field(
        default=5,
        metadata={"help": "Number of EM iterations until convergence."},
    )
    annotator_column: str | None = field(
        default=None,
        metadata={
            "help": "Column name in the dataset that identifies annotators. "
            "If None, samples are assigned to clusters (not annotators)."
        },
    )
    em_log_file: str = field(
        default="em_log.json",
        metadata={"help": "File to log EM iteration statistics."},
    )


def compute_preference_likelihood(model, tokenizer, chosen_text, rejected_text, device, max_length=512):
    """Compute the likelihood that the reward model prefers chosen over rejected.

    Returns w(phi, x, y1, y2) = exp(r(y1,x)) / (exp(r(y1,x)) + exp(r(y2,x)))
    as defined in Algorithm 2 of the paper.
    """
    chosen_inputs = tokenizer(
        chosen_text, return_tensors="pt", truncation=True, max_length=max_length, padding=True
    ).to(device)
    rejected_inputs = tokenizer(
        rejected_text, return_tensors="pt", truncation=True, max_length=max_length, padding=True
    ).to(device)

    with torch.no_grad():
        chosen_reward = model(**chosen_inputs).logits.squeeze(-1)
        rejected_reward = model(**rejected_inputs).logits.squeeze(-1)

    # Bradley-Terry preference probability
    log_prob = torch.nn.functional.logsigmoid(chosen_reward - rejected_reward)
    return log_prob.item()


def e_step_sample_level(models, tokenizer, dataset, device, max_length=512):
    """E-step: assign each sample to the cluster whose reward model best explains it.

    This is a simplified version for datasets without annotator IDs - each preference
    pair is assigned to the reward model that gives it the highest likelihood.
    """
    num_clusters = len(models)
    assignments = []

    for i in range(len(dataset)):
        example = dataset[i]

        # Get chosen and rejected texts
        if "chosen" in example and "rejected" in example:
            # Chat format: extract text
            if isinstance(example["chosen"], list):
                chosen_text = tokenizer.apply_chat_template(example["chosen"], tokenize=False)
                rejected_text = tokenizer.apply_chat_template(example["rejected"], tokenize=False)
            else:
                chosen_text = example["chosen"]
                rejected_text = example["rejected"]
        else:
            raise ValueError("Dataset must have 'chosen' and 'rejected' columns.")

        log_probs = []
        for model in models:
            log_prob = compute_preference_likelihood(
                model, tokenizer, chosen_text, rejected_text, device, max_length
            )
            log_probs.append(log_prob)

        # Hard cluster assignment: assign to the model with highest likelihood
        best_cluster = int(np.argmax(log_probs))
        assignments.append(best_cluster)

    return assignments


def e_step_annotator_level(models, tokenizer, dataset, annotator_column, device, max_length=512):
    """E-step with annotator-level clustering (as in Algorithm 2).

    Each annotator h is assigned to the cluster u that maximizes the product of
    preference likelihoods across all their annotated pairs.
    """
    num_clusters = len(models)

    # Group data by annotator
    annotator_data = {}
    for i in range(len(dataset)):
        example = dataset[i]
        annotator_id = example[annotator_column]
        if annotator_id not in annotator_data:
            annotator_data[annotator_id] = []
        annotator_data[annotator_id].append(i)

    # Assign each annotator to a cluster
    annotator_assignments = {}
    for annotator_id, indices in annotator_data.items():
        log_probs_per_cluster = [0.0] * num_clusters

        for idx in indices:
            example = dataset[idx]
            if isinstance(example["chosen"], list):
                chosen_text = tokenizer.apply_chat_template(example["chosen"], tokenize=False)
                rejected_text = tokenizer.apply_chat_template(example["rejected"], tokenize=False)
            else:
                chosen_text = example["chosen"]
                rejected_text = example["rejected"]

            for c, model in enumerate(models):
                log_prob = compute_preference_likelihood(
                    model, tokenizer, chosen_text, rejected_text, device, max_length
                )
                log_probs_per_cluster[c] += log_prob

        best_cluster = int(np.argmax(log_probs_per_cluster))
        annotator_assignments[annotator_id] = best_cluster

    # Convert annotator assignments to sample-level assignments
    assignments = []
    for i in range(len(dataset)):
        annotator_id = dataset[i][annotator_column]
        assignments.append(annotator_assignments[annotator_id])

    return assignments


def m_step(models, tokenizer, dataset, assignments, training_args, model_name_or_path, model_args, output_dir):
    """M-step: retrain each reward model on its assigned cluster data."""
    num_clusters = len(models)
    new_models = []

    for cluster_idx in range(num_clusters):
        # Filter dataset for this cluster
        cluster_indices = [i for i, a in enumerate(assignments) if a == cluster_idx]

        if len(cluster_indices) == 0:
            print(f"  Warning: Cluster {cluster_idx} has no samples, keeping previous model.")
            new_models.append(models[cluster_idx])
            continue

        cluster_dataset = dataset.select(cluster_indices)
        print(f"  Cluster {cluster_idx}: {len(cluster_dataset)} samples")

        # Initialize a fresh reward model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=1,
            trust_remote_code=model_args.trust_remote_code,
        )

        cluster_output_dir = os.path.join(output_dir, f"cluster_{cluster_idx}")

        # Clone training args for this cluster
        cluster_training_args = RewardConfig(
            output_dir=cluster_output_dir,
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            num_train_epochs=training_args.num_train_epochs,
            learning_rate=training_args.learning_rate,
            logging_steps=training_args.logging_steps,
            max_length=training_args.max_length,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            remove_unused_columns=False,
            report_to=training_args.report_to,
        )

        peft_config = get_peft_config(model_args)

        trainer = RewardTrainer(
            model=model,
            args=cluster_training_args,
            train_dataset=cluster_dataset,
            peft_config=peft_config,
        )
        trainer.train()

        # Save the cluster reward model
        trainer.save_model(cluster_output_dir)
        new_models.append(trainer.model)

    return new_models


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, RewardConfig, ModelConfig, EMArguments))
    script_args, training_args, model_args, em_args = parser.parse_args_into_dataclasses()

    output_dir = training_args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    ################
    # Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Load dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    train_dataset = dataset[script_args.dataset_train_split]

    ################
    # Initialize reward models
    ################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for cluster_idx in range(em_args.num_clusters):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            num_labels=1,
            trust_remote_code=model_args.trust_remote_code,
        )
        model = model.to(device)
        model.eval()
        models.append(model)

    ################
    # EM Loop
    ################
    em_log = []

    for em_iter in range(em_args.num_em_iterations):
        print(f"\n{'='*60}")
        print(f"EM Iteration {em_iter + 1}/{em_args.num_em_iterations}")
        print(f"{'='*60}")

        # E-step
        print("E-step: Assigning samples to clusters...")
        if em_args.annotator_column is not None:
            assignments = e_step_annotator_level(
                models, tokenizer, train_dataset, em_args.annotator_column, device,
                max_length=training_args.max_length,
            )
        else:
            assignments = e_step_sample_level(
                models, tokenizer, train_dataset, device,
                max_length=training_args.max_length,
            )

        # Log cluster distribution
        cluster_counts = {}
        for a in assignments:
            cluster_counts[a] = cluster_counts.get(a, 0) + 1
        print(f"  Cluster distribution: {cluster_counts}")

        # M-step
        print("M-step: Training cluster-specific reward models...")
        iter_output_dir = os.path.join(output_dir, f"em_iter_{em_iter}")
        models = m_step(
            models, tokenizer, train_dataset, assignments,
            training_args, model_args.model_name_or_path, model_args, iter_output_dir,
        )

        # Move models to device for next E-step
        for i in range(len(models)):
            models[i] = models[i].to(device)
            models[i].eval()

        em_log.append({
            "iteration": em_iter,
            "cluster_distribution": cluster_counts,
        })

    ################
    # Save final models and log
    ################
    final_output_dir = os.path.join(output_dir, "final")
    os.makedirs(final_output_dir, exist_ok=True)

    for cluster_idx, model in enumerate(models):
        model_path = os.path.join(final_output_dir, f"reward_model_{cluster_idx}")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print(f"Saved reward model {cluster_idx} to {model_path}")

    # Save EM log
    log_path = os.path.join(output_dir, em_args.em_log_file)
    with open(log_path, "w") as f:
        json.dump(em_log, f, indent=2)
    print(f"\nEM log saved to {log_path}")

    # Save reward model paths for use with MaxMinPPOTrainer
    paths_file = os.path.join(final_output_dir, "reward_model_paths.json")
    reward_model_paths = [
        os.path.join(final_output_dir, f"reward_model_{i}") for i in range(em_args.num_clusters)
    ]
    with open(paths_file, "w") as f:
        json.dump(reward_model_paths, f, indent=2)
    print(f"Reward model paths saved to {paths_file}")
