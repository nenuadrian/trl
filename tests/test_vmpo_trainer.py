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

from datasets import load_dataset
from transformers import AutoTokenizer

from trl import VMPOConfig, VMPOTrainer

from .testing_utils import TrlTestCase


class TestVMPOTrainer(TrlTestCase):
    def setup_method(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def test_train_online_with_reward_fn(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train[:4]")
        training_args = VMPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=1,
            max_new_tokens=16,
            vmpo_k=2,
            learning_rate=1e-5,
            report_to="none",
        )

        def reward_fn(prompts, completions):
            return [float(len(prompt) + len(completion)) for prompt, completion in zip(prompts, completions, strict=True)]

        trainer = VMPOTrainer(
            model=self.model_id,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dataset,
            reward_fn=reward_fn,
        )
        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_train_offline_from_pairwise_dataset(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train[:4]")
        training_args = VMPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=1,
            max_new_tokens=16,
            vmpo_k=2,
            learning_rate=1e-5,
            report_to="none",
        )

        trainer = VMPOTrainer(
            model=self.model_id,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dataset,
        )
        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
