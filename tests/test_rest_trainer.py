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

from datasets import Dataset
from transformers import AutoTokenizer

from trl import RESTConfig, RESTTrainer

from .testing_utils import TrlTestCase


class TestRESTTrainer(TrlTestCase):
    def setup_method(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _make_prompt_dataset(self, n=4):
        return Dataset.from_dict({"prompt": [f"What is {i} + {i}?" for i in range(n)]})

    def test_train_with_binary_reward_fn(self):
        """Test single iteration with binary reward function."""
        dataset = self._make_prompt_dataset()
        training_args = RESTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=1,
            num_iterations=1,
            num_samples_per_prompt=2,
            max_new_tokens=16,
            max_length=128,
            generation_batch_size=2,
            reward_threshold=0.0,
            learning_rate=1e-5,
            report_to="none",
        )

        def binary_reward(prompts, completions):
            return [1.0 if len(c) > 0 else 0.0 for c in completions]

        trainer = RESTTrainer(
            model=self.model_id,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dataset,
            reward_fn=binary_reward,
        )
        trainer.train()

    def test_train_with_continuous_reward_fn(self):
        """Test with continuous reward and reward-weighted loss."""
        dataset = self._make_prompt_dataset()
        training_args = RESTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=1,
            num_iterations=1,
            num_samples_per_prompt=2,
            max_new_tokens=16,
            max_length=128,
            generation_batch_size=2,
            reward_threshold=-1.0,
            reward_weighted_loss=True,
            learning_rate=1e-5,
            report_to="none",
        )

        def length_reward(prompts, completions):
            return [float(len(c)) for c in completions]

        trainer = RESTTrainer(
            model=self.model_id,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dataset,
            reward_fn=length_reward,
        )
        trainer.train()

    def test_multiple_iterations(self):
        """Test that multiple EM iterations run successfully."""
        dataset = self._make_prompt_dataset(n=2)
        training_args = RESTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=1,
            num_iterations=2,
            num_samples_per_prompt=2,
            max_new_tokens=8,
            max_length=64,
            generation_batch_size=2,
            reward_threshold=0.0,
            reset_model_each_iteration=True,
            learning_rate=1e-5,
            report_to="none",
        )

        def reward_fn(prompts, completions):
            return [1.0] * len(prompts)

        trainer = RESTTrainer(
            model=self.model_id,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dataset,
            reward_fn=reward_fn,
        )
        trainer.train()

    def test_max_solutions_cap(self):
        """Test that per-problem cap is applied correctly."""
        dataset = self._make_prompt_dataset(n=2)
        training_args = RESTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=1,
            num_iterations=1,
            num_samples_per_prompt=4,
            max_solutions_per_problem=1,
            max_new_tokens=8,
            max_length=64,
            generation_batch_size=2,
            reward_threshold=0.0,
            learning_rate=1e-5,
            report_to="none",
        )

        def reward_fn(prompts, completions):
            return [1.0] * len(prompts)

        trainer = RESTTrainer(
            model=self.model_id,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dataset,
            reward_fn=reward_fn,
        )
        trainer.train()
        # With 2 prompts and max_solutions_per_problem=1, the SFT dataset should have at most 2 samples
        assert len(trainer.train_dataset) <= 2

    def test_filtering_removes_low_reward(self):
        """Test that filtering correctly removes low-reward completions."""
        dataset = self._make_prompt_dataset(n=2)
        training_args = RESTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=1,
            num_iterations=1,
            num_samples_per_prompt=2,
            max_new_tokens=8,
            max_length=64,
            generation_batch_size=2,
            reward_threshold=100.0,  # Very high threshold - nothing should pass
            learning_rate=1e-5,
            report_to="none",
        )

        def reward_fn(prompts, completions):
            return [0.0] * len(prompts)  # All zero rewards

        trainer = RESTTrainer(
            model=self.model_id,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dataset,
            reward_fn=reward_fn,
        )
        # Should not crash even if all completions are filtered out
        trainer.train()
