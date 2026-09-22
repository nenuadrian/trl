from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl.experimental.online_dpo import (
    OnlineDPOConfig,
    OnlineDPOTrainer,
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
REWARD_MODEL = "trl-lib/Qwen2-0.5B-Reward"

NUM_EXAMPLES = 10_000

# ---------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ---------------------------------------------------------
# LMSYS -> prompt-only conversational dataset
# ---------------------------------------------------------

dataset = load_dataset(
    "lmsys/lmsys-chat-1m",
    split="train",
)

# Start with English prompts for a cleaner reward-model experiment.
dataset = dataset.filter(
    lambda x: x["language"] == "English"
)

def conversation_to_prompt(example):
    conversation = example["conversation"]

    # Find the final user turn.
    user_indices = [
        i
        for i, message in enumerate(conversation)
        if message["role"] == "user"
    ]

    if not user_indices:
        return {"prompt": None}

    last_user_idx = user_indices[-1]

    # Keep all context up to and including the final user message.
    prompt = conversation[: last_user_idx + 1]

    return {
        "prompt": prompt,
    }


dataset = dataset.map(
    conversation_to_prompt,
    remove_columns=dataset.column_names,
)

dataset = dataset.filter(
    lambda x: x["prompt"] is not None
)

dataset = dataset.shuffle(seed=42).select(
    range(NUM_EXAMPLES)
)

# ---------------------------------------------------------
# Train / eval split
# ---------------------------------------------------------

split = dataset.train_test_split(
    test_size=500,
    seed=42,
)

train_dataset = split["train"]
eval_dataset = split["test"]

print("Train examples:", len(train_dataset))
print("Eval examples:", len(eval_dataset))

print("\nExample prompt:")
print(train_dataset[0]["prompt"])


# ---------------------------------------------------------
# Policy
# ---------------------------------------------------------

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype="bfloat16",
)

model.config.pad_token_id = tokenizer.pad_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id


# ---------------------------------------------------------
# Online DPO
# ---------------------------------------------------------

args = OnlineDPOConfig(
    output_dir="./qwen25-05b-online-dpo-lmsys",

    # DPO
    beta=0.1,

    # Optimization
    learning_rate=5e-7,
    num_train_epochs=1,

    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,

    # Generation
    max_length=1024,
    max_new_tokens=256,

    temperature=0.8,
    top_p=0.95,

    # EOS
    missing_eos_penalty=1.0,

    # Efficiency
    gradient_checkpointing=True,

    # Logging
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,

    report_to="none",

    bf16=True,
    seed=42,
)


trainer = OnlineDPOTrainer(
    model=model,

    # Reward model is loaded automatically by TRL.
    reward_funcs=REWARD_MODEL,

    args=args,

    processing_class=tokenizer,

    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

trainer.save_model(
    "./qwen25-05b-online-dpo-lmsys"
)

tokenizer.save_pretrained(
    "./qwen25-05b-online-dpo-lmsys"
)
