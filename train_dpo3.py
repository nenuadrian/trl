from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "argilla/ultrafeedback-binarized-preferences-cleaned"
NUM_EXAMPLES = 5000


# ------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

dataset = load_dataset(
    DATASET_NAME,
    split="train",
).select(range(NUM_EXAMPLES))


def render_prompt(messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def get_assistant_content(messages):
    # UltraFeedback chosen/rejected are conversation lists.
    # We want the assistant's response without re-rendering it.
    for message in reversed(messages):
        if message["role"] == "assistant":
            return message["content"]

    raise ValueError("No assistant message found")


def convert_example(example):
    prompt = [
        {
            "role": "user",
            "content": example["prompt"],
        }
    ]

    return {
        "prompt": render_prompt(prompt),
        "chosen": get_assistant_content(example["chosen"]),
        "rejected": get_assistant_content(example["rejected"]),
    }


dataset = dataset.map(
    convert_example,
    remove_columns=dataset.column_names,
)


# ------------------------------------------------------------
# Sanity check
# ------------------------------------------------------------

print("\nDataset example:")
print("PROMPT:")
print(dataset[0]["prompt"])

print("\nCHOSEN:")
print(dataset[0]["chosen"])

print("\nREJECTED:")
print(dataset[0]["rejected"])


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
)


# ------------------------------------------------------------
# DPO configuration
# ------------------------------------------------------------

args = DPOConfig(
    output_dir="./dpo-qwen05b",

    num_train_epochs=1,

    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,

    learning_rate=5e-7,

    max_length=1024,

    logging_steps=10,
    save_steps=500,

    report_to="none",

    bf16=True,

    beta=0.1,

    seed=42,
)


# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------

trainer = DPOTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    processing_class=tokenizer,
)


# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

trainer.train()


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

trainer.save_model("./dpo-qwen05b")
tokenizer.save_pretrained("./dpo-qwen05b")
