from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "argilla/ultrafeedback-binarized-preferences-cleaned"
NUM_EXAMPLES = 5000


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

dataset = load_dataset(
    DATASET_NAME,
    split="train",
).select(range(NUM_EXAMPLES))


def convert_example(example):
    return {
        "prompt": [
            {
                "role": "user",
                "content": example["prompt"],
            }
        ],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


dataset = dataset.map(
    convert_example,
    remove_columns=dataset.column_names,
)

print("\nDataset example:")
print(dataset[0])


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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
