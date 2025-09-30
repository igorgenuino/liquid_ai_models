# -*- coding: utf-8 -*-
import torch
import transformers
import trl
import os
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForImageTextToText, AutoProcessor

# --- 1. Basic Setup ---
os.environ["WANDB_DISABLED"] = "true"

print(f"📦 PyTorch version: {torch.__version__}")
print(f"🤗 Transformers version: {transformers.__version__}")
print(f"📊 TRL version: {trl.__version__}")

# --- 2. Load Model and Processor for CPU ---
model_id = r"C:\Users\igmartin\projects\liquid_ai_models\LFM2-VL-450M"

print("📚 Loading processor...")
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    max_image_tokens=256,
)

print("🧠 Loading model for CPU...")
# KEY CHANGES FOR CPU:
# - Removed torch_dtype="bfloat16" (uses default float32)
# - Removed device_map="auto"
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    trust_remote_code=True,
)
# Manually set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on device: {device}")


print("\n✅ Local model loaded successfully!")
print(f"📖 Vocab size: {len(processor.tokenizer)}")
print(f"🔢 Parameters: {model.num_parameters():,}")
# Model size will be larger on CPU due to float32 precision
print(f"💾 Model size: ~{model.num_parameters() * 4 / 1e9:.1f} GB (float32)")

# --- 3. Load and Format Dataset ---
print("\n🔄 Loading and formatting dataset...")
raw_ds = load_dataset("akahana/Driver-Drowsiness-Dataset")
train_dataset = raw_ds["train"].select(range(500))
eval_dataset = raw_ds["test"].select(range(100))

system_message = (
    "You are a Vision Language Model specialized in analyzing face images and determining if a person is drowsy or not. "
    "Provide a concise answer based on the image and question."
)

def format_drowsiness_sample(sample):
    label_text = "Drowsy" if sample["label"] == 0 else "Non Drowsy"
    question_text = "Is the person in the image drowsy or not?"
    return [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": question_text},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": label_text}]},
    ]

train_dataset = [format_drowsiness_sample(s) for s in train_dataset]
eval_dataset = [format_drowsiness_sample(s) for s in eval_dataset]
print("✅ SFT Dataset formatted.")

# --- 4. Create Collate Function ---
def create_collate_fn(processor):
    def collate_fn(sample):
        # Move tensors to the correct device (CPU)
        batch = processor.apply_chat_template(sample, tokenize=True, return_dict=True, return_tensors="pt").to(device)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch
    return collate_fn

collate_fn = create_collate_fn(processor)

# --- 5. Setup PEFT LoRA ---
target_modules = [
    "q_proj", "v_proj", "fc1", "fc2", "linear",
    "gate_proj", "up_proj", "down_proj",
]
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=target_modules,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
#model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# NOTE: The hacks (model.tokenizer = ... and model.config._name_or_path = ...) have been removed.

# --- 6. Configure and Launch Training with transformers.Trainer ---

# We now import Trainer and TrainingArguments from the core transformers library
from transformers import Trainer, TrainingArguments
import math

# --- ADD THIS CALCULATION ---
# Calculate the number of steps per epoch to use for evaluation and saving
effective_batch_size = 1 * 16 # per_device_train_batch_size * gradient_accumulation_steps
steps_per_epoch = math.ceil(len(train_dataset) / effective_batch_size)
print(f"✅ Calculated steps per epoch: {steps_per_epoch}")
# --- END OF ADDITION ---


# Replace SFTConfig with TrainingArguments
training_args = TrainingArguments(
    output_dir="lfm2-vl-450-fine-tuning",
    #ideally 3 but 2 to be quicker
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    bf16=False,
    fp16=False,
    do_eval=True,
    eval_steps=steps_per_epoch,
    save_steps=steps_per_epoch,
    logging_steps=10,

    # --- THIS IS THE FINAL CHANGE ---
    # Disable loading the best model to bypass the versioning bug
    load_best_model_at_end=False,
    # --- END OF FINAL CHANGE ---

    report_to="none",
    gradient_checkpointing=False,
)

print("\n🏗️ Creating Trainer...")
# Replace SFTTrainer with the standard Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
)

print("\n🚀 Starting training on CPU (this will be slow)...")
trainer.train()

print("\n🎉 Training completed!")
trainer.save_model()
print(f"💾 Saving to: {training_args.output_dir}")