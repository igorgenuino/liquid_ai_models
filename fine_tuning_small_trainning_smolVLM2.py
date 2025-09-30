# -*- coding: utf-8 -*-
import torch
import transformers
import trl
import os
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments
import math

# --- 1. Basic Setup ---
os.environ["WANDB_DISABLED"] = "true"

print(f"📦 PyTorch version: {torch.__version__}")
print(f"🤗 Transformers version: {transformers.__version__}")
print(f"📊 TRL version: {trl.__version__}")

# --- 2. Load Model and Processor for CPU ---
# --- THIS IS THE FIRST CHANGE ---
# Changed the model_id to the new SmolVLM2 model
model_id = r"C:\Users\igmartin\projects\liquid_ai_models\SmolVLM2-500M-Video-Instruct"
# --- END OF CHANGE ---

print("📚 Loading processor...")
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    max_image_tokens=256,
)

print("🧠 Loading model for CPU...")
# --- THIS IS THE SECOND CHANGE ---
# We are loading the new model.
# Note that we are NOT including `torch_dtype` or `_attn_implementation`
# as those are for GPU usage. This is the correct way to load for CPU.
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    trust_remote_code=True,
)
# --- END OF CHANGE ---

# Manually set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on device: {device}")

print("\n✅ New model loaded successfully!")
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
def create_collate_fn(processor, device):
    def collate_fn(sample):
        # Move tensors to the correct device
        batch = processor.apply_chat_template(sample, tokenize=True, return_dict=True, return_tensors="pt").to(device)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch
    return collate_fn

collate_fn = create_collate_fn(processor, device)

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

# --- 6. Configure and Launch Training with transformers.Trainer ---
effective_batch_size = 1 * 16 # per_device_train_batch_size * gradient_accumulation_steps
steps_per_epoch = math.ceil(len(train_dataset) / effective_batch_size)
print(f"✅ Calculated steps per epoch: {steps_per_epoch}")

training_args = TrainingArguments(
    output_dir="smolvlm2-500m-drowsiness-finetune",
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
    # The 'evaluation_strategy' argument has been removed.
    eval_steps=steps_per_epoch,
    # The 'save_strategy' argument has been removed.
    save_steps=steps_per_epoch,
    logging_steps=10,
    load_best_model_at_end=False,
    report_to="none",
    gradient_checkpointing=False,
)

print("\n🏗️ Creating Trainer...")
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