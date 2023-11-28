from transformers import TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig
import torch

peft_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    lora_dropout=0.05, 
    target_modules=["q_proj", "v_proj"],
    bias="none", 
    task_type="CAUSAL_LM",
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing =True,
    num_train_epochs=1, 
    learning_rate=1e-4,
    fp16=True,
    save_total_limit=3,
    logging_steps=10,
    output_dir=".",
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
)
