#!/usr/bin/env python3
"""LoRA fine-tune of Qwen3.5-0.8B on tools/finetune/data (run on a Colab T4/A100).

    pip install "transformers>=4.57" "trl>=0.24" "peft>=0.17" datasets accelerate
    python train_lora.py --data data --out gc-planner-lora --epochs 2

Then export to GGUF Q4_0 and install on the board: see README.md in this folder.
Loss is computed on the assistant turn only (the plan), not on the system prompt.
"""
import argparse

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen3.5-0.8B")
ap.add_argument("--data", default="data")
ap.add_argument("--out", default="gc-planner-lora")
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--rank", type=int, default=16)
a = ap.parse_args()

ds = load_dataset("json", data_files={"train": f"{a.data}/train.jsonl", "validation": f"{a.data}/val.jsonl"})

cfg = SFTConfig(
    output_dir=a.out,
    num_train_epochs=a.epochs,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=a.lr,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",
    bf16=True,
    max_length=1024,
    assistant_only_loss=True,          # learn the plan, not the prompt
    report_to="none",
)
lora = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                  task_type="CAUSAL_LM")
trainer = SFTTrainer(model=a.base, args=cfg, train_dataset=ds["train"], eval_dataset=ds["validation"],
                     peft_config=lora)
trainer.train()
merged = trainer.model.merge_and_unload()
merged.save_pretrained(f"{a.out}/merged")
trainer.processing_class.save_pretrained(f"{a.out}/merged")
print(f"merged model in {a.out}/merged -> convert with llama.cpp (see README.md)")
