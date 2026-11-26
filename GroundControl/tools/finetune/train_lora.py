#!/usr/bin/env python3
"""LoRA fine-tune of Qwen3-1.7B on tools/finetune/data, as a script.

The Colab notebook (finetune_planner_colab.ipynb) runs these same steps and is the easy
path. Use this on any other machine with a CUDA GPU:

    pip install "transformers>=4.56" "trl>=0.19" "peft>=0.15" datasets accelerate
    python train_lora.py --data data --out gc-lora

It writes gc-lora/merged (fp16 Hugging Face weights). Convert that to GGUF Q4_0 and
install it on the board as README.md describes.
"""
import argparse
import json
import random

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
ap.add_argument("--data", default="data")
ap.add_argument("--out", default="gc-lora")
ap.add_argument("--epochs", type=float, default=1)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--general", type=float, default=0.2, help="alpaca-cleaned replay, as a fraction of train")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.base)


def to_pc(messages):
    """Prompt/completion in Qwen3's non-thinking chat format (the board runs reasoning off)."""
    prompt = tok.apply_chat_template(messages[:-1], add_generation_prompt=True,
                                     enable_thinking=False, tokenize=False)
    return {"prompt": prompt, "completion": messages[-1]["content"] + "<|im_end|>"}


def load(path):
    return [to_pc(json.loads(line)["messages"]) for line in open(path)]


train, val = load(f"{a.data}/train.jsonl"), load(f"{a.data}/val.jsonl")
extra = []
for ex in load_dataset("yahma/alpaca-cleaned", split="train").shuffle(seed=193):
    if len(extra) >= int(a.general * len(train)):
        break
    if len(ex["output"]) > 900:
        continue
    user = ex["instruction"] + (("\n\n" + ex["input"]) if ex["input"] else "")
    extra.append(to_pc([{"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": ex["output"]}]))
train += extra
random.Random(193).shuffle(train)
print(len(train), "train /", len(val), "val")

bf16 = torch.cuda.is_bf16_supported()
model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16 if bf16 else torch.float32)
cfg = SFTConfig(
    output_dir=a.out, num_train_epochs=a.epochs,
    per_device_train_batch_size=2, gradient_accumulation_steps=8,
    learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=15, logging_steps=25,
    eval_strategy="steps", eval_steps=200, save_strategy="no", max_length=1400,
    bf16=bf16, fp16=not bf16, gradient_checkpointing=True,
    completion_only_loss=True, report_to="none")
lora = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
trainer = SFTTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(train),
                     eval_dataset=Dataset.from_list(val), processing_class=tok, peft_config=lora)
trainer.train()
trainer.save_model(a.out)

merged = trainer.model.merge_and_unload().to(torch.float16)
merged.save_pretrained(f"{a.out}/merged", safe_serialization=True)
tok.save_pretrained(f"{a.out}/merged")
print("merged weights in", f"{a.out}/merged")
