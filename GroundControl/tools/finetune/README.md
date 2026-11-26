# Fine-tuning the planner

In-context prompting got the base Qwen3-1.7B to 78% on the held-out eval (docs/EVAL.md). The model knows the skills, but the few-shot examples pull it around, and they cost about 700 prefix tokens on every cold prompt (5,209 characters of prompt versus 2,976 without them). A LoRA trained on synthetic plans attacks both problems: the examples move into the weights, and the prompt gets shorter.

1. **Data** (laptop, seconds): `venv/bin/python tools/finetune/gen_data.py` writes `data/train.jsonl` and `data/val.jsonl` (6,555 / 345 examples), then zips them into `data/gc_finetune_data.zip`.
   1. Commands come from per-skill paraphrase templates, so every plan is exactly right. Each plan is checked with the planner's own `validate_plan`.
   2. Anything that matches `tools/eval/commands.yaml` or `dev.yaml` is dropped, so the eval stays held out.
   3. The system message is exactly `prompt.system_prompt(60, "think")`, the rules and skill list with no few-shot examples.
   4. The data also covers the other two jobs the same weights do on the board: composing answers from facts and writing choreography scores.
2. **Train** (Colab, about 1–2 hours on a free T4): open `finetune_planner_colab.ipynb` in Colab, pick a T4 GPU, Run all, and upload the zip when cell 2 asks.
   1. LoRA r16 on every projection layer, one epoch, loss on the completion only.
   2. 20% general instruction data (alpaca-cleaned) is mixed in, so the model doesn't forget how to write a normal sentence.
   3. The notebook merges the adapter, converts to GGUF and quantizes to Q4_0, the same format the board runs now.
   4. `train_lora.py` does the same training as a plain script, for any other CUDA machine.
3. **Install on the board**: `venv/bin/python tools/deploy.py --model ~/Downloads/gc-planner-Q4_0.gguf`.
   1. It copies the file to `/var/lib/arduino-app-cli/models/llamacpp/custom/` and adds a `[gc-planner-Q4_0]` section to the runner's `models.ini`.
   2. Then it restarts the llama.cpp runner and lists the models it serves. No Hugging Face account is involved.
4. **Evaluate**: `venv/bin/python tools/eval/run_eval.py --model gc-planner-Q4_0`. Selecting this model switches the planner to the `ft` prompt style automatically (`config.style_for`), which drops the few-shot examples to match training.
5. **Use it live**: add `gc-planner-Q4_0` to `LLM_MODELS` in `python/config.py` so the dashboard's model picker can switch between base and fine-tuned, or set `GC_LLM_MODEL=gc-planner-Q4_0` to make it the default.
