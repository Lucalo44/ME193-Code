# Fine-tuning the planner (Phase 11)

In-context prompting tops out around 60–70% on this 0.8B model (docs/EVAL.md). The model knows the skills, but it is easily pulled toward the few-shot examples. A small LoRA on synthetic plans is the fix the plan called for.

1. **Data** (laptop, seconds): `venv/bin/python tools/finetune/gen_data.py --n 4000` writes `data/train.jsonl` + `data/val.jsonl`. The commands are built from per-skill paraphrase templates, so every plan is exactly right. Anything matching `tools/eval/commands.yaml` or `dev.yaml` is dropped, so the eval stays held out. The system prompt has **no few-shot examples**, so after fine-tuning the board's prompt can drop them too (about 250 fewer prefix tokens).
2. **Train** (Colab, about 20 min on a T4): upload `data/` and `train_lora.py`, then run `python train_lora.py --data data --epochs 2`.
3. **Export to GGUF Q4_0** (Colab):
   ```bash
   git clone https://github.com/ggml-org/llama.cpp && pip install -r llama.cpp/requirements.txt
   python llama.cpp/convert_hf_to_gguf.py gc-planner-lora/merged --outtype f16 --outfile gc-planner-f16.gguf
   cmake -S llama.cpp -B build && cmake --build build --target llama-quantize -j
   ./build/bin/llama-quantize gc-planner-f16.gguf gc-planner-Q4_0.gguf Q4_0
   ```
4. **Install on the board**: upload the GGUF to a Hugging Face model repo, then
   `python3 tools/deploy.py --cmd "arduino-app-cli model install llamacpp:<user>/<repo>:Q4_0"`
   (App Lab 0.14 accepts a llama.cpp HF id or a direct file URL). Check the name it registers with `arduino-app-cli model list`.
5. **Evaluate**: `venv/bin/python tools/eval/run_eval.py --model <registered name>` and add the row to `docs/EVAL.md`. To use it live, put the model id in `app.yaml` and add it to `LLM_MODELS` in `python/config.py`, so the dashboard's model picker can switch between base and fine-tuned.
