# Fine-tuning the planner

In-context prompting got the base Qwen3-1.7B to 78% on the held-out eval (docs/EVAL.md). The model knows the skills, but the few-shot examples pull it around, and they cost about 700 prefix tokens on every cold prompt (5,209 characters of prompt versus 2,976 without them). A LoRA trained on synthetic plans attacks both problems: the examples move into the weights, and the prompt gets shorter.

1. **Data** (laptop, seconds): `venv/bin/python tools/finetune/gen_data.py` writes `data/train.jsonl` and `data/val.jsonl` (8,930 / 470 examples), then zips them into `data/gc_finetune_data.zip`.
   1. Commands come from per-skill paraphrase templates, so every plan is exactly right. Each plan is checked with the planner's own `validate_plan`.
   2. Anything in `tools/eval/commands.yaml` or `dev.yaml` is dropped, and so are near-duplicates (same words with different numbers or punctuation), so the eval keeps measuring generalisation.
   3. The system message is exactly `prompt.system_prompt(60, "think")`, the rules and skill list with no few-shot examples. Nothing new goes into the prompt, so prompts stay as fast as before.
   4. Coverage beyond single commands:
      1. two performers told what to do ("the car spins while the lego bot claps"), mapped to the dance modes;
      2. negations ("show a heart but don't send anything"), gibberish (no steps), and prompt injections;
      3. a lookup plus an action ("how tall is X? draw it"), "draw X" with no icon (writes the word), and post-dance sync reports with `{dance.*}` placeholders;
      4. turn idioms (quarter turn, about face), speed words (slowly, fast), plain matrix dances ("make the stick figure dance");
      5. how people type and talk: numbers as words, speech-to-text transcripts ("right hi", "for bars", "spin to times"), typos and texting shorthand;
      6. plans of up to 6 steps, whose think ends in "..." instead of cutting a skill name;
      7. answers composed from facts where the answer is in the last sentence, or missing ("Wikipedia doesn't say");
      8. choreography scores.
2. **Train** (Colab, about 2½–3 hours on a free T4): open `finetune_planner_colab.ipynb` in Colab, pick a T4 GPU, Run all, and upload the zip when cell 2 asks.
   1. LoRA r16 on every projection layer, one epoch, loss on the completion only.
   2. 20% general instruction data (alpaca-cleaned) is mixed in, so the model doesn't forget how to write a normal sentence.
   3. The notebook merges the adapter, converts to GGUF and quantizes to Q4_0, the same format the board runs now.
   4. `train_lora.py` does the same training as a plain script, for any other CUDA machine.
   5. **Or on the Tufts HPC cluster** (no session limit, A100/L40S/H100): with the VPN on and one SSH login kept open
      (`ssh -o ControlMaster=auto -o ControlPath=~/.ssh/sockets/%r@%h -o ControlPersist=6h <utln>@login-prod.pax.tufts.edu`),
      copy `train_lora.py`, `cluster/setup.sh`, `cluster/job.sbatch` and `data/*.jsonl` to `~/gc-ft`, run `bash setup.sh` once
      on the login node (adds the libraries to `~/mujoco-env` with PyTorch pinned, downloads the model, builds the quantizer),
      then `sbatch job.sbatch`. On an A100 80 GB the job trained, merged and quantized v2 in one go; resubmitting resumes from
      the last checkpoint.
3. **Install on the board**: `venv/bin/python tools/deploy.py --model ~/Downloads/gc-planner-Q4_0.gguf`.
   1. It copies the file to `/var/lib/arduino-app-cli/models/llamacpp/custom/` and writes an `.arduino_metadata.yaml` beside it, the same record App Lab's downloader writes. Without that file, App Lab doesn't list the model, and an app that names it won't start.
   2. App Lab names the model after its folder, so the id is `llamacpp:custom/gc-planner-Q4_0` in `app.yaml` and `custom/gc-planner-Q4_0` as the runner's model name. App Lab rebuilds the runner's `models.ini` from these records on every app start.
   3. No Hugging Face account is involved.
4. **Evaluate**: `venv/bin/python tools/eval/run_eval.py --model custom/gc-planner-Q4_0`. Selecting this model switches the planner to the `ft` prompt style automatically (`config.style_for`), which drops the few-shot examples to match training.
5. **Use it live**: it is the default now (`app.yaml` and `LLM_MODEL` in `python/config.py`). The base Qwen3-1.7B was deleted from the board to free 1 GB; to compare again, reinstall it with `arduino-app-cli model install llamacpp:unsloth/Qwen3-1.7B-GGUF:Q4_0` and add it back to `LLM_MODELS`.
