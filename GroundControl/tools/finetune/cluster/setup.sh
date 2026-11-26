#!/usr/bin/env bash
# Runs ON the Tufts HPC login node, once. Adds the fine-tuning libraries to the venv the
# MuJoCo work already built (~/mujoco-env: Python 3.13, PyTorch 2.11 for CUDA 12.8), and
# downloads everything the job needs, so the GPU job itself never needs the internet.
set -e
type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh 2>/dev/null || true
cd "$(dirname "$0")"
PYMOD=$(cat "$HOME/mujoco-env/python_module" 2>/dev/null || true)
[ -n "$PYMOD" ] && module load "$PYMOD"
source "$HOME/mujoco-env/bin/activate"

echo "== libraries (PyTorch pinned, so the MuJoCo setup keeps working) =="
python -c "import torch; print('torch', torch.__version__)" | tee /dev/stderr | sed 's/torch /torch==/' > constraints.txt
python -m pip install --no-cache-dir -q -c constraints.txt \
    "transformers>=4.56" "trl>=0.19" "peft>=0.15" "datasets>=3.0" "accelerate>=1.0" sentencepiece protobuf gguf
python -c "import transformers, trl, peft, datasets; print('transformers', transformers.__version__, '| trl', trl.__version__, '| peft', peft.__version__)"

echo "== base model and general instruction data (into ~/.cache/huggingface) =="
python - <<'EOF'
from huggingface_hub import snapshot_download
from datasets import load_dataset
print("model at", snapshot_download("Qwen/Qwen3-1.7B"))
print("alpaca-cleaned:", len(load_dataset("yahma/alpaca-cleaned", split="train")), "rows")
EOF

echo "== llama.cpp (the GGUF converter is plain Python; the quantizer needs a compiler) =="
[ -d llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp
for m in cmake gcc; do module load "$(module -t avail $m 2>&1 | grep -E "^$m/" | sort -V | tail -1)" 2>/dev/null || true; done
if command -v cmake >/dev/null && cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF >/dev/null 2>&1 \
   && cmake --build llama.cpp/build --target llama-quantize -j 8 >/dev/null 2>&1; then
  echo "quantizer OK (Q4_0 is made on the cluster)"
else
  echo "no quantizer here: the job stops at the f16 GGUF and the laptop quantizes it"
fi

echo "== disk =="
quota -s 2>/dev/null | tail -2 || true
du -sh "$HOME/.cache/huggingface" 2>/dev/null || true
echo "setup done"
