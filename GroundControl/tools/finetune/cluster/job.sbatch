#!/bin/bash
#SBATCH --job-name=gc-planner-v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=gc-ft-%j.out

# LoRA fine-tune of Qwen3-1.7B (Ground Control planner v2), then merge and convert to GGUF.
# Submit from ~/gc-ft:  sbatch job.sbatch     Watch:  tail -f gc-ft-<jobid>.out
# Resubmitting after a time limit or preemption resumes from the newest checkpoint.
set -e
cd "$SLURM_SUBMIT_DIR"
type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh 2>/dev/null || true
PYMOD=$(cat "$HOME/mujoco-env/python_module" 2>/dev/null || true)
[ -n "$PYMOD" ] && module load "$PYMOD"
source "$HOME/mujoco-env/bin/activate"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false   # setup.sh downloaded everything

echo "== $(date) on $(hostname) =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "${MEM:-0}" -ge 40000 ]; then B=8; A=2; elif [ "${MEM:-0}" -ge 20000 ]; then B=4; A=4; else B=2; A=8; fi
echo "batch $B x accumulation $A (effective 16)"

python train_lora.py --data data --out gc-lora-v2 --batch $B --accum $A --save-steps 100

echo "== GGUF $(date) =="
python llama.cpp/convert_hf_to_gguf.py gc-lora-v2/merged --outtype f16 --outfile gc-planner-v2-f16.gguf
if [ -x llama.cpp/build/bin/llama-quantize ]; then
  llama.cpp/build/bin/llama-quantize gc-planner-v2-f16.gguf gc-planner-v2-Q4_0.gguf Q4_0
  rm -f gc-planner-v2-f16.gguf
fi
ls -lh gc-planner-v2-*.gguf
echo "== done $(date) =="
