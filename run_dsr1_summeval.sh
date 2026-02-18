#!/usr/bin/env bash
source /home/divake/miniconda3/etc/profile.d/conda.sh
conda activate env_py311

MODEL="models/DeepSeek-R1-Distill-Qwen-32B"
DATA="summeval/data/summeval.json"
PROMPT_DIR="summeval/prompts/summeval"
SAVE_DIR="results/dsr1"
mkdir -p "$SAVE_DIR"

for dim in con coh flu rel; do
    echo ""
    echo "=================================================="
    echo "  Dimension: $dim   $(date)"
    echo "=================================================="
    python "evaluations and reprompt on server/qwen_eval.py" \
        --prompt_fp   "${PROMPT_DIR}/${dim}_detailed.txt" \
        --summeval_fp "$DATA" \
        --save_fp     "${SAVE_DIR}/dsr1_${dim}_summeval.json" \
        --model       "$MODEL" \
        --temperature 0 \
        --max_new_tokens 800
    echo "  Done: $dim at $(date)"
done

echo ""
echo "All 4 dimensions complete at $(date)"
