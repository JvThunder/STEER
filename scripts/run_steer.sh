#!/bin/bash
# Run main.py for all models and datasets sequentially.
# Supports multiple attack modes: csrt, gcg, steer, direct, etc.

SCRIPT="main.py"
OUTPUT_DIR="results/STEER"

DATASETS=("jbb" "harmbench" "advbench")

MODELS=(
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.2"
    "google/gemma-7b"
    "Qwen/Qwen3-8B"
    "zai-org/GLM-4-9B-0414"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)

run_model() {
    local model="$1"
    local dataset="$2"
    local mode="$3"
    shift 3
    local extra_args=("$@")

    echo ""
    echo "======================================================"
    echo "Running: $model | dataset: $dataset | mode: $mode"
    echo "======================================================"
    python "$SCRIPT" \
        --model "$model" \
        --dataset "$dataset" \
        --max-examples 50 \
        --mode "$mode" \
        --output-dir "$OUTPUT_DIR" \
        "${extra_args[@]}"
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "ERROR: $model failed with exit code $exit_code (dataset=$dataset mode=$mode)"
    fi
}