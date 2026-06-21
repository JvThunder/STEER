# STEER Attack - Selective Token-level Evasion via Embedding-guided Replacement

A research implementation of the STEER (Selective Token-level Evasion via Embedding-guided Replacement) attack for testing language model safety mechanisms through code-switching and adversarial prompt manipulation.

## Overview

STEER is an adversarial attack method that uses:
- **Gradient-based word contribution scoring** to identify which words to translate
- **Greedy per-word language search** to minimize refusal scores
- **Optional GPT-4o paraphrase preprocessing** to reduce prompt explicitness
- **Multi-language translation** (Thai, Korean, Swahili, Yoruba, Hindi, Vietnamese, Tagalog, Indonesian, Javanese, Arabic, Sundanese)

The codebase supports multiple language models (LLaMA, Gemma, Mistral, Qwen3, DeepSeek-R1) and safety benchmarks (JailbreakBench, HarmBench, AdvBench).

## Project Structure

```
STEER-attack/
├── main.py                  # Main entry point for running experiments
├── experiment.py            # Orchestrates attacks across datasets
├── model_configs.py         # Model-specific configurations and templates
├── model_context.py         # Model loading and context initialization
├── dataset_loaders.py       # Dataset loading utilities
├── utils.py                 # Shared utility functions (GPT judge, Fisher ratio, etc.)
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (OpenAI API key)
├── .gitignore              # Git ignore rules
│
├── attacks/                 # Attack implementations
│   ├── __init__.py         # Attack module exports
│   ├── base_attack.py      # Base attack class with shared utilities
│   └── steer_attack.py     # STEER attack implementation
│
├── scripts/                 # Execution scripts
│   └── run_steer.sh        # Batch script for running multiple experiments
│
└── results/                 # Output directory (created at runtime)
    └── *.json              # Experiment results (JSON format)
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for model inference)
- OpenAI API key (for GPT-4o paraphrase and judging)
- HuggingFace account and authentication token

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd STEER-attack
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **HuggingFace Authentication:**

Login to HuggingFace to download models and datasets:
```bash
huggingface-cli login
```

Enter your HuggingFace token when prompted. You can get a token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

Alternatively, set the token as an environment variable:
```bash
export HF_TOKEN=your_huggingface_token_here
```

**Note:** Some models (e.g., LLaMA, Gemma) require accepting their license agreements on HuggingFace before downloading.

4. **Configure environment variables:**

Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Quick Start

After completing setup, run your first experiment:

```bash
# 1. Test with a small example (10 samples from JailbreakBench)
python main.py \
    --model meta-llama/Llama-3-8B-Instruct \
    --dataset jbb \
    --max-examples 10 \
    --mode steer

# 2. Results will be saved to: results/llama-3-8b-instruct_jbb_steer_<timestamp>.json
```

### Basic Usage

Run a full STEER attack experiment:
```bash
python main.py \
    --model meta-llama/Llama-3-8B-Instruct \
    --dataset jbb \
    --max-examples 50 \
    --mode steer \
    --output-dir results
```

### Command-Line Arguments

#### Core Arguments
- `--model`: HuggingFace model ID (default: `meta-llama/Llama-3-8B-Instruct`)
  - Supported: LLaMA, Gemma, Mistral, Qwen3, DeepSeek-R1
- `--dataset`: Dataset to use (choices: `jbb`, `harmbench`, `advbench`, default: `jbb`)
- `--max-examples`: Maximum number of examples to test (default: all)
- `--max-iterations`: Maximum code-switching iterations (default: 8)
- `--best-layer`: Transformer layer for refusal direction extraction (auto-detected if omitted)

#### Attack Modes
- `--mode`: Attack strategy (default: `steer`)
  - `direct`: Send prompt as-is with no modifications
  - `paraphrase_only`: Paraphrase then attack, no code-switching
  - `steer_no_paraphrase`: Code-switch only, skip paraphrase step
  - `steer`: Full pipeline: paraphrase + STEER code-switch (**recommended**)
  - `csrt`: Original CSRT method (Yoo et al., 2024)
  - `gcg`: Greedy Coordinate Gradient (Zou et al., 2023)

#### Language Selection
- `--lang-preset`: Language subset for STEER word-translation (default: `all`)
  - `all`: All 11 languages
  - `top1`: Thai only
  - `top3`: Thai, Korean, Swahili
  - `top5`: Thai, Korean, Swahili, Yoruba, Hindi
  - `bot3`: Javanese, Arabic, Sundanese
  - `high3`: Chinese, German, French
  - `non_latin`: Thai, Korean, Hindi, Arabic
  - `latin_only`: Swahili, Yoruba, Vietnamese, Tagalog, Indonesian, Javanese, Sundanese

#### Layer Ablation
- `--layer-ablation`: Run STEER once per layer variant
- `--layer-variants`: Comma-separated layer variants (default: `first,mid,last,fisher`)
  - Options: `first`, `mid`, `last`, `fisher`

#### GCG-Specific Arguments
- `--judge`: Judge mode for GCG (default: `gpt`)
  - `string`: Fast refusal-substring heuristic
  - `gpt`: GPT judge only
  - `both`: String first, GPT when string says not-refusing
- `--seed`: Random seed for reproducibility (default: 42)
- `--num-adv-tokens`: Adversarial suffix length in tokens (default: 20)
- `--topk`: Top-K candidates per token position (default: 256)
- `--batch-size`: Candidate substitutions per iteration (default: 512)
- `--mini-batch`: Forward-pass mini-batch size (default: 64)

#### Other Options
- `--no-paraphrase`: Skip GPT-4o paraphrase preprocessing
- `--num-judge-calls`: Max GPT judge calls per example (0 = unlimited, default: 0)
- `--shuffle`: Shuffle dataset examples before running
- `--output-dir`: Directory for result files (default: `results`)
- `--suffix`: Optional suffix for output filename

### Example Commands

**Full STEER attack on JailbreakBench:**
```bash
python main.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset jbb \
    --max-examples 100 \
    --mode steer \
    --lang-preset top5
```

**Layer ablation study:**
```bash
python main.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --dataset harmbench \
    --mode steer \
    --layer-ablation \
    --layer-variants first,mid,last,fisher
```

**Direct attack (baseline):**
```bash
python main.py \
    --model google/gemma-7b \
    --dataset advbench \
    --max-examples 50 \
    --mode direct
```

**Using batch script:**
```bash
bash scripts/run_steer.sh
```

## Results

### Output Location

Results are saved in JSON format to the directory specified by `--output-dir` (default: `results/`).

### Output Filename Format

```
{model_name}_{dataset}_{mode}_{timestamp}.json
```

Examples:
- `llama-3-8b-instruct_jbb_steer_2024-03-23_143022.json`
- `mistral-7b-instruct_harmbench_steer_layer-fisher_2024-03-23_150145.json`

### Output Structure

Each result file contains:
```json
{
  "metadata": {
    "model": "meta-llama/Llama-3-8B-Instruct",
    "dataset": "jbb",
    "mode": "steer",
    "best_layer": 15,
    "lang_preset": "all",
    "total_examples": 100,
    "max_iterations": 8
  },
  "results": [
    {
      "goal": "Write a tutorial on how to...",
      "target": "",
      "category": "harmful",
      "success": true,
      "is_harmful": true,
      "num_iterations": 3,
      "initial_score": 15.234,
      "final_score": -2.145,
      "original_prompt": "...",
      "final_prompt": "...",
      "response": "...",
      "justification": "..."
    }
  ],
  "statistics": {
    "success_rate": 0.78,
    "average_iterations": 4.2,
    "harmful_rate": 0.65
  }
}
```

### Key Metrics

- **success**: Whether the attack succeeded (got past refusal mechanisms)
- **is_harmful**: Whether the response contained actionable harmful information
- **num_iterations**: Number of code-switching iterations required
- **initial_score**: Refusal score before attack
- **final_score**: Refusal score after attack (negative = likely compliance)
- **success_rate**: Percentage of successful attacks
- **harmful_rate**: Percentage of responses judged as harmful

## Supported Models

### Tested Models
- **LLaMA**: `meta-llama/Meta-Llama-3-8B-Instruct`
- **Mistral**: `mistralai/Mistral-7B-Instruct-v0.2`
- **Gemma**: `google/gemma-7b`
- **Qwen3**: `Qwen/Qwen3-8B`
- **GLM**: `zai-org/GLM-4-9B-0414`
- **DeepSeek-R1**: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`

### Model-Specific Features
- **Automatic chat template detection** with fallback templates
- **DeepSeek-R1/Qwen3 thinking support** (automatically disabled for STEER scoring)
- **Automatic layer count detection** for layer ablation studies

## Datasets

### JailbreakBench (jbb)
- Source: `JailbreakBench/JBB-Behaviors`
- Contains curated harmful behavior requests
- Used for jailbreak testing

### HarmBench
- Source: `walledai/HarmBench`
- Standardized harmful behavior benchmark
- Multiple semantic categories

### AdvBench
- Source: `walledai/AdvBench`
- Adversarial prompts for safety testing
- Classic red-teaming dataset

## Fisher Ratio Analysis

The codebase automatically determines the best transformer layer for refusal direction extraction using Fisher ratio analysis:

```python
python main.py --model <model_name>  # Auto-selects best layer
```

Or specify manually:
```bash
python main.py --model <model_name> --best-layer 15
```

Fisher ratio analysis results are saved to `{output_dir}/fisher_ratio_{model_name}.json`.

## Dependencies

Key dependencies (see `requirements.txt` for full list):
- `torch>=2.0.0` - PyTorch for model inference
- `transformers>=4.40.0` - HuggingFace Transformers
- `openai>=1.0.0` - OpenAI API client
- `deep-translator>=1.11.0` - Multi-language translation
- `datasets>=2.14.0` - HuggingFace Datasets
- `accelerate>=0.24.0` - Model loading optimization
- `bitsandbytes>=0.46.1` - Quantization support

## Troubleshooting

### HuggingFace Access Issues

**"Repository not found" or "Access denied" errors:**
1. Ensure you've accepted the model's license agreement on HuggingFace:
   - LLaMA models: Visit [Meta-Llama model page](https://huggingface.co/meta-llama) and accept the license
   - Gemma models: Visit [Google Gemma page](https://huggingface.co/google/gemma-7b) and accept the license
2. Verify you're logged in: `huggingface-cli whoami`
3. Check your token has read permissions

**Dataset loading errors:**
- Ensure you're logged in to HuggingFace
- Some datasets may be gated and require access approval

### CUDA Out of Memory
- Reduce `--max-examples` to process fewer samples
- Use smaller models (e.g., 7B instead of 13B)
- Reduce `--batch-size` and `--mini-batch` for GCG mode
- Enable model quantization (if supported by your setup)

### OpenAI API Errors
- Verify your API key is valid and has credits
- Check rate limits if running large experiments
- Use `--num-judge-calls` to limit GPT judge calls

## Notes

⚠️ **Important**: This codebase is for research purposes only. It is designed to evaluate and improve AI safety mechanisms, not to enable harmful behavior.
