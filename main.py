"""Entry point for the OOP-refactored STEER attack pipeline."""

import argparse
import os

from dotenv import load_dotenv
from openai import OpenAI

from model_context import load_model, init_context
from experiment import Experiment
from utils import find_best_layer


def _get_num_layers(model) -> int:
    """Return the number of transformer layers (excluding the embedding layer)."""
    cfg = model.config
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(cfg, attr):
            return getattr(cfg, attr)
    raise ValueError(
        f"Cannot determine number of layers from model config. "
        f"Available attributes: {[a for a in dir(cfg) if 'layer' in a.lower()]}"
    )

load_dotenv()

_OPENAI_CLIENT = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY")
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run code-switching jailbreak experiment (OOP version)"
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3-8B-Instruct",
        help=(
            "HuggingFace model ID. Supported families: LLaMA, Gemma, Mistral, "
            "Qwen3, DeepSeek-R1 (e.g. deepseek-ai/DeepSeek-R1-0528-Qwen3-8B)."
        ),
    )
    parser.add_argument(
        "--dataset", type=str, default="jbb",
        choices=["jbb", "harmbench", "advbench"],
        help="Dataset to use (default: jbb)",
    )
    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="Maximum number of examples to test (default: all)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=8,
        help="Maximum code-switching iterations (default: 8)",
    )
    parser.add_argument(
        "--best-layer", type=int, default=None,
        help=(
            "Transformer layer index for refusal-direction extraction. "
            "When omitted, automatically determined via Fisher-ratio analysis "
            "across all layers (recommended)."
        ),
    )
    parser.add_argument(
        "--mode", type=str, default="steer",
        choices=[
            "direct", "paraphrase_only", "steer_no_paraphrase",
            "steer", "csrt", "gcg",
        ],
        help=(
            "Attack mode: "
            "'direct' = send prompt as-is with no modifications; "
            "'paraphrase_only' = paraphrase then attack, no code-switching; "
            "'steer_no_paraphrase' = code-switch only, skip paraphrase step; "
            "'steer' = full pipeline: paraphrase + STEER code-switch (default); "
            "'csrt' = original CSRT method (Yoo et al., 2024); "
            "'gcg' = Greedy Coordinate Gradient (Zou et al., 2023)"
        ),
    )
    # ── GCG-specific arguments ────────────────────────────────────────────────
    parser.add_argument(
        "--judge", type=str, default="gpt",
        choices=["string", "gpt", "both"],
        help=(
            "GCG judge mode: "
            "'string' = fast refusal-substring heuristic (default, paper-comparable); "
            "'gpt' = GPT judge only; "
            "'both' = string first, GPT only when string says not-refusing"
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="GCG random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--num-adv-tokens", type=int, default=20,
        help="GCG adversarial suffix length in tokens (default: 20)",
    )
    parser.add_argument(
        "--topk", type=int, default=256,
        help="GCG top-K candidates per token position (default: 256)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512,
        help="GCG candidate substitutions sampled per iteration (default: 512)",
    )
    parser.add_argument(
        "--mini-batch", type=int, default=64,
        help="GCG forward-pass mini-batch size for candidate evaluation (default: 64)",
    )
    # ── Global paraphrase flag ─────────────────────────────────────────────────
    parser.add_argument(
        "--num-judge-calls", type=int, default=0,
        help="Max GPT judge calls per example (0 = unlimited, default: 0).",
    )
    parser.add_argument(
        "--no-paraphrase", action="store_true",
        help=(
            "Skip GPT-4o-mini goal paraphrase preprocessing for all attack modes. "
            "By default, paraphrase is enabled."
        ),
    )
    parser.add_argument(
        "--shuffle", action="store_true",
        help="Shuffle the dataset examples before running (default: in-order).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory where result files are saved (default: results).",
    )
    parser.add_argument(
        "--suffix", type=str, default="",
        help="Optional suffix appended to the output filename stem (default: empty).",
    )
    parser.add_argument(
        "--layer-ablation", action="store_true",
        help=(
            "Run STEER once per layer variant (see --layer-variants), "
            "loading the model only once. Each variant is saved with a "
            "'_layer-<variant>' suffix."
        ),
    )
    parser.add_argument(
        "--layer-variants", type=str, default="first,mid,last,fisher",
        help=(
            "Comma-separated layer variants to run when --layer-ablation is set. "
            "Options: first, mid, last, fisher (default: all four)."
        ),
    )
    parser.add_argument(
        "--lang-preset", type=str, default="all",
        choices=["all", "top1", "top3", "top5", "bot3", "high3", "non_latin", "latin_only"],
        help=(
            "Language subset for STEER word-translation search. "
            "Rankings are empirical (aggregated substitution counts, all models × all benchmarks). "
            "'all' = all 11 languages (default); "
            "'top1' = Thai; "
            "'top3' = Thai, Korean, Swahili; "
            "'top5' = Thai, Korean, Swahili, Yoruba, Hindi; "
            "'bot3' = Javanese, Arabic, Sundanese (empirical bottom 3); "
            "'high3' = Chinese, German, French (high internet-use languages); "
            "'non_latin' = Thai, Korean, Hindi, Arabic; "
            "'latin_only' = Swahili, Yoruba, Vietnamese, Tagalog, Indonesian, Javanese, Sundanese."
        ),
    )
    return parser


def _run_experiment(args, model, tokenizer, best_layer: int, suffix: str = "") -> None:
    ctx = init_context(args.model, model, tokenizer, best_layer)
    experiment = Experiment(ctx, openai_client=_OPENAI_CLIENT)
    experiment.run(
        dataset=args.dataset,
        max_examples=args.max_examples,
        max_iterations=args.max_iterations,
        mode=args.mode,
        judge_mode=args.judge,
        # GCG
        seed=args.seed,
        num_adv_tokens=args.num_adv_tokens,
        topk=args.topk,
        batch_size=args.batch_size,
        mini_batch=args.mini_batch,
        num_judge_calls=args.num_judge_calls,
        # Global paraphrase flag
        paraphrase=not args.no_paraphrase,
        shuffle=args.shuffle,
        output_dir=args.output_dir,
        lang_preset=args.lang_preset,
        suffix=suffix,
    )


def main() -> None:
    args = build_parser().parse_args()

    model, tokenizer = load_model(args.model)

    if args.layer_ablation:
        wanted = {v.strip() for v in args.layer_variants.split(",") if v.strip()}
        valid = {"first", "mid", "last", "fisher"}
        unknown = wanted - valid
        if unknown:
            raise ValueError(f"Unknown --layer-variants: {unknown}. Valid: {valid}")

        num_layers = _get_num_layers(model)
        print(f"Model has {num_layers} transformer layers (indices 1–{num_layers})")

        fisher_layer = None
        if "fisher" in wanted:
            fisher_layer = find_best_layer(
                model, tokenizer, args.model,
                output_dir=args.output_dir,
            )
            print(f"Fisher-selected layer: {fisher_layer}\n")

        all_variants = {
            "first":  1,
            "mid":    max(1, num_layers // 2),
            "last":   num_layers,
            "fisher": fisher_layer,
        }
        variants = {k: v for k, v in all_variants.items() if k in wanted}
        
        print("Layer variants to run:")
        for name, idx in variants.items():
            print(f"  {name:8s} → layer {idx}")
        print()

        for variant_name, layer_idx in variants.items():
            print(f"\n{'='*60}")
            print(f"Variant: {variant_name}  (layer {layer_idx} / {num_layers})")
            print(f"{'='*60}")
            _run_experiment(args, model, tokenizer, layer_idx,
                            suffix=args.suffix)
    else:
        best_layer = args.best_layer
        if best_layer is None:
            best_layer = find_best_layer(
                model, tokenizer, args.model,
                output_dir=args.output_dir,
            )
            print(f"Auto-selected best_layer={best_layer} for {args.model}")

        _run_experiment(args, model, tokenizer, best_layer, suffix=args.suffix)


if __name__ == "__main__":
    main()
