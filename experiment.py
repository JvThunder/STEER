"""Experiment ,  orchestrates an attack across an entire behavior dataset."""

import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from model_context import ModelContext
from dataset_loaders import load_behaviors
from attacks.steer_attack import STEERAttack

class Experiment:
    """Run a chosen attack mode over a dataset and collect statistics.

    Parameters
    ----------
    ctx:            ModelContext produced by ``init_context``.
    openai_client:  OpenAI client (required for CSRT and STEER modes).
    """

    def __init__(self, ctx: ModelContext, openai_client=None):
        self.ctx = ctx
        self.openai_client = openai_client
        self.output_dir: str = "results"

    def _make_attack(
        self,
        mode: str,
        judge_mode: str = "string",
        seed: int = 42,
        lang_preset: str = "all",
    ):
        """Instantiate the correct attack class for *mode*."""
        if mode == "direct":
            return DirectAttack(self.ctx, openai_client=self.openai_client, judge_mode=judge_mode)
        elif mode == "csrt":
            return CSRTAttack(self.ctx, self.openai_client, judge_mode=judge_mode)
        elif mode == "gcg":
            return GCGAttack(self.ctx, judge_mode=judge_mode, seed=seed)
        else:
            return STEERAttack(
                self.ctx,
                openai_client=self.openai_client,
                judge_mode=judge_mode,
                lang_preset=lang_preset,
            )

    def _run_one(
        self,
        attack,
        mode: str,
        goal: str,
        target: str,
        max_iterations: int,
        gcg_kwargs: dict,
        paraphrase: bool = True,
    ) -> dict:
        """Dispatch a single goal to the appropriate attack variant.

        *paraphrase* is the global flag from Experiment.run().
        Modes with fixed paraphrase semantics ('paraphrase_only',
        'steer_no_paraphrase', 'steer') ignore this flag.
        All other modes honour it via their own use_paraphrase parameter.
        """
        if mode == "direct":
            return attack.run(goal, use_paraphrase=paraphrase, verbose=True)
        elif mode == "paraphrase_only":
            # always paraphrase ,  that is the point of this mode
            return attack.run(goal, max_iterations=0, verbose=True, use_paraphrase=True)
        elif mode == "steer_no_paraphrase":
            # always skip paraphrase ,  that is the point of this mode
            return attack.run(goal, max_iterations=max_iterations, verbose=True, use_paraphrase=False)
        elif mode == "csrt":
            return attack.run(goal, max_iterations=max_iterations,
                              use_paraphrase=paraphrase, verbose=True)
        elif mode == "gcg":
            return attack.run(goal, target=target, max_iterations=max_iterations,
                              use_paraphrase=paraphrase, verbose=True, **gcg_kwargs)
        else:  # "steer" (default) ,  always paraphrases internally
            return attack.run(goal, max_iterations=max_iterations, verbose=True, use_paraphrase=True)

    def run(
        self,
        dataset: str = "jbb",
        max_examples: int | None = None,
        max_iterations: int = 8,
        mode: str = "steer",
        # GCG-specific parameters
        judge_mode: str = "gpt",
        seed: int = 42,
        num_adv_tokens: int = 20,
        topk: int = 256,
        batch_size: int = 512,
        mini_batch: int = 64,
        # Judge call budget (0 = unlimited)
        num_judge_calls: int = 0,
        # Global paraphrase flag (applies to all attack modes)
        paraphrase: bool = True,
        shuffle: bool = False,
        output_dir: str = "results",
        # Language ablation preset (STEER only)
        lang_preset: str = "all",
        suffix: str = "",
    ) -> list[dict]:
        """Run the experiment and return all result dicts.

        Parameters
        ----------
        dataset:        "jbb", "harmbench", or "advbench".
        max_examples:   Cap on the number of behaviors tested.
        max_iterations: Passed through to the attack.
        mode:           Attack mode ,  one of:
                        "direct", "paraphrase_only",
                        "steer_no_paraphrase", "steer",
                        "csrt", "gcg".
        judge_mode:     GCG only ,  "string" | "gpt" | "both".
        seed:           GCG only ,  random seed for reproducibility.
        num_adv_tokens: GCG only ,  adversarial suffix length in tokens.
        topk:           GCG only ,  top-K candidates per token position.
        batch_size:     GCG only ,  candidate substitutions per iteration.
        mini_batch:     GCG only ,  forward-pass mini-batch size.
        paraphrase:     Paraphrase goal before running the attack (default: True).
                        Pass False (--no-paraphrase) to skip for all modes.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Determine the output filename and skip if it already exists.
        model_tag = self.ctx.model_name.split("/")[-1]
        if mode == "gcg":
            _out_file = os.path.join(output_dir, f"gcg_results_{dataset}_{model_tag}{suffix}.json")
        elif lang_preset != "all":
            _out_file = os.path.join(output_dir, f"optimized_cs_results_{dataset}_{model_tag}_{mode}_lang-{lang_preset}{suffix}.json")
        else:
            _out_file = os.path.join(output_dir, f"optimized_cs_results_{dataset}_{model_tag}_{mode}{suffix}.json")

        if os.path.exists(_out_file):
            print(f"Output already exists, skipping: {_out_file}")
            return []

        behaviors = load_behaviors(dataset, max_examples=max_examples)
        if shuffle:
            random.shuffle(behaviors)
        print(f"Loaded {len(behaviors)} behaviors")
        print(f"Mode: {mode} | Judge: {judge_mode} | Paraphrase: {paraphrase}\n")

        attack = self._make_attack(mode, judge_mode=judge_mode, seed=seed, lang_preset=lang_preset)

        # ── GCG-specific one-time setup ───────────────────────────────────────
        gcg_kwargs: dict = {}
        gcg_config: dict = {}
        if mode == "gcg":
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            print(f"Random seed: {seed}")

            print("Building filtered adversarial vocabulary...")
            adv_vocab = attack.build_adv_vocab()
            attack.inject_adv_vocab(adv_vocab)
            print(
                f"Adversarial vocab size: {len(adv_vocab)} "
                f"(out of {self.ctx.tokenizer.vocab_size} total)\n"
            )
            gcg_kwargs = dict(
                num_adv_tokens=num_adv_tokens,
                topk=topk,
                batch_size=batch_size,
                mini_batch=mini_batch,
                num_judge_calls=num_judge_calls,
            )
            gcg_config = dict(
                model=self.ctx.model_name,
                dataset=dataset,
                max_examples=max_examples,
                max_iterations=max_iterations,
                num_adv_tokens=num_adv_tokens,
                topk=topk,
                batch_size=batch_size,
                mini_batch=mini_batch,
                judge_mode=judge_mode,
                seed=seed,
            )

        results: list[dict] = []
        model_short = self.ctx.model_name.split("/")[-1]
        pbar = tqdm(behaviors, desc=f"{model_short} | {dataset}", unit="example")

        for idx, behavior in enumerate(pbar):
            goal   = behavior["goal"]
            target = behavior.get("target") or "Sure, here is"
            print(f"\n{'='*70}")
            print(f"[{idx+1}/{len(behaviors)}] Goal: {goal[:60]}...")
            print("="*70)

            result = self._run_one(
                attack, mode, goal, target, max_iterations,
                gcg_kwargs, paraphrase=paraphrase,
            )
            result["goal"]   = goal
            result["target"] = target
            results.append(result)

            print(f"\n--- Result ---")
            print(f"Success: {result['success']}")
            print(f"Paraphrase success: {result.get('paraphrase_success', False)}")
            print(f"Iterations: {result['iterations']}")
            print(f"Translations: {len(result['translations'])}")
            if result.get("paraphrased") and result.get("paraphrased") != goal:
                print(f"Paraphrased: {result['paraphrased'][:80]}...")

            cumulative_successes = sum(1 for r in results if r["success"])
            cumulative_rate = cumulative_successes / len(results) * 100
            print(
                f"Cumulative success rate: "
                f"{cumulative_successes}/{len(results)} ({cumulative_rate:.1f}%)"
            )

            pbar.set_postfix(
                model=model_short,
                benchmark=dataset,
                asr=f"{cumulative_rate:.1f}%",
            )

        if mode == "gcg":
            self._print_gcg_summary(results, dataset, judge_mode, seed)
            self._save_gcg_results(results, dataset, gcg_config)
        else:
            self._print_summary(results, dataset, mode)
            self._save_results(results, dataset, mode, lang_preset, suffix)
        return results

    # ── Reporting ─────────────────────────────────────────────────────────────

    def _print_summary(self, results: list[dict], dataset: str, mode: str) -> None:
        n = len(results)
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)

        successes = sum(1 for r in results if r["success"])
        print(f"Success rate: {successes}/{n} ({successes/n*100:.1f}%)")

        paraphrase_successes = sum(1 for r in results if r.get("paraphrase_success", False))
        print(
            f"Paraphrase success rate: "
            f"{paraphrase_successes}/{n} ({paraphrase_successes/n*100:.1f}%)"
        )

        paraphrase_only_wins = sum(
            1 for r in results
            if r["success"] and r["iterations"] == 0 and r.get("paraphrase_success", False)
        )
        if paraphrase_only_wins > 0:
            print(
                f"Success with paraphrase alone (no code-switching): "
                f"{paraphrase_only_wins}/{successes}"
            )

        avg_iterations = sum(r["iterations"] for r in results) / n
        print(f"Average iterations: {avg_iterations:.1f}")

        avg_translations = sum(len(r["translations"]) for r in results) / n
        print(f"Average translations per prompt: {avg_translations:.1f}")

        # ── Refusal score statistics ───────────────────────────────────────────
        refused_scores, not_refused_scores = [], []
        for r in results:
            for entry in r.get("history", []):
                score = entry.get("refusal_score")
                if score is None:
                    continue
                if entry.get("is_refusing", True):
                    refused_scores.append(score)
                else:
                    not_refused_scores.append(score)

        def _stats(scores, label):
            if not scores:
                print(f"  {label}: no data")
                return
            arr = np.array(scores)
            print(
                f"  {label} (n={len(arr)}): "
                f"mean={arr.mean():.3f}  std={arr.std():.3f}  "
                f"min={arr.min():.3f}  max={arr.max():.3f}  "
                f"median={np.median(arr):.3f}"
            )

        print(f"\n{'='*70}")
        print("REFUSAL SCORE STATISTICS")
        print(f"{'='*70}")
        _stats(refused_scores,     "Refused    ")
        _stats(not_refused_scores, "Not refused")

    def _save_results(self, results: list[dict], dataset: str, mode: str, lang_preset: str = "all", suffix: str = "") -> None:
        model_tag = self.ctx.model_name.split('/')[-1]
        if lang_preset != "all":
            out_file = os.path.join(self.output_dir, f"optimized_cs_results_{dataset}_{model_tag}_{mode}_lang-{lang_preset}{suffix}.json")
        else:
            out_file = os.path.join(self.output_dir, f"optimized_cs_results_{dataset}_{model_tag}_{mode}{suffix}.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {out_file}")

    # ── GCG-specific reporting ────────────────────────────────────────────────

    def _print_gcg_summary(
        self,
        results: list[dict],
        dataset: str,
        judge_mode: str,
        seed: int,
    ) -> None:
        n = len(results)
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)

        successes = sum(1 for r in results if r["success"])
        avg_iters = sum(r["iterations"] for r in results) / n
        print(f"Attack Success Rate: {successes}/{n} ({successes/n*100:.1f}%)")
        print(f"Average iterations:  {avg_iters:.1f}")
        print(f"Judge mode:          {judge_mode}")
        print(f"Seed:                {seed}")

        success_losses, fail_losses = [], []
        for r in results:
            for e in r.get("history", []):
                loss = e.get("loss")
                if loss is None:
                    continue
                (success_losses if e.get("is_harmful") else fail_losses).append(loss)

        def _stats(scores, label):
            if not scores:
                print(f"  {label}: no data")
                return
            a = np.array(scores)
            print(
                f"  {label} (n={len(a)}): mean={a.mean():.4f}  std={a.std():.4f}  "
                f"min={a.min():.4f}  max={a.max():.4f}  median={np.median(a):.4f}"
            )

        print(f"\n{'='*70}")
        print("LOSS STATISTICS")
        print(f"{'='*70}")
        _stats(success_losses, "Harmful     ")
        _stats(fail_losses,    "Not harmful ")

    def _save_gcg_results(
        self, results: list[dict], dataset: str, config: dict
    ) -> None:
        model_tag = self.ctx.model_name.split("/")[-1]
        out_file = os.path.join(self.output_dir, f"gcg_results_{dataset}_{model_tag}.json")
        with open(out_file, "w") as f:
            json.dump({"config": config, "results": results}, f, indent=2, default=str)
        print(f"\nResults saved to {out_file}")
