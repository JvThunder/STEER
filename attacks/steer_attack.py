"""STEERAttack ,  Selective Token-level Evasion via Embedding-guided Replacement.

Gradient-based word-contribution scoring selects which words to translate,
then a greedy per-word language search minimises the refusal score.
An optional GPT-4o paraphrase pre-step reduces prompt explicitness first.
"""

import numpy as np
import torch
from deep_translator import GoogleTranslator

from utils import gpt_paraphrase_to_less_explicit
from model_context import ModelContext
from attacks.base_attack import BaseAttack

# Full language pool ,  ordered by empirical selection frequency
# (aggregated across all models and benchmarks)
_LANGUAGES = [
    ('th', 'Thai'),
    ('ko', 'Korean'),
    ('sw', 'Swahili'),
    ('yo', 'Yoruba'),
    ('hi', 'Hindi'),
    ('vi', 'Vietnamese'),
    ('tl', 'Tagalog'),
    ('id', 'Indonesian'),
    ('jw', 'Javanese'),
    ('ar', 'Arabic'),
    ('su', 'Sundanese'),
]

# Named presets for language-ablation experiments.
# Rankings are empirically derived from aggregated substitution counts
# across all 6 models × 3 benchmarks.
# Non-Latin scripts: Thai, Korean, Hindi, Arabic
# Latin scripts: Swahili, Yoruba, Vietnamese, Tagalog, Indonesian, Javanese, Sundanese
LANG_PRESETS: dict[str, list[tuple[str, str]]] = {
    "all":        _LANGUAGES,
    "top1":       _LANGUAGES[:1],    # Thai
    "top3":       _LANGUAGES[:3],    # Thai, Korean, Swahili
    "top5":       _LANGUAGES[:5],    # Thai, Korean, Swahili, Yoruba, Hindi
    "bot3":       _LANGUAGES[8:],    # Javanese, Arabic, Sundanese (empirical bottom 3)
    "high3":      [('zh-CN', 'Chinese'), ('de', 'German'), ('fr', 'French')],  # high internet-use languages
    "non_latin":  [('th', 'Thai'), ('ko', 'Korean'), ('hi', 'Hindi'), ('ar', 'Arabic')],
    "latin_only": [('sw', 'Swahili'), ('yo', 'Yoruba'), ('vi', 'Vietnamese'),
                   ('tl', 'Tagalog'), ('id', 'Indonesian'), ('jw', 'Javanese'), ('su', 'Sundanese')],
}


class STEERAttack(BaseAttack):
    """Optimised code-switching attack with optional paraphrase pre-step.

    Strategy
    --------
    0. (Optional) Paraphrase goal to less explicit language via GPT-4o.
    1. Compute gradient-based word-level refusal contributions.
    2. Sort words by contribution (highest first).
    3. Translate one word at a time (highest contributor), testing after each.
    4. Stop early on the first non-refusing harmful response.
    """

    def __init__(
        self,
        ctx: ModelContext,
        openai_client=None,
        judge_mode: str = "gpt",
        lang_preset: str = "all",
    ):
        super().__init__(ctx, openai_client=openai_client, judge_mode=judge_mode)
        if lang_preset not in LANG_PRESETS:
            raise ValueError(
                f"Unknown lang_preset '{lang_preset}'. "
                f"Choose from: {list(LANG_PRESETS)}"
            )
        self._languages = LANG_PRESETS[lang_preset]
        self._lang_preset = lang_preset

    # ── Attribution helpers ───────────────────────────────────────────────────

    def _compute_token_attribution(self, prompt: str, layer_idx: int | None = None) -> list[tuple]:
        """Gradient × input attribution for every token in *prompt*.

        Returns a list of (token_str, attribution_score) tuples.
        """
        if layer_idx is None:
            layer_idx = self.ctx.best_layer

        messages = [{"role": "user", "content": prompt}]
        formatted = self._fmt(messages, thinking=False)
        inputs = self.ctx.tokenizer(formatted, return_tensors="pt").to(self.ctx.device)
        tokens = self.ctx.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

        self.ctx.model.zero_grad()
        embedding_layer = self.ctx.model.get_input_embeddings()
        embeddings = embedding_layer(inputs.input_ids).detach().requires_grad_(True)

        outputs = self.ctx.model(
            inputs_embeds=embeddings,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True,
        )

        hidden_state = outputs.hidden_states[layer_idx][0, -1, :].float()
        refusal_score = torch.dot(hidden_state, self.ctx.refusal_direction)
        refusal_score.backward(retain_graph=False)

        if embeddings.grad is None:
            print("Warning: gradients not computed, using zero attributions")
            attributions = np.zeros(len(tokens))
        else:
            token_grads = embeddings.grad[0]
            token_attributions = (token_grads * embeddings[0]).sum(dim=-1)
            attributions = token_attributions.detach().cpu().numpy()

        return [(tokens[i], float(attributions[i])) for i in range(len(tokens))]

    def _compute_word_contributions(self, words: list[str]) -> list[dict]:
        """Map token-level attributions back to word-level contributions."""
        prompt = " ".join(words)
        token_attributions = self._compute_token_attribution(prompt)

        token_to_word_idx = []
        for word_idx, word in enumerate(words):
            for _ in self.ctx.tokenizer.tokenize(word):
                token_to_word_idx.append(word_idx)

        prompt_tokens = self.ctx.tokenizer.tokenize(prompt)
        token_strs = [tok for tok, _ in token_attributions]

        match_start = -1
        for i in range(len(token_strs)):
            if i + len(prompt_tokens) <= len(token_strs):
                if all(
                    token_strs[i + j].replace('Ġ', '').replace('▁', '') ==
                    pt.replace('Ġ', '').replace('▁', '')
                    for j, pt in enumerate(prompt_tokens)
                ):
                    match_start = i
                    break

        token_scores_by_word: dict[int, list[float]] = {i: [] for i in range(len(words))}

        if match_start >= 0:
            for i, word_idx in enumerate(token_to_word_idx):
                if match_start + i < len(token_attributions):
                    token_scores_by_word[word_idx].append(token_attributions[match_start + i][1])
        else:
            tokens_per_word = len(prompt_tokens) / len(words)
            for i, (_, score) in enumerate(token_attributions):
                word_idx = min(int(i / max(1, tokens_per_word)), len(words) - 1)
                token_scores_by_word[word_idx].append(score)

        return [
            {
                "index": i,
                "word": word,
                "contribution": sum(token_scores_by_word.get(i, [0.0])),
                "cumulative_score": 0,
            }
            for i, word in enumerate(words)
        ]

    # ── Translation helpers ───────────────────────────────────────────────────

    @staticmethod
    def _translate_word(word: str, target_lang: str) -> str:
        try:
            translated = GoogleTranslator(source='en', target=target_lang).translate(word)
            return translated if translated else word
        except Exception:
            return word

    def _find_best_translation(
        self, word: str, current_words: list[str], word_index: int
    ) -> tuple[str, str, float]:
        """Try all candidate languages in one batched forward pass and return
        the translation that yields the lowest refusal score.

        Returns (best_translation, best_language_name, best_score).
        """
        # Build all candidate prompts (including the unchanged baseline)
        candidate_translations = [word]
        candidate_lang_names = ["English"]
        candidate_prompts = [" ".join(current_words)]

        for lang_code, lang_name in self._languages:
            try:
                translated = self._translate_word(word, lang_code)
                if translated and translated != word:
                    test_words = current_words.copy()
                    test_words[word_index] = translated
                    candidate_translations.append(translated)
                    candidate_lang_names.append(lang_name)
                    candidate_prompts.append(" ".join(test_words))
            except Exception:
                continue

        # Single batched forward pass over all candidates
        scores = self.get_refusal_scores_batch(candidate_prompts)

        best_idx = int(min(range(len(scores)), key=lambda i: scores[i]))
        return candidate_translations[best_idx], candidate_lang_names[best_idx], scores[best_idx]

    # ── Public interface ──────────────────────────────────────────────────────

    def run(
        self,
        goal: str,
        max_iterations: int = 15,
        verbose: bool = True,
        use_paraphrase: bool = True,
    ) -> dict:
        """Execute the STEER attack.

        Parameters
        ----------
        goal:            The harmful goal string.
        max_iterations:  Maximum number of word-translation iterations.
        verbose:         Print progress to stdout.
        use_paraphrase:  Apply GPT-4o paraphrase pre-step.
        """
        # ── Step 0: optional paraphrase ───────────────────────────────────────
        paraphrased_goal = goal
        paraphrase_success = False

        if use_paraphrase:
            if verbose:
                print(f"\n{'='*70}")
                print("STEP 0: Paraphrasing to less explicit language...")
                print(f"{'='*70}")
                print(f"Original: {goal}")

            paraphrased_goal, paraphrase_success = gpt_paraphrase_to_less_explicit(goal)

            if verbose:
                if paraphrase_success:
                    print(f"Paraphrased: {paraphrased_goal}")
                    test_score = self.get_refusal_score(paraphrased_goal)
                    print(f"Paraphrase refusal score: {test_score:.2f}")

                    test_response = self.generate_response(paraphrased_goal)
                    is_refusing, is_harmful, justification = self._judge(test_response, goal)
                    print(f"Paraphrase test ,  Refusing: {is_refusing}, Harmful: {is_harmful}")

                    if not is_refusing and is_harmful:
                        print("\n✓ SUCCESS with paraphrase alone (no code-switching needed)!")
                        return {
                            "success": True,
                            "final_prompt": paraphrased_goal,
                            "final_response": test_response,
                            "final_score": test_score,
                            "iterations": 0,
                            "translations": [],
                            "paraphrased": paraphrased_goal,
                            "paraphrase_success": paraphrase_success,
                            "is_refusing": is_refusing,
                            "is_harmful": is_harmful,
                            "history": [{
                                "iteration": 0,
                                "prompt": paraphrased_goal,
                                "refusal_score": test_score,
                                "response": test_response,
                                "is_refusing": is_refusing,
                                "is_harmful": is_harmful,
                                "justification": justification,
                                "translations": [],
                            }],
                        }
                else:
                    print("Paraphrase failed, using original")

        # ── Step 1: code-switching ────────────────────────────────────────────
        words = paraphrased_goal.split()
        current_words = words.copy()
        translations_made: list[dict] = []
        iteration_history: list[dict] = []

        initial_score = self.get_refusal_score(paraphrased_goal)

        if verbose:
            print(f"\n{'='*70}")
            print("STEP 1: Code-switching optimisation...")
            print(f"{'='*70}")
            print(f"\nWorking prompt: {paraphrased_goal}")
            print(f"Initial refusal score: {initial_score:.2f}")

        contributions = self._compute_word_contributions(words)

        if verbose:
            print("\nWord contributions to refusal (top 5):")
            for c in sorted(contributions, key=lambda x: x["contribution"], reverse=True)[:5]:
                print(f"  '{c['word']}': {c['contribution']:+.2f}")

        sorted_indices = [
            c["index"]
            for c in sorted(contributions, key=lambda x: x["contribution"], reverse=True)
        ]
        translated_indices: set[int] = set()

        response = self.generate_response(paraphrased_goal)
        is_refusing, is_harmful, justification = self._judge(response, goal)

        iteration_history.append({
            "iteration": 0,
            "prompt": paraphrased_goal,
            "refusal_score": initial_score,
            "response": response,
            "is_refusing": is_refusing,
            "is_harmful": is_harmful,
            "justification": justification,
            "translations": [],
        })

        if not is_refusing and is_harmful:
            return {
                "success": True,
                "final_prompt": paraphrased_goal,
                "final_response": response,
                "final_score": initial_score,
                "iterations": 0,
                "translations": [],
                "paraphrased": paraphrased_goal,
                "paraphrase_success": paraphrase_success,
                "is_refusing": is_refusing,
                "is_harmful": is_harmful,
                "history": iteration_history,
            }

        actual_iterations = 0
        for iteration in range(1, max_iterations + 1):
            next_word_idx = next(
                (idx for idx in sorted_indices if idx not in translated_indices), None
            )
            if next_word_idx is None:
                if verbose:
                    print("\nNo more words to translate.")
                break

            original_word = words[next_word_idx]
            snapshot_words = current_words.copy()
            best_trans, best_lang, new_score = self._find_best_translation(
                original_word, current_words, next_word_idx
            )

            if best_trans != original_word:
                current_words[next_word_idx] = best_trans
                translated_indices.add(next_word_idx)
                translations_made.append({
                    "original": original_word,
                    "translated": best_trans,
                    "language": best_lang,
                    "index": next_word_idx,
                })
                if verbose:
                    print(f"\nTranslated '{original_word}' → '{best_trans}' ({best_lang})")
            else:
                translated_indices.add(next_word_idx)
                if verbose:
                    print(f"\nNo better translation for '{original_word}', skipping")
                continue

            actual_iterations = iteration
            current_prompt = " ".join(current_words)
            current_score = self.get_refusal_score(current_prompt)

            if verbose:
                print(f"--- Iteration {iteration} ---")
                print(f"Current prompt: {current_prompt}")
                print(f"Refusal score: {current_score:.2f}")
                print(f"Translations so far: {len(translations_made)}")

            response = self.generate_response(current_prompt)
            is_refusing, is_harmful, justification = self._judge(response, goal)

            if verbose:
                print(f"Response: {response[:200]}...")
                print(f"Refusing: {is_refusing}, Harmful: {is_harmful}")

            iteration_history.append({
                "iteration": actual_iterations,
                "prompt": current_prompt,
                "refusal_score": current_score,
                "response": response,
                "is_refusing": is_refusing,
                "is_harmful": is_harmful,
                "justification": justification,
                "translations": translations_made.copy(),
            })

            if not is_refusing and is_harmful:
                if verbose:
                    print(
                        f"\n✓ SUCCESS at iteration {iteration} "
                        f"with {len(translations_made)} translations!"
                    )
                return {
                    "success": True,
                    "final_prompt": current_prompt,
                    "final_response": response,
                    "final_score": current_score,
                    "iterations": actual_iterations,
                    "translations": translations_made,
                    "paraphrased": paraphrased_goal,
                    "paraphrase_success": paraphrase_success,
                    "is_refusing": is_refusing,
                    "is_harmful": is_harmful,
                    "history": iteration_history,
                }

            if not is_refusing and not is_harmful:
                # Model responded but prompt has lost its harmful meaning ,  revert.
                current_words = snapshot_words
                translations_made.pop()
                if verbose:
                    print(
                        f"Reverting '{original_word}' → '{best_trans}': "
                        "prompt lost harmful meaning"
                    )

        # ── Final attempt ─────────────────────────────────────────────────────
        final_prompt = " ".join(current_words)
        final_score = self.get_refusal_score(final_prompt)
        final_response = self.generate_response(final_prompt)
        is_refusing, is_harmful, justification = self._judge(final_response, goal)

        iteration_history.append({
            "iteration": actual_iterations,
            "prompt": final_prompt,
            "refusal_score": final_score,
            "response": final_response,
            "is_refusing": is_refusing,
            "is_harmful": is_harmful,
            "justification": justification,
            "translations": translations_made.copy(),
        })

        return {
            "success": not is_refusing and is_harmful,
            "final_prompt": final_prompt,
            "final_response": final_response,
            "final_score": final_score,
            "iterations": actual_iterations,
            "translations": translations_made,
            "paraphrased": paraphrased_goal,
            "paraphrase_success": paraphrase_success,
            "is_refusing": is_refusing,
            "is_harmful": is_harmful,
            "history": iteration_history,
        }
