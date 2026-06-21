"""
model_configs.py
~~~~~~~~~~~~~~~~
OOP model-specific configuration for the code-switching experiment.

Each ModelConfig subclass encapsulates:
  - Chat template (used as fallback when the tokenizer has none)
  - supports_thinking flag (Qwen3 / DeepSeek-R1 family)
  - preprocess_logits()   – any dtype fixups before softmax
  - get_fmt_kwargs()      – extra kwargs passed to apply_chat_template
  - apply_template()      – full formatting, with fallback template injection

Use get_config(model_name) to obtain the right instance automatically.
"""

from __future__ import annotations

import torch
from typing import Optional


# ── Fallback Jinja2 chat templates ────────────────────────────────────────────

_GEMMA_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n{% endif %}"
    "{% if message['role'] == 'assistant' %}<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)

_MISTRAL_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% endif %}"
    "{% if message['role'] == 'assistant' %}{{ message['content'] }}</s>{% endif %}"
    "{% endfor %}"
)

_QWEN_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n{% endif %}"
    "{% if message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n{% endif %}"
    "{% if message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

# DeepSeek-R1 uses the same ChatML format as Qwen, with <think>…</think> blocks
# injected by the model itself.  The fallback template is therefore identical;
# enable_thinking=False is passed via get_fmt_kwargs when thinking is disabled.
_DEEPSEEK_TEMPLATE = _QWEN_TEMPLATE

_LLAMA_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "{{ messages[0]['content'] }}<|eot_id|>"
    "{% set messages = messages[1:] %}{% endif %}"
    "{% if not messages[0]['role'] == 'system' %}<|begin_of_text|>{% endif %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
    "{% endif %}"
    "{% if message['role'] == 'assistant' %}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
)

_GENERIC_TEMPLATE = None  # handled in code


# ── Base class ────────────────────────────────────────────────────────────────

class ModelConfig:
    """Base configuration for a model family.

    Subclass and override the class-level attributes and/or methods to
    customise behaviour for a specific model family.
    """

    #: Canonical model-name keyword used for auto-detection (lower-case substring).
    family_key: str = ""

    #: Fallback Jinja2 chat template (used when the tokenizer has none set).
    chat_template: Optional[str] = None

    #: Whether this model supports an explicit thinking/reasoning toggle.
    supports_thinking: bool = False

    #: Whether to pass trust_remote_code=True when loading the model/tokenizer.
    trust_remote_code: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_fmt_kwargs(
        self,
        add_generation_prompt: bool = True,
        thinking: Optional[bool] = None,
    ) -> dict:
        """Return extra kwargs to pass to tokenizer.apply_chat_template.

        Subclasses that support thinking mode should override this to inject
        ``enable_thinking=False`` when ``thinking is False``.
        """
        return {"tokenize": False, "add_generation_prompt": add_generation_prompt}

    def preprocess_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Optional logit preprocessing before softmax (e.g. dtype cast).

        Override in subclasses that need it (e.g. Qwen3/DeepSeek BFloat16).
        """
        return logits

    def apply_template(
        self,
        tokenizer,
        messages: list[dict],
        add_generation_prompt: bool = True,
        thinking: Optional[bool] = None,
    ) -> str:
        """Format *messages* using the tokenizer's built-in template or the
        class-level fallback.  Injects the fallback template into the tokenizer
        object only when necessary (no-op if the tokenizer already has one).
        """
        if not tokenizer.chat_template and self.chat_template:
            tokenizer.chat_template = self.chat_template

        if tokenizer.chat_template:
            kwargs = self.get_fmt_kwargs(add_generation_prompt, thinking)
            return tokenizer.apply_chat_template(messages, **kwargs)

        # Last-resort generic formatter
        parts = []
        for m in messages:
            if m["role"] == "user":
                parts.append(f"### User:\n{m['content']}\n")
            elif m["role"] == "assistant":
                parts.append(f"### Assistant:\n{m['content']}\n")
        if add_generation_prompt:
            parts.append("### Assistant:\n")
        return "".join(parts)

    def extract_response(self, decoded: str) -> str:
        """Extract the model's response from a fully decoded generation string.

        The default implementation strips everything up to and including the
        last occurrence of the word "assistant" (case-insensitive), which
        covers most chat-formatted outputs.
        """
        lower = decoded.lower()
        idx = lower.rfind("assistant")
        if idx != -1:
            return decoded[idx + len("assistant"):].strip()
        return decoded.strip()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"family_key={self.family_key!r}, "
            f"supports_thinking={self.supports_thinking})"
        )


# ── Concrete model-family configs ─────────────────────────────────────────────

class LlamaConfig(ModelConfig):
    """Meta LLaMA / LLaMA-3 family."""

    family_key = "llama"
    chat_template = _LLAMA_TEMPLATE


class GemmaConfig(ModelConfig):
    """Google Gemma / Gemma-2 family."""

    family_key = "gemma"
    chat_template = _GEMMA_TEMPLATE


class MistralConfig(ModelConfig):
    """Mistral / Mixtral family."""

    family_key = "mistral"
    chat_template = _MISTRAL_TEMPLATE


class QwenConfig(ModelConfig):
    """Qwen / Qwen2 / Qwen3 family (includes thinking toggle for Qwen3)."""

    family_key = "qwen"
    chat_template = _QWEN_TEMPLATE
    supports_thinking = True  # Qwen3 supports enable_thinking

    def get_fmt_kwargs(
        self,
        add_generation_prompt: bool = True,
        thinking: Optional[bool] = None,
    ) -> dict:
        kwargs: dict = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
        if thinking is False:
            kwargs["enable_thinking"] = False
        return kwargs

    def preprocess_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.float()


class DeepSeekR1Config(ModelConfig):
    """DeepSeek-R1 family (reasoning models based on Qwen3 / other backbones).

    Covers model IDs matching ``deepseek`` (e.g.
    ``deepseek-ai/DeepSeek-R1-0528-Qwen3-8B``).

    Key behaviour:
    - The tokenizer's ``enable_thinking`` parameter is silently ignored by
      this model's template (both variants produce ``<｜Assistant｜>`` at the
      end).  The model's weights still strongly predict the ``<think>`` token
      (ID 151667) as the very first response token, so measuring logits at
      position -1 after the bare template gives ~zero probability for
      "Sorry"/"I"/etc. and an identically-zero refusal score.
    - Fix: when ``thinking=False`` (used for refusal scoring and forced direct
      generation), append an empty think block ``<think>\\n\\n</think>\\n`` so
      position -1 is *after* the thinking placeholder and the model predicts
      the actual first response word.
    """

    family_key = "deepseek"
    chat_template = _DEEPSEEK_TEMPLATE
    supports_thinking = True

    # Token strings for the thinking delimiters (same IDs as Qwen3).
    _THINK_OPEN  = "<think>"   # token 151667
    _THINK_CLOSE = "</think>"  # token 151668
    _EMPTY_THINK = "<think>\n\n</think>\n"

    def get_fmt_kwargs(
        self,
        add_generation_prompt: bool = True,
        thinking: Optional[bool] = None,
    ) -> dict:
        # enable_thinking is ignored by this tokenizer's template, but passing
        # it does no harm and keeps the intent explicit.
        kwargs: dict = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
        if thinking is False:
            kwargs["enable_thinking"] = False
        return kwargs

    def apply_template(
        self,
        tokenizer,
        messages: list[dict],
        add_generation_prompt: bool = True,
        thinking: Optional[bool] = None,
    ) -> str:
        text = super().apply_template(tokenizer, messages, add_generation_prompt, thinking)
        # Append an empty think block so that logits at position -1 reflect the
        # first real response token rather than the <think> token.
        if thinking is False and add_generation_prompt:
            text += self._EMPTY_THINK
        return text

    def extract_response(self, decoded: str) -> str:
        """Strip the thinking block and template prefix from generated text."""
        # If thinking tokens are present (not stripped), take content after </think>
        if self._THINK_CLOSE in decoded:
            return decoded.split(self._THINK_CLOSE)[-1].strip()
        # Fallback: split on the DeepSeek assistant marker
        marker = "<｜Assistant｜>"
        if marker in decoded:
            return decoded.split(marker)[-1].strip()
        return decoded.strip()

    def preprocess_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.float()


class GLMConfig(ModelConfig):
    """THUDM / Zhipu AI ChatGLM / GLM-4 family."""

    family_key = "glm"
    trust_remote_code = True

    def extract_response(self, decoded: str) -> str:
        # GLM-4 format: ...<|assistant|>\n{response}<|endoftext|>
        marker = "<|assistant|>"
        if marker in decoded:
            return decoded.split(marker)[-1].strip()
        return decoded.strip()


# ── Registry & factory ────────────────────────────────────────────────────────

# Ordered list of (substring, config_instance).  First match wins.
_REGISTRY: list[tuple[str, ModelConfig]] = [
    ("deepseek",  DeepSeekR1Config()),
    ("qwen",      QwenConfig()),
    ("llama",     LlamaConfig()),
    ("gemma",     GemmaConfig()),
    ("mistral",   MistralConfig()),
    ("glm",       GLMConfig()),
]

_DEFAULT_CONFIG = ModelConfig()


def get_config(model_name: str) -> ModelConfig:
    """Return the ModelConfig instance that matches *model_name*.

    Matching is done by looking for the family keyword (case-insensitive
    substring) in the model name.  Falls back to the generic ModelConfig.

    Example::

        cfg = get_config("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
        # → DeepSeekR1Config instance
    """
    name_lower = model_name.lower()
    for keyword, cfg in _REGISTRY:
        if keyword in name_lower:
            return cfg
    return _DEFAULT_CONFIG


def register_config(keyword: str, config: ModelConfig, priority: bool = False) -> None:
    """Register a custom ModelConfig at runtime.

    Args:
        keyword:  Lower-case substring that must appear in the model name.
        config:   ModelConfig (or subclass) instance.
        priority: If True, prepend so this config is checked before all others.
    """
    entry = (keyword.lower(), config)
    if priority:
        _REGISTRY.insert(0, entry)
    else:
        _REGISTRY.append(entry)
