"""Gemma-4 text tower wrapper exposing the 2.3-era ``GemmaLanguageModel`` contract.

``GemmaFeaturesExtractorV2`` (``feature_extractor.py``) is model-version
agnostic: it consumes ``(all_hidden_states: list[mx.array], attention_mask:
mx.array | None)`` -- a plain list of ``num_gemma_layers`` ``(B, T, D)``
tensors plus a ``(B, T)`` binary mask -- and does not touch the encoder
object itself. What ties an encoder into the pipeline is the *caller*
contract exercised by ``PromptEncoder.encode()``
(``ltx_pipelines_mlx/utils/blocks.py:129`` and the trainer's
``preprocess.py`` / ``trainer.py``):

    all_hidden_states, attention_mask = text_encoder.encode_all_layers(prompt, max_length=max_length)

``GemmaLanguageModel.encode_all_layers`` (``base_encoder.py:190-203``) is
the exact method surface this module reproduces for Gemma-4:

* ``tokenize(text, max_length=1024) -> (token_ids, attention_mask)`` --
  LEFT-padded to ``max_length`` with the tokenizer's native pad token id
  (this port's divergence, documented at the top of ``base_encoder.py``:
  upstream right-pads before the connector with an additive mask; we keep
  LEFT padding + a binary mask end-to-end). ``attention_mask`` is ``1`` on
  real tokens, ``0`` on padding.
* ``get_all_hidden_states(token_ids, attention_mask) -> list[mx.array]`` --
  ``num_gemma_layers`` tensors, one per layer (embeddings + every decoder
  layer's output for Gemma-3's 49; identically 49 for Gemma-4's tower:
  scaled embeddings + 48 decoder layers).
* ``encode_all_layers(text, max_length=1024) -> (list[mx.array], mx.array)``
  -- tokenize + forward in one call; this is the method
  ``PromptEncoder.encode()`` actually calls.
* ``encode(text, max_length=1024) -> mx.array`` -- convenience: just the
  last hidden state.

HARD REQUIREMENT (Task 3 review, ledger-recorded): HF's
``output_hidden_states=True`` tuple ties its *last* entry to the
post-final-norm ``last_hidden_state``
(``transformers/output_capturing.py:261-263``), and upstream's
``LTXGemmaTextEncoder.encode`` (see
``/private/tmp/.../upstream_gemma_base_encoder.py:68-71``) consumes exactly
that tuple. ``Gemma4TextModel.__call__`` (Task 3) intentionally returns the
**raw pre-norm** stack (see its docstring) so the tower stays a pure
port of ``output_hidden_states`` collection; this module is what applies
``tower.norm`` to the last entry before the stack reaches
``GemmaFeaturesExtractorV2``. Get this wrong and every 2.5 text embedding
is silently wrong -- entries 0..47 must stay byte-identical to the raw
tower output, and entry -1 must equal ``tower.norm(raw[-1])``, not
``raw[-1]`` itself.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.text_encoders.gemma.gemma4 import Gemma4TextModel

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# Mirrors GemmaLanguageModel's default (base_encoder.py) and
# GemmaAssets.TOKENIZER_MAX_LENGTH upstream.
TOKENIZER_MAX_LENGTH = 1024


class Gemma4TextEncoder(nn.Module):
    """Gemma-4 text tower + tokenizer, exposing the 2.3 encoder's method surface.

    Args:
        tower: A pre-built :class:`Gemma4TextModel`, or ``None`` to load
            later via :meth:`load`. Exposed as a constructor arg (rather
            than only via ``load``) so tests can inject a tiny tower
            without touching the real 16 GB pack.
        tokenizer: A pre-built HuggingFace tokenizer, or ``None`` to load
            later via :meth:`load` / :meth:`load_tokenizer`.
    """

    def __init__(
        self,
        tower: Gemma4TextModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ):
        super().__init__()
        self._tower = tower
        self._tokenizer = tokenizer

    @property
    def tower(self) -> Gemma4TextModel | None:
        """The underlying :class:`Gemma4TextModel`, or ``None`` if unloaded."""
        return self._tower

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase | None:
        """The underlying HuggingFace tokenizer, or ``None`` if unloaded."""
        return self._tokenizer

    def load_tokenizer(self, model_dir: str | Path) -> None:
        """Load the tokenizer from a pack directory's ``tokenizer.json``.

        Uses ``transformers.AutoTokenizer`` (a transitive dependency
        already pulled in by ``ltx-trainer``; no new project dependency).
        The pack's ``tokenizer_config.json`` already sets
        ``padding_side: "left"`` and ``pad_token: "<pad>"``, matching this
        port's LEFT-pad convention.

        Args:
            model_dir: Path to the pack directory (containing
                ``tokenizer.json`` and ``tokenizer_config.json``).
        """
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    def load(self, model_dir: str | Path) -> None:
        """Load both the tower and the tokenizer from a pack directory.

        Args:
            model_dir: Path to the pack directory.
        """
        self.load_tokenizer(model_dir)
        self._tower = Gemma4TextModel.load_from_pack(model_dir)

    def tokenize(self, text: str, max_length: int = TOKENIZER_MAX_LENGTH) -> tuple[mx.array, mx.array]:
        """Tokenize a text string with left-padding to ``max_length``.

        Mirrors ``GemmaLanguageModel.tokenize`` (``base_encoder.py:52-79``)
        exactly: strip, encode, keep the *last* ``max_length`` tokens on
        overflow (left-truncation matches left-padding), left-pad with the
        tokenizer's native pad token id.

        Args:
            text: Input text.
            max_length: Sequence length (padded/truncated to this length).

        Returns:
            ``(token_ids, attention_mask)``, each shape ``(1, max_length)``.
            ``attention_mask``: ``1`` for valid tokens, ``0`` for padding.
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() or load_tokenizer() first.")

        tokens = self._tokenizer.encode(text.strip())
        if len(tokens) > max_length:
            tokens = tokens[-max_length:]

        pad_length = max_length - len(tokens)
        pad_token = self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else 0
        padded_tokens = [pad_token] * pad_length + tokens
        attention_mask = [0] * pad_length + [1] * len(tokens)

        return mx.array([padded_tokens]), mx.array([attention_mask])

    def get_all_hidden_states(
        self,
        token_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> list[mx.array]:
        """Run the tower and return all layers' hidden states, last entry post-norm.

        Args:
            token_ids: ``(B, seq_len)`` token IDs.
            attention_mask: ``(B, seq_len)`` binary mask (1=valid, 0=padding).

        Returns:
            ``num_hidden_layers + 1`` ``(B, seq_len, hidden_dim)`` tensors
            (49 for the real pack). Entries ``0..-2`` are the tower's raw
            per-layer outputs, unchanged. Entry ``-1`` is
            ``tower.norm(raw[-1])`` -- the post-final-norm
            ``last_hidden_state``, matching what upstream's
            ``output_hidden_states=True`` tuple actually contains (see
            module docstring).
        """
        if self._tower is None:
            raise RuntimeError("Tower not loaded. Call load() first.")

        raw_states = self._tower(token_ids, attention_mask=attention_mask)
        hidden_states = list(raw_states)
        hidden_states[-1] = self._tower.norm(hidden_states[-1])
        return hidden_states

    def encode(self, text: str, max_length: int = TOKENIZER_MAX_LENGTH) -> mx.array:
        """Tokenize and extract the final (post-norm) hidden state in one call.

        Args:
            text: Input text.
            max_length: Padded sequence length.

        Returns:
            Hidden states of shape ``(1, max_length, hidden_dim)``.
        """
        token_ids, attention_mask = self.tokenize(text, max_length)
        all_states = self.get_all_hidden_states(token_ids, attention_mask=attention_mask)
        return all_states[-1]

    def encode_all_layers(self, text: str, max_length: int = TOKENIZER_MAX_LENGTH) -> tuple[list[mx.array], mx.array]:
        """Tokenize and extract ALL layer hidden states in one call.

        This is the method ``PromptEncoder.encode()``
        (``ltx_pipelines_mlx/utils/blocks.py:129``) and the trainer call.

        Args:
            text: Input text.
            max_length: Padded sequence length.

        Returns:
            ``(hidden_states, attention_mask)``:
            - ``hidden_states``: list of ``(1, max_length, hidden_dim)``
              tensors (49 total), last entry post-final-norm.
            - ``attention_mask``: ``(1, max_length)`` binary mask.
        """
        token_ids, attention_mask = self.tokenize(text, max_length)
        hidden_states = self.get_all_hidden_states(token_ids, attention_mask=attention_mask)
        return hidden_states, attention_mask

    @functools.cached_property
    def default_gemma4_t2v_system_prompt(self) -> str:
        """Load the default Gemma-4 T2V system prompt."""
        return _load_system_prompt("gemma4_t2v_system_prompt.txt")

    @functools.cached_property
    def default_gemma4_i2v_system_prompt(self) -> str:
        """Load the default Gemma-4 I2V system prompt."""
        return _load_system_prompt("gemma4_i2v_system_prompt.txt")


@functools.lru_cache(maxsize=2)
def _load_system_prompt(prompt_name: str) -> str:
    """Load a system prompt file from the prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / prompt_name
    with open(prompt_path) as f:
        return f.read()


def default_gemma4_t2v_system_prompt() -> str:
    """Standalone accessor mirroring ``default_gemma4_t2v_system_prompt()`` upstream."""
    return _load_system_prompt("gemma4_t2v_system_prompt.txt")


def default_gemma4_i2v_system_prompt() -> str:
    """Standalone accessor mirroring ``default_gemma4_i2v_system_prompt()`` upstream."""
    return _load_system_prompt("gemma4_i2v_system_prompt.txt")
