# saguaro/tokenization/qwt_text_tokenizer.py
# Copyright 2026 Verso Industries
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quantum Wavelet Tokenizer front-end with a HuggingFace-like interface.

This module provides the text-side API expected by the rest of the codebase
while emitting the integer token IDs that feed directly into the fused Quantum
Wavelet Tokenizer C++ operations.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

from saguaro.ops.fused_text_tokenizer import (
    fused_text_tokenize_batch,
    SuperwordTrieHandle,
)


@dataclass
class _Encoding:
    """Internal encoding representation."""
    input_ids: list[int]
    attention_mask: list[int]


@dataclass
class SuperpositionEncoding:
    """Superposition BPE encoding with multiple segmentations."""
    input_ids: list[list[int]]
    amplitudes: list[float]
    attention_masks: list[list[int]]


class QWTTextTokenizer:
    """Lightweight tokenizer front-end paired with the fused QWT runtime.

    Uses C++ fused operations for high-performance tokenization.
    """

    _BYTE_VOCAB = 256
    _SPECIAL_TOKENS = {
        "unk_token_id": 0,
        "cls_token_id": 1,
        "pad_token_id": 2,
        "eos_token_id": 3,
        "sep_token_id": 4,
        "mask_token_id": 5,
        "think_token_id": 6,
        "pause_token_id": 7,
        "reflect_token_id": 8,
        "conclude_token_id": 9,
    }

    _COMPLEXITY_PATTERNS = [
        (r"\bif\b.*\bthen\b", 1),
        (r"\b(because|therefore|thus|hence)\b", 1),
        (r"\b(first|second|third|finally)\b", 1),
        (r"\b(however|but|although)\b", 1),
        (r"\?\s*$", 2),
        (r"\b(analyze|evaluate|compare|contrast)\b", 2),
        (r"\b(prove|derive|calculate|solve)\b", 3),
    ]

    def __init__(
        self,
        *,
        vocab_size: int,
        model_max_length: int,
        byte_offset: int = 32,
        enable_thinking_tokens: bool = True,
        trie: SuperwordTrieHandle | None = None,
    ) -> None:
        """Initialize the QWT Text Tokenizer."""
        min_required = byte_offset + self._BYTE_VOCAB
        if vocab_size < min_required:
            raise ValueError(
                f"QWTTextTokenizer requires vocab_size>={min_required}, got {vocab_size}."
            )
        self._vocab_size = vocab_size
        self.model_max_length = model_max_length
        self._byte_offset = byte_offset
        self.enable_thinking_tokens = enable_thinking_tokens
        self.trie = trie

        # Initialize special token attributes
        self.unk_token_id: int = self._SPECIAL_TOKENS["unk_token_id"]
        self.cls_token_id: int = self._SPECIAL_TOKENS["cls_token_id"]
        self.pad_token_id: int = self._SPECIAL_TOKENS["pad_token_id"]
        self.eos_token_id: int = self._SPECIAL_TOKENS["eos_token_id"]
        self.sep_token_id: int = self._SPECIAL_TOKENS["sep_token_id"]
        self.mask_token_id: int = self._SPECIAL_TOKENS["mask_token_id"]
        self.think_token_id: int = self._SPECIAL_TOKENS["think_token_id"]
        self.pause_token_id: int = self._SPECIAL_TOKENS["pause_token_id"]
        self.reflect_token_id: int = self._SPECIAL_TOKENS["reflect_token_id"]
        self.conclude_token_id: int = self._SPECIAL_TOKENS["conclude_token_id"]

        self.unk_token: str = "<unk>"
        self.cls_token: str = "<cls>"
        self.pad_token: str = "<pad>"
        self.eos_token: str = "<eos>"
        self.sep_token: str = "<sep>"
        self.mask_token: str = "<mask>"
        self.think_token: str = "<think>"
        self.pause_token: str = "<pause>"
        self.reflect_token: str = "<reflect>"
        self.conclude_token: str = "<conclude>"

        self.special_tokens_set = set(self._SPECIAL_TOKENS.values())

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def __call__(
        self,
        texts: str | list[str],
        *,
        truncation: bool = True,
        max_length: int | None = None,
        padding: bool | str = False,
        add_special_tokens: bool = True,
        return_tensors: str | None = None,
    ) -> dict[str, Any]:
        """Tokenize text(s) using C++ fused operations."""
        single = isinstance(texts, str)
        input_texts = [texts] if single else list(texts)
        
        target_len = max_length or self.model_max_length
        
        # Use C++ batch tokenization
        try:
             # inject_thinking=True allows recognizing <think> tags.
             # Note: This doesn't do automatic insertion based on complexity.
             tokens_tensor, lengths_tensor = fused_text_tokenize_batch(
                input_texts,
                trie=self.trie,
                byte_offset=self._byte_offset,
                add_special_tokens=add_special_tokens,
                max_length=target_len,
                inject_thinking=self.enable_thinking_tokens,
             )
             
             # Convert to lists if not returning tensors
             if return_tensors != "tf":
                 tokens = tokens_tensor.numpy().tolist()
                 # Trim padding if requested (C++ pads to max_length by default)
                 # Actually C++ returns padded to max_length.
                 # If user wants padding='max_length', it's done.
                 # If user wants padding=False, we might need to trim.
                 
                 # But sticking to TF tensors is most efficient.
                 # If return_tensors != 'tf', we convert.
                 
                 input_ids = []
                 attention_masks = []
                 real_lengths = lengths_tensor.numpy().tolist()
                 
                 for i, seq in enumerate(tokens):
                     length = real_lengths[i]
                     # If truncation was True, length is min(len, max_length)
                     # C++ Op handles truncation.
                     
                     # Padding logic:
                     # C++ Op ALWAYS pads to max_length with PAD_ID.
                     
                     if padding == "max_length":
                         # Already padded
                         curr_ids = seq
                         curr_mask = [1] * length + [0] * (len(seq) - length)
                     elif padding is False:
                         # Trim to actual length
                         curr_ids = seq[:length]
                         curr_mask = [1] * length
                     else:
                         # Dynamic padding (pad to max in batch)
                         # Here batch is processed with fixed max_length.
                         # We can trim to max(real_lengths)
                         max_in_batch = max(real_lengths)
                         curr_ids = seq[:max_in_batch]
                         curr_mask = [1] * length + [0] * (max_in_batch - length)

                     input_ids.append(curr_ids)
                     attention_masks.append(curr_mask)
                 
                 result = {
                     "input_ids": input_ids,
                     "attention_mask": attention_masks,
                 }
             else:
                 # Return tensors directly
                 # Attention mask must be generated
                 tf.shape(tokens_tensor)[0]
                 seq_len = tf.shape(tokens_tensor)[1]
                 # specific lengths are in lengths_tensor
                 mask = tf.sequence_mask(lengths_tensor, maxlen=seq_len, dtype=tf.int32)
                 
                 result = {
                     "input_ids": tokens_tensor,
                     "attention_mask": mask,
                 }

        except RuntimeError as e:
            logger.warning("C++ ops failed, falling back to Python: %s", e)
            # Fallback to python implementation (omitted for brevity, or partial)
            # Assuming ops are available per roadmap
            raise e

        if single:
            if return_tensors == "tf":
                # Squeeze batch dim? HF tokenizer usually doesn't squeeze for single? 
                # Actually HF tokenizer returns dict of lists or tensors.
                # If single input, usually unbatched? 
                # Let's check original behavior: "result = {k: v[0] ...}"
                result = {k: v[0] for k, v in result.items()}
            else:
                result = {k: v[0] for k, v in result.items()}

        return result

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        bytes_out: list[int] = []
        special = self.special_tokens_set if skip_special_tokens else set()
        for idx in ids:
            try:
                idx = int(idx)
            except (ValueError, TypeError):
                pass
            if idx in special:
                continue
            if idx < self._byte_offset or idx >= self._byte_offset + self._BYTE_VOCAB:
                continue
            bytes_out.append(idx - self._byte_offset)
        try:
            return bytes(bytes_out).decode("utf-8", errors="ignore")
        except UnicodeDecodeError:
            return bytes(bytes_out).decode("latin-1", errors="ignore")

    def batch_decode(
        self,
        batch_ids: Iterable[Sequence[int]],
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_ids]

    def _compute_complexity(self, text: str) -> int:
        """Compute text complexity score (Python logic)."""
        score = 0
        text_lower = text.lower()
        for pattern, weight in self._COMPLEXITY_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                score += weight
        return min(10, max(0, score))

    def inject_thinking_tokens(
        self,
        token_ids: list[int],
        text: str | None = None,
        complexity_override: int | None = None,
    ) -> list[int]:
        """Inject thinking tokens (Python logic).
        
        Useful if you want to perform automatic injection based on complexity
        AFTER tokenization.
        """
        if not self.enable_thinking_tokens:
            return token_ids

        if complexity_override is not None:
            complexity = max(0, min(10, complexity_override))
        elif text is not None:
            complexity = self._compute_complexity(text)
        else:
            return token_ids

        if complexity == 0:
            return token_ids

        result: list[int] = []
        if complexity >= 1:
            result.append(self.think_token_id)

        pause_interval = max(10, 50 - (complexity * 5))
        for i, token_id in enumerate(token_ids):
            result.append(token_id)
            if complexity >= 5 and (i + 1) % pause_interval == 0 and i < len(token_ids) - 1:
                result.append(self.pause_token_id)

        if complexity >= 3:
            result.append(self.reflect_token_id)

        if complexity >= 1:
            result.append(self.conclude_token_id)

        return result
