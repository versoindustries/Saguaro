# saguaro/ops/fused_text_tokenizer.py
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

"""Python wrapper for fused text tokenization ops.

This module provides Python bindings for the high-performance C++ text
tokenization operations with SIMD optimization and trie-based n-gram merging.
"""

from __future__ import annotations

import logging
from typing import Sequence

import tensorflow as tf

logger = logging.getLogger(__name__)

# Native op module - loaded from consolidated binary
_text_tok_module = None
_ops_available = False

try:
    from saguaro.ops.lib_loader import load_saguaro_library
    
    _core = load_saguaro_library()
    if _core is not None:
        _text_tok_module = _core
        _ops_available = True
        logger.info("[FusedTextTokenizer] Loaded native ops from consolidated binary")
except ImportError as err:
    logger.warning("[FusedTextTokenizer] Native ops not available: %s", err)
except RuntimeError as err:
    logger.warning("[FusedTextTokenizer] Native ops load failed: %s", err)


def ops_available() -> bool:
    """Check if native text tokenizer ops are available."""
    return _ops_available


class SuperwordTrieHandle:
    """Resource handle for SuperwordTrie with Python-friendly API."""
    
    def __init__(self) -> None:
        """Create empty SuperwordTrie resource."""
        if not _ops_available:
            raise RuntimeError(
                "Native text tokenizer ops not available. "
            )
        self._handle = _text_tok_module.saguaro_trie_create()
        self._num_entries = 0
    
    @property
    def handle(self) -> tf.Tensor:
        """Get the raw resource handle for use in ops."""
        return self._handle
    
    @property
    def num_entries(self) -> int:
        """Get number of n-gram entries in the trie."""
        return self._num_entries
    
    def insert(self, ngram: Sequence[int], superword_id: int) -> None:
        """Insert a single n-gram to superword mapping."""
        ngram_tensor = tf.constant(list(ngram), dtype=tf.int32)
        id_tensor = tf.constant(superword_id, dtype=tf.int32)
        _text_tok_module.superword_trie_insert(self._handle, ngram_tensor, id_tensor)
        self._num_entries += 1
    
    def insert_batch(
        self,
        ngrams: Sequence[Sequence[int]],
        superword_ids: Sequence[int],
    ) -> None:
        """Insert multiple n-gram mappings efficiently."""
        if len(ngrams) != len(superword_ids):
            raise ValueError(
                f"ngrams and superword_ids must have same length: "
                f"{len(ngrams)} != {len(superword_ids)}"
            )
        
        if not ngrams:
            return
        
        # Build flat representation
        offsets = [0]
        tokens = []
        for ngram in ngrams:
            tokens.extend(ngram)
            offsets.append(len(tokens))
        
        offsets_tensor = tf.constant(offsets, dtype=tf.int32)
        tokens_tensor = tf.constant(tokens, dtype=tf.int32)
        ids_tensor = tf.constant(list(superword_ids), dtype=tf.int32)
        
        _text_tok_module.superword_trie_build_from_table(
            self._handle, offsets_tensor, tokens_tensor, ids_tensor
        )
        self._num_entries = len(ngrams)
    
    def clear(self) -> None:
        """Clear and rebuild an empty trie."""
        # Recreate handle (cheaper than modifying existing)
        self._handle = _text_tok_module.saguaro_trie_create()
        self._num_entries = 0


class StreamingNgramCounterHandle:
    """Resource handle for StreamingNgramCounter."""

    def __init__(self, min_ngram: int = 2, max_ngram: int = 5) -> None:
        """Create a StreamingNgramCounter resource."""
        if not _ops_available:
            raise RuntimeError("Native text tokenizer ops not available.")
        self._handle = _text_tok_module.streaming_ngram_count_create(
            min_ngram=min_ngram, max_ngram=max_ngram
        )

    @property
    def handle(self) -> tf.Tensor:
        """Get the raw resource handle."""
        return self._handle

    def count_batch(
        self,
        tokens: Sequence[int] | tf.Tensor,
        lengths: Sequence[int] | tf.Tensor,
    ) -> None:
        """Count n-grams from a batch of token sequences.
        
        Note: The C++ op expects 'lengths' (1D tensor), not splits.
        """
        if not isinstance(tokens, (tf.Tensor, tf.Variable)):
            tokens = tf.constant(tokens, dtype=tf.int32)
        if not isinstance(lengths, (tf.Tensor, tf.Variable)):
            lengths = tf.constant(lengths, dtype=tf.int32)

        _text_tok_module.streaming_ngram_count(self._handle, tokens, lengths)

    def get_top_k(
        self, k: int = 1000, min_freq: int = 1
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Get top-K most frequent n-grams."""
        ngrams, counts, splits = _text_tok_module.streaming_ngram_count_export(
            self._handle, min_frequency=min_freq, max_count=k
        )
        # Note: Export op returns (ngrams, counts, splits)
        # Highnoon wrapper expected (ngrams_flat, ngram_lengths, counts)
        # We should adapt to what the Op returns.
        # Register Op says: Output("ngrams"), Output("counts"), Output("splits")
        return ngrams, counts, splits


_EMPTY_TRIE = None

def _get_empty_trie() -> SuperwordTrieHandle:
    global _EMPTY_TRIE
    if _EMPTY_TRIE is None:
        _EMPTY_TRIE = SuperwordTrieHandle()
    return _EMPTY_TRIE


def fused_text_tokenize(
    text: str | bytes,
    trie: SuperwordTrieHandle | None = None,
    byte_offset: int = 32,
    add_special_tokens: bool = True,
    max_length: int = 131072,
    inject_thinking: bool = False,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Tokenize single text with SIMD optimization.
    
    Uses SAGUAROTextTokenize op.
    """
    if not _ops_available:
        raise RuntimeError("Native text tokenizer ops not available.")
    
    if isinstance(text, str):
        text = text.encode("utf-8")
    
    text_tensor = tf.constant(text.decode("utf-8", errors="replace"))
    
    if trie is None:
        trie = _get_empty_trie()
    handle = trie.handle
    
    return _text_tok_module.saguaro_text_tokenize(
        text_tensor,
        handle,
        byte_offset=byte_offset,
        add_special_tokens=add_special_tokens,
        max_length=max_length,
        inject_thinking=inject_thinking,
    )


def fused_text_tokenize_batch(
    texts: Sequence[str | bytes],
    trie: SuperwordTrieHandle | None = None,
    byte_offset: int = 32,
    add_special_tokens: bool = True,
    max_length: int = 131072,
    num_threads: int = 0,
    inject_thinking: bool = False,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Tokenize batch of texts in parallel with SIMD optimization.
    
    Uses SAGUAROTextTokenizeBatch op.
    """
    if not _ops_available:
        raise RuntimeError("Native text tokenizer ops not available.")
    
    # Convert to strings
    str_texts = []
    for t in texts:
        if isinstance(t, bytes):
            t = t.decode("utf-8", errors="replace")
        str_texts.append(t)
    
    texts_tensor = tf.constant(str_texts)
    
    if trie is None:
        trie = _get_empty_trie()
    handle = trie.handle
    
    return _text_tok_module.saguaro_text_tokenize_batch(
        texts_tensor,
        handle,
        byte_offset=byte_offset,
        add_special_tokens=add_special_tokens,
        max_length=max_length,
        num_threads=num_threads,
        inject_thinking=inject_thinking,
    )


def trie_apply_merges(
    tokens: Sequence[int],
    trie: SuperwordTrieHandle,
    superword_table: dict[tuple[int, ...], int] | None = None,
    max_ngram: int = 8,
) -> list[int]:
    """Apply superword merges using longest-match algorithm (Python fallback)."""
    if trie is None or trie.num_entries == 0:
        return list(tokens)
    
    if superword_table is None or len(superword_table) == 0:
        return list(tokens)
    
    # Optimized longest-match algorithm
    tokens_list = list(tokens)
    n_tokens = len(tokens_list)
    result: list[int] = []
    i = 0
    
    while i < n_tokens:
        # Try longest match first (greedy)
        matched = False
        for n in range(min(max_ngram, n_tokens - i), 1, -1):  # n >= 2
            ngram = tuple(tokens_list[i:i + n])
            superword_id = superword_table.get(ngram)
            if superword_id is not None:
                result.append(superword_id)
                i += n
                matched = True
                break
        
        if not matched:
            result.append(tokens_list[i])
            i += 1
    
    return result


def create_trie_from_superword_table(
    superword_table: dict[tuple[int, ...], int],
) -> SuperwordTrieHandle:
    """Create SuperwordTrieHandle from a superword table."""
    trie = SuperwordTrieHandle()
    
    if not superword_table:
        return trie
    
    ngrams = list(superword_table.keys())
    ids = [superword_table[ng] for ng in ngrams]
    trie.insert_batch(ngrams, ids)
    
    return trie


__all__ = [
    "ops_available",
    "SuperwordTrieHandle",
    "StreamingNgramCounterHandle",
    "fused_text_tokenize",
    "fused_text_tokenize_batch",
    "trie_apply_merges",
    "create_trie_from_superword_table",
]
