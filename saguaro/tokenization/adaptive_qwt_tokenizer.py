# saguaro/tokenization/adaptive_qwt_tokenizer.py
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

"""Adaptive QWT Tokenizer with Learnable Codebook.

This module extends the base QWTTextTokenizer with a learnable codebook
that grows the vocabulary by learning frequent n-grams from the training corpus.
This bridges the gap between user-configured vocab_size and actual token output.

Architecture:
    1. Base UTF-8 byte encoding (QWTTextTokenizer)
    2. + Learned n-gram tokens (via SuperwordMerger)
    = Full vocab utilization with sequence compression

Example:
    >>> tokenizer = AdaptiveQWTTokenizer(vocab_size=8000, model_max_length=256)
    >>> tokenizer.learn_from_corpus(["Hello world", "Machine learning"])
    >>> tokens = tokenizer("Hello world")
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from saguaro.tokenization.qwt_text_tokenizer import QWTTextTokenizer
from saguaro.tokenization.superword_merger import SuperwordMerger, SuperwordMergerConfig

logger = logging.getLogger(__name__)


class AdaptiveQWTTokenizer(QWTTextTokenizer):
    """Adaptive QWT Tokenizer with learnable n-gram codebook.

    Extends QWTTextTokenizer to support vocabulary expansion through
    learned n-gram tokens. The tokenizer maintains a base vocabulary
    from UTF-8 byte encoding and dynamically extends it with frequently
    occurring n-grams learned from the training corpus.
    """

    # Base vocab: special tokens (10) + byte offset (32) + 256 bytes + buffer
    _BASE_VOCAB_SIZE = 10 + 32 + 256 + 64  # = 362

    def __init__(
        self,
        *,
        vocab_size: int,
        model_max_length: int,
        max_vocab_size: int | None = None,
        byte_offset: int = 32,
        enable_thinking_tokens: bool = True,
        min_ngram_size: int = 2,
        max_ngram_size: int = 5,
    ) -> None:
        """Initialize the Adaptive QWT Tokenizer.

        Args:
            vocab_size: Target vocabulary size (will grow via learning).
            model_max_length: Maximum sequence length for the model.
            max_vocab_size: Hard cap on vocabulary size.
            byte_offset: Offset for byte values in vocabulary (default: 32).
            enable_thinking_tokens: Enable thinking token injection.
            min_ngram_size: Minimum n-gram size for codebook learning.
            max_ngram_size: Maximum n-gram size for codebook learning.
        """
        # Initialize base tokenizer with minimum required vocab
        base_vocab = max(self._BASE_VOCAB_SIZE, byte_offset + 256 + 64)
        super().__init__(
            vocab_size=base_vocab,
            model_max_length=model_max_length,
            byte_offset=byte_offset,
            enable_thinking_tokens=enable_thinking_tokens,
        )

        # Store target configuration
        self._target_vocab_size = vocab_size
        self._base_vocab_size = base_vocab
        self._codebook_capacity = max(0, vocab_size - base_vocab)

        # Hard cap
        self._max_vocab_size = max_vocab_size
        if max_vocab_size is not None and max_vocab_size < vocab_size:
            # Reduce codebook capacity to respect budget
            self._codebook_capacity = max(0, max_vocab_size - base_vocab)

        # N-gram configuration
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size

        # SuperwordMerger for learned tokens (initialized on first training)
        self._merger: SuperwordMerger | None = None
        self._trained = False

        logger.info(
            "[AdaptiveQWT] Initialized: base_vocab=%d, target=%d, codebook_capacity=%d",
            self._base_vocab_size,
            self._target_vocab_size,
            self._codebook_capacity,
        )

    @property
    def vocab_size(self) -> int:
        """Return the actual utilized vocabulary size."""
        if self._merger is not None:
            return self._base_vocab_size + self._merger.superword_count
        return self._base_vocab_size

    @property
    def target_vocab_size(self) -> int:
        return self._target_vocab_size

    @property
    def codebook_capacity(self) -> int:
        if self._merger is not None:
            return max(0, self._codebook_capacity - self._merger.superword_count)
        return self._codebook_capacity

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def merger(self) -> SuperwordMerger | None:
        return self._merger

    def learn_from_corpus(
        self,
        texts: Sequence[str],
        min_freq: int = 10,
        progress_callback: Any = None,
        num_workers: int = 4,
    ) -> int:
        """Learn frequent n-grams from corpus to populate codebook.
        """
        if self._codebook_capacity <= 0:
            logger.warning("[AdaptiveQWT] No codebook capacity.")
            return 0

        logger.info(
            "[AdaptiveQWT] Learning from %d texts (min_freq=%d, capacity=%d)",
            len(texts),
            min_freq,
            self._codebook_capacity,
        )

        # Parallel tokenization
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        token_sequences: list[list[int]] = []
        
        def tokenize_text(text: str) -> list[int]:
            """Tokenize a single text (thread-safe)."""
            encoding = self._encode_one(
                text,
                truncation=True,
                max_length=self.model_max_length,
                add_special_tokens=False,
            )
            # Handle dictionary output (if return_tensors='metrics' or none)
            ids = encoding['input_ids'] if isinstance(encoding, dict) else encoding.input_ids
            # Ensure it is a list of ints
            return ids if isinstance(ids, list) else ids.numpy().tolist()

        try:
            completed = 0
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(tokenize_text, t): i for i, t in enumerate(texts)}
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            token_sequences.append(result)
                    except Exception as e:
                        # Log first errors at WARNING to surface issues
                        if completed < 5:
                            logger.warning("[AdaptiveQWT] Tokenization error: %s", e)
                    
                    completed += 1
                    if completed % 5000 == 0:
                        logger.info("[AdaptiveQWT] Tokenized %d/%d texts...", completed, len(texts))
                        if progress_callback:
                            progress_callback(completed, len(texts))

        except KeyboardInterrupt:
            logger.warning("[AdaptiveQWT] Tokenization interrupted.")

        if not token_sequences:
            logger.warning("[AdaptiveQWT] No sequences tokenized, returning 0")
            self._trained = True 
            return 0

        # Create and train SuperwordMerger
        merger_config = SuperwordMergerConfig(
            min_frequency=min_freq,
            max_vocab_size=self._codebook_capacity,
            min_ngram_size=self._min_ngram_size,
            max_ngram_size=self._max_ngram_size,
            byte_offset=self._byte_offset,
        )
        self._merger = SuperwordMerger(
            base_vocab_size=self._base_vocab_size,
            config=merger_config,
        )

        # Train on tokenized sequences
        learned_count = self._merger.train(
            token_sequences,
            progress_callback=progress_callback,
        )

        self._trained = True
        return learned_count

    # Helper method for _encode_one since QWTTextTokenizer API might need adjustment
    def _encode_one(self, text, **kwargs):
        # The base QWTTextTokenizer.__call__ handles single strings by returning a dict.
        # We invoke it with return_tensors=None to get lists.
        # However, checking qwt_text_tokenizer.py again...
        # __call__ accepts `texts` (str or list).
        return super().__call__(text, return_tensors=None, **kwargs)

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
        """Tokenize text(s) with optional n-gram compression."""
        
        # Get base encoding from parent class
        # Note: We pass return_tensors=None to manipulate lists before conversion
        result = super().__call__(
            texts,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            add_special_tokens=add_special_tokens,
            return_tensors=None, 
        )

        # Apply n-gram merging if trained
        if self._merger is not None and self._merger.superword_count > 0:
            single = isinstance(texts, str)
            input_ids = result["input_ids"]

            if single:
                # Single sequence
                merged_ids = self._merger.apply(input_ids)
                result["input_ids"] = merged_ids
                result["attention_mask"] = [1] * len(merged_ids)
            else:
                # Batch of sequences
                merged_batch = []
                attention_batch = []
                for seq in input_ids:
                    merged = self._merger.apply(seq)
                    merged_batch.append(merged)
                    attention_batch.append([1] * len(merged))
                result["input_ids"] = merged_batch
                result["attention_mask"] = attention_batch

            # Re-pad if needed (merging may have shortened sequences)
            pad_target = None
            if padding == "max_length" or (padding is True and max_length):
                pad_target = max_length or self.model_max_length

            if pad_target is not None:
                if single:
                    result = self._pad_result(result, pad_target, single=True)
                else:
                    result = self._pad_result(result, pad_target, single=False)

        # Convert to tensors if requested
        if return_tensors == "tf":
            import tensorflow as tf
            result = {k: tf.constant(v, dtype=tf.int32) for k, v in result.items()}

        return result

    def _pad_result(
        self,
        result: dict[str, Any],
        target_length: int,
        single: bool,
    ) -> dict[str, Any]:
        """Pad result to target length after merging."""
        if single:
            ids = result["input_ids"]
            mask = result["attention_mask"]
            if len(ids) < target_length:
                pad_amount = target_length - len(ids)
                ids = ids + [self.pad_token_id] * pad_amount
                mask = mask + [0] * pad_amount
            elif len(ids) > target_length:
                ids = ids[:target_length]
                mask = mask[:target_length]
            result["input_ids"] = ids
            result["attention_mask"] = mask
        else:
            new_ids = []
            new_masks = []
            for ids, mask in zip(result["input_ids"], result["attention_mask"]):
                if len(ids) < target_length:
                    pad_amount = target_length - len(ids)
                    ids = list(ids) + [self.pad_token_id] * pad_amount
                    mask = list(mask) + [0] * pad_amount
                elif len(ids) > target_length:
                    ids = ids[:target_length]
                    mask = mask[:target_length]
                new_ids.append(ids)
                new_masks.append(mask)
            result["input_ids"] = new_ids
            result["attention_mask"] = new_masks
        return result

    def decode(
        self,
        ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs back to text."""
        # Expand superwords if merger is active
        if self._merger is not None:
            ids = self._merger.reverse_merge(ids)

        return super().decode(ids, skip_special_tokens=skip_special_tokens)

    def get_stats(self) -> dict[str, Any]:
        """Get tokenizer statistics."""
        stats = {
            "base_vocab_size": self._base_vocab_size,
            "target_vocab_size": self._target_vocab_size,
            "current_vocab_size": self.vocab_size,
            "codebook_capacity": self._codebook_capacity,
            "remaining_capacity": self.codebook_capacity,
            "is_trained": self._trained,
            "min_ngram_size": self._min_ngram_size,
            "max_ngram_size": self._max_ngram_size,
        }
        if self._merger is not None:
            stats["merger_stats"] = self._merger.get_stats()
        return stats

    def save_codebook(self, path: str) -> None:
        """Save the learned codebook to a file."""
        if self._merger is None or not self._trained:
            raise RuntimeError("Cannot save codebook: no training has been performed.")
        from pathlib import Path as PathLib
        save_path = PathLib(path)
        self._merger.save(save_path)
        logger.info("[AdaptiveQWT] Saved codebook to %s", path)

    def load_codebook(self, path: str) -> int:
        """Load a pre-trained codebook from file."""
        from pathlib import Path as PathLib

        load_path = PathLib(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Codebook file not found: {path}")

        self._merger = SuperwordMerger.load(load_path)
        self._trained = True
        return self._merger.superword_count


__all__ = ["AdaptiveQWTTokenizer"]
