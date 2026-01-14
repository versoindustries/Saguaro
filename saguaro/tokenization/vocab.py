import os
import json
import logging
from typing import List, Optional

from saguaro.tokenization.intelligent_controller import (
    IntelligentVocabController,
    VocabControllerConfig,
)

logger = logging.getLogger(__name__)


class CoherenceManager:
    """
    Manages the tokenizer vocabulary and Coherence (Repo-Specific) Adapters.

    Refactored to use HighNoon's IntelligentVocabController for:
    - Pretrained baseline loading (Verso Baseline)
    - Dynamic capacity tuning based on Code Volume (LOC).
    - Adaptive training from repo content.
    """

    def __init__(self, base_vocab_path: Optional[str] = None):
        # Initialize IntelligentVocabController with default config
        self.config = VocabControllerConfig(
            model_max_length=4096, min_ngram_freq=10, byte_offset=32
        )
        self.controller = IntelligentVocabController(self.config)

        # Load Verso Baseline immediately
        try:
            self.controller.load_pretrained("verso-baseline")
        except Exception as e:
            logger.warning(f"Could not load verso-baseline: {e}")

        # Resource handle logic handled dynamically

    def initialize(self):
        """
        Ensure the controller is ready.
        For IntelligentVocabController, initialization happens on creation/load.
        This method is kept for compatibility.
        """
        # Trigger handle creation if needed by accessing it
        _ = self.get_trie_handle()

    def get_trie_handle(self):
        """Returns the native resource handle."""
        # Access the internal merger from the controller's tokenizer
        tokenizer = self.controller.tokenizer
        merger = tokenizer.merger

        if merger and merger.has_native_trie:
            return merger._native_trie.handle

        # Return empty/null if no merger active (base vocab only)
        return None

    def calculate_vocab_budget(self, total_bytes: int) -> int:
        """
        Dynamically calculate target vocab size based on code volume.

        Formula:
        Small (<1MB): 2000 tokens (Scripts)
        Medium (<10MB): 10000 tokens (Libraries)
        Large (<100MB): 32000 tokens (Frameworks)
        Huge (>100MB): 64000 tokens (Monorepos)

        Note: The IndexEngine supports up to 131072 tokens.
        """
        if total_bytes < 1_000_000:  # < 1MB
            return 2000
        elif total_bytes < 10_000_000:  # < 10MB
            return 10000
        elif total_bytes < 100_000_000:  # < 100MB
            return 32000
        else:
            return 64000

    def calibrate(self, workset_files: List[str]):
        """
        Scan the provided files and train/update the Adapter Vocab.
        Merges new repo-specific tokens with the existing Verso Baseline.
        """
        if not workset_files:
            return

        # 1. Analyze Volume
        total_bytes = 0
        valid_files = []
        for fpath in workset_files:
            if os.path.exists(fpath):
                try:
                    size = os.path.getsize(fpath)
                    total_bytes += size
                    valid_files.append(fpath)
                except OSError:
                    pass

        # 2. Determine Budget
        budget = self.calculate_vocab_budget(total_bytes)
        logger.info(
            f"Calibrating Coherence Adapter on {len(valid_files)} files ({total_bytes / 1024 / 1024:.2f} MB). Target Budget: {budget}"
        )

        # 3. Collect Text Samples
        samples = []
        sample_limit = 5 * 1024 * 1024  # 5MB limit
        current_sample_size = 0

        import random

        random.shuffle(valid_files)

        for fpath in valid_files:
            if current_sample_size >= sample_limit:
                break
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read(100000)
                    if content:
                        samples.append(content)
                        current_sample_size += len(content)
            except Exception:
                continue

        if not samples:
            return

        # 4. Fusion Training Strategy
        # We want to Keep Baseline + Add Repo Tokens

        # A. Get Baseline Superwords
        baseline_merger = self.controller.tokenizer.merger
        baseline_entries = {}
        next_id = self.config.byte_offset + 256 + 10  # Default base

        if baseline_merger:
            # Copy existing
            for ngram, entry in baseline_merger._superword_table.items():
                baseline_entries[ngram] = entry
            next_id = baseline_merger._next_superword_id
            logger.info(f"Retaining {len(baseline_entries)} baseline tokens.")

        # B. Train Temporary Merger on Repo Samples
        # We use a raw SuperwordMerger to learn the frequencies
        from saguaro.tokenization.superword_merger import (
            SuperwordMerger,
            SuperwordMergerConfig,
            SuperwordEntry,
        )
        from saguaro.tokenization.qwt_text_tokenizer import QWTTextTokenizer

        # We need to tokenize samples to IDs first. The Merger trains on ID sequences.
        # Use the BASE tokenizer (bytes) for this to find byte-level n-grams?
        # Or do we use the *current* tokenizer?
        # HighNoon approach: Learn from Base IDs.

        # Create a simple base tokenizer (no merger)
        base_tokenizer = QWTTextTokenizer(
            vocab_size=self.controller.base_vocab_size,
            model_max_length=self.controller.config.model_max_length,
            trie=None,  # Pure byte-level
        )

        # Tokenize samples
        logger.info("Tokenizing samples for training...")
        tokenized_samples = []
        for text in samples:
            # Use basic encoding (no trie)
            res = base_tokenizer(text, add_special_tokens=False)
            tokenized_samples.append(res["input_ids"])

        # Train new merger
        repo_config = SuperwordMergerConfig(
            min_frequency=5,
            max_vocab_size=budget,  # Use calculated budget
            min_ngram_size=2,
            max_ngram_size=5,
        )
        repo_merger = SuperwordMerger(
            base_vocab_size=self.controller.base_vocab_size, config=repo_config
        )
        repo_merger.train(tokenized_samples)
        logger.info(f"Learned {repo_merger.superword_count} repo-specific tokens.")

        # C. Merge & Deduplicate
        # Add new tokens if they don't exist in baseline
        final_entries = baseline_entries.copy()
        added_count = 0

        # Sort repo tokens by frequency
        repo_tokens = list(repo_merger._superword_table.values())
        repo_tokens.sort(key=lambda x: -x.frequency)

        for entry in repo_tokens:
            if entry.token_ids not in final_entries:
                # Create new entry with correct ID
                new_entry = SuperwordEntry(
                    token_ids=entry.token_ids,
                    superword_id=next_id,
                    frequency=entry.frequency,
                    text_repr=entry.text_repr,
                )
                final_entries[entry.token_ids] = new_entry
                next_id += 1
                added_count += 1

        logger.info(
            f"Fused Vocab: {len(baseline_entries)} base + {added_count} repo = {len(final_entries)} total."
        )

        # D. Save Fused Adapter
        adapter_path = os.path.join(os.getcwd(), ".saguaro/vocab_adapter.json")

        # Construct state dict manually to utilize 'load_codebook'
        state = {
            "base_vocab_size": self.controller.base_vocab_size,
            "config": {
                "min_frequency": 5,
                "max_vocab_size": len(final_entries),
                "min_ngram_size": 2,
                "max_ngram_size": 5,
                "byte_offset": 32,
            },
            "superwords": [
                {
                    "token_ids": list(e.token_ids),
                    "superword_id": e.superword_id,
                    "frequency": e.frequency,
                    "text_repr": e.text_repr,
                }
                for e in final_entries.values()
            ],
            "next_id": next_id,
        }

        os.makedirs(os.path.dirname(adapter_path), exist_ok=True)
        with open(adapter_path, "w") as f:
            json.dump(state, f)

        # E. Reload Controller
        self.controller.load_codebook(adapter_path)

        # F. MEMORY OPTIMIZATION: Force cleanup of temporary training objects
        # This prevents orphaned merger/tokenizer objects from accumulating
        import gc
        del repo_merger
        del base_tokenizer
        del tokenized_samples
        del samples
        del repo_tokens
        gc.collect()
        logger.debug("Calibration cleanup complete.")
