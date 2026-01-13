#!/usr/bin/env python3
# scripts/train_baseline.py
# Copyright 2026 Verso Industries
#
# Train the Verso Baseline tokenizer codebook from curriculum datasets.
# This is OPTIONAL - Saguaro works without it using byte-level tokenization.
# The codebook improves tokenization efficiency by ~30-50%.
#
# Usage:
#   python scripts/train_baseline.py --curriculum verso-baseline
#   python scripts/train_baseline.py --corpus /path/to/code --fast

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saguaro.tokenization.intelligent_controller import IntelligentVocabController, VocabControllerConfig
from saguaro.data.dataset_loader import DatasetLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainBaseline")

def train_baseline(corpus_path: str = None, curriculum_name: str = None, output_path: str = "", fast: bool = False):
    """
    Train the Verso Baseline tokenizer.
    Can train from a local corpus path OR a HuggingFace curriculum.
    """
    if not corpus_path and not curriculum_name:
        logger.error("Must specify either --corpus or --curriculum")
        return

    logger.info("Initializing IntelligentVocabController...")
    
    # Config for a robust baseline
    config = VocabControllerConfig(
        model_max_length=8192,
        min_ngram_freq=50,       # High frequency for baseline
        sample_size=1_000_000,   # Large sample size
        min_ngram_size=2,
        max_ngram_size=5
    )
    
    controller = IntelligentVocabController(config)
    samples = []
    
    # Limit logic
    # Fast: 10MB, Full: 500MB (Local) or 1GB (Streaming)
    max_bytes = 10 * 1024 * 1024 if fast else 1024 * 1024 * 1024 
    current_bytes = 0

    # 1. Load Data
    if curriculum_name:
        logger.info(f"Loading Curriculum: {curriculum_name}")
        loader = DatasetLoader()
        
        logger.info(f"Streaming samples (Limit: {max_bytes/1024/1024:.0f} MB)...")
        for text in loader.stream_samples(curriculum_name, total_limit_bytes=max_bytes):
             samples.append(text)
             current_bytes += len(text)
             # Basic progress
             if len(samples) % 1000 == 0:
                 sys.stdout.write(f"  ...collected {len(samples)} samples ({current_bytes/1024/1024:.1f} MB)\r")
                 sys.stdout.flush()
                 
    elif corpus_path:
        files = []
        logger.info(f"Scanning corpus at {corpus_path}...")
        for root, _, filenames in os.walk(corpus_path):
            for f in filenames:
                if f.endswith(('.py', '.js', '.ts', '.cpp', '.h', '.cc', '.md', '.java', '.go', '.rs')):
                    files.append(os.path.join(root, f))
        
        logger.info(f"Found {len(files)} source files.")
        if len(files) == 0:
            logger.error("No valid source files found!")
            return

        import random
        random.shuffle(files)
        
        logger.info(f"Reading samples (Limit: {max_bytes/1024/1024:.0f} MB)...")
        for fpath in files:
            if current_bytes >= max_bytes:
                break
            try:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    if len(text) > 100:
                        samples.append(text)
                        current_bytes += len(text)
            except Exception:
                pass
            
    logger.info(f"\nCollected {len(samples)} samples ({current_bytes/1024/1024:.2f} MB).")
    
    if not samples:
        logger.error("No samples collected. Aborting.")
        return

    # 3. Train
    logger.info("Starting training (this may take a while)...")
    controller.train_codebook(samples, min_freq=config.min_ngram_freq)
    
    # 4. Save
    if not output_path:
        output_path = "saguaro/artifacts/codebooks/verso_baseline.json"
        
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    controller.save_codebook(output_path)
    logger.info(f"✅ Baseline saved to: {output_path}")
    logger.info(f"Vocab Stats: {controller.get_stats()}")

def main():
    parser = argparse.ArgumentParser(description="Train Saguaro Baseline Tokenizer")
    parser.add_argument("--corpus", help="Path to local corpus directory")
    parser.add_argument("--curriculum", help="Name of curriculum preset (e.g. verso-baseline)")
    parser.add_argument("--output", default="saguaro/artifacts/codebooks/verso_baseline.json", help="Output path")
    parser.add_argument("--fast", action="store_true", help="Fast mode (smaller sample)")
    
    args = parser.parse_args()
    
    train_baseline(args.corpus, args.curriculum, args.output, args.fast)

if __name__ == "__main__":
    main()
