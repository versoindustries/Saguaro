# saguaro/data/dataset_loader.py
# Copyright 2026 Verso Industries

import logging
import json
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

class DatasetLoader:
    """
    Adapts HuggingFace Datasets for Saguaro Tokenizer Training.
    Handles streaming from the 'Verso Baseline' curriculum.
    """
    
    # Datasets requiring specific configurations to avoid hanging on massive metadata
    # Format: "dataset_name": ["config1", "config2", ...] or "dataset_name": "single_config"
    DATASET_CONFIGS = {
        # The Stack v2 requires language subset - uses PascalCase config names
        "bigcode/the-stack-v2-dedup": ["Python", "JavaScript", "TypeScript", "Go", "Rust", "Java"],
        "bigcode/the-stack-v2": ["Python", "JavaScript", "TypeScript", "Go", "Rust"],
        "bigcode/starcoderdata": "python",
        # code_contests was renamed
        "google/code_contests": None,  # No config needed
        "deepmind/code_contests": None,  # Alias
        # Code search net needs language
        "code-search-net/code_search_net": ["python", "javascript", "go", "java"],
    }
    
    # Text field mapping for datasets with non-standard field names
    TEXT_FIELDS = {
        "bigcode/the-stack-v2-dedup": "content",
        "bigcode/the-stack-v2": "content", 
        "bigcode/starcoderdata": "content",
        "code-search-net/code_search_net": "func_code_string",
        "google/code_contests": "description",
        "deepmind/code_contests": "description",
        "lighteval/MATH": "problem",
        "meta-math/MetaMathQA": "query",
        "microsoft/orca-math-word-problems-200k": "question",
        "TIGER-Lab/MathInstruct": "instruction",
        "math-ai/StackMathQA": "question",
        "open-r1/Math-Verify": "problem",
        "HuggingFaceH4/ultrafeedback_binarized": "prompt",
        "Salesforce/xlam-function-calling-60k": "query",
        "HuggingFaceH4/ultrachat_200k": "prompt",
        "OpenAssistant/oasst_top1_2023-08-25": "text",
    }
    
    def __init__(self, curriculum_path: Optional[str] = None):
        if not curriculum_path:
            # Default to artifacts/curriculum_presets.json relative to this file
            base_dir = Path(__file__).parent.parent
            curriculum_path = base_dir / "artifacts" / "curriculum_presets.json"
            
        self.curriculum_path = Path(curriculum_path)
        self.presets = {}
        self._load_curriculum()
        
    def _load_curriculum(self):
        if not self.curriculum_path.exists():
            logger.warning(f"Curriculum file not found: {self.curriculum_path}")
            return
            
        try:
            with open(self.curriculum_path, 'r') as f:
                data = json.load(f)
                self.presets = data.get("presets", {})
        except Exception as e:
            logger.error(f"Failed to load curriculum: {e}")

    def _get_text_from_item(self, ds_name: str, item: dict) -> Optional[str]:
        """Extract text from dataset item using known field mappings."""
        # Check if we have a known text field for this dataset
        if ds_name in self.TEXT_FIELDS:
            field = self.TEXT_FIELDS[ds_name]
            text = item.get(field)
            if text:
                return text if isinstance(text, str) else str(text)
        
        # Fallback to common field names
        for field in ["text", "content", "code", "prompt", "question", "instruction"]:
            if field in item and item[field]:
                val = item[field]
                return val if isinstance(val, str) else str(val)
        
        return None

    def _stream_single_dataset(self, ds_name: str, config: Optional[str], 
                                budget_bytes: int, load_dataset_fn) -> Iterator[str]:
        """Stream from a single dataset with optional config."""
        config_str = f" (config: {config})" if config else ""
        logger.info(f"  → Streaming: {ds_name}{config_str}...")
        
        try:
            if config:
                ds = load_dataset_fn(ds_name, config, split="train", streaming=True, trust_remote_code=True)
            else:
                ds = load_dataset_fn(ds_name, split="train", streaming=True, trust_remote_code=True)
            
            ds_bytes = 0
            for item in ds:
                if ds_bytes >= budget_bytes:
                    break
                
                text = self._get_text_from_item(ds_name, item)
                if not text or len(text) < 50:  # Skip very short samples
                    continue
                
                yield text
                ds_bytes += len(text)
                
        except Exception as e:
            logger.warning(f"  ✗ Failed {ds_name}{config_str}: {e}")

    def stream_samples(self, preset_name: str, total_limit_bytes: int = 100_000_000) -> Iterator[str]:
        """
        Stream text samples from the specified curriculum preset.
        
        Handles datasets requiring specific configurations (like The Stack v2)
        and properly extracts text using known field mappings.
        
        Args:
            preset_name: Name of preset in curriculum (e.g., 'verso-baseline')
            total_limit_bytes: Soft limit on total bytes to yield
        
        Yields:
            Text strings
        """
        if preset_name not in self.presets:
            logger.error(f"Unknown preset: {preset_name}")
            return
            
        hf_datasets = self.presets[preset_name].get("hf_datasets", [])
        if not hf_datasets:
            logger.warning(f"No HF datasets defined for {preset_name}")
            return
            
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("HuggingFace 'datasets' library not installed.")
            logger.error("Run: pip install datasets")
            return

        # Expand datasets with configs into individual streams
        expanded_streams = []
        for ds_name in hf_datasets:
            if ds_name in self.DATASET_CONFIGS:
                configs = self.DATASET_CONFIGS[ds_name]
                if configs is None:
                    # Explicitly no config needed
                    expanded_streams.append((ds_name, None))
                elif isinstance(configs, list):
                    # Multiple configs - add each as separate stream
                    for cfg in configs:
                        expanded_streams.append((ds_name, cfg))
                else:
                    # Single config string
                    expanded_streams.append((ds_name, configs))
            else:
                # Unknown dataset - try without config
                expanded_streams.append((ds_name, None))

        logger.info(f"Streaming from {len(expanded_streams)} streams ({len(hf_datasets)} datasets)...")
        
        current_bytes = 0
        budget_per_stream = total_limit_bytes // max(len(expanded_streams), 1)
        
        for ds_name, config in expanded_streams:
            if current_bytes >= total_limit_bytes:
                break
            
            remaining_budget = min(budget_per_stream, total_limit_bytes - current_bytes)
            
            for text in self._stream_single_dataset(ds_name, config, remaining_budget, load_dataset):
                yield text
                current_bytes += len(text)
                
                if current_bytes >= total_limit_bytes:
                    break

        logger.info(f"Finished streaming. Total bytes: {current_bytes}")
