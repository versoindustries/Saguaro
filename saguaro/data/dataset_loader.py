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

    def stream_samples(self, preset_name: str, total_limit_bytes: int = 100_000_000) -> Iterator[str]:
        """
        Stream text samples from the specified curriculum preset.
        
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
            logger.error("HuggingFace 'datasets' library not installed. Cannot stream curriculum.")
            logger.error("Run: pip install datasets")
            return

        logger.info(f"Streaming samples from {len(hf_datasets)} datasets in '{preset_name}'...")
        
        current_bytes = 0
        
        # Generator that mixes datasets? 
        # Making a true mixture is hard with basic streaming utils.
        # We will iterate them sequentially but limit each to a uniform fraction for now,
        # or just round-robin if we could keep them open.
        
        # Simple strategy: Allocating budget per dataset
        budget_per_dataset = total_limit_bytes // len(hf_datasets)
        
        for ds_name in hf_datasets:
            if current_bytes >= total_limit_bytes:
                break
                
            logger.info(f"Loading stream: {ds_name}...")
            try:
                # Load streaming
                # We assume 'train' split usually exists
                ds = load_dataset(ds_name, split="train", streaming=True, trust_remote_code=True)
                
                ds_bytes = 0
                for item in ds:
                    if ds_bytes >= budget_per_dataset:
                        break
                        
                    # Extract text
                    # Common keys: 'text', 'content', 'code'
                    text = item.get("text") or item.get("content") or item.get("code")
                    if not text:
                        continue
                        
                    yield text
                    
                    text_len = len(text)
                    ds_bytes += text_len
                    current_bytes += text_len
                    
                    if current_bytes >= total_limit_bytes:
                        break
                        
            except Exception as e:
                logger.warning(f"Failed to stream {ds_name}: {e}")
                continue

        logger.info(f"Finished streaming. Total bytes: {current_bytes}")
