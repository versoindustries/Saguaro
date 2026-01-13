
import time
import os
import logging
from typing import List
from saguaro.indexing.engine import IndexEngine

logger = logging.getLogger(__name__)

class Watcher:
    def __init__(self, engine: IndexEngine, target_path: str, interval: int = 5):
        self.engine = engine
        self.target_path = target_path
        self.interval = interval
        self.running = False
        
    def scan_files(self) -> List[str]:
        all_files = []
        for root, dirs, files in os.walk(self.target_path):
            # Prune
            if '.saguaro' in dirs:
                dirs.remove('.saguaro')
            if '.git' in dirs:
                dirs.remove('.git')
            if 'venv' in dirs:
                dirs.remove('venv')
            
            for file in files:
                if file.endswith(('.py', '.cc', '.cpp', '.h', '.hpp', '.c', '.js', '.ts')):
                    all_files.append(os.path.join(root, file))
        return all_files

    def start(self):
        self.running = True
        logger.info(f"Starting SAGUARO Watcher on {self.target_path} (Interval: {self.interval}s)")
        
        while self.running:
            try:
                # 1. Scan for files
                all_files = self.scan_files()
                
                # 2. Check for updates (using Engine's tracker)
                # We peek at the tracker to see if we need to do anything before calling the heavy engine
                # Actually, engine.index_batch does the check efficiently now.
                
                # We can't just pass ALL 20k files to index_batch every 5 seconds if we want it super cheap?
                # filter_needs_indexing is fast (stat calls). 20k stats takes ~100ms.
                
                # Let's optimize: perform check here to log trigger, then pass to engine
                needed = self.engine.tracker.filter_needs_indexing(all_files)
                
                if needed:
                    logger.info(f"Detected changes in {len(needed)} files. Indexing...")
                    f_count, e_count = self.engine.index_batch(needed)
                    self.engine.commit()
                    logger.info(f"Update complete. {f_count} files processed.")
                
            except Exception as e:
                logger.error(f"Watcher error: {e}")
                # Resilience: Don't crash
                
            time.sleep(self.interval)
            
    def stop(self):
        self.running = False
