import time
import logging
from contextlib import contextmanager
from typing import Dict

logger = logging.getLogger("saguaro.profiler")


class Profiler:
    def __init__(self, threshold_ms: float = 100.0):
        self.threshold = threshold_ms
        self.stats: Dict[str, float] = {}

    @contextmanager
    def measure(self, name: str):
        start = time.time()
        try:
            yield
        finally:
            duration = (time.time() - start) * 1000
            self.stats[name] = duration
            if duration > self.threshold:
                logger.warning(f"SLOW OP [{name}]: {duration:.2f}ms")


profiler = Profiler()
