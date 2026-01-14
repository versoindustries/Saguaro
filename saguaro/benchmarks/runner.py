import time
import json
import logging
from typing import Dict, List, Any
import random

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Runs standardized performance and retrieval accuracy benchmarks for SAGUARO.
    supported datasets: 'CodeSearchNet' (simulated), 'SWE-bench' (simulated dummy), 'custom'.
    """

    def __init__(self, dataset_name: str, custom_path: str = None):
        self.dataset_name = dataset_name
        self.custom_path = custom_path

    def load_dataset(self) -> List[Dict[str, Any]]:
        """
        Load logical queries and expected results.
        For Phase 5 prototype, this generates synthetic data.
        """
        queries = []
        if self.dataset_name == "custom" and self.custom_path:
            try:
                with open(self.custom_path, "r") as f:
                    queries = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load custom dataset: {e}")

        else:
            # Generate synthetic benchmark queries
            for i in range(10):
                queries.append(
                    {
                        "query": f"synthetic query {i}",
                        "expected_tokens": ["def", "class"],
                        "difficulty": "medium",
                    }
                )
        return queries

    def run(self) -> Dict[str, Any]:
        """
        Execute the benchmark.
        """
        queries = self.load_dataset()
        logger.info(
            f"Running benchmark on {self.dataset_name} with {len(queries)} queries..."
        )

        results = {
            "dataset": self.dataset_name,
            "total_queries": len(queries),
            "successful_retrievals": 0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "accuracy_score": 0.0,
        }

        latencies = []

        # Simulate running queries against index
        # valid_query = logic usually calls saguaro.dni.server.DNIServer.query

        for q in queries:
            start_time = time.time()

            # TODO: Plug in actual DNIServer query here once DNI is fully importable
            # For now, simulate processing delay
            time.sleep(random.uniform(0.01, 0.05))

            end_time = time.time()
            latency = (end_time - start_time) * 1000
            latencies.append(latency)

            # Simulate generic success rate
            if random.random() > 0.1:
                results["successful_retrievals"] += 1

        if latencies:
            results["avg_latency_ms"] = sum(latencies) / len(latencies)
            latencies.sort()
            p95_idx = int(len(latencies) * 0.95)
            results["p95_latency_ms"] = latencies[p95_idx]

        results["accuracy_score"] = (
            results["successful_retrievals"] / len(queries) if queries else 0
        )

        return results

    def print_report(self, results: Dict[str, Any]):
        print(f"\nBenchmark Report: {results['dataset']}")
        print("=" * 40)
        print(f"Total Queries:      {results['total_queries']}")
        print(f"Success Rate:       {results['accuracy_score'] * 100:.1f}%")
        print(f"Avg Latency:        {results['avg_latency_ms']:.2f} ms")
        print(f"P95 Latency:        {results['p95_latency_ms']:.2f} ms")
        print("=" * 40)
