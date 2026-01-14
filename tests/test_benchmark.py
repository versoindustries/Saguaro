import unittest
import json
import tempfile
import os
import shutil
from saguaro.benchmarks.runner import BenchmarkRunner


class TestBenchmarkRunner(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.custom_data_path = os.path.join(self.test_dir, "data.json")
        with open(self.custom_data_path, "w") as f:
            json.dump([{"query": "test", "expected_tokens": ["foo"]}], f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_run_synthetic(self):
        runner = BenchmarkRunner("CodeSearchNet")
        results = runner.run()
        self.assertEqual(results["dataset"], "CodeSearchNet")
        self.assertEqual(results["total_queries"], 10)
        self.assertTrue("avg_latency_ms" in results)

    def test_run_custom(self):
        runner = BenchmarkRunner("custom", self.custom_data_path)
        results = runner.run()
        self.assertEqual(results["dataset"], "custom")
        self.assertEqual(results["total_queries"], 1)


if __name__ == "__main__":
    unittest.main()
