
import os
import yaml
import time
from typing import Dict, Any

class HealthDashboard:
    def __init__(self, saguaro_dir: str):
        self.saguaro_dir = saguaro_dir
        self.config_path = os.path.join(saguaro_dir, "config.yaml")
        # We assume vectors stored in 'vectors' subdir for now or check config
        
    def generate_report(self) -> Dict[str, Any]:
        """
        Generates health metrics for the SAGUARO Index.
        """
        report = {}
        
        # 1. Freshness (Last Incremental Update)
        # We can check modification time of index.pkl or vector store
        store_path = os.path.join(self.saguaro_dir, "vectors", "index.pkl")
        if os.path.exists(store_path):
            mtime = os.path.getmtime(store_path)
            report["freshness"] = {
                "last_update_ts": mtime,
                "last_update_fmt": time.ctime(mtime),
                "age_seconds": time.time() - mtime
            }
        else:
            report["freshness"] = {"status": "not_indexed"}

        # 2. Index Size vs Repo Size (Efficiency)
        # Repo size is hard without walking, but let's check vector store size
        if os.path.exists(store_path):
            idx_size_mb = os.path.getsize(store_path) / (1024 * 1024)
            # Metadata size
            meta_path = os.path.join(self.saguaro_dir, "vectors", "metadata.json")
            meta_size_mb = os.path.getsize(meta_path) / (1024 * 1024) if os.path.exists(meta_path) else 0
            
            report["storage"] = {
                "vector_index_mb": round(idx_size_mb, 2),
                "metadata_mb": round(meta_size_mb, 2),
                "total_mb": round(idx_size_mb + meta_size_mb, 2)
            }
            
        # 3. Dimensionality & Config
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
                report["config"] = {
                    "active_dim": config.get("active_dim", "Unknown"),
                    "total_dim": config.get("total_dim", "Unknown"),
                    "dark_space_ratio": config.get("dark_space_ratio", "Unknown")
                }
                
        # 4. Latency Check (Self-Test)
        # Run a micro-query if possible, or leave for CLI wrapper
        
        return report

    def print_dashboard(self):
        report = self.generate_report()
        
        print("\n=== SAGUARO Index Health Dashboard ===")
        
        # Freshness
        fresh = report.get("freshness", {})
        if "last_update_fmt" in fresh:
            age = fresh['age_seconds']
            status = "🟢 Healthy" if age < 3600 else "🟡 Stale (>1h)" if age < 86400 else "🔴 Ancient (>24h)"
            print(f"\n[Freshness] {status}")
            print(f"  Last Update: {fresh['last_update_fmt']}")
            print(f"  Age: {int(age // 60)} minutes")
        else:
            print("\n[Freshness] PENDING (No Index Found)")
            
        # Storage
        store = report.get("storage", {})
        if store:
            print("\n[Storage Footprint]")
            print(f"  Vector Index: {store['vector_index_mb']} MB")
            print(f"  Metadata:     {store['metadata_mb']} MB")
            print(f"  Total Size:   {store['total_mb']} MB")
            
        # Config
        conf = report.get("config", {})
        if conf:
            print("\n[Holographic Configuration]")
            print(f"  Active Dimension: {conf.get('active_dim')}")
            print(f"  Total Dimension:  {conf.get('total_dim')}")
            print(f"  Dark Space Ratio: {conf.get('dark_space_ratio')}")
            
        print("\n======================================")
