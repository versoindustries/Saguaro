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

        # 1. Freshness & Storage
        vectors_dir = os.path.join(self.saguaro_dir, "vectors")
        # Support both legacy index.pkl and new vectors.bin
        store_path = os.path.join(vectors_dir, "vectors.bin")
        if not os.path.exists(store_path):
            store_path = os.path.join(vectors_dir, "index.pkl")

        if os.path.exists(store_path):
            mtime = os.path.getmtime(store_path)
            report["freshness"] = {
                "last_update_ts": mtime,
                "last_update_fmt": time.ctime(mtime),
                "age_seconds": time.time() - mtime,
            }
            
            idx_size_mb = os.path.getsize(store_path) / (1024 * 1024)
            meta_path = os.path.join(vectors_dir, "metadata.json")
            meta_size_mb = (
                os.path.getsize(meta_path) / (1024 * 1024)
                if os.path.exists(meta_path)
                else 0
            )

            report["storage"] = {
                "vector_index_mb": round(idx_size_mb, 2),
                "metadata_mb": round(meta_size_mb, 2),
                "total_mb": round(idx_size_mb + meta_size_mb, 2),
            }
        else:
            report["freshness"] = {"status": "not_indexed"}

        # 2. Config & Performance
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f) or {}
                report["config"] = {
                    "active_dim": config.get("active_dim", "Unknown"),
                    "total_dim": config.get("total_dim", "Unknown"),
                    "dark_space_ratio": config.get("dark_space_ratio", "Unknown"),
                }
                
                # Memory metrics from last index
                last_idx = config.get("last_index", {})
                if last_idx:
                    report["performance"] = {
                        "peak_memory_mb": last_idx.get("peak_memory_mb"),
                        "indexed_files": last_idx.get("files"),
                        "indexed_entities": last_idx.get("entities"),
                    }

        # 3. Governance (Verification Coverage)
        tracking_file = os.path.join(self.saguaro_dir, "tracking.json")
        if os.path.exists(tracking_file):
            from saguaro.indexing.tracker import IndexTracker
            tracker = IndexTracker(tracking_file)
            total_files = len(tracker.state)
            verified_files = sum(1 for entry in tracker.state.values() if entry.get("verified", False))
            
            report["governance"] = {
                "total_tracked_files": total_files,
                "verified_files": verified_files,
                "coverage_percent": round((verified_files / total_files * 100), 1) if total_files > 0 else 0
            }

        return report

    def print_dashboard(self):
        report = self.generate_report()

        print("\n" + "═" * 50)
        print(" SAGUARO Enterprise Q-COS Health Dashboard ".center(50, "═"))
        print("═" * 50)

        # 1. System Health Status
        fresh = report.get("freshness", {})
        if "last_update_fmt" in fresh:
            age = fresh["age_seconds"]
            if age < 3600:
                status, icon = "Synchronized", "🟢"
            elif age < 86400:
                status, icon = "Stale (>1h)", "🟡"
            else:
                status, icon = "Diverged (>24h)", "🔴"
            
            print(f"\n{icon} System Status: {status}")
            print(f"  Last Pulse: {fresh['last_update_fmt']}")
        else:
            print("\n⚪ System Status: PENDING (Initial Index Required)")

        # 2. Governance & Compliance
        gov = report.get("governance", {})
        if gov:
            print("\n🛡️ Governance Audit")
            icon = "✅" if gov["coverage_percent"] > 90 else "⚠️"
            print(f"  Verification Coverage: {gov['coverage_percent']}% {icon}")
            print(f"  Verified Files:        {gov['verified_files']} / {gov['total_tracked_files']}")

        # 3. Memory & Performance
        perf = report.get("performance", {})
        if perf:
            print("\n⚡ Resource Intelligence")
            print(f"  Peak Indexing RAM: {perf['peak_memory_mb']:.1f} MB")
            print(f"  Indexed Assets:    {perf['indexed_entities']} entities across {perf['indexed_files']} files")

        # 4. Storage Footprint
        store = report.get("storage", {})
        if store:
            print("\n💾 Neural Storage")
            print(f"  Holographic Bundle: {store['vector_index_mb']} MB")
            print(f"  Semantic Metadata:  {store['metadata_mb']} MB")
            print(f"  Total Capacity:     {store['total_mb']} MB")

        # 5. Holographic Config
        conf = report.get("config", {})
        if conf:
            print("\n🧬 Quantum Configuration")
            print(f"  Manifold Dimension: {conf.get('active_dim')} (Active) / {conf.get('total_dim')} (Total)")
            dsr = conf.get('dark_space_ratio', 'Unknown')
            try:
                print(f"  Dark Space Buffer:  {float(dsr) * 100:.0f}%")
            except (ValueError, TypeError):
                print(f"  Dark Space Buffer:  {dsr}")

        print("\n" + "═" * 50)
