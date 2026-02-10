import os
import yaml
import json
import pytest
from saguaro.health import HealthDashboard

@pytest.fixture
def mock_saguaro_dir(tmp_path):
    saguaro_dir = tmp_path / ".saguaro"
    saguaro_dir.mkdir()
    
    # Create mock config
    config = {
        "active_dim": 8192,
        "total_dim": 16384,
        "dark_space_ratio": 0.4,
        "last_index": {
            "peak_memory_mb": 150.5,
            "files": 10,
            "entities": 100
        }
    }
    with open(saguaro_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
        
    # Create mock vector storage
    vectors_dir = saguaro_dir / "vectors"
    vectors_dir.mkdir()
    (vectors_dir / "vectors.bin").write_text("dummy")
    (vectors_dir / "metadata.json").write_text("[]")
    
    # Create mock tracking
    tracking = {
        "file1.py": {"mtime": 100, "hash": "h1", "verified": True},
        "file2.py": {"mtime": 100, "hash": "h2", "verified": False}
    }
    with open(saguaro_dir / "tracking.json", "w") as f:
        json.dump(tracking, f)
        
    return str(saguaro_dir)

def test_health_report_generation(mock_saguaro_dir):
    dashboard = HealthDashboard(mock_saguaro_dir)
    report = dashboard.generate_report()
    
    # Check sections
    assert "freshness" in report
    assert "storage" in report
    assert "config" in report
    assert "performance" in report
    assert "governance" in report
    
    # Check values
    assert report["config"]["active_dim"] == 8192
    assert report["performance"]["peak_memory_mb"] == 150.5
    assert report["governance"]["coverage_percent"] == 50.0  # 1/2 verified
    assert report["governance"]["total_tracked_files"] == 2

def test_print_dashboard(mock_saguaro_dir, capsys):
    dashboard = HealthDashboard(mock_saguaro_dir)
    dashboard.print_dashboard()
    captured = capsys.readouterr()
    
    assert "SAGUARO Enterprise Q-COS Health Dashboard" in captured.out
    assert "Governance Audit" in captured.out
    assert "Verification Coverage: 50.0%" in captured.out
    assert "Peak Indexing RAM: 150.5 MB" in captured.out
