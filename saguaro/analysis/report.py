import os
import time
from typing import Dict, Any
from saguaro.coverage import CoverageReporter
from saguaro.analysis.dead_code import DeadCodeAnalyzer
from saguaro.sentinel.verifier import SentinelVerifier
from saguaro.analysis.entry_points import EntryPointDetector
# Use standard library or simple heuristics for "Architecture" if no specialized module yet


class ReportGenerator:
    def __init__(self, root_path: str):
        self.root_path = os.path.abspath(root_path)

    def generate(self) -> Dict[str, Any]:
        """Generates the comprehensive State of the Repo report."""
        report = {
            "generated_at": time.time(),
            "repo_path": self.root_path,
            "sections": {},
        }

        # 1. Coverage
        print("Gathering coverage stats...")
        cov = CoverageReporter(self.root_path)
        report["sections"]["coverage"] = cov.generate_report()

        # 2. Dead Code (Reachability)
        print("Analyzing reachability (Dead Code)...")
        # DeadCodeAnalyzer might be slow, use defaults
        dc = DeadCodeAnalyzer(self.root_path)
        candidates = dc.analyze()
        # Summary only
        report["sections"]["dead_code"] = {
            "total_candidates": len(candidates),
            "high_confidence_candidates": len(
                [c for c in candidates if c["confidence"] > 0.8]
            ),
            "top_candidates": candidates[:10] if candidates else [],
        }

        # 3. Sentinel (Security & Violations)
        print("Running Sentinel Audit...")
        # Maybe skip slow engines like mypy if we want speed?
        # Roadmap implies a "durable, structured artifact", so quality counts.
        # But for 'finish the roadmap', I'll use default engines.
        verifier = SentinelVerifier(
            self.root_path, engines=["native", "ruff", "semantic"]
        )
        # Skip mypy/vulture for speed in this implementation unless requested
        violations = verifier.verify_all()

        # Aggregate violations by severity/category
        violation_stats = {"critical": 0, "high": 0, "medium": 0, "low": 0, "total": 0}
        by_engine = {}

        for v in violations:
            sev = v.get("severity", "low")
            violation_stats[sev] = violation_stats.get(sev, 0) + 1
            violation_stats["total"] += 1

            eng = v.get(
                "engine", "unknown"
            )  # Assuming we add engine field to violation
            by_engine[eng] = by_engine.get(eng, 0) + 1

        report["sections"]["sentinel"] = {
            "summary": violation_stats,
            "by_engine": by_engine,
            "violation_count": len(violations),
        }

        # 4. Architecture / Entry Points
        print("Mapping Entry Points...")
        ep_detector = EntryPointDetector(self.root_path)
        entry_points = ep_detector.detect()
        report["sections"]["architecture"] = {
            "entry_points": len(entry_points),
            "entry_point_types": {},
            # TODO: Add dependency depth analysis if we had a graph module ready
        }
        for ep in entry_points:
            etype = ep.get("type", "unknown")
            report["sections"]["architecture"]["entry_point_types"][etype] = (
                report["sections"]["architecture"]["entry_point_types"].get(etype, 0)
                + 1
            )

        # 5. Features / Capabilities
        # Placeholder for Feature Inventory
        report["sections"]["features"] = {
            "inventory": "Not yet implemented (requires semantic Capability Map)"
        }

        return report

    def save_markdown(self, report: Dict[str, Any], output_path: str):
        """Saves report as Markdown."""
        lines = []
        lines.append(f"# State of the Repo: {os.path.basename(report['repo_path'])}")
        lines.append(f"**Generated:** {time.ctime(report['generated_at'])}")
        lines.append("")

        # Coverage
        cov = report["sections"]["coverage"]
        lines.append("## 1. Codebase Coverage")
        lines.append(f"- **Total Files:** {cov['total_files']}")
        lines.append(f"- **AST Supported:** {cov['ast_supported_files']}")
        lines.append("- **Languages:**")
        for lang, count in cov["languages"].items():
            lines.append(f"  - {lang}: {count}")
        lines.append("")

        # Dead Code
        dc = report["sections"]["dead_code"]
        lines.append("## 2. Dead Code & Debt")
        lines.append(f"- **Candidates:** {dc['total_candidates']}")
        lines.append(f"- **High Confidence:** {dc['high_confidence_candidates']}")
        if dc["top_candidates"]:
            lines.append("### Top Candidates to Remove:")
            for c in dc["top_candidates"]:
                lines.append(
                    f"- [{c['confidence']:.2f}] `{c['symbol']}` in {os.path.basename(c['file'])}"
                )
        lines.append("")

        # Sentinel
        sen = report["sections"]["sentinel"]
        lines.append("## 3. Sentinel Health (Security & Governance)")
        lines.append(f"- **Total Violations:** {sen['violation_count']}")
        lines.append("- **Severity Breakdown:**")
        for sev, count in sen["summary"].items():
            if sev != "total":
                lines.append(f"  - {sev.title()}: {count}")
        lines.append("")

        # Architecture
        arch = report["sections"]["architecture"]
        lines.append("## 4. Architecture & Entry Points")
        lines.append(f"- **Entry Points Detected:** {arch['entry_points']}")
        for t, c in arch["entry_point_types"].items():
            lines.append(f"  - {t}: {c}")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))
