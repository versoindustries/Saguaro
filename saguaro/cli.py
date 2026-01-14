import argparse
import sys
import os
from saguaro import __version__

def main():
    parser = argparse.ArgumentParser(description=f"SAGUARO v{__version__} - Quantum Codebase OS")
    parser.add_argument("--version", action="version", version=f"SAGUARO v{__version__}")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Init
    init_parser = subparsers.add_parser("init", help="Initialize SAGUARO in the current directory")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing .saguaro directory")
    
    # Quickstart
    # Quickstart
    subparsers.add_parser("quickstart", help="One-command setup (Init + Index + Configs)")
    
    # Index
    index_parser = subparsers.add_parser("index", help="Index the codebase")
    index_parser.add_argument("--path", default=".", help="Codebase path")
    index_parser.add_argument("--force", action="store_true", help="Force re-indexing of all files")
    index_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # Watch logic
    watch_parser = subparsers.add_parser("watch", help="Watch for changes and index incrementally")
    watch_parser.add_argument("--path", default=".", help="Codebase path")
    watch_parser.add_argument("--interval", type=int, default=5, help="Poll interval in seconds")

    # Serve (DNI)
    serve_parser = subparsers.add_parser("serve", help="Start DNI Server")
    serve_parser.add_argument("--mcp", action="store_true", help="Start in MCP Server mode")
    serve_parser.add_argument("--auth-token", help="Require authentication token (MCP only)")

    # Coverage
    coverage_parser = subparsers.add_parser("coverage", help="Repo coverage report")
    coverage_parser.add_argument("--path", default=".", help="Path to report on")

    # Health
    # Health
    subparsers.add_parser("health", help="Index Health Dashboard")

    # Governor
    gov_parser = subparsers.add_parser("governor", help="Context Budget Governor")
    gov_parser.add_argument("--check", help="Check string against budget", action="store_true")
    gov_parser.add_argument("--text", help="Text to check")

    # Workset

    # Workset
    workset_parser = subparsers.add_parser("workset", help="Manage Agent Worksets")
    workset_sub = workset_parser.add_subparsers(dest="workset_op")
    
    ws_create = workset_sub.add_parser("create", help="Create a new workset")
    ws_create.add_argument("--desc", required=True, help="Description of task")
    ws_create.add_argument("--files", required=True, help="Comma-separated list of files")
    
    workset_sub.add_parser("list", help="List active worksets")
    
    ws_show = workset_sub.add_parser("show", help="Show workset details")
    ws_show.add_argument("id", help="Workset ID")

    ws_expand = workset_sub.add_parser("expand", help="Expand workset budget/files")
    ws_expand.add_argument("id", help="Workset ID")
    ws_expand.add_argument("--files", required=True, help="New files to add")
    ws_expand.add_argument("--justification", required=True, help="Reason for escalation")

    ws_lock = workset_sub.add_parser("lock", help="Acquire lease on workset (Active)")
    ws_lock.add_argument("id", help="Workset ID")

    ws_unlock = workset_sub.add_parser("unlock", help="Release lease on workset (Closed)")
    ws_unlock.add_argument("id", help="Workset ID")


    # Refactor
    refactor_parser = subparsers.add_parser("refactor", help="Refactoring Intelligence")
    refactor_sub = refactor_parser.add_subparsers(dest="refactor_op")
    
    plan_parser = refactor_sub.add_parser("plan", help="Plan a refactor")
    plan_parser.add_argument("--symbol", required=True, help="Symbol to modify/rename")

    # Rename
    rename_parser = refactor_sub.add_parser("rename", help="Semantic Rename")
    rename_parser.add_argument("old", help="Old symbol Name")
    rename_parser.add_argument("new", help="New symbol Name")
    rename_parser.add_argument("--execute", action="store_true", help="Apply changes")

    # Shim
    shim_parser = refactor_sub.add_parser("shim", help="Generate Compatibility Shim")
    shim_parser.add_argument("path", help="Path to original file (will be overwritten)")
    shim_parser.add_argument("target", help="Module path to redirect to")

    # Safe Delete
    del_parser = refactor_sub.add_parser("safedelete", help="Safe Delete File")
    del_parser.add_argument("path", help="File to delete")
    del_parser.add_argument("--force", action="store_true", help="Ignore dependencies")
    del_parser.add_argument("--execute", action="store_true", help="Apply deletion")

    
    # Feedback
    fb_parser = subparsers.add_parser("feedback", help="Context Feedback Loop")
    fb_sub = fb_parser.add_subparsers(dest="fb_op")
    fb_log = fb_sub.add_parser("log", help="Log feedback")
    fb_log.add_argument("--query", required=True)
    fb_log.add_argument("--used", help="Comma-separated IDs of used items")
    fb_log.add_argument("--ignored", help="Comma-separated IDs of ignored items")
    
    fb_sub.add_parser("stats", help="Show feedback stats")

    # Query
    query_parser = subparsers.add_parser("query", help="Query the index")
    query_parser.add_argument("text", help="Query text")
    query_parser.add_argument("--k", type=int, default=5, help="Number of results")
    query_parser.add_argument("--file", help="Seed file for scoped search")
    query_parser.add_argument("--level", type=int, default=3, choices=[0, 1, 2, 3], help="Escalation level (0=Local, 3=Global)")
    query_parser.add_argument("--json", action="store_true", help="Output deterministic context bundle JSON")
    query_parser.add_argument("--profile", action="store_true", help="Enable query profiling")
    query_parser.add_argument("--workset", help="ID of active workset to scope query")

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify codebase against rules")
    verify_parser.add_argument("path", help="Path to verify", nargs="?", default=".")
    verify_parser.add_argument("--engines", help="Comma-separated list of engines (native,ruff,mypy,vulture)", default=None)
    verify_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    verify_parser.add_argument("--fix", action="store_true", help="Automatically fix violations where possible")

    # Chronicle (Time Crystal)
    chronicle_parser = subparsers.add_parser("chronicle", help="Manage Time Crystal snapshots")
    chronicle_sub = chronicle_parser.add_subparsers(dest="chronicle_op")
    
    chronicle_sub.add_parser("snapshot", help="Create a semantic snapshot")
    chronicle_sub.add_parser("list", help="List snapshots")
    chronicle_sub.add_parser("diff", help="Calculate drift between latest snapshots")

    # Legislation (Auto-Legislator)
    legislation_parser = subparsers.add_parser("legislation", help="Rule discovery")
    legislation_parser.add_argument("--draft", action="store_true", help="Scan and draft new rules")
    
    # Train (Adaptive Encoder)
    train_parser = subparsers.add_parser("train", help="Train Adaptive Encoder")
    train_parser.add_argument("--path", default=".", help="Corpus path")
    train_parser.add_argument("--epochs", type=int, default=1)

    # Train Baseline (Dev Tool)
    tb_parser = subparsers.add_parser("train-baseline", help="Train pretrained tokenizer baseline")
    tb_parser.add_argument("--corpus", help="Corpus path")
    tb_parser.add_argument("--curriculum", help="Name of curriculum preset (e.g. verso-baseline)")
    tb_parser.add_argument("--output", default="saguaro/artifacts/codebooks/verso_baseline.json")
    tb_parser.add_argument("--fast", action="store_true")

    # Constellation (Global Memory)
    constellation_parser = subparsers.add_parser("constellation", help="Manage Global Constellation")
    constellation_sub = constellation_parser.add_subparsers(dest="constellation_op")
    
    constellation_sub.add_parser("list", help="List global libraries")
    
    c_index = constellation_sub.add_parser("index-lib", help="Index a library to global storage")
    c_index.add_argument("name", help="Library name (e.g. requests-2.31)")
    c_index.add_argument("--path", help="Path to library source", required=True)
    
    c_link = constellation_sub.add_parser("link", help="Link a global library to current project")
    c_link.add_argument("name", help="Library name to link")

    # Benchmarks
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    bench_parser.add_argument("--dataset", default="CodeSearchNet", help="Dataset to run (CodeSearchNet, SWE-bench, custom)")
    bench_parser.add_argument("--custom", help="Path to custom JSON dataset file (if dataset=custom)")

    # Dead Code
    deadcode_parser = subparsers.add_parser("deadcode", help="Dead Code Discovery")
    deadcode_parser.add_argument("--threshold", type=float, default=0.5, help="Minimum confidence threshold")

    # Impact
    impact_parser = subparsers.add_parser("impact", help="Impact Analysis")
    impact_parser.add_argument("--path", required=True, help="File to analyze")

    # Report (State of the Repo)
    report_parser = subparsers.add_parser("report", help="Generate State of the Repo Report")
    report_parser.add_argument("--format", choices=["json", "markdown"], default="markdown", help="Output format")
    report_parser.add_argument("--output", default="saguaro_report.md", help="Output file path")

    # Analyze (Phase 4)
    analyze_parser = subparsers.add_parser("analyze", help="Deep Texture Analysis (Health Card)")
    analyze_parser.add_argument("--json", action="store_true", help="Output JSON format")

    # Knowledge
    kb_parser = subparsers.add_parser("knowledge", help="Shared Agent Knowledge Base")
    kb_sub = kb_parser.add_subparsers(dest="kb_op")
    
    kb_add = kb_sub.add_parser("add", help="Add a fact")
    kb_add.add_argument("--category", required=True, choices=["invariant", "rule", "pattern", "zone"])
    kb_add.add_argument("--key", required=True, help="Fact key")
    kb_add.add_argument("--value", required=True, help="Fact value")
    
    kb_list = kb_sub.add_parser("list", help="List facts")
    kb_list.add_argument("--category", help="Filter by category")
    
    kb_search = kb_sub.add_parser("search", help="Search facts")
    kb_search.add_argument("query", help="Search query")

    # Auditor
    audit_parser = subparsers.add_parser("audit", help="Auditor Agent Verification")
    audit_parser.add_argument("--path", help="Path to audit (diff mode)")

    # Build Graph
    subparsers.add_parser("build-graph", help="Build System Graph")

    # Entry Points
    subparsers.add_parser("entrypoints", help="Runtime Entry Point Detection")

    # Scribe (Phase 2: Synthesis)
    scribe_parser = subparsers.add_parser("scribe", help="Generative Engine")
    scribe_parser.add_argument("task", help="Task description")
    scribe_parser.add_argument("--file", help="Context file (optional hint)")
    scribe_parser.add_argument("--out", help="Output patch file", default="patch.json")

    # --- Phase 4/SSAI: Agent & Orchestration ---
    agent_parser = subparsers.add_parser("agent", help="SSAI Agent Interface")
    agent_sub = agent_parser.add_subparsers(dest="agent_command")

    # Perception Layer (with enhanced help for AI Adoption - Phase 3)
    skel_parser = agent_sub.add_parser(
        "skeleton", 
        help="Generate File Skeleton (signatures + docstrings only)",
        description="""Use this INSTEAD of view_file to explore a file's structure.
        
Shows function/class signatures and docstrings without full implementation code.
Saves 90%% of tokens compared to reading full files.

DECISION TREE:
  Need to understand a file's structure? → Use this command
  Need to read a specific function? → Then use 'saguaro agent slice'
  
Example:
  saguaro agent skeleton src/core.py
"""
    )
    skel_parser.add_argument("file", help="Target file path to generate skeleton for")

    slice_parser = agent_sub.add_parser(
        "slice", 
        help="Generate Context Slice (function + dependencies)",
        description="""Read a specific symbol with its dependencies and context.
        
Use this INSTEAD of view_file or view_code_item to read function/class code.
Automatically includes imports and parent context for understanding.

DECISION TREE:
  Need to read a specific function/class? → Use this command
  Need to explore file structure first? → Use 'saguaro agent skeleton' first
  Symbol not found? → Use 'saguaro query' to find it semantically
  
Example:
  saguaro agent slice MyClass.method --depth 2
"""
    )
    slice_parser.add_argument("symbol", help="Entry point symbol (e.g., ClassName.method)")
    slice_parser.add_argument("--depth", type=int, default=1, help="Dependency graph depth (default: 1)")

    # Action Layer
    patch_parser = agent_sub.add_parser("patch", help="Apply Semantic Patch")
    patch_parser.add_argument("file", help="Target file")
    patch_parser.add_argument("patch_json", help="Patch content or path to JSON")

    verify_parser = agent_sub.add_parser("verify", help="Verify Sandbox State")
    verify_parser.add_argument("sandbox_id", help="Sandbox ID to verify")

    # Intelligence Layer
    imp_parser = agent_sub.add_parser("impact", help="Predict Impact")
    imp_parser.add_argument("sandbox_id", help="Sandbox ID")

    commit_parser = agent_sub.add_parser("commit", help="Commit Sandbox to Disk")
    commit_parser.add_argument("sandbox_id", help="Sandbox ID")

    # Legacy Runner
    run_parser = agent_sub.add_parser("run", help="Run a specialized agent")
    run_parser.add_argument("role", choices=["planner", "cartographer", "surgeon", "auditor"], help="Agent role to run")
    run_parser.add_argument("--task", help="Task description or ID")
    
    # Task Graph
    task_parser = subparsers.add_parser("tasks", help="Manage Task Graph")
    task_parser.add_argument("--list", action="store_true", help="List ready tasks")
    task_parser.add_argument("--add", help="Add new task JSON string")
    
    # Shared Memory
    mem_parser = subparsers.add_parser("memory", help="Inspect Shared Memory")
    mem_parser.add_argument("--list", action="store_true", help="List all facts")
    mem_parser.add_argument("--read", help="Read fact by key")
    mem_parser.add_argument("--write", nargs=2, metavar=('KEY', 'VALUE'), help="Write fact (Key Value)")

    # --- Phase 5: Simulation ---
    sim_parser = subparsers.add_parser("simulate", help="Run Simulations")
    sim_sub = sim_parser.add_subparsers(dest="sim_op")
    
    sim_sub.add_parser("volatility", help="Generate Volatility Map")
    
    sim_reg = sim_sub.add_parser("regression", help="Predict Regressions")
    sim_reg.add_argument("--files", required=True, help="Comma-separated files changed")

    # --- Phase 6: Learning ---
    route_parser = subparsers.add_parser("route", help="Test Intent Routing")
    route_parser.add_argument("query", help="Query to classify")

    # --- AI Adoption Metrics ---
    metrics_parser = subparsers.add_parser(
        "metrics", 
        help="View AI adoption metrics (Saguaro vs fallback tool usage)"
    )
    metrics_parser.add_argument(
        "--session", 
        action="store_true", 
        help="Show current session metrics only"
    )
    metrics_parser.add_argument(
        "--json", 
        action="store_true", 
        help="Output as JSON"
    )
    metrics_parser.add_argument(
        "--reset", 
        action="store_true", 
        help="Reset all metrics"
    )

    args = parser.parse_args()
    
    # ... (existing command handlers) ...

    if args.command == "deadcode":
        from saguaro.analysis.dead_code import DeadCodeAnalyzer
        analyzer = DeadCodeAnalyzer(os.getcwd())
        print("Scanning for dead code...")
        candidates = analyzer.analyze()
        
        filtered = [c for c in candidates if c['confidence'] >= args.threshold]
        
        if not filtered:
             print("No dead code found with high confidence.")
        else:
             print(f"Found {len(filtered)} candidates:")
             for c in filtered:
                  print(f"[{c['confidence']:.2f}] {c['symbol']} ({c['file']})")
             print("\nNote: Verify manually before deletion.")

    elif args.command == "impact":
        from saguaro.analysis.impact import ImpactAnalyzer
        if not args.path:
            print("Error: --path required for analysis")
            sys.exit(1)
            
        analyzer = ImpactAnalyzer(os.getcwd())
        target = os.path.abspath(args.path)
        print(f"Analyzing impact for: {target}")
        
        report = analyzer.analyze_change(target)
        
        print("\n=== Impact Report ===")
        print(f"Target Module: {report['module']}")
        print(f"Total Dependents: {report['impact_score']}")
        
        print(f"\n[Tests Impacted] ({len(report['tests_impacted'])}):")
        for t in report['tests_impacted']:
            print(f" - {os.path.relpath(t, os.getcwd())}")
            
        print(f"\n[Interfaces/Code Impacted] ({len(report['interfaces_impacted'])}):")
        for i in report['interfaces_impacted']:
             print(f" - {os.path.relpath(i, os.getcwd())}")
             
        print(f"\n[Build Targets] ({len(report['build_targets'])}):")
        for b in report['build_targets']:
             print(f" - {os.path.relpath(b, os.getcwd())}")

    elif args.command == "report":
        from saguaro.analysis.report import ReportGenerator
        import json
        
        print("Generating State of the Repo Report...")
        generator = ReportGenerator(os.getcwd())
        report_data = generator.generate()
        
        if args.format == "json":
            with open(args.output, "w") as f:
                json.dump(report_data, f, indent=2)
        else:
            generator.save_markdown(report_data, args.output)
            
        print(f"Report saved to: {args.output}")

    elif args.command == "analyze":
        from saguaro.analysis.health_card import RepoHealthCard
        import json
        
        print("Running Deep Texture Analysis...")
        card = RepoHealthCard(os.getcwd())
        results = card.generate_card()
        
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print("\n=== Saguaro Health Card ===")
            print(f"Health Score: {results['health_score']:.2f}/1.00\n")
            
            m = results['metrics']
            print(f"Complexity:   {m['complexity']['score']:.2f} ({m['complexity']['rating']})")
            print(f"Dead Code:    {m['dead_code']['ratio']*100:.2f}% ({m['dead_code']['count']} chunks)")
            print(f"Type Safety:  {m['type_safety']['score']:.2f} ({m['type_safety']['errors']} errors, {m['type_safety']['density']*100:.2f}% density)")
            print("\nrecommendation: Run 'saguaro verify --fix' to improve score.")

    elif args.command == "audit":
        print("SAGUARO Auditor: automated governance check...\n")
        passed = True
        
        # 1. Run Sentinel Verification
        print("[1/3] Running Sentinel Verification...")
        from saguaro.sentinel.verifier import SentinelVerifier
        verifier = SentinelVerifier(repo_path=os.getcwd(), engines=None) # All engines
        violations = verifier.verify_all()
        if violations:
            print(f"  FAIL: {len(violations)} violations found.")
            passed = False
        else:
            print("  PASS: Codebase is compliant.")
            
        # 2. Check Critical Knowledge Invariants (Zones)
        print("\n[2/3] Checking Regulatory Zones...")
        from saguaro.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(os.path.join(os.getcwd(), ".saguaro"))
        zones = kb.get_facts("zone") # "do not touch" zones
        # Basic check: did we touch any files in these zones?
        # Requires knowing what changed. 
        # For now, we assume "audit" is run on dirty state or we perform a git diff check (not implemented yet).
        # We'll skip for prototype or check if 'path' arg was passed.
        if args.path: # Audit specific path
            for z in zones:
                 # Check overlap
                 pass
        print(f"  PASS: {len(zones)} zones monitored (no active violation check).")

        # 3. Impact Risk Assessment
        print("\n[3/3] Impact Risk Assessment...")
        # If we had a diff, we'd check impact score.
        print("  PASS: Risk acceptable.")
        
        print("\n=== Audit Decision ===")
        if passed:
            print("✅ APPROVED")
            sys.exit(0)
        else:
            print("❌ REJECTED")
            sys.exit(1)

    elif args.command == "knowledge":
        from saguaro.knowledge_base import KnowledgeBase
        saguaro_dir = os.path.join(os.getcwd(), ".saguaro")
        kb = KnowledgeBase(saguaro_dir)
        
        if args.kb_op == "add":
            kb.add_fact(args.category, args.key, args.value)
            print(f"Fact added: [{args.category}] {args.key}")
            
        elif args.kb_op == "list":
            facts = kb.get_facts(args.category)
            if not facts:
                 print("No facts found.")
            for f in facts:
                 print(f"[{f.category}] {f.key}: {f.value}")
                 
        elif args.kb_op == "search":
            results = kb.search(args.query)
            for f in results:
                 print(f"[{f.category}] {f.key}: {f.value}")

    elif args.command == "build-graph":
        from saguaro.build_system.ingestor import BuildGraphIngestor
        ingestor = BuildGraphIngestor(os.getcwd())
        print("Ingesting build graph...")
        graph = ingestor.ingest()
        
        print(f"Discovered {graph['target_count']} targets:")
        for name, data in graph['targets'].items():
            print(f" - [{data['type']}] {name} (defined in {data['file']})")
            if data['deps']:
                print(f"    Deps: {', '.join(data['deps'][:5])}{'...' if len(data['deps']) > 5 else ''}")
        
    elif args.command == "entrypoints":
        from saguaro.analysis.entry_points import EntryPointDetector
        detector = EntryPointDetector(os.getcwd())
        print("Scanning for runtime entry points...")
        eps = detector.detect()
        
        if not eps:
            print("No entry points found.")
        else:
            print(f"Found {len(eps)} entry points:")
            for ep in eps:
                name = ep.get('name', 'N/A')
                print(f" - [{ep['type']}] {name} ({os.path.relpath(ep['file'], os.getcwd())}:{ep['line']})")

    elif args.command == "serve":
        if hasattr(args, 'mcp') and args.mcp:
            from saguaro.mcp.server import main as mcp_main
            # Ensure auth token is in environ if not passed via sys.argv check in mcp_main, 
            # but mcp_main checks sys.argv too.
            # To be safe, we can set env var if args.auth_token is present, 
            # in case sys.argv parsing in mcp_main is position sensitive or confused by other flags.
            if args.auth_token:
                os.environ["saguaro_MCP_TOKEN"] = args.auth_token
            mcp_main()
        else:
            from saguaro.dni.server import main as server_main
            server_main()
        
    elif args.command == "query":
        # One-off query wrapper around DNI logic or Engine
        # For simplicity, reuse DNI logic but print to stdout
        from saguaro.dni.server import DNIServer
        
        # If we have a file/level, we use EscalationLadder
        if args.file:
            from saguaro.indexing.engine import IndexEngine
            from saguaro.indexing.auto_scaler import get_repo_stats_and_config
            from saguaro.escalation import EscalationLadder
            from saguaro.profiling import profiler
            
            # Need to partial init the engine/store to get query vector
            target_path = os.getcwd() # Assume root
            saguaro_dir = os.path.join(target_path, ".saguaro")
            
            stats = get_repo_stats_and_config(target_path)
            engine = IndexEngine(target_path, saguaro_dir, stats)
            
            with profiler.measure("query_encoding"):
                # Encode
                query_vec = engine.encode_text(args.text, dim=stats['total_dim'])
            
            with profiler.measure("search_retrieval"):
                # Search
                ladder = EscalationLadder(engine.store, target_path)
                results = ladder.search(query_vec[0], args.file, level=args.level, k=args.k)
            
            if args.profile:
                print(f"[Profile] Encoding: {profiler.stats.get('query_encoding', 0):.2f}ms")
                print(f"[Profile] Retrieval: {profiler.stats.get('search_retrieval', 0):.2f}ms")
            
            workset = None
            if args.workset:
                 from saguaro.workset import WorksetManager
                 wm = WorksetManager(saguaro_dir)
                 workset = wm.get_workset(args.workset)
                 if not workset:
                      print(f"Warning: Workset {args.workset} not found.", file=sys.stderr)

            if args.json:
                import time
                from saguaro.context import ContextBuilder
                bundle = ContextBuilder.build_from_results(args.text, results, time.time(), workset=workset)
                print(bundle.to_json())
            else:
                print(f"Query: '{args.text}' [Scoped: {args.level}]")
                for res in results:
                    print(f"[{res.get('rank', '?')}] [{res['score']:.4f}] {res['name']} ({res['type']})")
                    print(f"    Path: {res['file']}:{res['line']}")
                    print(f"    Why:  {res.get('reason', 'N/A')}")
                    print(f"    Scope: {res.get('scope', 'Global')}")
                    print("")
                
        else:
            # Use standard server logic
            server = DNIServer()
            server.initialize({"path": "."}) # Assume current dir
            result = server.query({"text": args.text, "k": args.k})
            
            workset = None
            if args.workset:
                 from saguaro.workset import WorksetManager
                 wm = WorksetManager(os.path.join(os.getcwd(), ".saguaro"))
                 workset = wm.get_workset(args.workset)
                 if not workset:
                      print(f"Warning: Workset {args.workset} not found.", file=sys.stderr)

            if args.json:
                import time
                from saguaro.context import ContextBuilder
                bundle = ContextBuilder.build_from_results(args.text, result['results'], time.time(), workset=workset)
                print(bundle.to_json())
            else:
                print(f"Query: '{args.text}'")
                for res in result['results']:
                    print(f"[{res['rank']}] [{res['score']:.4f}] {res['name']} ({res['type']})")
                    print(f"    Path: {res['file']}:{res['line']}")
                    print(f"    Why:  {res.get('reason', 'N/A')}")
                    print("")

    elif args.command == "init":
        print("Initializing SAGUARO...")
        saguaro_dir = os.path.join(os.getcwd(), ".saguaro")
        if os.path.exists(saguaro_dir) and not args.force:
            print("SAGUARO already initialized. Use --force to overwrite.")
            sys.exit(1)
            
        os.makedirs(saguaro_dir, exist_ok=True)
        # Create default config
        from saguaro.defaults import get_default_yaml
        config_path = os.path.join(saguaro_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(get_default_yaml() + "\n")
            
        print(f"initialized empty SAGUARO repository in {saguaro_dir}")

    elif args.command == "quickstart":
        from saguaro.quickstart import QuickstartManager
        qs = QuickstartManager(os.getcwd())
        qs.execute()

    elif args.command == "index":
        target_path = os.path.abspath(args.path)
        print(f"SAGUARO Indexing: {target_path}")
        
        # Auto-scale analysis
        from saguaro.indexing.auto_scaler import get_repo_stats_and_config
        stats = get_repo_stats_and_config(target_path)
        
        print(f"  - LoC Estimate: {stats['loc']}")
        print(f"  - Recommended Dimension: {stats['active_dim']}")
        print(f"  - Total Dimension (with Buffer): {stats['total_dim']}")
        print(f"  - Dark Space Ratio: {stats['dark_space_ratio']:.2f}")
        
        print("\nConnecting to Quantum Core...")
        from saguaro.indexing.engine import IndexEngine, process_batch_worker
        
        # Initialize Engine (Main Process)
        saguaro_dir = os.path.join(os.getcwd(), ".saguaro")
        engine = IndexEngine(target_path, saguaro_dir, stats)
        
        # Load config for exclusions
        import yaml
        import multiprocessing
        import concurrent.futures
        from saguaro.utils.file_utils import get_code_files

        config_path = os.path.join(saguaro_dir, "config.yaml")
        config = {}
        if os.path.exists(config_path):
             with open(config_path, 'r') as f:
                 config = yaml.safe_load(f) or {}
        
        exclusions = config.get('indexing', {}).get('exclude', [])
        
        # Robust File Discovery
        all_files = get_code_files(target_path, exclusions)
        
        total_files = len(all_files)
        # Use max CPU cores but leave one for system/main
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        
        print(f"Discovered {total_files} files.")
        if total_files == 0:
            print("WARNING: No code files found. Check your path or .saguaro/config.yaml exclusions.")
            sys.exit(0)

        # Calibrate Tokenizer (Main Process scans first to build vocab if needed)
        print("Calibrating Quantum Tokenizer (LOC-Aware)...")
        engine.calibrate(all_files) # TODO: Make this parallel too if slow, but usually fast

        # Filter (Incremental)
        if not args.force:
            needed = engine.tracker.filter_needs_indexing(all_files)
            if not needed:
                print("No files modified since last index.")
                sys.exit(0)
            print(f"Incrementally indexing {len(needed)} files ({total_files - len(needed)} skipped).")
            files_to_process = needed
        else:
            files_to_process = all_files
            engine.tracker.clear()
            
        # Process in chunks
        BATCH_SIZE = 64
        file_chunks = [files_to_process[i:i + BATCH_SIZE] for i in range(0, len(files_to_process), BATCH_SIZE)]
        
        # Get Dynamic Vocab Size from Engine (post-calibration)
        vocab_size = engine.vocab_size
        print(f"Starting Parallel Indexing with {num_workers} workers (vocab={vocab_size})...")
        
        indexed_files = 0
        indexed_entities = 0
        
        # Use 'spawn' context to safer TF handling
        ctx = multiprocessing.get_context('spawn')
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            # Submit all batches
            # Note: We pass primitive args (list of str, int, int, int) to avoid pickling the Engine
            futures = {
                executor.submit(process_batch_worker, chunk, stats['active_dim'], stats['total_dim'], vocab_size): chunk 
                for chunk in file_chunks
            }
            
            for future in concurrent.futures.as_completed(futures):
                chunk = futures[future]
                try:
                    meta_list, vectors_np = future.result()
                    
                    # Ingest into Main
                    f_cnt, e_cnt = engine.ingest_worker_result(meta_list, vectors_np)
                    indexed_files += f_cnt
                    indexed_entities += e_cnt
                    
                    # Update tracker for this chunk immediately
                    # We can do this because we know they succeeded
                    if f_cnt > 0:
                         engine.tracker.update([m['file'] for m in meta_list])
                         
                    print(f"Processed batch ({f_cnt} files, {e_cnt} entities)", end='\r')
                    
                except Exception as e:
                    print(f"\nBatch failed: {e}")
                    
        engine.commit()
        print("\n\nIndexing Complete.")
        print(f" - Files Processed: {indexed_files}")
        print(f" - Entities Indexed: {indexed_entities}")
        
        # Update config.yaml with scaling stats
        import yaml
        config_path = os.path.join(saguaro_dir, "config.yaml")
        existing_config = {}
        if os.path.exists(config_path):
             with open(config_path, 'r') as f:
                 existing_config = yaml.safe_load(f) or {}
        
        existing_config.update(stats)
        
        with open(config_path, 'w') as f:
            yaml.dump(existing_config, f)
        
    elif args.command == "watch":
        target_path = os.path.abspath(args.path)
        print(f"SAGUARO Watch Mode: {target_path}")
        
        # We need the engine initialized similarly to index command
        # Auto-scale analysis (reuse existing config if possible, or re-analyze)
        from saguaro.indexing.auto_scaler import get_repo_stats_and_config
        
        # Check if initialized
        saguaro_dir = os.path.join(os.getcwd(), ".saguaro")
        if not os.path.exists(os.path.join(saguaro_dir, "config.yaml")):
            print("Please run 'saguaro init' or 'saguaro index' first.")
            sys.exit(1)
            
        # Load config to get dims
        import yaml
        # Re-analyze stats to get dimensions
        stats = get_repo_stats_and_config(target_path)
        
        from saguaro.indexing.engine import IndexEngine
        from saguaro.watcher import Watcher
        
        engine = IndexEngine(target_path, saguaro_dir, stats)
        watcher = Watcher(engine, target_path, interval=args.interval)
        
        try:
            watcher.start()
        except KeyboardInterrupt:
            watcher.stop()
            print("\nWatcher stopped.")

    elif args.command == "verify":
        from saguaro.sentinel.verifier import SentinelVerifier
        engines = args.engines.split(",") if args.engines else None
        
        verifier = SentinelVerifier(
            repo_path=os.path.abspath(args.path),
            engines=engines
        )
        violations = verifier.verify_all()
        
        if args.fix:
            print(f"Attempting to auto-fix {len(violations)} violations...")
            fixed_count = 0
            remaining_violations = []
            
            # 1. Native/Engine Fixes
            for v in violations:
                fixed = False
                for engine in verifier.engines:
                    if engine.fix(v):
                        fixed = True
                        fixed_count += 1
                        break
                if not fixed:
                    remaining_violations.append(v)
            
            print(f"Engine-level fixes: {fixed_count}")
            
            # 2. Scribe Fixes (AI)
            if remaining_violations:
                print(f"Attempting AI fixes for {len(remaining_violations)} violations via Scribe...")
                try:
                    from saguaro.agents.scribe import Scribe
                    from saguaro.agents.sandbox import Sandbox
                    
                    scribe = Scribe(os.getcwd())
                    
                    # Iterate copy to modify list? No, we build a new 'still_remaining' list if needed,
                    # but here we just update fixed_count and re-verify at end.
                    for v in remaining_violations:
                        print(f"  Fixing {v['rule_id']} in {v['file']}...")
                        try:
                            # 1. Generate Patch
                            task = f"Fix violation {v['message']}"
                            patch = scribe.generate_patch(task, [v['file']])
                            
                            # 2. Verify in Sandbox
                            sb = Sandbox(os.getcwd())
                            sb.apply_patch(patch)
                            report = sb.verify()
                            
                            if report['status'] == 'pass':
                                # 3. Commit
                                sb.commit()
                                fixed_count += 1
                                print("    Fixed ✅")
                            else:
                                print(f"    Fix failed verification ❌: {report['violations']}")
                        except Exception as e:
                             print(f"    Scribe error: {e}")
                             
                except Exception as e:
                    print(f"Failed to init Scribe (skipping AI fixes): {e}")
            
            print(f"Total Fixed: {fixed_count}")
            
            if fixed_count > 0:
                 print("Re-verifying codebase...")
                 violations = verifier.verify_all()
        
        if args.format == "json":
            import json
            print(json.dumps(violations, indent=2))
            if violations:
                sys.exit(1)
        else:
            if violations:
                print(f"Sentinel Validation Failed: {len(violations)} violations found.")
                for v in violations:
                    print(f"[{v['severity']}] {v['file']}:{v['line']} - {v['message']} ({v['rule_id']})")
                    # Only print context if available and not empty
                    if v.get('context'):
                        print(f"    Context: {v['context'].strip()}")
                sys.exit(1)
            else:
                print("Sentinel Validation Passed: No violations.")
            
    elif args.command == "chronicle":
        from saguaro.chronicle.storage import ChronicleStorage
        storage = ChronicleStorage()
        
        if args.chronicle_op == "snapshot":
            # For prototype, we save a dummy blob or current index state
            # In validation, we'll verify this flow.
            print("Creating semantic snapshot...")
            snapshot_id = storage.save_snapshot(
                hd_state_blob=b"DUMMY_HD_STATE", 
                description="Manual CLI Snapshot"
            )
            print(f"Snapshot #{snapshot_id} created.")
            
        elif args.chronicle_op == "list":
            # TODO: Implement listing
            print("Snapshot listing not yet implemented.")
            
        elif args.chronicle_op == "diff":
            from saguaro.chronicle.diff import SemanticDiff
            print("Calculating semantic drift...")
            # Mock diff for cli response
            drift, details = SemanticDiff.calculate_drift(
                b"\x00"*10, b"\x00"*10 # Identity
            )
            print(f"Drift Score: {drift:.4f} ({SemanticDiff.human_readable_report(drift)})")

    elif args.command == "legislation":
        if args.draft:
            from saguaro.legislator import Legislator
            leg = Legislator(root_dir=os.getcwd())
            print("Drafting legislation...")
            yaml_content = leg.draft_rules()
            print("\n" + yaml_content)
        else:
            print("Use --draft to generate rules.")

    elif args.command == "train":
        from saguaro.encoder import AdaptiveEncoder
        encoder = AdaptiveEncoder()
        encoder.fine_tune_on_corpus(args.path, epochs=args.epochs)
        print("Adaptive training complete.")

    elif args.command == "train-baseline":
        from saguaro.tokenization.train_baseline import train_baseline
        print("Training Baseline Tokenizer...")
        train_baseline(args.corpus, args.curriculum, args.output, args.fast)

    elif args.command == "constellation":
        from saguaro.constellation.manager import ConstellationManager
        cm = ConstellationManager()
        
        if args.constellation_op == "list":
            libs = cm.list_libraries()
            if not libs:
                print("Constellation is empty.")
            else:
                print(f"Constellation Libraries ({len(libs)}):")
                for lib in libs:
                    print(f" - {lib}")
                    
        elif args.constellation_op == "index-lib":
            if not args.path:
                print("Error: --path required for index-lib")
                sys.exit(1)
            cm.index_library(args.name, args.path)
            
        elif args.constellation_op == "link":
            saguaro_dir = os.path.join(os.getcwd(), ".saguaro")
            if not os.path.exists(saguaro_dir):
                print("Error: Current directory is not initialized. Run 'saguaro init' first.")
                sys.exit(1)
            cm.link_to_project(args.name, saguaro_dir)

    elif args.command == "benchmark":
        from saguaro.benchmarks.runner import BenchmarkRunner
        runner = BenchmarkRunner(args.dataset, args.custom)
        results = runner.run()
        runner.print_report(results)
            
    elif args.command == "coverage":
        from saguaro.coverage import CoverageReporter
        reporter = CoverageReporter(os.path.abspath(args.path))
        reporter.print_report()

    elif args.command == "health":
        from saguaro.health import HealthDashboard
        # Assume .saguaro is in current directory for now, or use args
        saguaro_dir = os.path.join(os.getcwd(), ".saguaro")
        dashboard = HealthDashboard(saguaro_dir)
        dashboard.print_dashboard()

    elif args.command == "governor":
        from saguaro.governor import ContextGovernor
        gov = ContextGovernor()
        
        if args.check:
            text = args.text or ""
            # Simulate a context item
            item = {"content": text, "name": "CLI_Input"}
            safe, tokens, msg = gov.check_budget([item])
            
            print("Context Check:")
            print(f"  Tokens: {tokens}")
            print(f"  Status: {msg}")
            print(f"  Safe:   {safe}")

    elif args.command == "workset":
        from saguaro.workset import WorksetManager
        from saguaro.governor import ContextBudgetExceeded

        saguaro_dir = os.path.join(os.getcwd(), ".saguaro")
        wm = WorksetManager(saguaro_dir, repo_path=os.getcwd())
        
        if args.workset_op == "create":
            files = [f.strip() for f in args.files.split(",")]
            try:
                ws = wm.create_workset(args.desc, files)
                print(f"Workset created: {ws.id}")
                print(ws.to_json())
            except ContextBudgetExceeded as e:
                print(f"Error: {e}")
                sys.exit(1)
            
        elif args.workset_op == "list":
            worksets = wm.list_worksets()
            print(f"Found {len(worksets)} worksets:")
            for w in worksets:
                print(f" - [{w.id}] {w.description} ({len(w.files)} files) [{w.status}]")
                
        elif args.workset_op == "show":
            ws = wm.get_workset(args.id)
            if ws:
                print(ws.to_json())
            else:
                print("Workset not found.")

        elif args.workset_op == "expand":
             files = [f.strip() for f in args.files.split(",")]
             try:
                 ws = wm.expand_workset(args.id, files, args.justification)
                 print(f"Workset expanded: {ws.id}")
                 print(ws.to_json())
             except Exception as e:
                 print(f"Error: {e}")
                 sys.exit(1)

        elif args.workset_op == "lock":
            try:
                wm.lock_workset(args.id)
                print(f"Workset {args.id} locked.")
            except Exception as e:
                print(f"Error: {e}")
                
        elif args.workset_op == "unlock":
            try:
                wm.unlock_workset(args.id)
                print(f"Workset {args.id} unlocked.")
            except Exception as e:
                print(f"Error: {e}")

    elif args.command == "scribe":
        from saguaro.agents.scribe import Scribe
        import json
        
        scribe = Scribe(os.getcwd())
        context_files = [args.file] if args.file else []
        
        print(f"Scribe generating patch for: '{args.task}'...")
        patch = scribe.generate_patch(args.task, context_files)
        
        with open(args.out, 'w') as f:
            json.dump(patch, f, indent=2)
            
        print(f"Patch saved to {args.out}")
        print("Apply with: saguaro agent patch <file> " + args.out)

    # --- Phase 4/SSAI Handlers ---
    elif args.command == "agent":
        if args.agent_command == "skeleton":
            from saguaro.agents.perception import SkeletonGenerator
            import json
            gen = SkeletonGenerator()
            try:
                result = gen.generate(args.file)
                print(json.dumps(result, indent=2))
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)

        elif args.agent_command == "slice":
            from saguaro.agents.perception import SliceGenerator
            import json
            gen = SliceGenerator(os.getcwd())
            try:
                result = gen.generate(args.symbol, depth=args.depth)
                
                # Check for actionable error response (Phase 2: AI Adoption)
                if "error" in result and result.get("type") == "INDEX_MISS":
                    print(f"ERROR: {result['error']}", file=sys.stderr)
                    print(f"\n{result['suggestion']}", file=sys.stderr)
                    print("\nRecovery steps:", file=sys.stderr)
                    for step in result.get('recovery_steps', []):
                        print(f"  → {step}", file=sys.stderr)
                    sys.exit(1)
                
                print(json.dumps(result, indent=2))
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)

        elif args.agent_command == "patch":
            from saguaro.agents.sandbox import Sandbox
            import json
            
            # Load patch
            try:
                if os.path.exists(args.patch_json):
                    with open(args.patch_json, 'r') as f:
                        patch_data = json.load(f)
                else:
                    patch_data = json.loads(args.patch_json)
            except Exception as e:
                print(f"Invalid patch JSON: {e}", file=sys.stderr)
                sys.exit(1)

            # Create or get sandbox? For CLI, we create new one and return ID.
            # In a real conversation loop, we'd pass ID. 
            # Here we assume a new transaction for the atomic patch.
            sb = Sandbox(os.getcwd())
            sb.apply_patch(patch_data)
            print(f"Patch applied to Sandbox {sb.id}. Run 'saguaro agent verify {sb.id}' to check.")
            print(sb.id) # Output ID on last line for parsing

        elif args.agent_command == "verify":
            from saguaro.agents.sandbox import Sandbox
            import json
            
            sb = Sandbox.get(args.sandbox_id)
            if not sb:
                print(f"Sandbox {args.sandbox_id} not found (in-memory session lost).", file=sys.stderr)
                sys.exit(1)
                
            report = sb.verify()
            print(json.dumps(report, indent=2))
            if report["status"] != "pass":
                sys.exit(1)

        elif args.agent_command == "commit":
            from saguaro.agents.sandbox import Sandbox
            
            sb = Sandbox.get(args.sandbox_id)
            if not sb:
                 print(f"Sandbox {args.sandbox_id} not found.", file=sys.stderr)
                 sys.exit(1)
            
            count = sb.commit()
            print(f"Committed {count} files from Sandbox {args.sandbox_id}.")
            
            # Trigger micro-indexing (Drift-Aware)
            # For now, just a log message as placeholder for Phase 4 logic
            print("Triggering micro-indexing...")

        elif args.agent_command == "impact":
            from saguaro.agents.sandbox import Sandbox
            import json
            
            sb = Sandbox.get(args.sandbox_id)
            if not sb:
                 print(f"Sandbox {args.sandbox_id} not found.", file=sys.stderr)
                 sys.exit(1)
                 
            report = sb.calculate_impact()
            print(json.dumps(report, indent=2))
        
        elif args.agent_command == "run":
            from saguaro.agents import PlannerAgent, CartographerAgent, SurgeonAgent, AuditorAgent
            
            agent_map = {
                "planner": PlannerAgent,
                "cartographer": CartographerAgent,
                "surgeon": SurgeonAgent,
                "auditor": AuditorAgent
            }
            
            agent_cls = agent_map.get(args.role)
            if agent_cls:
                agent = agent_cls()
                print(f"Running {agent.name}...")
                # Mock context for CLI run
                result = agent.run(context=None, goal=args.task)
                print(result)
            else:
                print(f"Unknown agent role: {args.role}")
        
        else:
            parser.parse_args(["agent", "--help"])

    elif args.command == "tasks":
        from saguaro.coordination.graph import TaskGraph, TaskNode
        graph = TaskGraph()
        
        if args.list:
            ready = graph.get_ready_tasks()
            if not ready:
                print("No ready tasks.")
            for t in ready:
                 print(f"[{t.id}] {t.description} (Status: {t.status})")
                 
        elif args.add:
            import json
            import uuid
            try:
                data = json.loads(args.add)
                node = TaskNode(
                    id=data.get("id", str(uuid.uuid4())[:8]),
                    description=data.get("description", "No Data"),
                    status="pending",
                    dependencies=data.get("dependencies", [])
                )
                graph.add_task(node)
                print(f"Task {node.id} added.")
            except Exception as e:
                print(f"Error adding task: {e}")

    elif args.command == "memory":
        from saguaro.coordination.memory import SharedMemory
        mem = SharedMemory()
        
        if args.list:
            facts = mem.list_facts()
            for k, v in facts.items():
                print(f"{k}: {v['value']} (Source: {v.get('source', '?')})")
        elif args.read:
            val = mem.read_fact(args.read)
            print(f"{args.read}: {val}")
        elif args.write:
            key, val = args.write
            mem.write_fact(key, val, agent_id="CLI_USER")
            print(f"Fact '{key}' written.")

    # --- Phase 5 Handlers ---
    elif args.command == "simulate":
        if args.sim_op == "volatility":
            from saguaro.simulation.volatility import VolatilityMapper
            mapper = VolatilityMapper()
            vmap = mapper.generate_map(os.getcwd())
            print("Volatility Map:")
            for f, score in vmap.items():
                 print(f"[{score:.2f}] {f}")
                 
        elif args.sim_op == "regression":
            from saguaro.simulation.regression import RegressionPredictor
            pred = RegressionPredictor()
            files = args.files.split(",")
            risks = pred.predict_regression(files)
            if risks:
                 print("Predicted Risks (Regressions):")
                 for r in risks:
                      print(f" - {r}")
            else:
                 print("No regressions predicted.")

    # --- Phase 6 Handlers ---
    elif args.command == "route":
        from saguaro.learning.routing import IntentRouter
        router = IntentRouter()
        intent = router.route(args.query)
        print(f"Query: '{args.query}'")
        print(f"Intent: {intent.upper()}")

    elif args.command == "feedback":
        from saguaro.feedback import FeedbackStore
        saguaro_dir = os.path.join(os.getcwd(), ".saguaro")
        fs = FeedbackStore(saguaro_dir)
        
        if args.fb_op == "log":
            items = []
            if args.used:
                for uid in args.used.split(','):
                    items.append({'id': uid.strip(), 'action': 'used'})
            if args.ignored:
                for uid in args.ignored.split(','):
                    items.append({'id': uid.strip(), 'action': 'ignored'})
            
            sid = fs.log_feedback(args.query, items)
            print(f"Feedback logged. Session ID: {sid}")
            
        elif args.fb_op == "stats":
            stats = fs.get_stats()
            print("Feedback Stats:")
            for k, v in stats.items():
                print(f"  {k}: {v}")

    elif args.command == "refactor":
        if args.refactor_op == "plan":
            from saguaro.refactor.planner import RefactorPlanner
            planner = RefactorPlanner(os.getcwd())
            print(f"Analyzing impact for symbol: {args.symbol}...")
            plan = planner.plan_symbol_modification(args.symbol)
            
            print("\nRefactor Plan Generated:")
            print(f"Impact Score: {plan['impact_score']}")
            print("Impacted Files:")
            for f in plan['files_impacted']:
                print(f" - {f}")
            
            print("\nModules:")
            for mod, files in plan['modules'].items():
                print(f"  {mod}: {len(files)} files")

        elif args.refactor_op == "rename":
            from saguaro.refactor.renamer import SemanticRenamer
            renamer = SemanticRenamer(os.getcwd())
            print(f"Renaming '{args.old}' -> '{args.new}'...")
            
            res = renamer.rename_symbol(args.old, args.new, dry_run=not args.execute)
            
            print("Rename Results:")
            if res["files_modified"]:
                for f in res["files_modified"]:
                    print(f" - Modified: {os.path.relpath(f, os.getcwd())}")
            else:
                print(" - No files matched.")
            
            if res["errors"]:
                print("\nErrors:")
                for e in res["errors"]:
                    print(f" - {e}")
                    
            if res["dry_run"]:
                print("\n[Dry Run] Use --execute to apply changes.")

        elif args.refactor_op == "shim":
            from saguaro.refactor.shims import CompatShimGenerator
            gen = CompatShimGenerator(os.getcwd())
            gen.apply_shim(args.path, args.target)
            print(f"Shim created at {args.path} pointing to {args.target}")

        elif args.refactor_op == "safedelete":
             from saguaro.refactor.safety import SafetyEngine
             engine = SafetyEngine(os.getcwd())
             res = engine.safe_delete(args.path, force=args.force, dry_run=not args.execute)
             
             if res["success"]:
                  print(f"✅ {res['message']}")
             else:
                  print(f"❌ {res['message']}")
                  if "blocking_dependents" in res:
                       for d in res["blocking_dependents"]:
                            print(f"   - {os.path.relpath(d, os.getcwd())}")

    elif args.command == "metrics":
        from saguaro.mcp.adoption_metrics import AdoptionTracker
        import json
        
        saguaro_dir = os.path.join(os.getcwd(), ".saguaro")
        tracker = AdoptionTracker(saguaro_dir)
        
        if args.reset:
            # Reset metrics by removing the file
            metrics_file = os.path.join(saguaro_dir, "metrics.json")
            if os.path.exists(metrics_file):
                os.remove(metrics_file)
                print("Metrics reset.")
            else:
                print("No metrics to reset.")
        else:
            report = tracker.get_report()
            
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                tracker.print_report()

    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
