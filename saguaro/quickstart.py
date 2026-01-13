import os
import sys
import subprocess
import textwrap

class QuickstartManager:
    def __init__(self, root_path: str):
        self.root_path = os.path.abspath(root_path)
        self.venv_bin = os.path.join(self.root_path, "venv", "bin") if os.path.exists(os.path.join(self.root_path, "venv")) else None
        
    def run_command(self, cmd_args, description):
        print(f"\n[Step] {description}...")
        try:
            subprocess.run(cmd_args, check=True, cwd=self.root_path)
            print("  ✅ Done.")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed with exit code {e.returncode}")
            sys.exit(1)
            
    def execute(self):
        print("🤠 Howdy! Welcome to SAGUARO Quickstart.")
        print(f"Targeting: {self.root_path}")
        print("---------------------------------------")
        
        # 1. Initialize
        saguaro_exec = [sys.executable, "-m", "saguaro.cli"]
        self.run_command(saguaro_exec + ["init"], "Initializing SAGUARO Repository")
        
        # 2. Index
        print("\n[Step] Indexing Codebase (this might take a moment)...")
        # We run indexing in-process or via subprocess? Subprocess is safer for memory isolation if prototype
        self.run_command(saguaro_exec + ["index", "--path", self.root_path], "Building Quantum Index")
        
        # 3. Generate Configs
        print("\n[Step] Generating Integration Configs...")
        self.generate_configs()
        
        print("\n🎉 SAGUARO is ready to serve!")
        print("---------------------------------------")
        print("Next Steps:")
        print("1. Start the MCP Server for Claude/IDEs:")
        print("   $ saguaro serve --mcp")
        print("2. Or use the CLI internally:")
        print("   $ saguaro query 'auth logic'")
        
    def generate_configs(self):
        # 1. Antigravity / Agent Config (Generic)
        agent_config = textwrap.dedent(f"""
        # .agent/saguaro_tools.md
        
        ## SAGUARO Tools
        Use these tools for high-fidelity code understanding.
        
        - **Query**: `{sys.executable} -m saguaro.cli query "<text>"`
        - **Verify**: `{sys.executable} -m saguaro.cli verify .`
        """).strip()
        
        print("\n  [Config] Generated Agent Instructions:")
        print(textwrap.indent(agent_config, "    "))
        
        # 2. Claude Desktop (MCP) - Example JSON
        mcp_config = textwrap.dedent(f"""
        {{
          "mcpServers": {{
            "saguaro": {{
              "command": "{sys.executable}",
              "args": ["-m", "saguaro.cli", "serve", "--mcp"],
              "env": {{
                "PYTHONPATH": "{self.root_path}"
              }}
            }}
          }}
        }}
        """).strip()
        
        print("\n  [Config] Claude Desktop Config (add to claude_desktop_config.json):")
        print(textwrap.indent(mcp_config, "    "))
