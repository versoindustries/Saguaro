
import sys
import json
import subprocess
import os

def main():
    """
    Bridge script for VSCode/Codex to interact with SAGUARO SSAI.
    Usage: python codex_bridge.py <command> <args>
    Output: Markdown formatted for Inline Chat.
    """
    if len(sys.argv) < 2:
        print("Usage: codex_bridge.py <command> [args...]")
        sys.exit(1)
        
    command = sys.argv[1]
    args = sys.argv[2:]
    
    # Map commands to SAGUARO CLI
    saguaro_bin = os.path.join(os.path.dirname(os.path.dirname(__file__)), "venv", "bin", "saguaro")
    
    if command == "explain":
        # Codex "explain this context" -> SAGUARO Slice ??
        # Or query
        symbol = args[0]
        cmd = [saguaro_bin, "agent", "slice", symbol]
        
    elif command == "scan":
        file_path = args[0]
        cmd = [saguaro_bin, "agent", "skeleton", file_path]
        
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
        
    try:
        # Run SAGUARO
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"**Error**:\n{res.stderr}")
            sys.exit(1)
            
        data = json.loads(res.stdout)
        
        # Format for VSCode Markdown
        if command == "scan":
            print(f"### Skeleton: {data.get('file_path')}")
            for sym in data.get('symbols', []):
                print(f"- `{sym['name']}` ({sym['type']})")
                if sym.get('docstring'):
                    print(f"  - *{sym['docstring'].splitlines()[0]}*")
                    
        elif command == "explain":
             print(f"### Context Slice: {data.get('focus_symbol')}")
             for item in data.get('content', []):
                 print(f"#### {item['name']} ({item['role']})")
                 print("```python")
                 print(item.get('code') or item.get('signature'))
                 print("```")
                 
    except Exception as e:
        print(f"Bridge Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
