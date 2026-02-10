import sys
import os
import pytest

# Ensure we can import saguaro
sys.path.append(os.getcwd())

from saguaro.sentinel.engines.external import ExternalUtil, EngineError

def test_run_subproc_failfast():
    # 1. Test success
    output = ExternalUtil.run_subproc(["echo", "hello"], ".", check=True)
    assert "hello" in output

    # 2. Test intentional failure (exit 1)
    # Use a command that definitely returns non-zero, like ls on a non-existent path
    with pytest.raises(EngineError) as excinfo:
        ExternalUtil.run_subproc(["ls", "/non/existent/path/for/saguaro/testing"], ".", check=True)
    assert "exit code" in str(excinfo.value)

    # 3. Test non-existent command
    with pytest.raises(EngineError) as excinfo:
        ExternalUtil.run_subproc(["saguaro-hypothetical-missing-cmd"], ".", check=True)
    assert "Subprocess execution failed" in str(excinfo.value)
