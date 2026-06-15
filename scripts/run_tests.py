"""Run the stable local test suite for this project.

Use this script instead of calling ``uv run`` directly when you already have
the project virtual environment activated. It also forces UTF-8 output so the
Chinese and emoji test messages do not fail on Windows consoles.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

STABLE_TESTS = [
    "test/test_memory.py",
    "test/test_skills.py",
    "test/test_hooks.py",
    "test/test_tool_policy.py",
    "test/test_tools.py",
    "test/test_team.py",
    "test/test_mcp_config.py",
    "test/test_mcp_registry.py",
    "test/test_repo_intel_server.py",
    "test/test_agent_mcp.py",
    "test/test_agent_teams.py",
    "test/test_react_loop.py",
    "test/test_run_traces.py",
    "test/test_agent_evals.py",
    "test/test_subagent.py",
]


def main() -> int:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    print(f"Python: {sys.executable}")
    print(f"Root:   {ROOT}")
    print()

    for test_path in STABLE_TESTS:
        path = ROOT / test_path
        if not path.exists():
            print(f"Missing test file: {test_path}")
            return 1

        print(f"==> {test_path}", flush=True)
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT,
            env=env,
        )
        if result.returncode != 0:
            print()
            print(f"FAILED: {test_path} exited with code {result.returncode}")
            return result.returncode
        print()

    print("All stable local tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
