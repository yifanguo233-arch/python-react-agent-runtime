import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from internal_mcp.config import load_mcp_server_configs


def test_load_mcp_server_configs_reads_enabled_stdio_servers(tmp_path: Path):
    config_dir = tmp_path / ".mcp"
    config_dir.mkdir()
    config_path = config_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "name": "filesystem",
                        "transport": "stdio",
                        "command": "python",
                        "args": ["server.py"],
                        "env": {"DEMO": "1", "MCP_REPO_ROOT": "../"},
                        "enabled": True,
                        "timeout_seconds": 15,
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    configs = load_mcp_server_configs(str(tmp_path))

    assert len(configs) == 1
    assert configs[0].name == "filesystem"
    assert configs[0].command == sys.executable
    assert configs[0].args == [str((tmp_path / "server.py").resolve())]
    assert configs[0].env == {"DEMO": "1", "MCP_REPO_ROOT": str(tmp_path.resolve())}
    assert configs[0].timeout_seconds == 15


if __name__ == "__main__":
    tmp_dir = Path(__file__).resolve().parent / "_tmp_mcp_config"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()
    try:
        test_load_mcp_server_configs_reads_enabled_stdio_servers(tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print("OK test_mcp_config passed")
