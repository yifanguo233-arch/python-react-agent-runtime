import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from internal_mcp.client import MCPClientManager
from internal_mcp.registry import MCPToolRegistry
from internal_mcp.types import MCPServerConfig, MCPToolSpec


class StubClientManager:
    def __init__(self):
        self.calls = []

    def call_tool(self, server_name, tool_name, arguments):
        self.calls.append((server_name, tool_name, arguments))
        return "mcp-result"


def test_mcp_registry_wraps_tools_with_prefixed_public_names():
    client_manager = StubClientManager()
    registry = MCPToolRegistry(client_manager)
    tool_specs = [
        MCPToolSpec(
            server_name="Repo Intel",
            tool_name="find-symbol",
            description="查找符号定义",
            input_schema={"type": "object"},
        )
    ]

    tools = registry.load_tools(tool_specs)

    assert "mcp_repo_intel_find_symbol" in tools
    assert registry.is_mcp_tool("mcp_repo_intel_find_symbol") is True
    result = tools["mcp_repo_intel_find_symbol"]({"symbol_name": "ReActAgent"})
    assert result == "mcp-result"
    assert client_manager.calls == [("Repo Intel", "find-symbol", {"symbol_name": "ReActAgent"})]


def test_repo_intel_server_finds_symbol_through_mcp_wrapper():
    root_dir = Path(__file__).resolve().parent.parent
    server_path = root_dir / "internal_mcp" / "servers" / "repo_intel_server.py"
    client_manager = MCPClientManager()
    try:
        tool_specs = client_manager.load_servers(
            [
                MCPServerConfig(
                    name="repo_intel",
                    transport="stdio",
                    command=sys.executable,
                    args=[str(server_path)],
                    env={"MCP_REPO_ROOT": str(root_dir)},
                    timeout_seconds=20,
                )
            ]
        )
        registry = MCPToolRegistry(client_manager)
        tools = registry.load_tools(tool_specs)

        if "mcp_repo_intel_find_symbol" not in tools:
            errors = "; ".join(state.last_error for state in client_manager.get_server_states())
            if "WinError 5" in errors or "拒绝访问" in errors:
                print("SKIP live stdio MCP spawn blocked by Windows sandbox")
                return
            raise AssertionError(f"repo_intel MCP tools 未加载成功：{errors}")

        result = json.loads(
            tools["mcp_repo_intel_find_symbol"](
                {"symbol_name": "ReActAgent", "max_results": 5}
            )
        )

        paths = {item["path"] for item in result["matches"]}
        states = client_manager.get_server_states()
        assert "agent.py" in paths
        assert len(states) == 1
        assert states[0].connected is True
        assert states[0].tool_count >= 4
    finally:
        client_manager.shutdown()


if __name__ == "__main__":
    test_mcp_registry_wraps_tools_with_prefixed_public_names()
    test_repo_intel_server_finds_symbol_through_mcp_wrapper()
    print("OK test_mcp_registry passed")
