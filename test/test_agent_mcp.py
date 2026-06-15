import os
import sys
import types
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

tools_mock = types.ModuleType("tools")
for name in ["read_file", "write_to_file", "run_terminal_command", "list_directory", "search_in_files", "web_search", "query_knowledge_base"]:
    setattr(tools_mock, name, MagicMock())
sys.modules["tools"] = tools_mock

from agent import ReActAgent
from internal_mcp.types import MCPServerState, MCPToolSpec


class StubTeamManager:
    def __init__(self):
        self.config = {"members": [{"name": "alice", "role": "researcher", "status": "idle"}]}
        self.shutdown_requests = {}
        self.plan_requests = {}

    def send_message(self, sender, to, content, msg_type="message"):
        return f"{sender}->{to}:{content}:{msg_type}"

    def respond_shutdown(self, sender, request_id, approve, reason=""):
        return f"shutdown:{sender}:{request_id}:{approve}:{reason}"

    def submit_plan(self, sender, plan):
        return f"plan:{sender}:{plan}"


class StubMCPRegistry:
    def __init__(self):
        self.specs = {
            "mcp_repo_intel_find_symbol": MCPToolSpec(
                server_name="repo_intel",
                tool_name="find_symbol",
                description="查找仓库符号",
            )
        }

    def is_mcp_tool(self, tool_name: str) -> bool:
        return tool_name in self.specs

    def get_tool_specs(self):
        return dict(self.specs)


class StubMCPClientManager:
    def get_server_states(self):
        return [MCPServerState(name="repo_intel", transport="stdio", connected=True, tool_count=4)]


def test_get_status_includes_mcp_section_and_teammates_exclude_mcp_tools():
    agent = ReActAgent.__new__(ReActAgent)
    agent.model = "qwen2.5:3b"
    agent.team_manager = StubTeamManager()
    agent.mcp_registry = StubMCPRegistry()
    agent.mcp_client_manager = StubMCPClientManager()
    agent.mcp_server_configs = [object()]
    agent.tools = {
        "read_file": lambda path: path,
        "mcp_repo_intel_find_symbol": lambda arguments=None: arguments,
    }
    agent.ALWAYS_HIDDEN_PROMPT_TOOLS = set()
    agent.TEAM_PROMPT_TOOLS = {"task"}
    agent.GENERAL_QUESTION_TOOLS = {"web_search", "query_knowledge_base"}
    agent._make_bound_tool = ReActAgent._make_bound_tool.__get__(agent, ReActAgent)
    agent._is_mcp_tool = ReActAgent._is_mcp_tool.__get__(agent, ReActAgent)
    agent._get_mcp_status_lines = ReActAgent._get_mcp_status_lines.__get__(agent, ReActAgent)
    agent._is_general_question = ReActAgent._is_general_question.__get__(agent, ReActAgent)
    agent._is_multi_agent_request = ReActAgent._is_multi_agent_request.__get__(agent, ReActAgent)
    agent._is_memory_save_request = ReActAgent._is_memory_save_request.__get__(agent, ReActAgent)
    agent._build_prompt_tool_map = ReActAgent._build_prompt_tool_map.__get__(agent, ReActAgent)
    agent.get_status = ReActAgent.get_status.__get__(agent, ReActAgent)
    agent.build_teammate_tools = ReActAgent.build_teammate_tools.__get__(agent, ReActAgent)

    status = agent.get_status()
    lead_prompt_tools = agent._build_prompt_tool_map("请查找 ReActAgent 符号定义")
    teammate_tools = agent.build_teammate_tools("alice")

    assert "# MCP Status" in status
    assert "- 已配置 server：1" in status
    assert "- 已加载 MCP tools：1" in status
    assert "- server repo_intel: connected, tools=4" in status
    assert "mcp_repo_intel_find_symbol" in lead_prompt_tools
    assert "mcp_repo_intel_find_symbol" not in teammate_tools
    assert "read_file" in teammate_tools


def test_parse_action_supports_named_arguments_and_literal_dicts():
    agent = ReActAgent.__new__(ReActAgent)
    agent.parse_action = ReActAgent.parse_action.__get__(agent, ReActAgent)
    agent._parse_action_value = ReActAgent._parse_action_value.__get__(agent, ReActAgent)
    agent._parse_single_arg = ReActAgent._parse_single_arg.__get__(agent, ReActAgent)

    tool_name, args, kwargs = agent.parse_action('search_in_files(keyword="TODO", directory="E:/Duan-Code")')
    assert tool_name == "search_in_files"
    assert args == []
    assert kwargs == {"keyword": "TODO", "directory": "E:/Duan-Code"}

    tool_name, args, kwargs = agent.parse_action('mcp_repo_intel_find_symbol({"symbol_name": "ReActAgent"})')
    assert tool_name == "mcp_repo_intel_find_symbol"
    assert args == [{"symbol_name": "ReActAgent"}]
    assert kwargs == {}


def test_general_question_prompt_tool_map_is_restricted():
    agent = ReActAgent.__new__(ReActAgent)
    agent.tools = {
        "read_file": lambda path: path,
        "web_search": lambda query, max_results=3: query,
        "query_knowledge_base": lambda question, top_k=3: question,
        "save_memory": lambda name, description, mem_type, content: content,
        "task": lambda prompt: prompt,
    }
    agent.ALWAYS_HIDDEN_PROMPT_TOOLS = {"save_memory"}
    agent.TEAM_PROMPT_TOOLS = {"task"}
    agent.GENERAL_QUESTION_TOOLS = {"web_search", "query_knowledge_base"}
    agent._is_general_question = ReActAgent._is_general_question.__get__(agent, ReActAgent)
    agent._should_skip_planning = ReActAgent._should_skip_planning.__get__(agent, ReActAgent)
    agent._is_multi_agent_request = ReActAgent._is_multi_agent_request.__get__(agent, ReActAgent)
    agent._is_memory_save_request = ReActAgent._is_memory_save_request.__get__(agent, ReActAgent)
    agent._build_prompt_tool_map = ReActAgent._build_prompt_tool_map.__get__(agent, ReActAgent)

    prompt_tools = agent._build_prompt_tool_map("我想转agent开发，我需要补充什么知识？")

    assert set(prompt_tools) == {"web_search", "query_knowledge_base"}
    assert agent._should_skip_planning("我想转agent开发，我需要补充什么知识？") is True


def test_memory_tool_is_only_exposed_for_explicit_save_requests():
    agent = ReActAgent.__new__(ReActAgent)
    agent.tools = {
        "read_file": lambda path: path,
        "save_memory": lambda name, description, mem_type, content: content,
        "task": lambda prompt: prompt,
    }
    agent.ALWAYS_HIDDEN_PROMPT_TOOLS = {"save_memory"}
    agent.TEAM_PROMPT_TOOLS = {"task"}
    agent.GENERAL_QUESTION_TOOLS = {"web_search", "query_knowledge_base"}
    agent._is_general_question = ReActAgent._is_general_question.__get__(agent, ReActAgent)
    agent._is_multi_agent_request = ReActAgent._is_multi_agent_request.__get__(agent, ReActAgent)
    agent._is_memory_save_request = ReActAgent._is_memory_save_request.__get__(agent, ReActAgent)
    agent._should_ignore_memory = ReActAgent._should_ignore_memory.__get__(agent, ReActAgent)
    agent._memory_ignored_section = ReActAgent._memory_ignored_section.__get__(agent, ReActAgent)
    agent._build_prompt_tool_map = ReActAgent._build_prompt_tool_map.__get__(agent, ReActAgent)

    normal_tools = agent._build_prompt_tool_map("读取项目里的 README")
    memory_tools = agent._build_prompt_tool_map("请记住：我以后都喜欢简洁回答")

    assert "save_memory" not in normal_tools
    assert "save_memory" in memory_tools
    assert agent._should_ignore_memory("这次请 ignore memory 再分析")
    assert "未注入 .memory/" in agent._memory_ignored_section()


def test_minimax_model_routes_to_openai_compatible_client():
    agent = ReActAgent.__new__(ReActAgent)
    agent.model = "minimax/MiniMax-M2.7"
    agent._is_minimax_model = ReActAgent._is_minimax_model.__get__(agent, ReActAgent)
    agent._minimax_model_name = ReActAgent._minimax_model_name.__get__(agent, ReActAgent)
    agent.dispatch_model = ReActAgent.dispatch_model.__get__(agent, ReActAgent)
    agent.call_minimax_model = ReActAgent.call_minimax_model.__get__(agent, ReActAgent)

    response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content="<final_answer>ok</final_answer>")
            )
        ]
    )
    completions = MagicMock()
    completions.create.return_value = response
    agent.minimax_client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=completions)
    )
    messages = [{"role": "user", "content": "hi"}]

    result = agent.dispatch_model(messages)

    assert result == "<final_answer>ok</final_answer>"
    assert messages[-1] == {"role": "assistant", "content": "<final_answer>ok</final_answer>"}
    completions.create.assert_called_once()
    assert completions.create.call_args.kwargs["model"] == "MiniMax-M2.7"


if __name__ == "__main__":
    test_get_status_includes_mcp_section_and_teammates_exclude_mcp_tools()
    test_parse_action_supports_named_arguments_and_literal_dicts()
    test_general_question_prompt_tool_map_is_restricted()
    test_memory_tool_is_only_exposed_for_explicit_save_requests()
    test_minimax_model_routes_to_openai_compatible_client()
    print("OK test_agent_mcp passed")
