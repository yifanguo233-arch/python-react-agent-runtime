import os
import sys
import types
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

tools_mock = types.ModuleType("tools")
for name in [
    "read_file",
    "write_to_file",
    "run_terminal_command",
    "list_directory",
    "search_in_files",
    "web_search",
    "query_knowledge_base",
]:
    setattr(tools_mock, name, MagicMock())
sys.modules["tools"] = tools_mock

from agent import ReActAgent
from hooks import HookRunner


def _make_react_agent(responses, tools):
    agent = ReActAgent.__new__(ReActAgent)
    agent.tools = tools
    agent.project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    agent.hook_runner = HookRunner()
    agent.last_messages = None
    agent.dispatch_count = 0

    agent.render_system_prompt = lambda tpl, tool_map=None, extra_vars=None: "react system prompt"

    def dispatch_model(messages):
        idx = min(agent.dispatch_count, len(responses) - 1)
        agent.dispatch_count += 1
        agent.last_messages = messages
        content = responses[idx]
        messages.append({"role": "assistant", "content": content})
        return content

    agent.dispatch_model = dispatch_model
    agent._react_loop = ReActAgent._react_loop.__get__(agent, ReActAgent)
    agent._compress_history = ReActAgent._compress_history.__get__(agent, ReActAgent)
    agent.parse_action = ReActAgent.parse_action.__get__(agent, ReActAgent)
    agent._parse_action_value = ReActAgent._parse_action_value.__get__(agent, ReActAgent)
    agent._parse_single_arg = ReActAgent._parse_single_arg.__get__(agent, ReActAgent)
    agent._run_tool_with_hooks = ReActAgent._run_tool_with_hooks.__get__(agent, ReActAgent)
    agent._append_observation = ReActAgent._append_observation.__get__(agent, ReActAgent)
    agent._extract_path_argument = ReActAgent._extract_path_argument.__get__(agent, ReActAgent)
    agent._validate_path = ReActAgent._validate_path.__get__(agent, ReActAgent)
    agent._format_action_call = ReActAgent._format_action_call.__get__(agent, ReActAgent)
    agent._is_tool_failure = ReActAgent._is_tool_failure.__get__(agent, ReActAgent)
    agent._recover_from_tool_failure = ReActAgent._recover_from_tool_failure.__get__(agent, ReActAgent)
    agent._finalize_after_round_limit = ReActAgent._finalize_after_round_limit.__get__(agent, ReActAgent)
    agent._get_arg_value = ReActAgent._get_arg_value.__get__(agent, ReActAgent)
    agent._display_evidence_path = ReActAgent._display_evidence_path.__get__(agent, ReActAgent)
    agent._read_file_budget = ReActAgent._read_file_budget.__get__(agent, ReActAgent)
    agent._read_file_cache_key = ReActAgent._read_file_cache_key.__get__(agent, ReActAgent)
    agent._maybe_block_duplicate_read_file = ReActAgent._maybe_block_duplicate_read_file.__get__(agent, ReActAgent)
    agent._remember_read_file_observation = ReActAgent._remember_read_file_observation.__get__(agent, ReActAgent)
    agent._record_tool_evidence = ReActAgent._record_tool_evidence.__get__(agent, ReActAgent)
    agent._is_truncated_read_observation = ReActAgent._is_truncated_read_observation.__get__(agent, ReActAgent)
    agent._append_evidence_ledger_observation = ReActAgent._append_evidence_ledger_observation.__get__(agent, ReActAgent)
    agent._run_log_dir = ReActAgent._run_log_dir.__get__(agent, ReActAgent)
    agent._run_log_slug = ReActAgent._run_log_slug.__get__(agent, ReActAgent)
    agent._start_run_log = ReActAgent._start_run_log.__get__(agent, ReActAgent)
    agent._append_run_log = ReActAgent._append_run_log.__get__(agent, ReActAgent)
    agent._finish_run_log = ReActAgent._finish_run_log.__get__(agent, ReActAgent)
    return agent


def _observations(agent):
    return [
        message["content"]
        for message in agent.last_messages
        if message.get("role") == "user" and "<observation>" in message.get("content", "")
    ]


def _run_react_loop_quietly(agent, user_input, context="", tool_map=None, max_rounds=15):
    output = StringIO()
    with redirect_stdout(output):
        result = agent._react_loop(user_input, context, tool_map=tool_map or agent.tools, max_rounds=max_rounds)
    agent.last_stdout = output.getvalue()
    return result


def test_react_loop_action_observation_final_answer():
    calls = []

    def lookup(name):
        calls.append(name)
        return f"lookup result for {name}"

    agent = _make_react_agent(
        [
            '<action>lookup("alpha")</action>',
            "<final_answer>alpha is handled</final_answer>",
        ],
        {"lookup": lookup},
    )

    result = _run_react_loop_quietly(agent, "handle alpha")

    assert result == "alpha is handled"
    assert calls == ["alpha"]
    assert any("lookup result for alpha" in item for item in _observations(agent))
    assert agent.dispatch_count == 2


def test_react_loop_reports_unknown_tool_as_observation():
    agent = _make_react_agent(
        [
            '<action>missing_tool("x")</action>',
            "<final_answer>reported missing tool</final_answer>",
        ],
        {"known_tool": lambda value: value},
    )

    result = _run_react_loop_quietly(agent, "try missing tool")
    observations = _observations(agent)

    assert result == "reported missing tool"
    assert any("missing_tool" in item and "known_tool" in item for item in observations)


def test_react_loop_reports_action_parse_error_as_observation():
    agent = _make_react_agent(
        [
            "<action>lookup(</action>",
            "<final_answer>parse error was reported</final_answer>",
        ],
        {"lookup": lambda value: value},
    )

    result = _run_react_loop_quietly(agent, "bad action")
    observations = _observations(agent)

    assert result == "parse error was reported"
    assert any("Action" in item and "syntax" in item for item in observations)


def test_react_loop_rejects_multiple_actions_in_one_turn():
    calls = []

    def first():
        calls.append("first")
        return "first result"

    def second():
        calls.append("second")
        return "second result"

    agent = _make_react_agent(
        [
            "<action>first()</action>\n<action>second()</action>",
            "<final_answer>multiple action format was reported</final_answer>",
        ],
        {"first": first, "second": second},
    )

    result = _run_react_loop_quietly(agent, "multiple actions")
    observations = _observations(agent)

    assert result == "multiple action format was reported"
    assert calls == []
    assert any("每轮只能输出一个" in item for item in observations)


def test_react_loop_converts_tool_exception_to_observation():
    def explode():
        raise RuntimeError("boom")

    agent = _make_react_agent(
        [
            "<action>explode()</action>",
            "<final_answer>exception was observed</final_answer>",
        ],
        {"explode": explode},
    )

    result = _run_react_loop_quietly(agent, "explode once")
    observations = _observations(agent)

    assert result == "exception was observed"
    assert any("boom" in item for item in observations)


def test_react_loop_recovers_after_repeated_tool_failure():
    def unstable():
        raise RuntimeError("still broken")

    agent = _make_react_agent(
        [
            "<action>unstable()</action>",
            "<action>unstable()</action>",
            "<final_answer>stopped retrying unstable</final_answer>",
        ],
        {"unstable": unstable},
    )

    result = _run_react_loop_quietly(agent, "repeat failure")
    observations = _observations(agent)

    assert result == "stopped retrying unstable"
    assert sum("still broken" in item for item in observations) >= 2
    assert any("unstable" in item and "2" in item for item in observations)


def test_react_loop_forces_final_answer_after_round_budget():
    calls = []

    def lookup(name):
        calls.append(name)
        return f"lookup result for {name}"

    agent = _make_react_agent(
        [
            '<action>lookup("alpha")</action>',
            "<final_answer>summarized from available observations</final_answer>",
        ],
        {"lookup": lookup},
    )

    result = _run_react_loop_quietly(agent, "handle alpha", max_rounds=1)
    observations = _observations(agent)

    assert result == "summarized from available observations"
    assert calls == ["alpha"]
    assert agent.dispatch_count == 2
    assert any("ReAct" in item and "<final_answer>" in item for item in observations)


def test_react_loop_blocks_duplicate_read_file_with_same_budget():
    calls = []
    read_target = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "README.md")

    def read_file(file_path, max_chars=1000):
        calls.append((file_path, max_chars))
        return "README body"

    agent = _make_react_agent(
        [
            f'<action>read_file({read_target!r}, max_chars=1000)</action>',
            f'<action>read_file({read_target!r}, max_chars=1000)</action>',
            "<final_answer>duplicate read was blocked</final_answer>",
            "<final_answer>final after evidence ledger</final_answer>",
        ],
        {"read_file": read_file},
    )

    result = _run_react_loop_quietly(agent, "read twice")
    observations = _observations(agent)

    assert result == "final after evidence ledger"
    assert calls == [(read_target, 1000)]
    assert any("已在本任务中读取过" in item and "已阻止重复读取" in item for item in observations)
    assert any("已观察证据清单" in item and "read_file: README.md" in item for item in observations)


def test_react_loop_injects_evidence_ledger_before_accepting_final_answer():
    read_target = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "agent.py")

    def read_file(file_path, max_chars=1000):
        return "partial agent source\n\n... [文件过大，仅显示前 1KB，完整文件约 47KB]"

    agent = _make_react_agent(
        [
            f'<action>read_file({read_target!r}, max_chars=1000)</action>',
            "<final_answer>draft with risky line numbers</final_answer>",
            "<final_answer>revised without unsupported line numbers</final_answer>",
        ],
        {"read_file": read_file},
    )

    result = _run_react_loop_quietly(agent, "summarize truncated file")
    observations = _observations(agent)

    assert result == "revised without unsupported line numbers"
    assert agent.dispatch_count == 3
    assert any("已观察证据清单" in item and "agent.py" in item for item in observations)
    assert any("状态=已截断" in item and "不要引用未观察到的具体行号" in item for item in observations)


def test_run_log_records_action_observation_and_final_answer():
    def lookup(name):
        return f"lookup result for {name}"

    agent = _make_react_agent(
        [
            '<action>lookup("alpha")</action>',
            "<final_answer>alpha is handled</final_answer>",
        ],
        {"lookup": lookup},
    )

    project_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "tmp", "run_log_project")
    os.makedirs(project_dir, exist_ok=True)
    agent.project_directory = project_dir
    agent._start_run_log("demo task")
    result = _run_react_loop_quietly(agent, "demo task")
    log_path = agent.current_run_log_path
    returned = agent._finish_run_log(result)

    with open(log_path, "r", encoding="utf-8") as f:
        log_content = f.read()

    assert returned == "alpha is handled"
    assert "## User Task" in log_content
    assert "demo task" in log_content
    assert "## Action" in log_content
    assert "lookup('alpha')" in log_content
    assert "## Observation" in log_content
    assert "lookup result for alpha" in log_content
    assert "## Final Answer" in log_content
    assert "alpha is handled" in log_content


def test_tool_failure_detection_does_not_misclassify_source_text():
    agent = _make_react_agent([], {})
    source_text = (
        "def example():\n"
        "    observation = \"路径 'x' 不在项目目录内，只允许操作 repo 下的文件\"\n"
        "    return observation\n"
    )

    assert agent._is_tool_failure(source_text) is False
    assert agent._is_tool_failure("路径 'C:/outside' 不在项目目录内，只允许操作 C:/repo 下的文件") is True


def test_explicit_skill_react_loop_uses_larger_round_budget():
    agent = ReActAgent.__new__(ReActAgent)
    captured = {}
    agent.session_history = []
    agent.session_started = True
    agent.project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    agent.current_memory_section = "- none"
    agent._start_run_log = lambda user_input: None
    agent._finish_run_log = lambda answer: answer
    agent._load_memory_section = lambda user_input: "- none"
    agent._build_session_context = lambda: ""
    agent._handle_special_command = lambda user_input: None
    agent._build_selected_skill_context = lambda selected_skill: "\nskill body"
    agent._build_prompt_tool_map = lambda user_input, selected_skill=None, hinted_skill=None: {"read_file": lambda path: path}
    def fake_react_loop(user_input, context, tool_map=None, max_rounds=15):
        captured["max_rounds"] = max_rounds
        return "done"

    agent._react_loop = fake_react_loop

    class FakeRegistry:
        def get_manifest(self, name):
            if name == "analyze-code":
                return {"name": "analyze-code", "description": "demo"}
            return None

        def list_manifests(self):
            return [{"name": "analyze-code"}]

    agent.skill_registry = FakeRegistry()
    agent.run = ReActAgent.run.__get__(agent, ReActAgent)

    with redirect_stdout(StringIO()):
        result = agent.run("/analyze-code analyze this repo")

    assert result == "done"
    assert captured["max_rounds"] == 30


if __name__ == "__main__":
    test_react_loop_action_observation_final_answer()
    print("OK normal action -> observation -> final_answer")
    test_react_loop_reports_unknown_tool_as_observation()
    print("OK unknown tool is reported as observation")
    test_react_loop_reports_action_parse_error_as_observation()
    print("OK malformed action is reported as observation")
    test_react_loop_rejects_multiple_actions_in_one_turn()
    print("OK multiple actions in one turn are rejected")
    test_react_loop_converts_tool_exception_to_observation()
    print("OK tool exception is converted to observation")
    test_react_loop_recovers_after_repeated_tool_failure()
    print("OK repeated tool failure triggers recovery")
    test_react_loop_forces_final_answer_after_round_budget()
    print("OK round budget exhaustion forces a final answer")
    test_react_loop_blocks_duplicate_read_file_with_same_budget()
    print("OK duplicate read_file calls with the same budget are blocked")
    test_react_loop_injects_evidence_ledger_before_accepting_final_answer()
    print("OK evidence ledger is injected before final answer")
    test_run_log_records_action_observation_and_final_answer()
    print("OK run log records action, observation, and final answer")
    test_tool_failure_detection_does_not_misclassify_source_text()
    print("OK source text is not misclassified as tool failure")
    test_explicit_skill_react_loop_uses_larger_round_budget()
    print("OK explicit skill tasks use larger ReAct round budget")
    print("test_react_loop passed")
