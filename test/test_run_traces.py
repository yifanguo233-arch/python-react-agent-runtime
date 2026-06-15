import builtins
import json
import os
import shutil
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent import ReActAgent
from hooks import HookRunner
from run_traces import TraceStore, format_run_list, format_run_report


TMP_ROOT = Path(__file__).resolve().parents[1] / "tmp" / "trace_viewer_tests"


def _make_react_agent(project_dir: Path, responses, tools):
    agent = ReActAgent.__new__(ReActAgent)
    agent.tools = tools
    agent.project_directory = str(project_dir)
    agent.hook_runner = HookRunner()
    agent.model = "trace-test-model"
    agent.dispatch_count = 0
    agent.render_system_prompt = lambda tpl, tool_map=None, extra_vars=None: "trace system prompt"

    def dispatch_model(messages):
        idx = min(agent.dispatch_count, len(responses) - 1)
        agent.dispatch_count += 1
        content = responses[idx]
        messages.append({"role": "assistant", "content": content})
        return content

    agent.dispatch_model = dispatch_model
    for name in [
        "_react_loop",
        "_compress_history",
        "parse_action",
        "_parse_action_value",
        "_parse_single_arg",
        "_run_tool_with_hooks",
        "_append_observation",
        "_extract_path_argument",
        "_validate_path",
        "_format_action_call",
        "_is_tool_failure",
        "_recover_from_tool_failure",
        "_finalize_after_round_limit",
        "_get_arg_value",
        "_display_evidence_path",
        "_read_file_budget",
        "_read_file_cache_key",
        "_maybe_block_duplicate_read_file",
        "_remember_read_file_observation",
        "_record_tool_evidence",
        "_is_truncated_read_observation",
        "_append_evidence_ledger_observation",
        "_run_log_dir",
        "_run_log_slug",
        "_start_run_log",
        "_append_run_log",
        "_finish_run_log",
        "_record_trace_event",
    ]:
        setattr(agent, name, getattr(ReActAgent, name).__get__(agent, ReActAgent))
    return agent


def test_trace_store_formats_run_report():
    root = TMP_ROOT / "store_only"
    shutil.rmtree(root, ignore_errors=True)
    store = TraceStore(root / ".runs" / "traces.sqlite3")
    run_id = store.start_run("demo task", str(root), "model-a", str(root / ".runs" / "demo.log"))
    store.add_event(run_id, "plan", "1. Read README\n2. Summarize", label="Plan")
    store.add_event(run_id, "thought", "Need evidence", label="Thought")
    store.add_event(run_id, "action", "read_file('README.md')", tool_name="read_file")
    store.add_event(run_id, "human_approval", "tool approved", tool_name="run_terminal_command", human_approval=True)
    store.add_event(run_id, "tool_result", "README content", tool_name="read_file", latency_ms=3.2)
    store.add_event(run_id, "evidence_ledger", "1. read_file: README.md", label="Evidence Ledger")
    store.finish_run(run_id, "final answer")

    listing = format_run_list(store.list_runs())
    report = format_run_report(store, run_id)

    assert run_id in listing
    assert "avg_tool_latency=3.2ms" in listing
    assert "Plan:" in report
    assert "human_approval=yes" in report
    assert "latency=3.2ms" in report
    assert "Evidence Ledger:" in report
    assert "Final Answer:" in report


def test_agent_react_loop_writes_sqlite_trace():
    project_dir = TMP_ROOT / "agent_trace"
    shutil.rmtree(project_dir, ignore_errors=True)
    project_dir.mkdir(parents=True)
    target = project_dir / "README.md"
    target.write_text("trace readme", encoding="utf-8")

    def read_file(file_path, max_chars=1000):
        return "README trace content"

    agent = _make_react_agent(
        project_dir,
        [
            f"<thought>Need file evidence</thought><action>read_file({str(target)!r}, max_chars=1000)</action>",
            "<final_answer>draft answer</final_answer>",
            "<final_answer>final answer with README.md evidence</final_answer>",
        ],
        {"read_file": read_file},
    )

    with redirect_stdout(StringIO()):
        agent._start_run_log("trace read task")
        result = agent._react_loop("trace read task", "", tool_map=agent.tools)
        agent._finish_run_log(result)

    store = TraceStore(project_dir / ".runs" / "traces.sqlite3")
    run_id = store.latest_run_id()
    events = store.get_events(run_id)
    event_types = [event["event_type"] for event in events]
    report = format_run_report(store, run_id)

    assert result == "final answer with README.md evidence"
    assert "thought" in event_types
    assert "action" in event_types
    assert "tool_result" in event_types
    assert "observation" in event_types
    assert "evidence_ledger" in event_types
    assert any(event["latency_ms"] is not None for event in events if event["event_type"] == "tool_result")
    assert "README.md" in report


def test_terminal_tool_denial_records_policy_audit_without_execution():
    project_dir = TMP_ROOT / "approval_trace"
    shutil.rmtree(project_dir, ignore_errors=True)
    project_dir.mkdir(parents=True)
    calls = []

    def run_terminal_command(command, timeout=60):
        calls.append(command)
        return "should not execute"

    agent = _make_react_agent(
        project_dir,
        ['<action>run_terminal_command("rm -rf tmp")</action>'],
        {"run_terminal_command": run_terminal_command},
    )
    original_input = builtins.input
    try:
        builtins.input = lambda prompt="": "n"
        with redirect_stdout(StringIO()):
            agent._start_run_log("deny dangerous command")
            result = agent._react_loop("deny dangerous command", "", tool_map=agent.tools)
            agent._finish_run_log(result)
    finally:
        builtins.input = original_input

    store = TraceStore(project_dir / ".runs" / "traces.sqlite3")
    events = store.get_events(store.latest_run_id())
    policy_events = [event for event in events if event["event_type"] == "tool_policy"]

    assert calls == []
    assert policy_events
    metadata = json.loads(policy_events[-1]["metadata_json"])
    assert metadata["tool"] == "run_terminal_command"
    assert metadata["risk"] == "high"
    assert metadata["approved"] is False
    assert metadata["blocked"] is True
    assert "deny_pattern" in metadata["blocked_reason"]
    assert policy_events[-1]["human_approval"] == 0
    assert not [event for event in events if event["event_type"] == "tool_result"]


def test_terminal_tool_policy_approval_prompts_once_and_executes():
    project_dir = TMP_ROOT / "approval_once"
    shutil.rmtree(project_dir, ignore_errors=True)
    project_dir.mkdir(parents=True)
    calls = []

    def run_terminal_command(command, timeout=60):
        calls.append(command)
        return "terminal ok"

    agent = _make_react_agent(
        project_dir,
        [
            '<action>run_terminal_command("echo ok")</action>',
            "<final_answer>done</final_answer>",
        ],
        {"run_terminal_command": run_terminal_command},
    )
    prompts = []
    original_input = builtins.input
    try:
        builtins.input = lambda prompt="": prompts.append(prompt) or "y"
        with redirect_stdout(StringIO()):
            agent._start_run_log("approve harmless command")
            result = agent._react_loop("approve harmless command", "", tool_map=agent.tools)
            agent._finish_run_log(result)
    finally:
        builtins.input = original_input

    store = TraceStore(project_dir / ".runs" / "traces.sqlite3")
    events = store.get_events(store.latest_run_id())
    summary = store.list_runs(limit=1)[0]
    policy_events = [event for event in events if event["event_type"] == "tool_policy"]
    metadata = json.loads(policy_events[-1]["metadata_json"])

    assert result == "done"
    assert calls == ["echo ok"]
    assert len(prompts) == 1
    assert metadata["tool"] == "run_terminal_command"
    assert metadata["risk"] == "high"
    assert metadata["approved"] is True
    assert metadata["blocked"] is False
    assert summary.approval_count == 1
    assert [event for event in events if event["event_type"] == "tool_result"]


if __name__ == "__main__":
    test_trace_store_formats_run_report()
    test_agent_react_loop_writes_sqlite_trace()
    test_terminal_tool_denial_records_policy_audit_without_execution()
    test_terminal_tool_policy_approval_prompts_once_and_executes()
    print("OK test_run_traces passed")
