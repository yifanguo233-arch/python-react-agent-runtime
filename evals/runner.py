from __future__ import annotations

import argparse
import builtins
import contextlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import ReActAgent
from hooks import HookRunner


CASES_PATH = Path(__file__).with_name("cases.json")
MCP_TOOL_NAME = "mcp_repo_intel_find_symbol"


@dataclass
class ActionAttempt:
    tool_name: str
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    raw: str = ""


@dataclass
class ToolCall:
    tool_name: str
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class EvalTrace:
    case_id: str
    final_answer: str
    model_turns: int
    actions: list[ActionAttempt]
    tool_calls: list[ToolCall]
    observations: list[str]
    confirm_prompts: list[str]
    stdout: str = ""

    @property
    def tool_error_count(self) -> int:
        errors = sum(1 for call in self.tool_calls if call.error)
        known_tools = {call.tool_name for call in self.tool_calls}
        for action in self.actions:
            if action.tool_name.startswith("missing_") and action.tool_name not in known_tools:
                errors += 1
        return errors


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    reasons: list[str]
    trace: EvalTrace


@dataclass
class EvalReport:
    total: int
    passed: int
    avg_steps: float
    tool_error_rate: float
    repeated_read_blocked: int
    results: list[CaseResult]

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    def to_text(self, verbose: bool = False) -> str:
        lines = [
            f"total={self.total}",
            f"pass={self.passed}",
            f"pass_rate={_format_percent(self.pass_rate)}",
            f"avg_steps={self.avg_steps:.1f}",
            f"tool_error_rate={_format_percent(self.tool_error_rate)}",
            f"repeated_read_blocked={self.repeated_read_blocked}",
        ]
        failures = [result for result in self.results if not result.passed]
        if verbose or failures:
            lines.append("")
            lines.append("cases:")
            for result in self.results:
                status = "PASS" if result.passed else "FAIL"
                detail = "; ".join(result.reasons) if result.reasons else "ok"
                lines.append(f"- {status} {result.case_id}: {detail}")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            {
                "total": self.total,
                "pass": self.passed,
                "pass_rate": round(self.pass_rate, 4),
                "avg_steps": round(self.avg_steps, 2),
                "tool_error_rate": round(self.tool_error_rate, 4),
                "repeated_read_blocked": self.repeated_read_blocked,
                "cases": [
                    {
                        "id": result.case_id,
                        "passed": result.passed,
                        "reasons": result.reasons,
                        "steps": result.trace.model_turns,
                        "tool_errors": result.trace.tool_error_count,
                    }
                    for result in self.results
                ],
            },
            ensure_ascii=False,
            indent=2,
        )


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if not isinstance(cases, list):
        raise ValueError("Eval cases must be a JSON list.")
    return cases


def run_eval_suite(cases: list[dict[str, Any]] | None = None) -> EvalReport:
    loaded_cases = cases if cases is not None else load_cases()
    results = [run_case(case) for case in loaded_cases]
    action_count = sum(len(result.trace.actions) for result in results)
    tool_error_count = sum(result.trace.tool_error_count for result in results)
    avg_steps = sum(result.trace.model_turns for result in results) / len(results) if results else 0.0
    repeated_read_blocked = sum(_repeated_read_blocked_count(result.trace) for result in results)
    return EvalReport(
        total=len(results),
        passed=sum(1 for result in results if result.passed),
        avg_steps=avg_steps,
        tool_error_rate=(tool_error_count / action_count) if action_count else 0.0,
        repeated_read_blocked=repeated_read_blocked,
        results=results,
    )


def run_case(case: dict[str, Any]) -> CaseResult:
    responses = case.get("responses") or _default_responses(case)
    trace_state: dict[str, Any] = {
        "tool_calls": [],
        "confirm_prompts": [],
        "model_turns": 0,
        "last_messages": [],
    }
    agent = _make_agent(case, responses, trace_state)
    output = StringIO()
    original_input = builtins.input
    confirm_responses = list(case.get("confirm_responses") or _default_confirm_responses(case))

    def fake_input(prompt: str = "") -> str:
        trace_state["confirm_prompts"].append(str(prompt))
        if confirm_responses:
            return str(confirm_responses.pop(0))
        return "y"

    try:
        builtins.input = fake_input
        with contextlib.redirect_stdout(output):
            final_answer = agent._react_loop(
                case.get("prompt", case["id"]),
                "",
                tool_map=agent.tools,
                max_rounds=int(case.get("max_rounds", 12)),
            )
    finally:
        builtins.input = original_input

    messages = trace_state["last_messages"]
    trace = EvalTrace(
        case_id=case["id"],
        final_answer=final_answer,
        model_turns=trace_state["model_turns"],
        actions=_extract_actions(agent, messages),
        tool_calls=trace_state["tool_calls"],
        observations=_extract_observations(messages),
        confirm_prompts=trace_state["confirm_prompts"],
        stdout=output.getvalue(),
    )
    reasons = grade_case(case, trace)
    return CaseResult(case_id=case["id"], passed=not reasons, reasons=reasons, trace=trace)


def grade_case(case: dict[str, Any], trace: EvalTrace) -> list[str]:
    kind = case["kind"]
    graders: dict[str, Callable[[dict[str, Any], EvalTrace], list[str]]] = {
        "read_file_required": _grade_read_file_required,
        "no_duplicate_read": _grade_no_duplicate_read,
        "load_skill_first": _grade_load_skill_first,
        "dangerous_command_confirmation": _grade_dangerous_command_confirmation,
        "mcp_find_symbol_summary": _grade_mcp_find_symbol_summary,
        "tool_error_recovery": _grade_tool_error_recovery,
    }
    if kind not in graders:
        return [f"unknown eval kind: {kind}"]
    return graders[kind](case, trace)


def _make_agent(case: dict[str, Any], responses: list[str], trace_state: dict[str, Any]) -> ReActAgent:
    agent = ReActAgent.__new__(ReActAgent)
    agent.model = "eval-scripted"
    agent.project_directory = str(ROOT)
    agent.hook_runner = HookRunner()
    agent.current_run_log_path = None
    agent.render_system_prompt = lambda tpl, tool_map=None, extra_vars=None: "eval system prompt"

    response_queue = list(responses)

    def dispatch_model(messages: list[dict[str, str]]) -> str:
        trace_state["model_turns"] += 1
        trace_state["last_messages"] = messages
        content = response_queue.pop(0) if response_queue else "<final_answer>no scripted response left</final_answer>"
        messages.append({"role": "assistant", "content": content})
        return content

    agent.dispatch_model = dispatch_model
    agent.tools = _build_tool_map(trace_state)

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
        "_append_run_log",
    ]:
        setattr(agent, name, getattr(ReActAgent, name).__get__(agent, ReActAgent))

    return agent


def _build_tool_map(trace_state: dict[str, Any]) -> dict[str, Callable]:
    def record(name: str, args: tuple[Any, ...], kwargs: dict[str, Any], error: str = "") -> None:
        trace_state["tool_calls"].append(ToolCall(name, list(args), dict(kwargs), error=error))

    def read_file(file_path: str, max_chars: int = 1200) -> str:
        record("read_file", (file_path,), {"max_chars": max_chars})
        rel = _relative_target(file_path)
        return f"Contents for {rel}. ReActAgent tools skills MCP evidence."

    def load_skill(name: str) -> str:
        record("load_skill", (name,), {})
        return f"<skill name=\"{name}\">Use the skill instructions before tool work.</skill>"

    def run_terminal_command(command: str, timeout: int = 60) -> str:
        record("run_terminal_command", (command,), {"timeout": timeout})
        return f"command would run: {command}"

    def mcp_repo_intel_find_symbol(arguments: dict[str, Any] | None = None) -> str:
        payload = dict(arguments or {})
        symbol = str(payload.get("symbol_name", "UnknownSymbol"))
        record(MCP_TOOL_NAME, (payload,), {})
        return json.dumps(
            {
                "symbol_name": symbol,
                "matches": [
                    {
                        "path": _symbol_path(symbol),
                        "name": symbol,
                        "qualname": symbol,
                        "kind": "class" if symbol[:1].isupper() else "function",
                        "line": 42,
                    }
                ],
                "truncated": False,
            },
            ensure_ascii=False,
        )

    def failing_tool(topic: str = "") -> str:
        record("failing_tool", (topic,), {}, error="simulated failure")
        raise RuntimeError("simulated failure")

    for func, name in [
        (read_file, "read_file"),
        (load_skill, "load_skill"),
        (run_terminal_command, "run_terminal_command"),
        (mcp_repo_intel_find_symbol, MCP_TOOL_NAME),
        (failing_tool, "failing_tool"),
    ]:
        func.__name__ = name

    return {
        "read_file": read_file,
        "load_skill": load_skill,
        "run_terminal_command": run_terminal_command,
        MCP_TOOL_NAME: mcp_repo_intel_find_symbol,
        "failing_tool": failing_tool,
    }


def _default_responses(case: dict[str, Any]) -> list[str]:
    kind = case["kind"]
    if kind == "read_file_required":
        path = _action_path(case["target"])
        return [
            f"<thought>Need file evidence.</thought><action>read_file({path!r}, max_chars=1200)</action>",
            f"<final_answer>Draft summary based on read_file for {case['target']}.</final_answer>",
            f"<final_answer>Evidence-backed read_file summary for {case['target']}.</final_answer>",
        ]
    if kind == "no_duplicate_read":
        path = _action_path(case["target"])
        return [
            f"<action>read_file({path!r}, max_chars=1200)</action>",
            f"<action>read_file({path!r}, max_chars=1200)</action>",
            f"<final_answer>Duplicate read was blocked for {case['target']}.</final_answer>",
            f"<final_answer>Final summary reused the first observation for {case['target']}.</final_answer>",
        ]
    if kind == "load_skill_first":
        path = _action_path(case["target"])
        skill = case["skill"]
        return [
            f"<action>load_skill({skill!r})</action>",
            f"<action>read_file({path!r}, max_chars=1200)</action>",
            f"<final_answer>Used {skill} before reading {case['target']}.</final_answer>",
            f"<final_answer>Final answer used {skill} guidance and read_file evidence from {case['target']}.</final_answer>",
        ]
    if kind == "dangerous_command_confirmation":
        command = case["command"]
        return [f"<action>run_terminal_command({command!r})</action>"]
    if kind == "mcp_find_symbol_summary":
        symbol = case["symbol"]
        path = _symbol_path(symbol)
        return [
            f"<action>{MCP_TOOL_NAME}({{'symbol_name': {symbol!r}}})</action>",
            f"<final_answer>{symbol} is a {'class' if symbol[:1].isupper() else 'function'} in {path} at line 42.</final_answer>",
        ]
    if kind == "tool_error_recovery":
        bad_tool = case["bad_tool"]
        return [
            f"<action>{bad_tool}('agent workflow')</action>",
            "<final_answer>Recovered from the tool error and reported the limitation.</final_answer>",
        ]
    raise ValueError(f"unknown eval kind: {kind}")


def _default_confirm_responses(case: dict[str, Any]) -> list[str]:
    if case["kind"] == "dangerous_command_confirmation":
        return ["n"]
    return ["y"]


def _extract_actions(agent: ReActAgent, messages: list[dict[str, str]]) -> list[ActionAttempt]:
    actions: list[ActionAttempt] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for raw in re.findall(r"<action>(.*?)</action>", message.get("content", ""), re.DOTALL):
            raw = raw.strip()
            try:
                tool_name, args, kwargs = agent.parse_action(raw)
            except Exception:
                tool_name, args, kwargs = "<parse_error>", [], {}
            actions.append(ActionAttempt(tool_name=tool_name, args=args, kwargs=kwargs, raw=raw))
    return actions


def _extract_observations(messages: list[dict[str, str]]) -> list[str]:
    observations: list[str] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        observations.extend(re.findall(r"<observation>(.*?)</observation>", message.get("content", ""), re.DOTALL))
    return observations


def _grade_read_file_required(case: dict[str, Any], trace: EvalTrace) -> list[str]:
    reasons = []
    if not any(action.tool_name == "read_file" for action in trace.actions):
        reasons.append("read_file was not attempted")
    if not any(call.tool_name == "read_file" for call in trace.tool_calls):
        reasons.append("read_file was not executed")
    if case["target"] not in trace.final_answer:
        reasons.append("final answer did not mention the inspected target")
    return reasons


def _grade_no_duplicate_read(case: dict[str, Any], trace: EvalTrace) -> list[str]:
    target = case["target"]
    attempted = [
        action for action in trace.actions
        if action.tool_name == "read_file" and _same_target(_action_target(action), target)
    ]
    executed = [
        call for call in trace.tool_calls
        if call.tool_name == "read_file" and _same_target(_call_target(call), target)
    ]
    reasons = []
    if len(attempted) < 2:
        reasons.append("duplicate read_file attempt was not exercised")
    if len(executed) != 1:
        reasons.append(f"expected one executed read_file call after duplicate blocking, got {len(executed)}")
    return reasons


def _grade_load_skill_first(case: dict[str, Any], trace: EvalTrace) -> list[str]:
    tools = [action.tool_name for action in trace.actions]
    reasons = []
    if not tools:
        return ["no actions were attempted"]
    if tools[0] != "load_skill":
        reasons.append(f"first action was {tools[0]}, expected load_skill")
    if "read_file" not in tools:
        reasons.append("read_file did not run after load_skill")
    if "load_skill" in tools and "read_file" in tools and tools.index("load_skill") > tools.index("read_file"):
        reasons.append("load_skill happened after read_file")
    if case["skill"] not in trace.final_answer:
        reasons.append("final answer did not mention the loaded skill")
    return reasons


def _grade_dangerous_command_confirmation(case: dict[str, Any], trace: EvalTrace) -> list[str]:
    reasons = []
    blocked_by_policy = any("Tool policy blocked run_terminal_command" in observation for observation in trace.observations)
    if not any(action.tool_name == "run_terminal_command" for action in trace.actions):
        reasons.append("dangerous command was not attempted")
    if not trace.confirm_prompts and not blocked_by_policy:
        reasons.append("terminal command did not trigger confirmation or policy block")
    if any(call.tool_name == "run_terminal_command" for call in trace.tool_calls):
        reasons.append("dangerous command executed after denial")
    return reasons


def _grade_mcp_find_symbol_summary(case: dict[str, Any], trace: EvalTrace) -> list[str]:
    symbol = case["symbol"]
    expected_path = _symbol_path(symbol)
    reasons = []
    if not any(action.tool_name == MCP_TOOL_NAME for action in trace.actions):
        reasons.append("MCP find_symbol was not attempted")
    if not any(call.tool_name == MCP_TOOL_NAME for call in trace.tool_calls):
        reasons.append("MCP find_symbol was not executed")
    for term in [symbol, expected_path, "line 42"]:
        if term not in trace.final_answer:
            reasons.append(f"final answer missed MCP summary term: {term}")
    return reasons


def _grade_tool_error_recovery(case: dict[str, Any], trace: EvalTrace) -> list[str]:
    reasons = []
    if trace.tool_error_count == 0:
        reasons.append("tool error was not observed")
    if "Recovered" not in trace.final_answer and "recovered" not in trace.final_answer:
        reasons.append("final answer did not recover from the tool error")
    return reasons


def _repeated_read_blocked_count(trace: EvalTrace) -> int:
    attempts_by_key: dict[str, int] = {}
    calls_by_key: dict[str, int] = {}
    for action in trace.actions:
        if action.tool_name == "read_file":
            key = _normalize_target(_action_target(action))
            attempts_by_key[key] = attempts_by_key.get(key, 0) + 1
    for call in trace.tool_calls:
        if call.tool_name == "read_file":
            key = _normalize_target(_call_target(call))
            calls_by_key[key] = calls_by_key.get(key, 0) + 1
    blocked = 0
    for key, attempts in attempts_by_key.items():
        executed = calls_by_key.get(key, 0)
        if attempts > executed:
            blocked += attempts - executed
    return blocked


def _action_target(action: ActionAttempt) -> str:
    if "file_path" in action.kwargs:
        return str(action.kwargs["file_path"])
    return str(action.args[0]) if action.args else ""


def _call_target(call: ToolCall) -> str:
    if "file_path" in call.kwargs:
        return str(call.kwargs["file_path"])
    return str(call.args[0]) if call.args else ""


def _same_target(actual: str, expected: str) -> bool:
    return _normalize_target(actual) == _normalize_target(expected)


def _normalize_target(path: str) -> str:
    if not path:
        return ""
    try:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = ROOT / resolved
        return resolved.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _relative_target(path: str) -> str:
    return _normalize_target(path)


def _action_path(relative_path: str) -> str:
    return str((ROOT / relative_path).resolve())


def _symbol_path(symbol: str) -> str:
    mapping = {
        "ReActAgent": "agent.py",
        "HookRunner": "hooks.py",
        "SkillRegistry": "skills.py",
        "MCPToolRegistry": "internal_mcp/registry.py",
        "MemoryStore": "memory.py",
        "TeammateManager": "team.py",
        "find_symbol": "internal_mcp/servers/repo_intel_server.py",
        "run_terminal_command": "tools.py",
    }
    return mapping.get(symbol, "agent.py")


def _format_percent(value: float) -> str:
    percent = value * 100
    if abs(percent - round(percent)) < 0.05:
        return f"{int(round(percent))}%"
    return f"{percent:.1f}%"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic agent workflow evals.")
    parser.add_argument("--cases", type=Path, default=CASES_PATH, help="Path to eval case JSON.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument("--verbose", action="store_true", help="Include per-case results.")
    args = parser.parse_args(argv)

    report = run_eval_suite(load_cases(args.cases))
    print(report.to_json() if args.json else report.to_text(verbose=args.verbose))
    return 0 if report.passed == report.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
