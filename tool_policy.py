from __future__ import annotations

import json
import os
import re
import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


POLICY_CONFIG_FILES = (".tool_policy.json", ".tool_policy.yaml", ".tool_policy.yml")


DEFAULT_POLICY_CONFIG = {
    "tools": {
        "run_terminal_command": {
            # 终端命令能力强、风险高：默认需要人工确认，并先用 deny_patterns 拦截明显危险命令。
            # 命中 deny_patterns 时会直接 blocked，不会进入人工确认，更不会执行 subprocess.run。
            "risk": "high",
            "require_approval": True,
            "deny_patterns": [
                r"^\s*$",
                r"\brm\s+-rf\b",
                r"\bgit\s+reset\s+--hard\b",
                r"\bgit\s+clean\s+-fdx\b",
                r"\bdel\s+/s\s+/q\b",
                r"\brmdir\s+/s\s+/q\b",
                r"\bRemove-Item\b.*\b-Recurse\b",
            ],
        },
        "read_file": {
            "risk": "low",
            "require_approval": False,
            "allowed_roots": ["."],
        },
        "write_to_file": {
            "risk": "medium",
            "require_approval": False,
            "allowed_roots": ["."],
        },
        "list_directory": {
            "risk": "low",
            "require_approval": False,
            "allowed_roots": ["."],
        },
        "search_in_files": {
            "risk": "low",
            "require_approval": False,
            "allowed_roots": ["."],
        },
    }
}


@dataclass
class ToolPolicyRule:
    risk: str = "low"
    require_approval: bool = False
    deny_patterns: list[str] = field(default_factory=list)
    allowed_roots: list[str] = field(default_factory=list)


@dataclass
class ToolPolicyDecision:
    tool: str
    risk: str = "low"
    require_approval: bool = False
    approved: bool | None = None
    blocked: bool = False
    blocked_reason: str = ""
    matched_pattern: str = ""
    allowed_roots: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "risk": self.risk,
            "require_approval": self.require_approval,
            "approved": self.approved,
            "blocked": self.blocked,
            "blocked_reason": self.blocked_reason,
            "matched_pattern": self.matched_pattern,
            "allowed_roots": list(self.allowed_roots),
        }


class ToolPermissionPolicy:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = _merge_policy_config(DEFAULT_POLICY_CONFIG, config or {})
        self.rules = self._load_rules(self.config)

    @classmethod
    def from_project(cls, project_directory: str) -> "ToolPermissionPolicy":
        return cls(load_tool_policy_config(project_directory))

    def evaluate(
        self,
        tool_name: str,
        args: list[Any],
        kwargs: dict[str, Any],
        project_directory: str,
    ) -> ToolPolicyDecision:
        # ToolPolicy 的职责是做“执行前决策”，不执行工具本身。
        # 输出的 decision 会告诉 agent.py：允许、需要审批，还是直接阻断。
        rule = self._find_rule(tool_name)
        decision = ToolPolicyDecision(
            tool=tool_name,
            risk=rule.risk,
            require_approval=rule.require_approval,
            approved=(None if rule.require_approval else True),
            allowed_roots=list(rule.allowed_roots),
        )

        # 对 run_terminal_command，下面会取出真实命令字符串；
        # 对文件类工具，则会把 args/kwargs 序列化后用于策略匹配。
        text = _tool_input_text(tool_name, args, kwargs)
        for pattern in rule.deny_patterns:
            if _pattern_matches(pattern, text):
                # 例如 text = "rm -rf tmp" 会命中 \brm\s+-rf\b。
                # 一旦命中，直接返回 blocked decision，调用方不能执行真实工具。
                decision.blocked = True
                decision.blocked_reason = f"deny_pattern matched: {pattern}"
                decision.matched_pattern = pattern
                decision.approved = False
                return decision

        if rule.allowed_roots:
            target_path = _path_argument(tool_name, args, kwargs)
            if target_path and not _path_within_allowed_roots(target_path, rule.allowed_roots, project_directory):
                decision.blocked = True
                decision.blocked_reason = f"path outside allowed_roots: {target_path}"
                decision.approved = False

        return decision

    def _find_rule(self, tool_name: str) -> ToolPolicyRule:
        if tool_name in self.rules:
            return self.rules[tool_name]

        wildcard_matches = [
            (pattern, rule)
            for pattern, rule in self.rules.items()
            if _is_glob_pattern(pattern) and fnmatch.fnmatchcase(tool_name, pattern)
        ]
        if wildcard_matches:
            wildcard_matches.sort(key=lambda item: _glob_specificity(item[0]), reverse=True)
            return wildcard_matches[0][1]

        return ToolPolicyRule()

    def _load_rules(self, config: dict[str, Any]) -> dict[str, ToolPolicyRule]:
        tools = config.get("tools", {})
        rules: dict[str, ToolPolicyRule] = {}
        if not isinstance(tools, dict):
            return rules
        for tool_name, raw_rule in tools.items():
            if not isinstance(raw_rule, dict):
                continue
            rules[str(tool_name)] = ToolPolicyRule(
                risk=str(raw_rule.get("risk", "low")),
                require_approval=bool(raw_rule.get("require_approval", False)),
                deny_patterns=[str(item) for item in raw_rule.get("deny_patterns", [])],
                allowed_roots=[str(item) for item in raw_rule.get("allowed_roots", [])],
            )
        return rules


def load_tool_policy_config(project_directory: str) -> dict[str, Any]:
    for filename in POLICY_CONFIG_FILES:
        path = Path(project_directory) / filename
        if not path.is_file():
            continue
        loaded = _load_policy_file(path)
        return _validate_policy_config(loaded, path.name)
    return {}


def _load_policy_file(path: Path) -> Any:
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ValueError(f"{path.name} requires PyYAML; use .tool_policy.json if PyYAML is not installed.") from exc

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _validate_policy_config(loaded: Any, source_name: str) -> dict[str, Any]:
    if not isinstance(loaded, dict):
        raise ValueError(f"{source_name} must contain an object.")
    return loaded


def _merge_policy_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base))
    base_tools = merged.setdefault("tools", {})
    for tool_name, override_rule in (override.get("tools") or {}).items():
        if not isinstance(override_rule, dict):
            continue
        current = dict(base_tools.get(tool_name, {}))
        current.update(override_rule)
        base_tools[tool_name] = current
    return merged


def _tool_input_text(tool_name: str, args: list[Any], kwargs: dict[str, Any]) -> str:
    if tool_name == "run_terminal_command":
        # 终端工具的核心风险在 command 本身，所以只抽取命令字符串做策略匹配。
        command = kwargs.get("command", args[0] if args else "")
        return str(command)
    return json.dumps({"args": args, "kwargs": kwargs}, ensure_ascii=False, default=str)


def _pattern_matches(pattern: str, text: str) -> bool:
    try:
        return re.search(pattern, text, re.IGNORECASE) is not None
    except re.error:
        return pattern.lower() in text.lower()


def _is_glob_pattern(pattern: str) -> bool:
    return any(char in pattern for char in "*?[")


def _glob_specificity(pattern: str) -> int:
    return sum(1 for char in pattern if char not in "*?[")


def _path_argument(tool_name: str, args: list[Any], kwargs: dict[str, Any]) -> str | None:
    if tool_name in ("read_file", "write_to_file"):
        return str(kwargs.get("file_path", args[0] if args else "")) or None
    if tool_name == "list_directory":
        return str(kwargs.get("path", args[0] if args else "")) or None
    if tool_name == "search_in_files":
        return str(kwargs.get("directory", args[1] if len(args) >= 2 else "")) or None
    return None


def _path_within_allowed_roots(path: str, roots: list[str], project_directory: str) -> bool:
    target = Path(path)
    if not target.is_absolute():
        target = Path(project_directory) / target
    try:
        target_abs = target.resolve()
    except OSError:
        target_abs = Path(os.path.abspath(str(target)))

    allowed: list[Path] = []
    for root in roots:
        candidate = Path(root)
        if not candidate.is_absolute():
            candidate = Path(project_directory) / candidate
        try:
            allowed.append(candidate.resolve())
        except OSError:
            allowed.append(Path(os.path.abspath(str(candidate))))

    for root in allowed:
        try:
            target_abs.relative_to(root)
            return True
        except ValueError:
            continue
    return False
