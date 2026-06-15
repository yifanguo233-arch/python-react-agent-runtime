import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tool_policy import ToolPermissionPolicy


TMP_ROOT = Path(__file__).resolve().parents[1] / "tmp" / "tool_policy_tests"


def _fresh_dir(name: str) -> Path:
    path = TMP_ROOT / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


def test_default_policy_blocks_dangerous_terminal_commands():
    project_dir = _fresh_dir("default_block")
    policy = ToolPermissionPolicy()

    decision = policy.evaluate("run_terminal_command", ["rm -rf tmp"], {}, str(project_dir))
    record = decision.to_record()

    assert decision.blocked is True
    assert record["tool"] == "run_terminal_command"
    assert record["risk"] == "high"
    assert record["approved"] is False
    assert "deny_pattern" in record["blocked_reason"]


def test_project_config_limits_write_allowed_roots():
    project_dir = _fresh_dir("allowed_roots")
    (project_dir / "safe").mkdir()
    (project_dir / ".tool_policy.json").write_text(
        json.dumps(
            {
                "tools": {
                    "write_to_file": {
                        "risk": "medium",
                        "allowed_roots": ["safe"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    policy = ToolPermissionPolicy.from_project(str(project_dir))
    inside = policy.evaluate("write_to_file", [str(project_dir / "safe" / "ok.txt"), "ok"], {}, str(project_dir))
    outside = policy.evaluate("write_to_file", [str(project_dir / "other.txt"), "no"], {}, str(project_dir))

    assert inside.blocked is False
    assert inside.approved is True
    assert outside.blocked is True
    assert outside.approved is False
    assert "outside allowed_roots" in outside.blocked_reason


def test_default_policy_limits_file_tools_to_project_root():
    project_dir = _fresh_dir("default_file_roots")
    sibling = TMP_ROOT / "outside_read.txt"
    sibling.write_text("outside", encoding="utf-8")
    policy = ToolPermissionPolicy()

    read_decision = policy.evaluate("read_file", [str(sibling)], {}, str(project_dir))
    list_decision = policy.evaluate("list_directory", [str(TMP_ROOT)], {}, str(project_dir))

    assert read_decision.blocked is True
    assert read_decision.approved is False
    assert "outside allowed_roots" in read_decision.blocked_reason
    assert list_decision.blocked is True


def test_policy_supports_wildcard_rules_for_mcp_tools():
    project_dir = _fresh_dir("mcp_wildcard")
    policy = ToolPermissionPolicy(
        {
            "tools": {
                "mcp_*": {
                    "risk": "medium",
                    "require_approval": True,
                }
            }
        }
    )

    decision = policy.evaluate("mcp_repo_intel_find_symbol", [{"symbol_name": "ReActAgent"}], {}, str(project_dir))

    assert decision.risk == "medium"
    assert decision.require_approval is True
    assert decision.approved is None
    assert decision.blocked is False


if __name__ == "__main__":
    test_default_policy_blocks_dangerous_terminal_commands()
    test_project_config_limits_write_allowed_roots()
    test_default_policy_limits_file_tools_to_project_root()
    test_policy_supports_wildcard_rules_for_mcp_tools()
    print("OK test_tool_policy passed")
