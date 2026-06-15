import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evals.runner import load_cases, run_eval_suite


def test_eval_suite_has_standard_task_volume():
    cases = load_cases()
    assert 30 <= len(cases) <= 50
    assert len(cases) == 50
    assert {case["kind"] for case in cases} >= {
        "read_file_required",
        "no_duplicate_read",
        "load_skill_first",
        "dangerous_command_confirmation",
        "mcp_find_symbol_summary",
    }


def test_eval_suite_report_metrics():
    report = run_eval_suite(load_cases())
    text = report.to_text()

    assert report.total == 50
    assert report.passed == 50
    assert report.repeated_read_blocked >= 7
    assert report.tool_error_rate > 0
    assert "total=50" in text
    assert "pass=50" in text
    assert "pass_rate=100%" in text
    assert "avg_steps=" in text
    assert "tool_error_rate=" in text
    assert "repeated_read_blocked=" in text


if __name__ == "__main__":
    test_eval_suite_has_standard_task_volume()
    test_eval_suite_report_metrics()
    print("OK test_agent_evals passed")

