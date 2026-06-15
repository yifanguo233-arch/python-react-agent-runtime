import os
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tools
from agent import ReActAgent
from hooks import HookRunner
from tools import (
    list_directory,
    query_knowledge_base,
    read_file,
    run_terminal_command,
    search_in_files,
    web_search,
    write_to_file,
)


TMP_ROOT = Path(__file__).resolve().parents[1] / "tmp"


def _case_dir(name):
    path = TMP_ROOT / "tool_tests" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_file_read_write_success_and_errors():
    root = _case_dir("file_io")
    file_path = root / "nested" / "hello.txt"

    write_result = write_to_file(str(file_path), "Hello\\nWorld")
    content = read_file(str(file_path))
    missing = read_file(str(root / "missing.txt"))
    directory = read_file(str(root))

    assert write_result.startswith("写入成功：")
    assert "Hello\nWorld" in content
    assert missing.startswith("文件读取失败：")
    assert directory.startswith("文件读取失败：")


def test_run_terminal_command_returns_stdout_and_failure_details():
    success_cmd = f'"{sys.executable}" -c "print(\'tool_stdout\')"'
    failure_cmd = (
        f'"{sys.executable}" -c "import sys; '
        "print('out'); print('err', file=sys.stderr); sys.exit(3)\""
    )

    success = run_terminal_command(success_cmd)
    failure = run_terminal_command(failure_cmd)
    empty = run_terminal_command("")

    assert "命令执行成功" in success
    assert "exit_code: 0" in success
    assert "tool_stdout" in success
    assert "命令执行失败" in failure
    assert "exit_code: 3" in failure
    assert "out" in failure and "err" in failure
    assert empty.startswith("命令执行失败：")


def test_list_directory_limits_and_ignores_generated_dirs():
    root = _case_dir("list_directory")
    (root / ".git").mkdir(exist_ok=True)
    (root / ".git" / "hidden.txt").write_text("hidden", encoding="utf-8")
    (root / ".venv_broken").mkdir(exist_ok=True)
    (root / ".venv_broken" / "site_package.py").write_text("hidden", encoding="utf-8")
    for index in range(8):
        (root / f"file_{index}.txt").write_text(str(index), encoding="utf-8")

    full = list_directory(str(root), max_entries=20)
    limited = list_directory(str(root), max_entries=3)
    missing = list_directory(str(root / "missing"))

    assert "file_0.txt" in full
    assert ".git" not in full and "hidden.txt" not in full
    assert ".venv_broken" not in full and "site_package.py" not in full
    assert "结果过多" in limited
    assert missing.startswith("目录列出失败：")


def test_search_in_files_limits_results_and_ignores_generated_dirs():
    root = _case_dir("search_in_files")
    (root / "a.txt").write_text("needle one\nneedle two\nneedle three", encoding="utf-8")
    (root / ".venv").mkdir(exist_ok=True)
    (root / ".venv" / "ignored.txt").write_text("needle hidden", encoding="utf-8")
    (root / ".venv_anaconda_torch_fail").mkdir(exist_ok=True)
    (root / ".venv_anaconda_torch_fail" / "ignored_too.txt").write_text("needle hidden", encoding="utf-8")

    result = search_in_files("needle", str(root), max_results=2)
    empty_keyword = search_in_files("", str(root))
    missing_dir = search_in_files("needle", str(root / "missing"))

    assert result.count("needle") == 2
    assert "结果过多" in result
    assert "ignored.txt" not in result
    assert "ignored_too.txt" not in result
    assert empty_keyword.startswith("文件搜索失败：")
    assert missing_dir.startswith("文件搜索失败：")


def test_web_search_uses_mock_ddgs_without_network():
    original_ddgs = tools._DDGS

    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, query, max_results=3):
            assert query == "agent tools"
            assert max_results == 2
            return [
                {"title": "Result A", "body": "Body A"},
                {"title": "Result B", "body": "Body B"},
            ]

    try:
        tools._DDGS = FakeDDGS
        result = web_search("agent tools", max_results=2)
        invalid = web_search("agent tools", max_results="bad")
    finally:
        tools._DDGS = original_ddgs

    assert "结果1" in result and "Result A" in result
    assert "结果2" in result and "Body B" in result
    assert invalid.startswith("搜索失败：")


def test_query_knowledge_base_uses_mock_components():
    original_get_components = tools._get_rag_components

    class FakeEmbedding:
        def tolist(self):
            return [[0.1, 0.2]]

    class FakeModel:
        def encode(self, texts):
            assert texts == ["agent tools"]
            return FakeEmbedding()

    class FakeCollection:
        def query(self, query_embeddings, n_results):
            assert query_embeddings == [[0.1, 0.2]]
            assert n_results == 3
            return {
                "documents": [["agent tool calling content", "unrelated"]],
                "metadatas": [[
                    {
                        "source": "doc.md",
                        "file_type": "md",
                        "title": "Agent",
                        "heading_path": "Agent / Tools",
                        "chunk_index": 1,
                    },
                    {"source": "other.md", "chunk_index": 2},
                ]],
            }

    try:
        tools._get_rag_components = lambda: (FakeModel(), FakeCollection())
        result = query_knowledge_base("agent tools", top_k=1)
        invalid = query_knowledge_base("agent tools", top_k="bad")
    finally:
        tools._get_rag_components = original_get_components

    assert "片段1" in result
    assert "doc.md" in result
    assert "agent tool calling content" in result
    assert invalid.startswith("知识库查询失败：")


def test_agent_path_guard_covers_directory_tools():
    root = _case_dir("path_guard")
    project = root / "repo"
    sibling = root / "repo_other"
    project.mkdir(exist_ok=True)
    sibling.mkdir(exist_ok=True)

    agent = ReActAgent.__new__(ReActAgent)
    agent.project_directory = str(project)
    agent.hook_runner = HookRunner()
    agent.tools = {
        "list_directory": lambda path: "SHOULD_NOT_RUN",
        "search_in_files": lambda keyword, directory: "SHOULD_NOT_RUN",
    }
    agent._run_tool_with_hooks = ReActAgent._run_tool_with_hooks.__get__(agent, ReActAgent)
    agent._append_observation = ReActAgent._append_observation.__get__(agent, ReActAgent)
    agent._extract_path_argument = ReActAgent._extract_path_argument.__get__(agent, ReActAgent)
    agent._validate_path = ReActAgent._validate_path.__get__(agent, ReActAgent)
    agent._format_action_call = ReActAgent._format_action_call.__get__(agent, ReActAgent)

    messages = []
    with redirect_stdout(StringIO()):
        list_result, _ = agent._run_tool_with_hooks(
            "list_directory",
            [str(sibling)],
            {},
            messages,
            available_tools=agent.tools,
        )
        search_result, _ = agent._run_tool_with_hooks(
            "search_in_files",
            ["needle", str(sibling)],
            {},
            messages,
            available_tools=agent.tools,
        )
        allowed_result, _ = agent._run_tool_with_hooks(
            "list_directory",
            [str(project)],
            {},
            messages,
            available_tools={"list_directory": lambda path: "IN_PROJECT"},
        )

    assert "SHOULD_NOT_RUN" not in list_result
    assert "SHOULD_NOT_RUN" not in search_result
    assert str(sibling) in list_result
    assert str(sibling) in search_result
    assert allowed_result == "IN_PROJECT"


if __name__ == "__main__":
    test_file_read_write_success_and_errors()
    print("OK file read/write success and readable errors")
    test_run_terminal_command_returns_stdout_and_failure_details()
    print("OK terminal command returns stdout, stderr, and exit code")
    test_list_directory_limits_and_ignores_generated_dirs()
    print("OK list_directory limits output and ignores generated dirs")
    test_search_in_files_limits_results_and_ignores_generated_dirs()
    print("OK search_in_files limits output and ignores generated dirs")
    test_web_search_uses_mock_ddgs_without_network()
    print("OK web_search is tested without network")
    test_query_knowledge_base_uses_mock_components()
    print("OK query_knowledge_base is tested without Chroma or embeddings")
    test_agent_path_guard_covers_directory_tools()
    print("OK agent path guard covers directory tools")
    print("test_tools passed")
