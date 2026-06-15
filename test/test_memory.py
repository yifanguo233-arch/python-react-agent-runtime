import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memory import MemoryStore, _safe_name


def _make_temp_dir(name: str) -> str:
    temp_dir = os.path.join(os.path.dirname(__file__), f"_tmp_{name}")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    return temp_dir


def test_save_memory_creates_file_and_index():
    temp_dir = _make_temp_dir("memory_store")
    try:
        store = MemoryStore(os.path.join(temp_dir, ".memory"))
        result = store.save_memory(
            "prefer_tabs",
            "User prefers tabs for indentation",
            "user",
            "The user explicitly prefers tabs over spaces when editing source files.",
        )
        assert "memory 已保存" in result
        assert "memory 索引已更新" in result
        assert "保存规则" in result
        assert os.path.isfile(os.path.join(temp_dir, ".memory", "prefer_tabs.md"))
        assert os.path.isfile(os.path.join(temp_dir, ".memory", "MEMORY.md"))
        print("OK memory file and index created")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_build_memory_section_contains_saved_memory():
    temp_dir = _make_temp_dir("memory_section")
    try:
        store = MemoryStore(os.path.join(temp_dir, ".memory"))
        store.save_memory(
            "incident_board",
            "Issue board lives in Linear",
            "reference",
            "Project incidents are usually tracked in the Linear incident board.",
        )
        section = store.build_memory_section()
        assert "incident_board" in section
        assert "reference" in section
        assert "Linear incident board" in section
        assert "长期记忆目录" in section
        assert "不要参考 memory" in section
        print("OK memory section built")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_invalid_memory_type_rejected():
    temp_dir = _make_temp_dir("memory_type")
    try:
        store = MemoryStore(os.path.join(temp_dir, ".memory"))
        try:
            store.save_memory("bad", "bad", "task", "should fail")
            raise AssertionError("非法 memory type 应抛错")
        except ValueError as e:
            assert "memory type" in str(e)
        print("OK memory type validation")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_describe_available_returns_index_like_list():
    temp_dir = _make_temp_dir("memory_list")
    try:
        store = MemoryStore(os.path.join(temp_dir, ".memory"))
        store.save_memory(
            "approved_pattern",
            "This retry pattern was explicitly approved",
            "feedback",
            "The user accepted this retry strategy as the right pattern for flaky network calls.",
        )
        description = store.describe_available()
        assert "approved_pattern" in description
        assert "feedback" in description
        print("OK memory description list")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_chinese_memory_name_stays_readable():
    temp_dir = _make_temp_dir("memory_chinese_name")
    try:
        store = MemoryStore(os.path.join(temp_dir, ".memory"))
        store.save_memory(
            "中文偏好",
            "用户希望保留中文 memory 名",
            "user",
            "中文 memory 名应该生成可读文件名，而不是被清空。",
        )
        assert _safe_name("中文偏好") == "中文偏好"
        assert os.path.isfile(os.path.join(temp_dir, ".memory", "中文偏好.md"))
        print("OK Chinese memory file name")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_empty_memory_section_explains_storage_and_ignore():
    temp_dir = _make_temp_dir("memory_empty_section")
    try:
        store = MemoryStore(os.path.join(temp_dir, ".memory"))
        section = store.build_memory_section()
        assert "暂无可用长期记忆" in section
        assert "长期记忆目录" in section
        assert "ignore memory" in section
        assert "不要参考 memory" in section
        print("OK empty memory section explains storage and ignore")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_save_memory_creates_file_and_index()
    test_build_memory_section_contains_saved_memory()
    test_invalid_memory_type_rejected()
    test_describe_available_returns_index_like_list()
    test_chinese_memory_name_stays_readable()
    test_empty_memory_section_explains_storage_and_ignore()
    print("\ntest_memory passed")
