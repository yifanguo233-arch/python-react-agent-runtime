import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skills import SkillRegistry, load_skill, load_skills, match_skill


def test_load_skills_returns_manifests_only():
    manifests = load_skills()
    assert manifests, "应该至少发现一个 skill"
    assert all("name" in skill and "description" in skill for skill in manifests)
    assert all("steps" not in skill for skill in manifests)
    print("OK skill discovery returns lightweight manifests")


def test_registry_describe_available():
    registry = SkillRegistry()
    description = registry.describe_available()
    assert "analyze-code" in description
    assert "list-files" in description
    print("OK skill registry describes available skills")


def test_match_skill_by_keyword():
    skill = match_skill("请帮我分析项目结构和风险点")
    assert skill is not None
    assert skill["name"] == "analyze-code"
    print("OK keyword matching finds analyze-code")


def test_load_skill_returns_full_body_and_resource_index():
    content = load_skill("analyze-code")
    assert "<skill name=\"analyze-code\">" in content
    assert "## Steps" in content
    assert "reference.md" in content
    assert "reference.md 的正文内容" not in content
    print("OK load_skill returns full body and only resource paths")


def test_discovery_reads_frontmatter_only():
    tmp_dir = os.path.join(os.path.dirname(__file__), "_tmp_skill_registry")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    try:
        skill_dir = os.path.join(tmp_dir, "body-only")
        os.makedirs(skill_dir)
        skill_md = os.path.join(skill_dir, "SKILL.md")
        with open(skill_md, "w", encoding="utf-8") as f:
            f.write(
                """---
name: manifest-only
description: Lightweight manifest
keywords:
  - manifest-key
---

# Body Title

## Description
Body description that should not become the manifest.

## Keywords
- body-only-key

## Steps
secret body detail
"""
            )

        registry = SkillRegistry(tmp_dir)
        manifests = registry.list_manifests()
        assert len(manifests) == 1
        assert manifests[0]["name"] == "manifest-only"
        assert manifests[0]["description"] == "Lightweight manifest"
        assert "secret body detail" not in repr(manifests[0])
        assert registry.match("manifest-key")["name"] == "manifest-only"
        assert registry.match("body-only-key") is None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("OK skill discovery reads frontmatter only")


def test_analyze_code_long_horizon_demo_contract():
    skill = match_skill("请分析这个项目，输出项目结构、核心模块、执行链路、风险点和改进建议")
    content = load_skill("analyze-code")

    assert skill is not None
    assert skill["name"] == "analyze-code"
    for expected in [
        "每一轮只能输出一个",
        "read_file(\"README.md\")",
        "不要用 `run_terminal_command(\"type ...\")`",
        "list_directory(\".\", max_entries=300)",
        "search_in_files(\"class ReActAgent\"",
        "一旦 README、目录、关键搜索、7 个必读文件都已有 observation",
        "下一轮必须输出 `<final_answer>`",
        "agent.py",
        "tools.py",
        "skills.py",
        "memory.py",
        "internal_mcp/client.py",
        "internal_mcp/registry.py",
        "项目结构总览",
        "核心模块职责",
        "主要执行链路",
        "已观察证据清单",
    ]:
        assert expected in content
    print("OK analyze-code long-horizon demo contract")


if __name__ == "__main__":
    test_load_skills_returns_manifests_only()
    test_registry_describe_available()
    test_match_skill_by_keyword()
    test_load_skill_returns_full_body_and_resource_index()
    test_discovery_reads_frontmatter_only()
    test_analyze_code_long_horizon_demo_contract()
    print("\ntest_skills passed")
