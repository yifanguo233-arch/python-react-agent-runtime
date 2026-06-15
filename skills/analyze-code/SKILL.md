---
name: analyze-code
description: 分析项目结构、核心模块、执行链路、风险点和改进建议
keywords:
  - 分析项目
  - 项目结构
  - 核心模块
  - 执行链路
  - 风险点
  - 改进建议
  - 代码分析
  - 仓库分析
  - long-horizon
  - repo analysis
  - review
  - analyze code
---

# analyze-code

## Description
用于做基于真实 observation 的仓库分析。目标不是猜测项目，而是读取一组固定的最小证据，搜索关键入口，然后按结构化报告收口。

## Keywords
- 分析项目
- 项目结构
- 核心模块
- 执行链路
- 风险点
- 改进建议
- 代码分析
- 仓库分析
- long-horizon
- repo analysis
- review
- analyze code

## Demo Prompt
```text
/analyze-code 请分析这个项目，输出项目结构、核心模块、执行链路、风险点和改进建议。请先读取 README.md，列目录，搜索关键文件，读取 agent.py、tools.py、skills.py、memory.py 和 internal_mcp/ 相关文件，再基于真实 observation 输出报告。
```

## Steps
0. 工具使用纪律
   - 每一轮只能输出一个 `<action>...</action>`。
   - 读取文件只能使用 `read_file("path")` 或 `read_file("path", max_chars=N)`。
   - 列目录只能使用 `list_directory("path", max_entries=N)`。
   - 搜索文本只能使用 `search_in_files("keyword", "directory", max_results=N)`。
   - 不要用 `run_terminal_command("type ...")`、`cat`、`dir`、`ls` 来读取文件或列目录。
   - 不要重复读取已经读取过的文件；如果 observation 已经足够，立刻总结。

1. 最小证据集
   - 先读取 `read_file("README.md")`。如文件较长，可用 `read_file("README.md", max_chars=20000)`。
   - 再执行 `list_directory(".", max_entries=300)`，只用于识别目录形态，不要钻进生成目录。
   - 最多执行 3 次搜索，优先：
     - `search_in_files("class ReActAgent", ".", max_results=10)`
     - `search_in_files("def _run_tool_with_hooks", ".", max_results=10)`
     - `search_in_files("class SkillRegistry", ".", max_results=10)`
   - 然后读取且只读取一次这些必读文件：
     - `agent.py`
     - `tools.py`
     - `skills.py`
     - `memory.py`
     - `internal_mcp/config.py`
     - `internal_mcp/client.py`
     - `internal_mcp/registry.py`

2. 可选补充
   - 只有当最小证据集仍无法解释执行链路时，最多再读取 2 个补充文件。
   - 推荐补充候选：`hooks.py`、`team.py`、`rag/document_pipeline.py`、`rag/build_index.py`、`internal_mcp/servers/repo_intel_server.py`。
   - 不要读取 `skills/analyze-code/SKILL.md`、`reference.md` 或脚本来扩写报告，除非用户明确要求分析技能自身。

3. 强制收口规则
   - 一旦 README、目录、关键搜索、7 个必读文件都已有 observation，下一轮必须输出 `<final_answer>`。
   - 不要继续调用 `read_file`、`search_in_files`、`list_directory`。
   - 如果某个文件缺失，也要基于已获得 observation 总结，并在风险点里说明缺口。

## Final Report Shape
最终答案必须包含以下小节，并明确指出结论来自已读取 observation：

1. 项目结构总览
2. 核心模块职责
3. 主要执行链路
4. Agent 主循环读取证据
5. Skills / Memory / MCP / RAG / Hooks / Team 机制
6. 风险点
7. 改进建议
8. 已观察证据清单

## Quality Bar
- 报告必须基于真实 observation，不要编造未读文件内容。
- 风险点要具体到模块或函数层级，避免空泛评价。
- 改进建议要按优先级排列，说明为什么值得改。
- 如果触发轮数预算或信息不足，仍然输出报告，并把不足写在“已观察证据清单”或“风险点”中。
