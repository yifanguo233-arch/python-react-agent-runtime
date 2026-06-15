# Reference: analyze-code

## Analysis Dimensions

- Repository shape: top-level files, feature directories, tests, generated/cache directories.
- Execution chain: CLI entry, prompt rendering, model dispatch, action parsing, tool execution, observation feedback, final answer.
- Tool layer: file tools, terminal command tool, search tool, web search, local knowledge-base query.
- Skill system: lightweight discovery, manifest matching, on-demand SKILL.md loading, resource disclosure.
- Memory system: `.memory/` storage, memory index, prompt injection.
- MCP system: config loading, stdio client connection, registry wrapping, internal repo intelligence server.
- RAG system: document parsing, chunking, embedding, Chroma persistence, query recall and reranking.
- Hook system: SessionStart, PreToolUse, PostToolUse.
- Team system: teammate threads, inbox messages, shutdown protocol, plan approval.

## Report Template

```text
# 项目分析报告

## 项目一句话定位
...

## 项目结构概览
...

## 核心执行链路
用户任务 -> <action> -> 工具调用 -> <observation> -> <final_answer>

## 核心模块职责
- agent.py: ...
- tools.py: ...
- skills.py: ...
- memory.py: ...
- internal_mcp/: ...
- rag/: ...

## 风险点
1. ...
2. ...
3. ...

## 优先改进建议
1. ...
2. ...
3. ...

## 本次读取过的证据文件
- README.md
- agent.py
- tools.py
...
```

## Review Checklist

- Did the Agent read `README.md` before summarizing?
- Did it inspect the directory tree?
- Did it search for core symbols?
- Did it read the files it claims to summarize?
- Did it separate observed facts from recommendations?
- Did it produce concrete risks and prioritized next steps?
