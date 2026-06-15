# AI Agent 面试演示方案（最终版）

目标：证明下面这些能力：

- 这是一个 Agent Runtime，不是普通聊天
- 它有 Plan-and-Execute + ReAct 执行闭环
- 它有统一工具层，能真实调用工具
- 它有 Tool Policy 和权限边界
- 它有 Hook 生命周期扩展
- 它有 Trace，可复盘、可筛项目
- 它有 Skill 机制，不只是底层工具调用

---

## 1. 演示前准备

### 1.1 中央 Trace 库环境变量

```powershell
$env:AGENT_TRACE_DB="C:\my_project\multi-agent-ecommerce-system\.runs\traces.sqlite3"
把 trace 写到 multi-agent-ecommerce-system 这个项目自己的库

$env:AGENT_TRACE_DB="C:\my_project\agent-traces\traces.sqlite3"
直接写入中心库
```
---

## 2. 第一段演示：在另一个项目里跑正常任务

### 2.1 启动 Agent

```powershell
.\.venv\Scripts\python.exe agent.py C:\my_project\multi-agent-ecommerce-system
```

### 2.2 输入任务

```text
请读取C:/my_project/multi-agent-ecommerce-system/agents目录，概括各个 agent 的职责，并生成 summary.md 文件。
```

### 2.3 这一步重点看什么

- 能看到 `Plan`
- 能看到 `Thought`
- 能看到 `Action`
- 能看到 `Observation`
- 能看到真实工具调用
- 能看到最终生成结果
- 能看到 `PostToolUse Hook`
---

## 3. 第二段演示：查看 Trace

在自己的库
```powershell
.\.venv\Scripts\python.exe scripts\view_runs.py --db C:\my_project\multi-agent-ecommerce-system\.runs\traces.sqlite3 list
.\.venv\Scripts\python.exe scripts\view_runs.py --db C:\my_project\multi-agent-ecommerce-system\.runs\traces.sqlite3 show
```

如果想查看某一条指定 Trace，把上面任意一个 `show` 改成下面这种形式：

```powershell
.\.venv\Scripts\python.exe scripts\view_runs.py --db C:\my_project\multi-agent-ecommerce-system\.runs\traces.sqlite3 show ce6119256d0b
```

合并到中心库
.\.venv\Scripts\python.exe scripts\merge_traces.py --dest C:\my_project\agent-traces\traces.sqlite3 C:\my_project\multi-agent-ecommerce-system


在中心库
.\.venv\Scripts\python.exe scripts\view_runs.py --db C:\my_project\agent-traces\traces.sqlite3 list
只看这个项目
.\.venv\Scripts\python.exe scripts\view_runs.py --db C:\my_project\agent-traces\traces.sqlite3 --project C:\my_project\multi-agent-ecommerce-system list
展示具体
.\.venv\Scripts\python.exe scripts\view_runs.py --db C:\my_project\agent-traces\traces.sqlite3 show f30c95e2162d




## 4. 第三段演示：回到本项目，看工具分类

### 4.1 启动本项目 Agent

```powershell
.\.venv\Scripts\python.exe agent.py .
```

### 4.2 输入任务

```text
阅读这个项目，告诉我有什么工具，分类展示
```
---
## 5. 第四段演示：越权访问被拦截

### 5.1 输入任务

```text
请列出 C:/Users/lenovo/Desktop 目录下的文件。
```

### 5.2 这一步重点看什么

- 它尝试调用文件类工具
- 在本仓库的 Agent Runtime 里，会被 `Tool Policy` 的 `allowed_roots` 拦截
- Observation 里预期能看到类似 `Tool policy blocked list_directory: path outside allowed_roots: C:/Users/lenovo/Desktop`
- Trace 里还会单独记录一条 `Tool Policy` 事件

## 6. 第五段演示：触发 Skill

### 6.1 输入任务

```text
/analyze-code 请分析这个项目，输出项目结构、核心模块、执行链路、风险点和改进建议。
```

关键词触发示例：

```text
请对这个项目做一次 analyze code，输出改进建议。
```

## 7. 补充演示：Direct Answer 路径

### 7.1 输入任务

```text
什么是 ReAct？它和 Plan-and-Execute 的区别是什么？
```
