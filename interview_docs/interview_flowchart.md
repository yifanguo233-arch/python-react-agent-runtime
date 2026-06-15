# AI Agent 整体流程讲解稿

---

## 1. 整体流程总览

```text
用户输入任务
  ↓
main(project_directory, model)
  ↓
创建 ReActAgent
  ↓
Runtime 装配
  - 注册本地工具
  - 注册内部能力工具
  - 加载 Memory
  - 加载 Team / MCP
  - 加载 Tool Policy
  - 加载 Hook
  - 初始化 Trace / Run Log
  ↓
ReActAgent.run(user_input)
  ↓
SessionStart Hook
  ↓
构造会话上下文 + 记忆上下文
  ↓
任务分流
  - 特殊命令
  - Skill
  - Direct Answer
  - Plan-and-Execute
  - Pure ReAct fallback
  ↓
进入执行链路
  - plan()
  - 用户确认
  - execute_step()
  - 或 _react_loop()
  ↓
统一 Tool Gateway
  - Tool Policy
  - Human approval
  - Path validation
  - Hook
  - Real tool execution
  - Observation 回填
  ↓
Final Answer
  ↓
写入 session history / trace / run log
```

---

## 2. 入口和 Runtime 装配

```text
main(project_directory, model)
  ↓
创建 ReActAgent
  ↓
初始化 self.tools
  ↓
注册本地工具
  ↓
注册内部能力工具
  - load_skill
  - save_memory
  - task
  - team 相关能力
  ↓
初始化 MemoryStore
  ↓
初始化 TeammateManager
  ↓
加载 MCP 工具
  ↓
初始化 ToolPermissionPolicy
  ↓
初始化 HookRunner
  ↓
初始化 TraceStore / run log
```

---

## 3. run() 里的任务分流

```text
run(user_input)
  ↓
SessionStart Hook
  ↓
启动 run log / trace
  ↓
构造 session context
  ↓
判断任务类型
  ├─ 特殊命令
  ├─ Skill
  ├─ Direct Answer
  ├─ Plan-and-Execute
  └─ Pure ReAct fallback
```

### 3.1 特殊命令路径

```text
用户输入 /status /team /inbox
  ↓
_handle_special_command()
  ↓
直接返回结果
```

### 3.2 Skill 路径

```text
命中 skill
  ↓
加载 skill 正文
  ↓
注入任务说明 / 模板
  ↓
进入 _react_loop()
```

### 3.3 Direct Answer 路径

```text
任务偏概念问答 / 建议型问题
  ↓
跳过 plan 和工具执行
  ↓
进入 _direct_answer()
  ↓
直接生成答案
```

### 3.4 Plan-and-Execute 路径

```text
任务复杂
  ↓
plan()
  ↓
模型输出多个 <step>
  ↓
展示给用户确认
  ↓
逐步 execute_step()
```

### 3.5 Pure ReAct fallback 路径

```text
planning 失败
或用户拒绝当前计划
  ↓
降级到 _react_loop()
  ↓
边想边做
```

---

## 4. Plan-and-Execute 主流程

```text
复杂任务进入 plan()
  ↓
模型输出多个 <step>
  ↓
Runtime 解析步骤列表
  ↓
写入 trace 的 plan 事件
  ↓
向用户展示执行计划
  ↓
用户确认
  ↓
逐个 step 调 execute_step()
  ↓
汇总每个 step 的结果
  ↓
生成整体 final answer
```

### 4.1 规划阶段和执行阶段分离

```text
plan()
  ↓
只负责拆步骤
  ↓
不执行真实工具

execute_step()
  ↓
负责每一步的真实执行
```

### 4.2 step 之间的上下文衔接

```text
step1 结果
  ↓
写入后续上下文
  ↓
step2 使用 step1 结果
  ↓
继续推进
```

---

## 5. execute_step() 内部 ReAct 循环

```text
execute_step(step, context, original_task)
  ↓
构造 React Prompt
  ↓
注入原始任务 + 当前 step + 前序结果
  ↓
进入 step 内部循环
  ↓
模型输出
  - thought
  - action
  - final_answer
```

### 5.1 Thought

```text
模型输出 <thought>
  ↓
控制台打印 Thought
  ↓
Trace 记录 Thought
```

### 5.2 协议校验

```text
检查协议合法性
  ├─ 是否伪造 <observation>
  └─ 是否同时输出 <action> 和 <final_answer>
  ↓
不合法则重试
```

### 5.3 final_answer

```text
模型输出 <final_answer>
  ↓
当前 step 完成
  ↓
返回 step 结果
```

### 5.4 action

```text
模型输出 <action>
  ↓
解析工具名和参数
  ↓
进入 _run_tool_with_hooks()
```

---

## 6. 纯 ReAct 模式

```text
_react_loop()
  ↓
模型输出 thought / action / final_answer
  ↓
如果是 action
  - parse_action
  - 进入 Tool Gateway
  - 拿到 observation
  - 回填给模型
  ↓
如果是 final_answer
  - 检查 evidence ledger
  - 返回最终答案
```

### 6.1 Observation 回传

```text
工具执行结果
  ↓
封装成 <observation>
  ↓
加入消息历史
  ↓
成为模型下一轮判断依据
```

### 6.2 Evidence Ledger

```text
真实观察到的工具结果
  ↓
写入 evidence ledger
  ↓
约束 final answer 引用范围
```

---

## 7. Tool Gateway

```text
模型给出 action
  ↓
Runtime 解析 action
  ↓
进入统一 Tool Gateway
  ↓
记录 trace: action
  ↓
检查工具是否存在
  ↓
Tool Policy 决策
  ↓
必要时人工审批
  ↓
文件类工具做路径边界校验
  ↓
PreToolUse Hook
  ↓
执行真实工具函数
  ↓
PostToolUse Hook
  ↓
记录 tool_result
  ↓
结果作为 observation 回填
```

---

## 8. Tool Policy

```text
工具调用进入 Tool Policy
  ↓
执行前决策
  ├─ allow
  ├─ require approval
  └─ block
```

### 8.1 deny pattern

```text
命中危险模式
  - rm -rf
  - rmdir /s /q
  - git reset --hard
  ↓
直接拦截
```

### 8.2 allowed_roots

```text
文件类工具
  - read_file
  - list_directory
  - write_to_file
  - search_in_files
  ↓
检查路径是否在允许目录内
  ↓
越界则拦截
```

### 8.3 Human-in-the-loop

```text
高风险工具
  - run_terminal_command
  ↓
即使未命中 deny
  ↓
仍需人工审批
```

---

## 9. Hook 生命周期

```text
SessionStart
  ↓
PreToolUse
  ↓
Real Tool Execution
  ↓
PostToolUse
```

### 9.1 Hook 在链路中的位置

```text
任务开始
  ↓
SessionStart Hook
  ↓
每次工具执行前
  ↓
PreToolUse Hook
  ↓
每次工具执行后
  ↓
PostToolUse Hook
```

---

## 10. Trace、Run Log 和复盘

```text
任务开始
  ↓
_start_run_log()
  ↓
生成 .runs/*.log
  ↓
TraceStore.start_run()
  ↓
写 runs 表
  ↓
后续每一轮继续写 events 表
  - thought
  - action
  - tool_policy
  - tool_result
  - evidence_ledger
  ↓
任务结束时 finish_run()
  ↓
写 final_answer / finished_at / status
```

---

## 11. Memory

### 11.1 短期记忆

```text
当前运行中的任务和结果
  ↓
写入 session_history
  ↓
注入当前任务上下文
```

### 11.2 长期记忆

```text
用户明确要求保存
  ↓
开放 save_memory
  ↓
写入 .memory/
  ↓
下次运行继续使用
```

---

## 12. Multi-Agent / Subagent

### 12.1 Subagent

```text
主 Agent 调用 task()
  ↓
派生临时子智能体
  ↓
独立上下文
  ↓
更受限工具集
  ↓
最大轮数保护
```

### 12.2 Teammate / Multi-Agent

```text
进入团队协作
  ↓
TeammateManager 管理队友
  ↓
MessageBus 收发消息
  ↓
队友通过 inbox 协作
```
