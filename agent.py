# 1. python 内置标准库
import ast                                # 解析Python源代码为抽象语法树，用于分析/修改代码结构
import inspect                            # 检查函数、类、模块的信息（比如获取函数源码、参数）
import json
import os                                 # 操作系统交互：读取环境变量、文件路径、创建文件夹等
import re                                 # 正则表达式：文本查找、替换、匹配（比如提取关键词、过滤内容）
from datetime import datetime
from string import Template               # 字符串模板：方便批量替换文本中的变量
import sys
import time
from typing import Any, List, Callable, Tuple  # 类型注解：标注变量/函数类型，让代码更易读、防错

# 2. 第三方库 (需要用pip安装才能使用)
import click                              # 命令行工具：快速创建可在终端运行的命令、参数、选项
from dotenv import load_dotenv            # 加载.env文件：把私密配置（密钥、账号）存在文件里，不写死在代码
import platform                           # 获取系统信息：判断是Windows、Mac还是Linux
from openai import OpenAI

# 3. 自定义模块 (项目中自己写的)
from internal_mcp.client import MCPClientManager
from internal_mcp.config import load_mcp_server_configs
from internal_mcp.registry import MCPToolRegistry
from prompt_template import react_system_prompt_template, plan_system_prompt_template, subagent_system_prompt_template, direct_answer_system_prompt_template

from memory import MemoryStore
from team import TeammateManager
from hooks import HookRunner, build_default_hook_runner
from run_traces import TraceStore
from tool_policy import ToolPermissionPolicy, ToolPolicyDecision
from tools import read_file, write_to_file, run_terminal_command, list_directory, search_in_files, web_search, query_knowledge_base
from skills import get_skill_registry, match_skill

class SubagentContext:
    """子智能体上下文：独立消息列表 + 限制工具集 + 最大轮数保护"""
    def __init__(self, prompt: str, tools: dict, agent: 'ReActAgent', max_turns: int = 20):
        self.messages = [
            {"role": "system", "content": agent.render_system_prompt(subagent_system_prompt_template, tool_map=tools)},
            {"role": "user", "content": f"<question>{prompt}</question>"}
        ]
        self.tools = tools          # 子智能体可用工具（不含 task，防递归）
        self.agent = agent          # 引用父智能体，用于调用 dispatch_model 等
        self.max_turns = max_turns
        self.tool_failures: dict[str, int] = {}

    def run(self) -> str:
        """执行子智能体 ReAct 循环，返回结果摘要"""
        for turn in range(self.max_turns):
            content = self.agent.dispatch_model(self.messages)

            if self.agent._react_protocol_observation(self.messages, content):
                continue

            if "<final_answer>" in content:
                match = re.search(r"<final_answer>(.*?)</final_answer>", content, re.DOTALL)
                return match.group(1).strip() if match else "子任务完成"

            action_matches = re.findall(r"<action>(.*?)</action>", content, re.DOTALL)
            if not action_matches:
                self.messages.append({"role": "user", "content": "<observation>格式错误：必须输出 <action>...</action>，请重新输出。</observation>"})
                continue
            if len(action_matches) > 1:
                self.messages.append({"role": "user", "content": "<observation>格式错误：每轮只能输出一个 <action>...</action>。请只选择下一步最重要的一个工具调用，并重新输出。</observation>"})
                continue

            action = action_matches[0].strip()
            try:
                tool_name, args, kwargs = self.agent.parse_action(action)
            except Exception as e:
                self.messages.append({"role": "user", "content": f"<observation>Action 解析失败：{e}</observation>"})
                continue

            if tool_name not in self.tools:
                available = ', '.join(self.tools.keys())
                self.messages.append({"role": "user", "content": f"<observation>工具 '{tool_name}' 不存在，可用：{available}</observation>"})
                continue

            observation, should_stop = self.agent._run_tool_with_hooks(
                tool_name,
                args,
                kwargs,
                self.messages,
                available_tools=self.tools,
                cancel_message="子任务被取消",
            )
            if should_stop:
                return observation
            recovery_result = self.agent._recover_from_tool_failure(tool_name, observation, self.tool_failures, self.messages)
            if recovery_result is not None:
                return recovery_result

        return "子智能体达到最大轮数，未能完成任务"


class ReActAgent:
    # Callable意味可调用的函数
    def __init__(
        self,
        tools: List[Callable],
        model: str,
        project_directory: str,
        hook_runner: HookRunner | None = None,
        trace_db_path: str | None = None,
    ):
        # Step 0：Runtime 装配阶段。
        # 在真正处理用户任务前，先把模型、本地工具、Skills、Memory、MCP、Trace、ToolPolicy 都挂到 Agent 上。
        # Runtime 装配：把模型、工具、skills、memory、MCP、trace 和 policy 统一挂到同一个 Agent 实例。
        # 后续所有任务执行都会复用这些组件，避免在每个流程分支里重复初始化。
        # 把传入的工具函数列表转成字典
        # key是函数名 values是函数本身  方便后续直接通过名字调用工具
        self.tools = { func.__name__: func for func in tools }
        self.model = model
        self.project_directory = project_directory
        self.trace_db_path = self._resolve_trace_db_path(trace_db_path)
        self.hook_runner = hook_runner or build_default_hook_runner()
        self.session_started = False
        # Skills / Memory / Team / MCP 都是在 runtime 层统一管理的。
        self.skill_registry = get_skill_registry()
        self.memory_store = MemoryStore(os.path.join(self.project_directory, ".memory"))
        self.current_memory_section = self.memory_store.build_memory_section()
        self.team_manager = TeammateManager(os.path.join(self.project_directory, ".team"), self)
        self.mcp_client_manager = MCPClientManager()
        self.mcp_registry = MCPToolRegistry(self.mcp_client_manager)
        self.mcp_server_configs = load_mcp_server_configs(self.project_directory)
        load_dotenv()
        if not self._is_minimax_model():
            raise ValueError(
                f"Only MiniMax models are supported now, but received '{model}'. "
                "Please use a MiniMax model, for example: minimax/MiniMax-M2.7"
            )
        self.client = None
        self.minimax_client = OpenAI(
            api_key=self.get_minimax_api_key(),
            base_url=self.get_minimax_base_url(),
        )
        # 把内部能力也注册成工具，让模型可以通过 ReAct action 调用。
        # tools 是 Runtime 暴露给模型的能力边界；没注册的函数模型不能直接调用。
        self.tools["load_skill"] = self.load_skill
        self.tools["save_memory"] = self.save_memory
        self.tools["spawn_teammate"] = self.spawn_teammate
        self.tools["list_teammates"] = self.list_teammates
        self.tools["send_message"] = self.send_message
        self.tools["broadcast_message"] = self.broadcast_message
        self.tools["read_team_inbox"] = self.read_team_inbox
        self.tools["get_status"] = self.get_status
        self.tools["request_shutdown"] = self.request_shutdown
        self.tools["review_plan"] = self.review_plan
        # MCP 工具和本地 Python 工具合并进同一个 tool map，后面走统一的 action 网关。
        self.tools.update(self._load_mcp_tools())
        excluded_subagent_tools = {
            "task",
            "spawn_teammate",
            "list_teammates",
            "send_message",
            "broadcast_message",
            "read_team_inbox",
            "request_shutdown",
            "review_plan",
        }
        self.subagent_tools = {
            name: func
            for name, func in self.tools.items()
            if name not in excluded_subagent_tools and not self._is_mcp_tool(name)
        }
        # 将 task 方法注册为工具，让父智能体可以调用子智能体
        self.tools["task"] = self.task
        self.session_history: list[dict] = []  # 会话级记忆：记录本次启动中的所有任务和答案
        self.current_run_log_path: str | None = None
        self.trace_store: TraceStore | None = None
        self.trace_run_id: str | None = None
        # Tool Permission Policy 是工具调用前的安全层：风险等级、审批、路径边界、危险命令阻断。
        self.tool_policy = ToolPermissionPolicy.from_project(self.project_directory)
        self.tool_policy_records: list[dict[str, Any]] = []

    MAX_HISTORY_MESSAGES = 20  # 超过此数量时触发历史压缩（不含 system 消息）
    MAX_SESSION_HISTORY = 5  # 注入上下文时最多使用最近 N 条历史
    TOOL_FAILURE_PREFIXES = (
        "工具执行错误：",
        "搜索失败：",
        "知识库查询失败：",
        "文件读取失败：",
        "文件写入失败：",
        "目录列出失败：",
        "文件搜索失败：",
        "命令执行失败",
        "命令执行超时",
    )
    ALWAYS_HIDDEN_PROMPT_TOOLS = {"save_memory"}
    # save_memory 默认隐藏，只有用户明确要求保存时才开放，避免模型把临时过程写进长期记忆。
    TEAM_PROMPT_TOOLS = {
        "task",
        "spawn_teammate",
        "list_teammates",
        "send_message",
        "broadcast_message",
        "read_team_inbox",
        "request_shutdown",
        "review_plan",
    }
    GENERAL_QUESTION_TOOLS = {"web_search", "query_knowledge_base"}

    def _validate_react_output_protocol(self, content: str) -> str | None:
        """Reject model outputs that fake runtime-only tags or mix action/final in one turn."""
        action_count = len(re.findall(r"<action>(.*?)</action>", content, re.DOTALL))
        has_final = "<final_answer>" in content
        has_observation = "<observation>" in content

        if has_observation:
            return (
                "格式错误：不要自行输出 <observation>。"
                "<observation> 只能由系统在真实工具执行后回填。"
            )
        if has_final and action_count:
            return (
                "格式错误：同一轮不能同时输出 <action> 和 <final_answer>。"
                "如果要调用工具，只输出 <thought> 和一个 <action>；"
                "如果要结束任务，只输出 <thought> 和 <final_answer>。"
            )
        return None

    def _build_session_context(self) -> str:
        """将最近 N 条会话历史格式化为字符串，注入当前任务上下文"""
        recent = self.session_history[-self.MAX_SESSION_HISTORY:]
        if not recent:
            return ""
        lines = ["以下是本次会话中已完成的历史任务（供参考）："]
        for i, record in enumerate(recent, 1):
            lines.append(f"[历史任务 {i}] 用户：{record['task']}")
            lines.append(f"           结果：{record['answer'][:200]}{'...' if len(record['answer']) > 200 else ''}")
        return "\n".join(lines)

    def _build_selected_skill_context(self, selected_skill: dict | None) -> str:
        if not selected_skill:
            return ""
        skill_name = selected_skill["name"]
        skill_body = self.skill_registry.load_skill(skill_name)
        return (
            f"\n\n已显式选择技能：{skill_name}。"
            "该技能正文已由系统预加载，请直接遵循其中的步骤、脚本路径和动作示例执行。"
            "注意：技能名不是工具名，不要直接调用 skill 名称或自行发明同名工具；"
            "如果需要执行命令，请严格使用技能正文中给出的现有工具调用格式。"
            f"\n\n{skill_body}"
        )

    def _react_protocol_observation(self, messages: list, content: str) -> bool:
        protocol_error = self._validate_react_output_protocol(content)
        if protocol_error is None:
            return False
        messages.append({"role": "user", "content": f"<observation>{protocol_error}</observation>"})
        return True

    def _run_log_dir(self) -> str:
        return os.path.join(self.project_directory, ".runs")

    def _resolve_trace_db_path(self, trace_db_path: str | None) -> str:
        explicit = trace_db_path or os.getenv("AGENT_TRACE_DB", "").strip()
        if explicit:
            return os.path.abspath(explicit)
        return os.path.join(self._run_log_dir(), "traces.sqlite3")

    def _run_log_slug(self, user_input: str) -> str:
        text = re.sub(r"[^a-zA-Z0-9_-]+", "_", user_input.strip().lower())
        return text.strip("_")[:40] or "task"

    def _extract_absolute_paths(self, text: str) -> list[str]:
        candidates = re.findall(r"[A-Za-z]:[\\/][^\"'\r\n]+", text)
        cleaned: list[str] = []
        for candidate in candidates:
            value = candidate.strip().rstrip("，。；;,.!?)]}>'\"")
            value = value.replace("/", os.sep).replace("\\", os.sep)
            if value and value not in cleaned:
                cleaned.append(value)
        return cleaned

    def _project_root_markers(self) -> tuple[str, ...]:
        return (
            ".git",
            "pyproject.toml",
            "requirements.txt",
            "package.json",
            "README.md",
            "README-2.md",
            "agent.py",
            "main.py",
        )

    def _guess_project_root(self, path: str) -> str | None:
        if not path:
            return None
        candidate = os.path.abspath(path)
        if not os.path.exists(candidate):
            return None
        if os.path.isfile(candidate):
            candidate = os.path.dirname(candidate)

        common_source_dirs = {"agents", "src", "app", "python", "services", "scripts", "tests", "docs"}
        if os.path.basename(candidate).lower() in common_source_dirs:
            parent = os.path.dirname(candidate)
            if parent:
                candidate = parent

        current = candidate
        markers = self._project_root_markers()
        while True:
            if any(os.path.exists(os.path.join(current, marker)) for marker in markers):
                return current
            parent = os.path.dirname(current)
            if not parent or parent == current:
                break
            current = parent
        return candidate

    def _infer_target_project(self, user_input: str) -> str | None:
        runtime_project = os.path.abspath(self.project_directory)
        for path in self._extract_absolute_paths(user_input):
            guessed = self._guess_project_root(path)
            if not guessed:
                continue
            guessed_abs = os.path.abspath(guessed)
            if guessed_abs != runtime_project:
                return guessed_abs
        return None

    def _start_run_log(self, user_input: str) -> None:
        try:
            os.makedirs(self._run_log_dir(), exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self._run_log_dir(), f"{timestamp}_{self._run_log_slug(user_input)}.log")
            target_project = self._infer_target_project(user_input)
            self.current_run_log_path = path
            with open(path, "w", encoding="utf-8") as f:
                f.write("# Agent Run Log\n\n")
                f.write(f"started_at: {datetime.now().isoformat(timespec='seconds')}\n")
                f.write(f"runtime_project: {self.project_directory}\n")
                f.write(f"target_project: {target_project or self.project_directory}\n")
                f.write(f"trace_db: {self.trace_db_path}\n")
                f.write(f"model: {getattr(self, 'model', 'unknown')}\n\n")
                f.write("## User Task\n")
                f.write(f"{user_input}\n\n")
            self.trace_store = TraceStore(self.trace_db_path)
            self.trace_run_id = self.trace_store.start_run(
                task=user_input,
                project_directory=self.project_directory,
                model=getattr(self, "model", "unknown"),
                log_path=path,
                target_project=target_project,
            )
        except OSError as exc:
            self.current_run_log_path = None
            print(f"\n\nRun log 初始化失败：{exc}")

    def _append_run_log(self, section: str, content: str) -> None:
        path = getattr(self, "current_run_log_path", None)
        if not path:
            return
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"\n## {section}\n")
                f.write(str(content).rstrip())
                f.write("\n")
        except OSError as exc:
            print(f"\n\nRun log 写入失败：{exc}")
            self.current_run_log_path = None

        if section != "Action":
            event_type = section.lower().replace(" ", "_")
            self._record_trace_event(event_type, str(content), label=section)

    def _finish_run_log(self, final_answer: str) -> str:
        self._append_run_log("Final Answer", final_answer)
        store = getattr(self, "trace_store", None)
        run_id = getattr(self, "trace_run_id", None)
        if store and run_id:
            store.finish_run(run_id, final_answer)
        path = getattr(self, "current_run_log_path", None)
        if path:
            print(f"\n\nRun log: {path}")
        return final_answer

    def _record_trace_event(
        self,
        event_type: str,
        content: str,
        *,
        label: str = "",
        tool_name: str | None = None,
        latency_ms: float | None = None,
        human_approval: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        store = getattr(self, "trace_store", None)
        run_id = getattr(self, "trace_run_id", None)
        if not store or not run_id:
            return
        try:
            store.add_event(
                run_id,
                event_type,
                content,
                label=label,
                tool_name=tool_name,
                latency_ms=latency_ms,
                human_approval=human_approval,
                metadata=metadata,
            )
        except Exception as exc:
            print(f"\n\nTrace write failed: {exc}")

    def _get_tool_policy(self) -> ToolPermissionPolicy:
        policy = getattr(self, "tool_policy", None)
        if policy is None:
            policy = ToolPermissionPolicy.from_project(self.project_directory)
            self.tool_policy = policy
        return policy

    def _record_tool_policy_decision(self, decision: ToolPolicyDecision, action_call: str) -> None:
        record = decision.to_record()
        records = getattr(self, "tool_policy_records", None)
        if records is None:
            records = []
            self.tool_policy_records = records
        records.append(record)

        content = json.dumps(record, ensure_ascii=False, default=str)
        self._record_trace_event(
            "tool_policy",
            content,
            label="Tool Policy",
            tool_name=decision.tool,
            human_approval=decision.approved if decision.require_approval else None,
            metadata={**record, "action": action_call},
        )

    def run(self, user_input: str):
        # Step 1：用户输入进入主执行入口。
        # 面试讲主流程时，从这个函数开始：分流在这里，真正工具执行会继续下钻到 execute_step/_react_loop。
        # 核心流程入口：一次用户任务从这里进入完整执行链路。
        # 主干顺序：memory/hook -> 特殊命令 -> skill -> 普通问答 -> plan -> step-by-step ReAct。
        # 这里把“任务分流”和“真实执行”分开处理，避免简单问题也进入重工具流程。
        original_task = user_input
        self._start_run_log(original_task)
        selected_skill = None
        hinted_skill = None
        session_hook_message = ""

        # Step 2：加载长期记忆 Memory。
        # 每次任务都重新构建 memory section；如果用户明确要求忽略 memory，这里会返回忽略说明。
        self.current_memory_section = self._load_memory_section(original_task)

        # Step 3：首次会话运行 SessionStart Hook。
        # Hook 可以阻止任务，也可以给当前任务附加额外上下文。
        if not self.session_started:
            session_result = self.hook_runner.run("SessionStart", {
                "user_input": original_task,
                "project_directory": self.project_directory,
            })
            self.session_started = True
            if session_result["exit_code"] == 1:
                return self._finish_run_log(session_result["message"] or "会话被 Hook 阻止")
            if session_result["exit_code"] == 2:
                session_hook_message = session_result["message"]

        # Step 4：加载本次运行期的短期会话历史。
        # 这是 session_history，不是 .memory/ 长期记忆。
        # 显示会话历史摘要（若有）
        session_ctx = self._build_session_context()
        if session_ctx:
            print(f"\n\n 会话记忆已加载（{len(self.session_history)} 条历史）")

        # Step 5：先处理 /status、/team、/inbox 这类特殊命令。
        # 命中特殊命令会直接返回，不进入 plan/ReAct。
        special_command_result = self._handle_special_command(user_input)
        if special_command_result is not None:
            return self._finish_run_log(special_command_result)

        # Step 6：判断 Skill。
        # Slash 命令是显式选择；关键词匹配只是提示模型“可能需要 load_skill”。
        # Skill 触发分两类：
        # 1. 用户显式输入 /skill-name；2. 普通任务文本命中关键词后提示模型先 load_skill。
        # Skill 只是任务说明和经验包，不是可直接调用的工具函数。
        # 优先级1：slash 命令精确触发（/skill-name [可选附加描述]）
        if user_input.startswith("/"):
            parts = user_input[1:].split(None, 1)
            skill_name = parts[0]
            selected_skill = self.skill_registry.get_manifest(skill_name)
            if selected_skill:
                print(f"\n\n Slash 命令触发技能：/{selected_skill['name']} — {selected_skill.get('description', '')}")
                user_input = parts[1] if len(parts) > 1 else selected_skill.get("description", skill_name)
            else:
                available = ', '.join('/' + s['name'] for s in self.skill_registry.list_manifests())
                print(f"\n\n 未找到技能 /{skill_name}，可用技能：{available}")
                return self._finish_run_log("未知 slash 命令")

        # 优先级2：关键词模糊匹配
        if not selected_skill:
            hinted_skill = match_skill(user_input, self.skill_registry.list_manifests())
            if hinted_skill:
                print(f"\n\n 匹配到相关技能：{hinted_skill['name']} — {hinted_skill.get('description', '')}")

        skill_hint = ""
        selected_skill_context = self._build_selected_skill_context(selected_skill)
        if selected_skill:
            skill_hint = (
                f"\n\n已显式选择技能：{selected_skill['name']}。"
                "技能正文已经提供在当前任务上下文中，请优先严格遵循其中的说明。"
            )
        elif hinted_skill:
            skill_hint = (
                f"\n\n可能相关技能：{hinted_skill['name']}。"
                f"如果它确实适用于当前任务，请先调用 load_skill(\"{hinted_skill['name']}\") 再继续执行。"
            )

        # Step 7：构造最终任务上下文，并按任务类型动态裁剪可见工具。
        # 这里决定模型在 prompt 里能看到哪些工具，避免无关场景乱用高级/高风险工具。
        task_for_execution = f"{user_input}{skill_hint}{selected_skill_context}"
        if session_hook_message:
            task_for_execution = f"{task_for_execution}\n\n{session_hook_message}"
        prompt_tool_map = self._build_prompt_tool_map(task_for_execution, selected_skill, hinted_skill)
        if selected_skill:
            # Step 8A：显式 Skill 任务直接进入 ReAct。
            # 因为 Skill 正文已经给出执行流程，所以这里不再额外 plan。
            # 显式选择技能时，技能正文已经进入上下文，直接让 ReAct 根据说明执行。
            print("\n\n 已显式选择技能，跳过通用任务规划，按技能正文直接执行...")
            result = self._react_loop(task_for_execution, context=session_ctx, tool_map=prompt_tool_map, max_rounds=60)
            self.session_history.append({"task": original_task, "answer": result})
            return self._finish_run_log(result)
        if self._should_skip_planning(task_for_execution):
            # Step 8B：通用问答走直答路径。
            # 概念解释/建议类问题不强行进入 Agent 工具流程，减少无意义 plan 和工具调用。
            # 普通问答不进入复杂 Agent workflow，避免为了简单问题强行调工具。
            print("\n\n 识别为通用问答，跳过任务规划，优先直接回答...")
            result = self._direct_answer(task_for_execution)
            self.session_history.append({"task": original_task, "answer": result})
            return self._finish_run_log(result)

        # Step 9：复杂任务进入 Plan-and-Execute 的 Plan 阶段。
        # plan() 只拆步骤；真正执行文件读取、搜索、命令等动作在后面的 ReAct 阶段。
        # 复杂任务先 plan，再让用户确认；确认后才进入逐步执行，这是外层 Plan-and-Execute。
        # plan 只负责拆步骤，不直接碰文件或跑命令；真实动作都留到每个 step 内部执行。
        steps = self.plan(task_for_execution, tool_map=prompt_tool_map)
        if not steps:
            # Step 9 fallback：规划失败时降级为纯 ReAct。
            # 这样模型没输出合法 <step> 时，任务仍然能继续推进。
            # 如果模型没有给出可解析步骤，就退回纯 ReAct，保证任务还能继续推进。
            print("\n\n 规划失败，降级为纯 ReAct 模式执行...")
            result = self._react_loop(task_for_execution, context=session_ctx, tool_map=prompt_tool_map)
            self.session_history.append({"task": original_task, "answer": result})
            return self._finish_run_log(result)

        print("\n\n 执行计划：")

        for i, step in enumerate(steps, 1):
            # 每个 step 内部仍然是 ReAct：模型决定 action，runtime 执行工具并返回 observation。
            print(f"  Step {i}: {step}")

        # Step 10：Human-in-the-loop 计划确认。
        # 用户确认后才执行计划；拒绝时改走直接 ReAct。
        confirm = input("\n\n是否按此计划执行？（Y/N，直接回车确认）").strip().lower()
        self._record_trace_event(
            "human_approval",
            "plan approved" if confirm != "n" else "plan denied",
            label="Plan Approval",
            human_approval=(confirm != "n"),
            metadata={"approval_type": "plan", "response": confirm},
        )
        if confirm == 'n':
            print("\n\n计划已取消，切换为直接对话模式...")
            result = self._react_loop(task_for_execution, context=session_ctx, tool_map=prompt_tool_map)
            self.session_history.append({"task": original_task, "answer": result})
            return self._finish_run_log(result)

        # Step 11：逐步执行每个 step。
        # 每个 step 通过 execute_step() 进入 ReAct 小循环；前一步结果会成为后一步上下文。
        # 执行阶段：依次执行每个步骤，传递上下文（携带会话历史）
        # 前一步结果会拼进 context，后续 step 可以基于真实 observation 继续做判断。
        context = session_ctx
        step_results: list[tuple[str, str]] = []
        for i, step in enumerate(steps, 1):
            print(f"\n\n{'='*50}")
            print(f"▶️  执行 Step {i}/{len(steps)}: {step}")
            print(f"{'='*50}")
            result = self.execute_step(step, context, task_for_execution, tool_map=prompt_tool_map)
            step_results.append((step, result))
            context += f"\n[Step {i} 结果] {result}"

        failed_steps = [(step, result) for step, result in step_results if self._is_step_result_failure(result)]
        if failed_steps:
            # Step 11.5：失败步骤显式暴露。
            # 有失败就不要包装成成功，避免 final_answer 误导用户。
            # 有失败步骤时不伪装成成功完成，直接把失败点和返回结果暴露出来。
            failure_lines = ["任务未成功完成。以下步骤执行失败："]
            for step, result in failed_steps:
                failure_lines.append(f"- 失败步骤：{step}")
                failure_lines.append(f"  返回结果：{result}")
            final_answer = "\n".join(failure_lines)
            self.session_history.append({"task": original_task, "answer": final_answer})
            return self._finish_run_log(final_answer)

        # Step 12：所有 step 完成后汇总 final_answer。
        # 汇总阶段只能基于各 step 的真实结果，不能再编造没有 observation 支持的成功。
        print("\n\n 所有步骤执行完成，正在汇总...")
        summary_messages = [
            {"role": "system", "content": self.render_system_prompt(direct_answer_system_prompt_template)},
            # 汇总阶段只能基于每个 step 的真实结果回答，不能补写没有观察到的成功。
            {"role": "user", "content": f"<question>{user_input}</question>\n\n以下是各步骤的真实执行结果，请仅基于这些结果给出最终结论，不要声称任何未明确出现的成功：\n{context}\n\n请只输出 <thought>...</thought> 和 <final_answer>...</final_answer>"}
        ]
        final_content = self.dispatch_model(summary_messages)
        final_match = re.search(r"<final_answer>(.*?)</final_answer>", final_content, re.DOTALL)
        final_answer = final_match.group(1).strip() if final_match else context
        self.session_history.append({"task": original_task, "answer": final_answer})
        return self._finish_run_log(final_answer)

    def plan(self, user_input: str, tool_map: dict[str, Callable] | None = None) -> list:
        # Step 9.1：Plan 阶段的具体实现。
        # 只要求模型输出多个 <step>，这里不会执行任何工具。
        # Plan 阶段：把复杂任务拆成可执行步骤。
        # 模型输出多个 <step>，runtime 解析成列表，并写入 trace。
        # 这里让模型只规划自然语言步骤，避免在计划阶段提前生成具体工具调用。
        print("\n\n 正在规划任务...")
        session_ctx = self._build_session_context()
        context_hint = f"\n\n{session_ctx}" if session_ctx else ""
        messages = [
            {"role": "system", "content": self.render_system_prompt(plan_system_prompt_template, tool_map=tool_map)},
            {"role": "user", "content": f"任务：{user_input}{context_hint}"}
        ]
        content = self.dispatch_model(messages)
        # 只接受显式 <step> 标签，减少模型输出闲聊文本对执行链路的影响。
        steps = re.findall(r"<step>(.*?)</step>", content, re.DOTALL)
        parsed_steps = [s.strip() for s in steps if s.strip()]
        if parsed_steps:
            plan_text = "\n".join(f"{i}. {step}" for i, step in enumerate(parsed_steps, 1))
            self._record_trace_event(
                "plan",
                plan_text,
                label="Plan",
                metadata={"steps": parsed_steps, "raw_plan": content},
            )
        return parsed_steps

    def execute_step(self, step: str, context: str, original_task: str, tool_map: dict[str, Callable] | None = None) -> str:
        # Step 11.1：执行单个计划步骤。
        # 每个 step 内部都是一个 ReAct 小循环：模型输出 action，Runtime 执行工具并回填 observation。
        # 单个 step 的执行仍然走 ReAct：计划只给方向，具体工具由模型按 observation 决定。
        context_hint = f"\n\n以下是前面步骤的执行结果，可作为当前步骤的上下文：\n{context}" if context else ""
        available_tools = tool_map or self.tools
        system_msg = {"role": "system", "content": self.render_system_prompt(react_system_prompt_template, tool_map=available_tools)}
        messages = [
            system_msg,
            {"role": "user", "content": f"<question>原始任务：{original_task}\n\n当前步骤：{step}{context_hint}</question>"}
        ]
        max_rounds = 20
        tool_failures: dict[str, int] = {}
        for _ in range(max_rounds):
            # Step 11.2：调用模型，让模型决定本轮是 final_answer 还是 action。
            # 每轮都先压缩历史，避免长任务把上下文撑爆。
            self._compress_history(messages)
            content = self.dispatch_model(messages)

            thought_match = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL)
            if thought_match:
                print(f"\n\n🧠 Thought: {thought_match.group(1).strip()}")

            if thought_match:
                self._record_trace_event("thought", thought_match.group(1).strip(), label="Thought")

            if self._react_protocol_observation(messages, content):
                continue

            if "<final_answer>" in content:
                # Step 11.3：当前 step 已完成。
                final_match = re.search(r"<final_answer>(.*?)</final_answer>", content, re.DOTALL)
                return final_match.group(1).strip() if final_match else "步骤完成"

            # Step 11.4：提取模型输出的 <action>。
            # action 仍然只是文本，下一步必须 parse_action 后进入工具网关。
            action_matches = re.findall(r"<action>(.*?)</action>", content, re.DOTALL)
            if not action_matches:
                messages.append({"role": "user", "content": "<observation>格式错误：你必须输出 <action>...</action> 标签，请重新按格式输出。</observation>"})
                continue
            if len(action_matches) > 1:
                messages.append({"role": "user", "content": "<observation>格式错误：每轮只能输出一个 <action>...</action>。请只选择下一步最重要的一个工具调用，并重新输出。</observation>"})
                continue

            action = action_matches[0].strip()
            try:
                # Step 11.5：解析 action 文本为 tool_name / args / kwargs。
                # 模型只产出 action 文本，真正的函数名和参数由 runtime 解析。
                tool_name, args, kwargs = self.parse_action(action)
            except Exception as e:
                messages.append({"role": "user", "content": f"<observation>Action 格式解析失败：{e}。请确保格式为 tool_name(\"arg1\", \"arg2\") 或 tool_name(name=\"value\")。</observation>"})
                continue

            # Step 11.6：进入统一工具网关执行工具。
            # 这里会经过 ToolPolicy、路径校验、Hook，最后才可能真正调用工具函数。
            observation, should_stop = self._run_tool_with_hooks(tool_name, args, kwargs, messages, available_tools=available_tools, cancel_message="步骤被用户取消")
            if should_stop:
                return observation
            # 工具连续失败时及时收口，避免同一个错误被模型反复重试。
            recovery_result = self._recover_from_tool_failure(tool_name, observation, tool_failures, messages)
            if recovery_result is not None:
                return recovery_result

        return "步骤达到最大执行轮数"

    def _react_loop(self, user_input: str, context: str, tool_map: dict[str, Callable] | None = None, max_rounds: int = 30) -> str:
        # Step R：纯 ReAct 执行入口。
        # 用于显式 Skill、plan 失败降级、用户拒绝 plan 后的直接执行等场景。
        # ReAct 工具调用入口流程：
        # 1. 模型只生成文本：<thought>...</thought> 或 <action>tool(args)</action>。
        # 2. Runtime 提取 action，并用 parse_action 解析出工具名和参数；这里还没有执行工具。
        # 3. 所有真实工具调用必须进入 _run_tool_with_hooks，统一经过 trace、ToolPolicy、Hook。
        # 4. 工具返回值会作为 observation 回填给模型，模型再决定下一步或输出 final_answer。
        # 关键边界：模型负责“提出调用意图”，Runtime 才负责“是否允许以及如何执行”。
        context_hint = f"\n\n{context}" if context else ""
        available_tools = tool_map or self.tools
        system_msg = {"role": "system", "content": self.render_system_prompt(react_system_prompt_template, tool_map=available_tools)}
        messages = [
            system_msg,
            {"role": "user", "content": f"<question>{user_input}</question>{context_hint}"}
        ]

        tool_failures: dict[str, int] = {}
        # evidence_ledger 记录已经观察过的证据，最终回答前会再喂回模型做一次约束。
        evidence_ledger: list[dict[str, Any]] = []
        read_file_cache: dict[str, dict[str, Any]] = {}
        evidence_ledger_injected = False
        for _ in range(max_rounds):
            # Step R1：每轮调用模型，得到 thought/action/final_answer。
            self._compress_history(messages)
            content = self.dispatch_model(messages)

            thought_match = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL)
            if thought_match:
                print(f"\n\n🧠 Thought: {thought_match.group(1).strip()}")

            if thought_match:
                self._record_trace_event("thought", thought_match.group(1).strip(), label="Thought")

            if self._react_protocol_observation(messages, content):
                continue

            if "<final_answer>" in content:
                # Step R2：模型想收口时，先注入 evidence ledger 做证据约束。
                # 这样最终答案只能基于真实 observation，减少凭空编造。
                # final_answer 前注入 evidence ledger，约束最终回答只能引用真实 observation。
                final_match = re.search(r"<final_answer>(.*?)</final_answer>", content, re.DOTALL)
                if evidence_ledger and not evidence_ledger_injected:
                    # 第一次看到 final_answer 时先不直接返回，先要求模型按证据清单修正。
                    self._append_evidence_ledger_observation(messages, evidence_ledger)
                    messages.append({
                        "role": "user",
                        "content": (
                            "<observation>请根据上面的已观察证据清单修正 final_answer。"
                            "证据来源必须与清单一致；如果 read_file 结果已截断，不要引用未观察到的具体行号或未显示代码。"
                            "现在请只输出 <thought>...</thought> 和 <final_answer>...</final_answer>。</observation>"
                        ),
                    })
                    evidence_ledger_injected = True
                    continue
                return final_match.group(1).strip() if final_match else "任务完成"

            # Step R3：没有 final_answer，就提取本轮唯一 action。
            action_matches = re.findall(r"<action>(.*?)</action>", content, re.DOTALL)
            if not action_matches:
                # 格式错误也作为 observation 回填，让模型自己修正下一轮输出格式。
                print("\n\n格式错误：缺少 <action> 标签，要求模型重试...")
                messages.append({"role": "user", "content": "<observation>格式错误：你必须输出 <action>...</action> 标签，请重新按格式输出。</observation>"})
                continue
            if len(action_matches) > 1:
                print("\n\n格式错误：一次输出了多个 <action>，要求模型重试...")
                messages.append({"role": "user", "content": "<observation>格式错误：每轮只能输出一个 <action>...</action>。请只选择下一步最重要的一个工具调用，并重新输出。</observation>"})
                continue

            action = action_matches[0].strip()
            try:
                # Step R4：解析 action，仍然不执行。
                # action 是模型输出的文本，例如 read_file("agent.py")，这里解析成函数名和参数。
                tool_name, args, kwargs = self.parse_action(action)
            except Exception as e:
                print(f"\n\n⚠️ Action 格式解析失败：{e}，反馈重试...")
                messages.append({"role": "user", "content": f"<observation>Action 格式解析失败：{e}。请确保格式为 tool_name(\"arg1\", \"arg2\") 或 tool_name(name=\"value\")。</observation>"})
                continue

            # Step R5：通过工具网关执行 action，并把结果作为 observation 回填。
            observation, should_stop = self._run_tool_with_hooks(
                tool_name,
                args,
                kwargs,
                messages,
                available_tools=available_tools,
                cancel_message="操作被用户取消",
                evidence_ledger=evidence_ledger,
                read_file_cache=read_file_cache,
            )
            if should_stop:
                print("\n\n操作已取消。")
                return observation
            # ReAct 的灵活性需要失败预算约束，否则模型可能在坏工具上打转。
            recovery_result = self._recover_from_tool_failure(tool_name, observation, tool_failures, messages)
            if recovery_result is not None:
                return recovery_result

        # Step R6：达到最大轮数后强制收口。
        return self._finalize_after_round_limit(messages, "ReAct 循环达到最大执行轮数", evidence_ledger=evidence_ledger)

    def _finalize_after_round_limit(self, messages: list, limit_message: str, evidence_ledger: list[dict[str, Any]] | None = None) -> str:
        if evidence_ledger:
            self._append_evidence_ledger_observation(messages, evidence_ledger)
        # Step R6.1：轮数耗尽后的强制收口。
        # 明确告诉模型不要再调用工具，只能基于已有 observation 输出 final_answer。
        # 达到轮数上限时强制收口，要求模型基于已有 observation 给出有限结论。
        self._append_observation(
            messages,
            (
                f"{limit_message}。工具调查预算已经用完。"
                "请不要再调用工具；必须基于已有 observation 输出 <final_answer>。"
                "如果信息不足，请在 final_answer 中明确说明缺失信息和下一步建议。"
            ),
        )
        content = self.dispatch_model(messages)
        final_match = re.search(r"<final_answer>(.*?)</final_answer>", content, re.DOTALL)
        if final_match:
            return final_match.group(1).strip()
        return limit_message

    def _compress_history(self, messages: list):
        non_system = [m for m in messages if m["role"] != "system"]
        if len(non_system) <= self.MAX_HISTORY_MESSAGES:
            return
        system_msgs = [m for m in messages if m["role"] == "system"]
        first_user = next((m for m in messages if m["role"] == "user"), None)
        recent = non_system[-(self.MAX_HISTORY_MESSAGES // 2):]
        kept = system_msgs + ([first_user] if first_user and first_user not in recent else []) + recent
        messages[:] = kept
        print(f"\n\n🗜️ 历史已压缩，保留 {len(messages)} 条消息")

    def _append_observation(self, messages: list, observation: str):
        # Step O：Observation 回填。
        # 工具结果、格式错误、策略阻断等都会变成 <observation>，让模型下一轮基于真实反馈继续。
        messages.append({"role": "user", "content": f"<observation>{observation}</observation>"})
        self._append_run_log("Observation", observation)

    def _run_tool_with_hooks(
        self,
        tool_name: str,
        args: list,
        kwargs: dict[str, Any] | None,
        messages: list,
        available_tools: dict | None = None,
        cancel_message: str = "操作被用户取消",
        evidence_ledger: list[dict[str, Any]] | None = None,
        read_file_cache: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[str, bool]:
        # Step T：统一工具调用网关。
        # parse_action 之后，所有真实工具调用都必须经过这里；这是权限、安全、审计的中心。
        # 工具调用网关：所有 action 解析后的真实工具调用都集中经过这里。
        # 统一处理 trace action -> policy 审批/阻断 -> 路径校验 -> hook -> 执行工具 -> observation。
        # 这层网关保证模型不能绕过权限、审计和 hook 直接操作环境。
        # 例如模型输出 run_terminal_command("rm -rf tmp") 时：
        # parse_action 只会解析出工具名和参数；这里的 ToolPolicy 会先匹配危险命令 deny_patterns。
        # 命中后直接返回 blocked observation，不会走到下面真正调用 tool_map[tool_name](...) 的位置。
        tool_map = available_tools or self.tools
        kwargs = kwargs or {}
        action_call = self._format_action_call(tool_name, args, kwargs)
        # Step T1：记录 action trace。
        # 先记录“模型想调用什么”，即使后面被阻断，也能审计。
        self._record_trace_event(
            "action",
            action_call,
            label="Action",
            tool_name=tool_name,
            metadata={"args": args, "kwargs": kwargs},
        )

        # Step T2：检查工具是否存在。
        # 模型可能幻觉工具名；不存在时返回 observation，而不是让程序崩溃。
        if tool_name not in tool_map:
            # 不存在的工具也要回填 observation，而不是让异常直接中断整个 Agent。
            available = ', '.join(tool_map.keys())
            observation = (
                f"工具 '{tool_name}' 不存在，可用工具：{available}。"
                "注意：技能名不是工具名；如果当前任务依赖某个技能，请遵循已加载的技能正文，"
                "或调用 load_skill(\"skill-name\") 读取技能说明后，再使用现有工具完成任务。"
            )
            self._append_observation(messages, observation)
            return observation, False

        # Step T3：Tool Permission Policy 执行前决策。
        # 这里判断风险等级、deny_patterns、allowed_roots；被阻断就不会执行真实工具。
        policy_decision = self._get_tool_policy().evaluate(tool_name, args, kwargs, self.project_directory)
        if policy_decision.blocked:
            # 高风险或命中 deny pattern 的工具调用会被直接阻断，例如危险删除命令。
            # 注意：被阻断时只把原因作为 observation 回填给模型，不执行真实工具函数。
            self._record_tool_policy_decision(policy_decision, action_call)
            observation = f"Tool policy blocked {tool_name}: {policy_decision.blocked_reason}"
            print(f"\n\n Observation: {observation}")
            self._append_observation(messages, observation)
            return observation, False

        if policy_decision.require_approval:
            # Step T4：高风险工具人工确认。
            # 例如普通终端命令没命中黑名单，但仍需要用户 Y/N 批准。
            # 高风险但未直接阻断的工具调用，需要用户确认，并把审批结果写入 trace。
            approval_prompt = f"\n\nTool policy requires approval for {tool_name} (risk={policy_decision.risk}). Continue? (Y/N)"
            approval_response = input(approval_prompt).strip().lower()
            policy_decision.approved = approval_response in {"y", "yes"}
            self._record_tool_policy_decision(policy_decision, action_call)
            if not policy_decision.approved:
                return cancel_message, True
        else:
            policy_decision.approved = True
            self._record_tool_policy_decision(policy_decision, action_call)

        file_path = self._extract_path_argument(tool_name, args, kwargs)
        if file_path is not None:
            # Step T5：文件路径安全边界。
            # 文件类工具必须限制在 project_directory 内，防止越权读写。
            # 文件类工具再做一层项目目录边界检查，避免 Agent 读写项目外路径。
            if not self._validate_path(str(file_path)):
                observation = f"路径 '{file_path}' 不在项目目录内，只允许操作 {self.project_directory} 下的文件"
                print(f"\n\n Observation：{observation}")
                self._append_observation(messages, observation)
                return observation, False
            duplicate_observation = self._maybe_block_duplicate_read_file(tool_name, file_path, args, kwargs, read_file_cache)
            if duplicate_observation is not None:
                # 重复读取会浪费轮数，也容易让模型陷入无效探索，直接提示基于已有结果总结。
                print(f"\n\n Observation：{duplicate_observation}")
                self._append_observation(messages, duplicate_observation)
                return duplicate_observation, False

        # Step T6：PreToolUse Hook。
        # Hook 可以在工具执行前阻断或追加 observation。
        pre = self.hook_runner.run("PreToolUse", {
            "tool_name": tool_name,
            "input": {"args": args, "kwargs": kwargs},
        })
        # Hook 是轻量扩展点：工具执行前后可以拦截、补充日志或修改 observation。
        if pre["exit_code"] == 1:
            observation = pre["message"] or f"Hook 阻止了工具 {tool_name} 的执行"
            print(f"\n\n Observation：{observation}")
            self._append_observation(messages, observation)
            return observation, False
        if pre["exit_code"] == 2 and pre["message"]:
            self._append_observation(messages, pre["message"])

        print(f"\n\n Action: {action_call}")
        self._append_run_log("Action", action_call)

        # Step T7：真正执行工具函数。
        # 只有通过前面的 policy、审批、路径和 hook 后，才会抵达这一行。
        tool_start = time.perf_counter()
        try:
            # 到这里才真正调用本地工具函数；前面所有步骤都是校验和审计。
            # 因此危险 action 如果在 policy/hook 阶段被拦截，就不会抵达这一行。
            observation = tool_map[tool_name](*args, **kwargs)
        except Exception as e:
            observation = f"工具执行错误：{str(e)}"

        latency_ms = (time.perf_counter() - tool_start) * 1000

        # Step T8：PostToolUse Hook。
        # 工具执行后可以做日志、结果改写或追加信息。
        post = self.hook_runner.run("PostToolUse", {
            "tool_name": tool_name,
            "input": {"args": args, "kwargs": kwargs},
            "output": observation,
        })
        if post["exit_code"] == 1:
            observation = post["message"] or observation
        elif post["exit_code"] == 2 and post["message"]:
            observation = f"{observation}\n\n{post['message']}"

        print(f"\n\n Observation：{observation}")
        self._record_trace_event(
            "tool_result",
            str(observation),
            label="Tool Result",
            tool_name=tool_name,
            latency_ms=latency_ms,
            metadata={"args": args, "kwargs": kwargs, "action": action_call},
        )
        # Step T9：把工具结果作为 observation 回填给模型，并记录 evidence。
        # ReAct 下一轮会基于这个 observation 决定继续调用工具还是输出 final_answer。
        self._append_observation(messages, observation)
        # observation 会回填给模型，同时作为 evidence/trace 的来源，方便后续调试和评估。
        self._record_tool_evidence(evidence_ledger, tool_name, args, kwargs, observation)
        self._remember_read_file_observation(read_file_cache, tool_name, file_path, args, kwargs, observation)
        return observation, False

    def _get_arg_value(self, args: list, kwargs: dict[str, Any], name: str, position: int, default: Any = None) -> Any:
        if name in kwargs:
            return kwargs[name]
        if len(args) > position:
            return args[position]
        return default

    def _display_evidence_path(self, file_path: str) -> str:
        abs_path = os.path.abspath(str(file_path))
        project_directory = os.path.abspath(self.project_directory)
        try:
            relative = os.path.relpath(abs_path, project_directory)
            if not relative.startswith("..") and not os.path.isabs(relative):
                return relative.replace(os.sep, "/")
        except ValueError:
            pass
        return str(file_path)

    def _read_file_budget(self, args: list, kwargs: dict[str, Any]) -> int | None:
        raw_budget = self._get_arg_value(args, kwargs, "max_chars", 1, 40 * 1024)
        try:
            budget = int(raw_budget)
        except (TypeError, ValueError):
            return None
        return budget if budget > 0 else None

    def _read_file_cache_key(self, file_path: str) -> str:
        return os.path.abspath(str(file_path))

    def _maybe_block_duplicate_read_file(
        self,
        tool_name: str,
        file_path: str,
        args: list,
        kwargs: dict[str, Any],
        read_file_cache: dict[str, dict[str, Any]] | None,
    ) -> str | None:
        if tool_name != "read_file" or read_file_cache is None:
            return None
        budget = self._read_file_budget(args, kwargs)
        if budget is None:
            return None
        cache_key = self._read_file_cache_key(str(file_path))
        previous = read_file_cache.get(cache_key)
        if not previous:
            return None
        previous_budget = int(previous.get("max_chars") or 0)
        if budget > previous_budget:
            return None
        display_path = previous.get("path") or self._display_evidence_path(str(file_path))
        return (
            f"文件 '{display_path}' 已在本任务中读取过（已读 max_chars={previous_budget}）。"
            f"本次 max_chars={budget} 未超过已读预算，已阻止重复读取；"
            "请基于已有 observation 总结，不要重复读取。"
        )

    def _remember_read_file_observation(
        self,
        read_file_cache: dict[str, dict[str, Any]] | None,
        tool_name: str,
        file_path: str | None,
        args: list,
        kwargs: dict[str, Any],
        observation: str,
    ) -> None:
        if read_file_cache is None or tool_name != "read_file" or file_path is None:
            return
        if self._is_tool_failure(str(observation)):
            return
        budget = self._read_file_budget(args, kwargs)
        if budget is None:
            return
        cache_key = self._read_file_cache_key(str(file_path))
        previous = read_file_cache.get(cache_key)
        if previous and int(previous.get("max_chars") or 0) >= budget:
            return
        read_file_cache[cache_key] = {
            "path": self._display_evidence_path(str(file_path)),
            "max_chars": budget,
        }

    def _record_tool_evidence(
        self,
        evidence_ledger: list[dict[str, Any]] | None,
        tool_name: str,
        args: list,
        kwargs: dict[str, Any],
        observation: str,
    ) -> None:
        if evidence_ledger is None:
            return
        entry: dict[str, Any] | None = None
        if tool_name == "read_file":
            file_path = self._get_arg_value(args, kwargs, "file_path", 0, "")
            entry = {
                "tool": "read_file",
                "target": self._display_evidence_path(str(file_path)),
                "max_chars": self._read_file_budget(args, kwargs),
                "truncated": self._is_truncated_read_observation(str(observation)),
            }
        elif tool_name == "list_directory":
            path = self._get_arg_value(args, kwargs, "path", 0, "")
            max_entries = self._get_arg_value(args, kwargs, "max_entries", 1, None)
            entry = {
                "tool": "list_directory",
                "target": self._display_evidence_path(str(path)),
                "max_entries": max_entries,
                "truncated": "结果过多" in str(observation),
            }
        elif tool_name == "search_in_files":
            keyword = self._get_arg_value(args, kwargs, "keyword", 0, "")
            directory = self._get_arg_value(args, kwargs, "directory", 1, "")
            max_results = self._get_arg_value(args, kwargs, "max_results", 2, None)
            entry = {
                "tool": "search_in_files",
                "target": f"{keyword!r} in {self._display_evidence_path(str(directory))}",
                "max_results": max_results,
                "truncated": "结果过多" in str(observation),
            }
        if entry is not None:
            evidence_ledger.append(entry)

    def _is_truncated_read_observation(self, observation: str) -> bool:
        return "仅显示前" in observation and "完整文件约" in observation

    def _append_evidence_ledger_observation(self, messages: list, evidence_ledger: list[dict[str, Any]]) -> None:
        if not evidence_ledger:
            return
        # 证据清单只记录工具真实返回过的范围，用来约束最终答案不要补写未观察内容。
        lines = ["已观察证据清单（框架自动记录，最终报告必须以此为准）："]
        for index, entry in enumerate(evidence_ledger, 1):
            tool_name = entry.get("tool", "")
            target = entry.get("target", "")
            parts = [f"{index}. {tool_name}: {target}"]
            if entry.get("max_chars") is not None:
                parts.append(f"max_chars={entry['max_chars']}")
            if entry.get("max_entries") is not None:
                parts.append(f"max_entries={entry['max_entries']}")
            if entry.get("max_results") is not None:
                parts.append(f"max_results={entry['max_results']}")
            if tool_name == "read_file":
                if entry.get("truncated"):
                    parts.append("状态=已截断，只能引用已显示内容；不要引用未观察到的具体行号或未显示代码")
                else:
                    parts.append("状态=未截断")
            elif entry.get("truncated"):
                parts.append("状态=结果被截断")
            lines.append("；".join(parts))
        lines.append(
            "证据使用规则：不要把 read_file 结果写成 search_in_files；"
            "只有 search_in_files observation 中出现的行号才可当作行号证据；"
            "被截断的 read_file 结果不能支持未显示代码的精确断言。"
        )
        ledger_text = "\n".join(lines)
        self._record_trace_event(
            "evidence_ledger",
            ledger_text,
            label="Evidence Ledger",
            metadata={"entries": evidence_ledger},
        )
        self._append_observation(messages, ledger_text)

    def _is_tool_failure(self, observation: str) -> bool:
        if observation.startswith(self.TOOL_FAILURE_PREFIXES):
            return True
        return (
            (observation.startswith("路径 '") and "不在项目目录内" in observation)
            or (observation.startswith("工具 '") and "不存在" in observation)
        )

    def _is_step_result_failure(self, result: str) -> bool:
        failure_markers = (
            "步骤达到最大执行轮数",
            "ReAct 循环达到最大执行轮数",
            "工具执行错误：",
            "工具 '",
            "不在项目目录内",
            "操作被用户取消",
            "步骤被用户取消",
            "子任务被取消",
        )
        return any(marker in result for marker in failure_markers)

    def _recover_from_tool_failure(self, tool_name: str, observation: str, failure_counts: dict[str, int], messages: list) -> str | None:
        if not self._is_tool_failure(observation):
            return None
        failure_counts[tool_name] = failure_counts.get(tool_name, 0) + 1
        if failure_counts[tool_name] < 2:
            return None
        # Step F：工具连续失败恢复。
        # 同一工具连续失败两次后，要求模型不要机械重试，而是总结或说明缺失信息。
        # 连续失败两次后不再机械重试，把控制权交回模型做降级总结或说明缺口。
        recovery_observation = (
            f"工具 '{tool_name}' 已连续失败 {failure_counts[tool_name]} 次。"
            f"最后一次错误：{observation}。"
            "不要继续重试同一个工具或同类失败工具；如果已有信息足够，请直接输出 <final_answer>；否则明确说明缺失信息和下一步建议。"
        )
        self._append_observation(messages, recovery_observation)
        content = self.dispatch_model(messages)
        final_match = re.search(r"<final_answer>(.*?)</final_answer>", content, re.DOTALL)
        if final_match:
            return final_match.group(1).strip()
        return recovery_observation

    def task(self, prompt: str) -> str:
        print(f"\n\n🔹 派生子智能体：{prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        subagent = SubagentContext(prompt=prompt, tools=self.subagent_tools, agent=self)
        result = subagent.run()
        print(f"\n\n🔹 子智能体返回：{result[:100]}{'...' if len(result) > 100 else ''}")
        return result

    def load_skill(self, name: str) -> str:
        # 这里是 Agent 暴露给模型的工具入口，真正的 Skill 解析在 skills.py。
        print(f"\n\n📚 加载技能：{name}")
        return self.skill_registry.load_skill(name)

    def save_memory(self, name: str, description: str, mem_type: str, content: str) -> str:
        """
        保存长期记忆。仅当用户明确要求记住稳定偏好、项目事实、反馈或参考信息时使用。
        不要保存一次性任务过程、未经验证的推断、命令输出或敏感信息。
        mem_type 必须是 user、feedback、project、reference 之一。
        """
        print(f"\n\n🧠 保存长期记忆：{name} [{mem_type}]")
        result = self.memory_store.save_memory(name, description, mem_type, content)
        # 保存后立即重建当前 prompt 可用的 memory section，后续轮次可以读到新记忆。
        self.current_memory_section = self.memory_store.build_memory_section()
        return result

    def _is_general_question(self, user_input: str) -> bool:
        lowered = user_input.lower()
        project_markers = (
            "代码", "文件", "目录", "项目", "仓库", "函数", "类", "模块", "bug", "报错",
            "测试", "运行", "实现", "修改", "重构", "调试", "read_file", "write_to_file",
            ".py", "agent.py", "tools.py", "搜索项目", "查看仓库",
        )
        general_markers = (
            "什么", "如何", "为什么", "建议", "学习", "知识", "路线", "介绍", "区别",
            "原理", "概念", "怎么", "需要补充", "我想转", "适合", "总结",
        )
        return any(marker in user_input or marker in lowered for marker in general_markers) and not any(marker in user_input or marker in lowered for marker in project_markers)

    def _should_skip_planning(self, user_input: str) -> bool:
        return self._is_general_question(user_input)

    def _direct_answer(self, user_input: str) -> str:
        messages = [
            {"role": "system", "content": self.render_system_prompt(direct_answer_system_prompt_template)},
            {"role": "user", "content": f"<question>{user_input}</question>"},
        ]
        content = self.dispatch_model(messages)
        final_match = re.search(r"<final_answer>(.*?)</final_answer>", content, re.DOTALL)
        if final_match:
            return final_match.group(1).strip()
        return content.strip()

    def _is_multi_agent_request(self, user_input: str) -> bool:
        lowered = user_input.lower()
        markers = ("multi-agent", "multi agent", "多 agent", "多智能体", "队友", "researcher", "coder", "tester")
        return any(marker in lowered or marker in user_input for marker in markers)

    def _build_prompt_tool_map(self, user_input: str, selected_skill: dict | None = None, hinted_skill: dict | None = None) -> dict[str, Callable]:
        # 每次任务动态决定可见工具，避免模型在不相关场景里乱用高阶能力。
        if self._is_general_question(user_input):
            general_tools = {
                name: func
                for name, func in self.tools.items()
                if name in self.GENERAL_QUESTION_TOOLS
            }
            return general_tools or dict(self.tools)

        hidden_tools = set(self.ALWAYS_HIDDEN_PROMPT_TOOLS)
        if self._is_memory_save_request(user_input):
            # 用户明确表达要记住时，才把 save_memory 暴露给模型。
            hidden_tools.discard("save_memory")
        if not self._is_multi_agent_request(user_input):
            hidden_tools.update(self.TEAM_PROMPT_TOOLS)
        if not selected_skill and not hinted_skill:
            # 没有命中 Skill 时不暴露 load_skill，减少模型无目的加载技能。
            hidden_tools.add("load_skill")

        return {
            name: func
            for name, func in self.tools.items()
            if name not in hidden_tools
        }

    def _extract_path_argument(self, tool_name: str, args: list, kwargs: dict[str, Any]) -> str | None:
        if tool_name in ("read_file", "write_to_file"):
            if "file_path" in kwargs:
                return str(kwargs["file_path"])
            if args:
                return str(args[0])
        if tool_name == "list_directory":
            if "path" in kwargs:
                return str(kwargs["path"])
            if args:
                return str(args[0])
        if tool_name == "search_in_files":
            if "directory" in kwargs:
                return str(kwargs["directory"])
            if len(args) >= 2:
                return str(args[1])
        return None

    def _format_action_call(self, tool_name: str, args: list, kwargs: dict[str, Any]) -> str:
        parts = [repr(arg) for arg in args]
        parts.extend(f"{key}={repr(value)}" for key, value in kwargs.items())
        return f"{tool_name}({', '.join(parts)})"

    def spawn_teammate(self, name: str, role: str, prompt: str) -> str:
        return self.team_manager.spawn(name, role, prompt)

    def list_teammates(self) -> str:
        return self.team_manager.list_teammates()

    def send_message(self, teammate: str, content: str, msg_type: str = "message") -> str:
        return self.team_manager.send_message("lead", teammate, content, msg_type)

    def broadcast_message(self, content: str, msg_type: str = "message") -> str:
        return self.team_manager.broadcast_message("lead", content, msg_type)

    def read_team_inbox(self, name: str = "lead") -> str:
        return self.team_manager.read_inbox(name)

    def _load_mcp_tools(self) -> dict[str, Callable]:
        # MCP server 负责暴露外部工具，Registry 把它们包装成本地可调用函数。
        tool_specs = self.mcp_client_manager.load_servers(self.mcp_server_configs)
        return self.mcp_registry.load_tools(tool_specs)

    def _is_mcp_tool(self, tool_name: str) -> bool:
        # 子 Agent 默认不继承 MCP 工具，避免外部能力被不受控地层层传递。
        return self.mcp_registry.is_mcp_tool(tool_name)

    def _get_mcp_status_lines(self) -> list[str]:
        states = self.mcp_client_manager.get_server_states()
        connected = sum(1 for state in states if state.connected)
        lines = [
            "# MCP Status",
            f"- 已配置 server：{len(self.mcp_server_configs)}",
            f"- 已连接 server：{connected}",
            f"- 已加载 MCP tools：{len(self.mcp_registry.get_tool_specs())}",
        ]
        for state in states:
            status = "connected" if state.connected else "disconnected"
            detail = f", last_error={state.last_error}" if state.last_error else ""
            lines.append(f"- server {state.name}: {status}, tools={state.tool_count}{detail}")
        return lines

    def get_status(self) -> str:
        members = self.team_manager.config.get("members", [])
        active = [member for member in members if member["status"] == "working"]
        idle = [member for member in members if member["status"] == "idle"]
        shutdown = [member for member in members if member["status"] == "shutdown"]
        pending_shutdown = sum(1 for request in self.team_manager.shutdown_requests.values() if request["status"] == "pending")
        pending_plan = sum(1 for request in self.team_manager.plan_requests.values() if request["status"] == "pending")
        backend = self.get_backend_name()
        lines = [
            "# Agent Status",
            f"- 模型后端：{backend}",
            "- Multi-agent 支持：已启用",
            f"- 队友总数：{len(members)}",
            f"- working：{len(active)}",
            f"- idle：{len(idle)}",
            f"- shutdown：{len(shutdown)}",
            f"- 待处理 shutdown 请求：{pending_shutdown}",
            f"- 待审批计划：{pending_plan}",
            "- 可用团队命令：/status, /team, /inbox [name]",
        ]
        lines.extend(["", *self._get_mcp_status_lines()])
        return "\n".join(lines)

    def get_multi_agent_usage_guide(self) -> str:
        return (
            "\n🤝 Multi-agent 使用说明\n"
            "- 当前版本已支持 multi-agent；只有创建队友后，才会进入实际团队协作。\n"
            "- 想触发 multi-agent，可直接说：请用多 agent 模式处理这个任务，并创建 researcher/coder/tester 队友。\n"
            "- 查看当前状态：/status\n"
            "- 查看团队成员：/team\n"
            "- 查看收件箱：/inbox 或 /inbox alice\n"
            "- 如需接入外部 MCP server，请在项目目录下准备 .mcp/config.json\n"
        )

    def request_shutdown(self, teammate: str) -> str:
        return self.team_manager.request_shutdown(teammate)

    def review_plan(self, request_id: str, approve: bool, feedback: str = "") -> str:
        return self.team_manager.review_plan(request_id, approve, feedback)

    def _handle_special_command(self, user_input: str) -> str | None:   #判断特殊命令（状态等），可直接返回
        if not user_input.startswith("/"):   #不是以/为开头，一定不是特殊命令
            return None
        parts = user_input.split(None, 1)
        command = parts[0].lower()
        if command == "/status":
            return self.get_status()
        if command == "/team":
            return self.list_teammates()
        if command == "/inbox":
            target = parts[1].strip() if len(parts) > 1 else "lead"
            return self.read_team_inbox(target)

        return None

    def _make_bound_tool(self, name: str, doc: str, func: Callable) -> Callable:
        func.__name__ = name
        func.__doc__ = doc
        return func

    def build_teammate_tools(self, teammate_name: str) -> dict[str, Callable]:
        excluded = {
            "task",
            "spawn_teammate",
            "list_teammates",
            "broadcast_message",
            "read_team_inbox",
            "request_shutdown",
            "review_plan",
            "send_message",
        }
        tools = {name: func for name, func in self.tools.items() if name not in excluded and not self._is_mcp_tool(name)}
        tools["send_message"] = self._make_bound_tool(
            "send_message",
            "向领导或其他队友发送消息。",
            lambda to, content, msg_type="message": self.team_manager.send_message(teammate_name, to, content, msg_type),
        )
        tools["respond_shutdown"] = self._make_bound_tool(
            "respond_shutdown",
            "响应领导发来的 shutdown request。",
            lambda request_id, approve, reason="": self.team_manager.respond_shutdown(teammate_name, request_id, approve, reason),
        )
        tools["submit_plan"] = self._make_bound_tool(
            "submit_plan",
            "向领导提交计划审批请求。",
            lambda plan: self.team_manager.submit_plan(teammate_name, plan),
        )
        return tools

    def _should_ignore_memory(self, user_input: str) -> bool:
        # 给用户一个显式逃生口：本次任务不参考长期记忆。
        lowered = user_input.lower()
        return (
            "ignore memory" in lowered
            or "without memory" in lowered
            or "no memory" in lowered
            or "忽略 memory" in user_input
            or "忽略之前的记忆" in user_input
            or "不要参考 memory" in user_input
            or "不要使用 memory" in user_input
            or "不要使用记忆" in user_input
            or "别参考记忆" in user_input
            or "忽略之前的memory" in lowered
        )

    def _is_memory_save_request(self, user_input: str) -> bool:
        # 用简单关键词判断保存意图，只在明确表达长期保存时开放 save_memory。
        lowered = user_input.lower()
        markers = (
            "remember this",
            "save memory",
            "save this memory",
            "记住",
            "帮我记住",
            "保存为 memory",
            "保存到 memory",
            "保存长期记忆",
            "以后都",
            "下次也",
        )
        return any(marker in lowered or marker in user_input for marker in markers)

    def _memory_ignored_section(self) -> str:
        return (
            "- 用户本次要求忽略长期记忆，未注入 .memory/ 中的内容。\n"
            "- 如果用户后续没有要求忽略，下一次任务会重新从 .memory/ 构建 memory section。"
        )

    def _load_memory_section(self, user_input: str) -> str:
        # memory 每次任务重新构建，保证 .memory/ 的改动能及时进入 prompt。
        if self._should_ignore_memory(user_input):
            return self._memory_ignored_section()
        return self.memory_store.build_memory_section()

    def get_skill_list(self) -> str:
        """返回轻量技能目录，供系统提示词常驻展示"""
        return self.skill_registry.describe_available()

    # 给 AI 生成工具使用说明书
    def get_tool_list(self, tool_map: dict | None = None) -> str:
        """生成工具列表字符串，包含函数签名和简要说明"""
        tool_descriptions = []
        tools = tool_map or self.tools
        for func in tools.values():
            name = func.__name__
            signature = str(inspect.signature(func))
            doc = inspect.getdoc(func)
            tool_descriptions.append(f"- {name}{signature}: {doc}")
        return "\n".join(tool_descriptions)
    
    # 返回一段完整 Ready-to-use 的 AI 系统提示词
    def render_system_prompt(self, system_prompt_template: str, tool_map: dict | None = None, extra_vars: dict | None = None) -> str:
        """渲染系统提示模板，替换变量"""
        """
        os.listdir(self.project_directory)  列出项目文件夹里所有文件名
        os.path.join(路径, 文件名)
        转成绝对路径(完整路径，如 /user/project/main.py)
        """
        tool_list = self.get_tool_list(tool_map)
        skill_list = self.get_skill_list()
        file_list = ", ".join(
            os.path.abspath(os.path.join(self.project_directory, f))
            for f in os.listdir(self.project_directory)
        )
        # 把一段带占位符的模板字符串，填上真实数据
        variables = dict(
            operating_system=self.get_operating_system_name(),
            tool_list=tool_list,
            skill_list=skill_list,
            memory_section=getattr(self, "current_memory_section", "- 暂无可用长期记忆"),
            file_list=file_list
        )
        if extra_vars:
            variables.update(extra_vars)
        return Template(system_prompt_template).substitute(variables)
        
    def _is_minimax_model(self) -> bool:
        model_name = self.model.lower()
        return model_name.startswith("minimax") or model_name.startswith("codex-minimax")

    def _minimax_model_name(self) -> str:
        if self.model.lower().startswith("minimax/"):
            return self.model.split("/", 1)[1]
        return self.model

    def get_minimax_api_key(self) -> str:
        load_dotenv()
        api_key = os.getenv("MINIMAX_API_KEY")
        if not api_key:
            raise ValueError("MINIMAX_API_KEY is missing. Add it to .env or set it in your shell.")
        return api_key

    def get_minimax_base_url(self) -> str:
        load_dotenv()
        return os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1").rstrip("/")

    def get_backend_name(self) -> str:
        return f"MiniMax API ({self.get_minimax_base_url()})"
    
    def dispatch_model(self, messages):
        """MiniMax-only dispatch."""
        if not self._is_minimax_model():
            raise ValueError(
                f"Model '{self.model}' is not supported. This runtime only allows MiniMax models."
            )
        return self.call_minimax_model(messages)

    def call_minimax_model(self, messages):
        print("\n\nMiniMax ", end="", flush=True)
        response = self.minimax_client.chat.completions.create(
            model=self._minimax_model_name(),
            messages=messages,
            temperature=0.7,
        )
        content = response.choices[0].message.content or ""
        print(content)
        messages.append({"role": "assistant", "content": content})
        return content
    
    def parse_action(self, code_str: str) -> Tuple[str, list[Any], dict[str, Any]]:
        # Step A：Action Parsing。
        # 把模型输出的 action 文本解析成 tool_name / args / kwargs；注意这里只解析，不执行。
        # Action 解析：把模型输出的 action 文本解析成真实函数调用。
        # 这里只接受“函数调用表达式”，例如 read_file("agent.py")，避免执行任意代码。
        # 注意这里不 eval 模型输出，只解析 AST 和字面量参数。
        # 例：run_terminal_command("rm -rf tmp") 会变成：
        # tool_name="run_terminal_command", args=["rm -rf tmp"], kwargs={}。
        # 是否允许执行由 _run_tool_with_hooks + ToolPermissionPolicy 决定。
        try:
            # Step A1：用 AST 解析表达式，避免 eval 执行模型生成的任意代码。
            expression = ast.parse(code_str, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid function call syntax: {exc.msg}") from exc

        call = expression.body
        # Step A2：只允许普通函数调用，例如 read_file("README.md")。
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            raise ValueError("Action 必须是函数调用，例如 tool_name(\"arg\")")

        # Step A3：提取函数名、位置参数和关键字参数。
        # 参数优先用 ast.literal_eval 解析，仍然不会执行代码。
        func_name = call.func.id
        args = [self._parse_action_value(code_str, arg) for arg in call.args]
        kwargs: dict[str, Any] = {}
        for keyword in call.keywords:
            if keyword.arg is None:
                raise ValueError("暂不支持 **kwargs 展开")
            kwargs[keyword.arg] = self._parse_action_value(code_str, keyword.value)

        return func_name, args, kwargs

    def _parse_action_value(self, source: str, node: ast.AST):
        try:
            return ast.literal_eval(node)
        except (ValueError, SyntaxError):
            raw = ast.get_source_segment(source, node)
            if raw is None:
                raise ValueError("参数必须是可解析的字面量")
            return self._parse_single_arg(raw)
    
    def _parse_single_arg(self, arg_str: str):
        """解析单个参数"""
        arg_str = arg_str.strip()
        # 如果是字符串字面量
        if (arg_str.startswith('"') and arg_str.endswith('"')) or \
           (arg_str.startswith("'") and arg_str.endswith("'")):
            # 移除外层引号并处理转义字符
            inner_str = arg_str[1:-1]
            # 处理常见的转义字符
            inner_str = inner_str.replace('\\"', '"').replace("\\'", "'")
            inner_str = inner_str.replace('\\n', '\n').replace('\\t', '\t')
            inner_str = inner_str.replace('\\r', '\r').replace('\\\\', '\\')
            return inner_str
        
        # 尝试使用 ast.literal_eval 解析其他类型
        try:
            return ast.literal_eval(arg_str)
        except (SyntaxError, ValueError):
            # 如果解析失败，返回原始字符串
            return arg_str
        
    def _validate_path(self, file_path: str) -> bool:
        # Step T5.1：Agent 层路径校验。
        # 这是 ToolPolicy allowed_roots 之外的第二层项目目录边界。
        """校验文件路径是否在项目目录内，防止越权访问"""
        abs_path = os.path.abspath(file_path)
        project_directory = os.path.abspath(self.project_directory)
        try:
            return os.path.commonpath([project_directory, abs_path]) == project_directory
        except ValueError:
            return False

    def get_operating_system_name(self):
        os_map = {
            "Darwin": "macOS",
            "Windows": "Windows",
            "Linux": "Linux"
        }

        return os_map.get(platform.system(), "Unknown")

@click.command()
@click.argument('project_directory',
                type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--model', default='minimax/MiniMax-M2.7',
              show_default=True,
              help='MiniMax model name. Example: minimax/MiniMax-M2.7')
@click.option('--trace-db',
              type=click.Path(dir_okay=False, resolve_path=True),
              default=None,
              help='Optional central trace SQLite path. Falls back to AGENT_TRACE_DB, then project-local .runs/traces.sqlite3')
def main(project_directory, model, trace_db):
    project_dir = os.path.abspath(project_directory)   #用户输入的路径变绝对路径
    if os.name == "nt":
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")  #如果当前系统是 Windows，就把终端输出和错误输出的编码改成 UTF-8，避免中文乱码或编码报错。

    tools = [read_file, write_to_file, run_terminal_command, list_directory, search_in_files, web_search, query_knowledge_base]
    agent = ReActAgent(
        tools=tools,
        model=model,
        project_directory=project_dir,
        trace_db_path=trace_db,
    )

    backend = agent.get_backend_name()
    print("\n🤖 Agent 已启动，输入 'exit' 或 'quit' 退出对话")
    print(f"🧠 模型：{model}  ({backend})")
    print(f"📁 工作目录：{project_dir}")
    print(f"🗂️ Trace DB：{agent.trace_db_path}")
    print("=" * 50)

    while True:
        try:
            task = input("\n请输入任务：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 已退出")
            break

        if not task:
            continue
        if task.lower() in ("exit", "quit", "q", "退出"):
            print("\n\n👋 已退出")
            break

        final_answer = agent.run(task)
        print(f"\n\n✅ Final Answer：{final_answer}")
        print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
