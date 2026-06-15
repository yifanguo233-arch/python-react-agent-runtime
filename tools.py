import os
import re
import subprocess


MAX_READ_SIZE = 40 * 1024  # 40KB，超过此大小的文件只读取前部分
DEFAULT_MAX_DIRECTORY_ENTRIES = 300
DEFAULT_MAX_SEARCH_RESULTS = 50

# 工具层：Agent 和真实环境交互的接口。
# 模型只能输出 action 文本，真正的文件读写、搜索、命令执行都在这里实现。

IGNORED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".runs",
    ".team",
    ".venv",
    ".idea",
    ".vscode",
    "__pycache__",
    "chroma_db",
    "node_modules",
    "tmp",
}
IGNORED_DIR_PREFIXES = (
    ".venv_",
    "tmp_",
    "chroma_db",
)

_DDGS = None
_chromadb = None
_SentenceTransformer = None


def _get_ddgs_class():
    # 联网搜索依赖按需导入，普通本地任务不需要承担这个启动成本。
    global _DDGS
    if _DDGS is None:
        from ddgs import DDGS as ImportedDDGS
        _DDGS = ImportedDDGS
    return _DDGS


def _get_rag_dependencies():
    # RAG 相关依赖比较重，只在真正查询知识库时再加载。
    global _chromadb, _SentenceTransformer
    if _chromadb is None:
        import chromadb as imported_chromadb
        _chromadb = imported_chromadb
    if _SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as ImportedSentenceTransformer
        _SentenceTransformer = ImportedSentenceTransformer
    return _chromadb, _SentenceTransformer


def _is_ignored_dir(name: str) -> bool:
    return name in IGNORED_DIRS or any(name.startswith(prefix) for prefix in IGNORED_DIR_PREFIXES)

def read_file(file_path, max_chars: int = MAX_READ_SIZE):
    # 文件读取工具：让 Agent 能基于真实项目文件回答，而不是凭模型记忆猜。
    """用于读取文件内容（默认最多读取前40KB，可用 max_chars 指定更小预算）"""
    if not file_path:
        return "文件读取失败：file_path 不能为空"
    if not os.path.exists(file_path):
        return f"文件读取失败：路径不存在：{file_path}"
    if os.path.isdir(file_path):
        return f"文件读取失败：目标是目录：{file_path}"
    try:
        max_chars = int(max_chars)
    except (TypeError, ValueError):
        return "文件读取失败：max_chars 必须是整数"
    if max_chars <= 0:
        return "文件读取失败：max_chars 必须大于 0"
    read_size = min(max_chars, MAX_READ_SIZE)
    try:
        file_size = os.path.getsize(file_path)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(read_size)
        if file_size > read_size:
            shown_kb = max(read_size // 1024, 1)
            content += f"\n\n... [文件过大，仅显示前 {shown_kb}KB，完整文件约 {file_size // 1024}KB]"
        return content
    except OSError as exc:
        return f"文件读取失败：{exc}"


def write_to_file(file_path, content):
    # 文件写入工具：Agent 修改文件时最终会落到这里；权限边界主要在 agent.py/tool_policy.py 做。
    """将指定内容写入指定文件"""
    if not file_path:
        return "文件写入失败：file_path 不能为空"
    if os.path.isdir(file_path):
        return f"文件写入失败：目标是目录：{file_path}"
    try:
        parent_dir = os.path.dirname(os.path.abspath(file_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(str(content).replace("\\n", "\n"))
        return f"写入成功：{file_path}"
    except OSError as exc:
        return f"文件写入失败：{exc}"


def run_terminal_command(command, timeout: int = 60):
    # 命令执行工具：能力强、风险也高，所以 Agent 调用前会经过 Tool Permission Policy 审批。
    # 这个函数只负责“真正执行命令”；能不能调用到这里，由 agent.py 的工具网关和 tool_policy.py 决定。
    """用于执行终端命令，timeout 为超时秒数（默认60秒）"""
    if not isinstance(command, str) or not command.strip():
        return "命令执行失败：command 不能为空"
    try:
        timeout = int(timeout)
    except (TypeError, ValueError):
        return "命令执行失败：timeout 必须是整数秒"
    if timeout <= 0:
        return "命令执行失败：timeout 必须大于 0"

    try:
        run_result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return f"命令执行超时：超过 {timeout}s，已终止"

    status = "命令执行成功" if run_result.returncode == 0 else "命令执行失败"
    stdout = run_result.stdout.rstrip() if run_result.stdout else "<empty>"
    stderr = run_result.stderr.rstrip() if run_result.stderr else "<empty>"
    return "\n".join([
        status,
        f"exit_code: {run_result.returncode}",
        f"stdout:\n{stdout}",
        f"stderr:\n{stderr}",
    ])


def list_directory(path, max_entries: int = DEFAULT_MAX_DIRECTORY_ENTRIES):
    # 目录工具：适合先了解项目结构，避免 Agent 一上来盲读文件。
    """列出指定目录下的所有文件和子目录（自动排除 .git/.venv 等目录）"""
    if not path:
        return "目录列出失败：path 不能为空"
    if not os.path.exists(path):
        return f"目录列出失败：路径不存在：{path}"
    if not os.path.isdir(path):
        return f"目录列出失败：目标不是目录：{path}"
    try:
        max_entries = int(max_entries)
    except (TypeError, ValueError):
        return "目录列出失败：max_entries 必须是整数"
    if max_entries <= 0:
        return "目录列表为空"

    result = []
    truncated = False
    base_path = os.path.abspath(path)
    for root, dirs, files in os.walk(base_path, topdown=True):
        dirs[:] = sorted(d for d in dirs if not _is_ignored_dir(d))
        files = sorted(files)
        level = root.replace(base_path, "").count(os.sep)
        indent = "  " * level
        result.append(f"{indent}{os.path.basename(root)}/")
        if len(result) >= max_entries:
            truncated = True
            break
        sub_indent = "  " * (level + 1)
        for file in files:
            result.append(f"{sub_indent}{file}")
            if len(result) >= max_entries:
                truncated = True
                break
        if truncated:
            break
    if truncated:
        result.append(f"... [结果过多，仅显示前 {max_entries} 条]")
    return "\n".join(result) if result else "目录列表为空"


def search_in_files(keyword, directory, max_results: int = DEFAULT_MAX_SEARCH_RESULTS):
    # 文本搜索工具：用于定位符号、函数名、关键词，是代码分析类任务的高频工具。
    # 这是精确关键词检索，不做 embedding；找代码符号和报错文本时比语义检索更可靠。
    """在指定目录下的所有文件中搜索包含关键词的行（自动排除 .git/.venv 等目录）"""
    if not keyword:
        return "文件搜索失败：keyword 不能为空"
    if not directory:
        return "文件搜索失败：directory 不能为空"
    if not os.path.exists(directory):
        return f"文件搜索失败：路径不存在：{directory}"
    if not os.path.isdir(directory):
        return f"文件搜索失败：目标不是目录：{directory}"
    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        return "文件搜索失败：max_results 必须是整数"
    if max_results <= 0:
        return "搜索结果为空"

    matches = []
    truncated = False
    for root, dirs, files in os.walk(directory, topdown=True):
        dirs[:] = sorted(d for d in dirs if not _is_ignored_dir(d))
        for file in sorted(files):
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, start=1):
                        # 直接保留文件路径和行号，方便 Agent 后续基于真实位置继续读文件。
                        if keyword in line:
                            matches.append(f"{file_path}:{line_num}: {line.rstrip()}")
                            if len(matches) >= max_results:
                                truncated = True
                                break
                    if truncated:
                        break
            except (OSError, UnicodeError):
                continue
        if truncated:
            break
    if truncated:
        matches.append(f"... [结果过多，仅显示前 {max_results} 条]")
    return "\n".join(matches) if matches else f"未找到包含 '{keyword}' 的内容"


def web_search(query: str, max_results: int = 3) -> str:
    # 联网搜索是辅助工具，不是 Duan-Code 主线；主线还是本地 ReAct + Tool Use。
    """联网搜索工具（ddgs），无需API Key，国内可直接使用"""
    try:
        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            return "搜索失败：max_results 必须是整数"

        if max_results <= 0:
            return "搜索结果为空"

        DDGS = _get_ddgs_class()
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return "搜索结果为空"

        output = []
        for idx, res in enumerate(results, 1):
            title = res.get("title", "无标题")
            body = res.get("body", "无摘要")
            output.append(f"【结果{idx}】\n标题：{title}\n内容：{body}\n")

        return "\n".join(output)

    except Exception as e:
        return f"搜索失败：{str(e)}"


_rag_model = None
_rag_collection = None

def _get_rag_components():
    """懒加载 RAG 模型和 ChromaDB collection，避免每次调用都重新初始化"""
    global _rag_model, _rag_collection
    chromadb, SentenceTransformer = _get_rag_dependencies()
    if _rag_model is None:
        # 查询侧和建库侧使用同一个 embedding 模型，保证向量空间一致。
        _rag_model = SentenceTransformer("all-MiniLM-L6-v2")
    if _rag_collection is None:
        # ChromaDB 只保存已经构建好的本地知识库索引；这里不负责临时建库。
        chroma_dir = os.path.join(os.path.dirname(__file__), "rag", "chroma_db")
        client = chromadb.PersistentClient(path=chroma_dir)
        _rag_collection = client.get_collection("knowledge_base")
    return _rag_model, _rag_collection


def _query_terms(text: str) -> list[str]:
    # 提取中英文、数字和代码常见符号，给召回后的轻量重排使用。
    return [term.lower() for term in re.findall(r"[A-Za-z0-9_\-\u4e00-\u9fff]+", text) if len(term.strip()) >= 2]


def _match_term_count(terms: list[str], text: str) -> int:
    searchable = text.lower()
    return sum(1 for term in terms if term in searchable)


def _keyword_overlap_score(question: str, document: str, metadata: dict) -> tuple[int, int]:
    # 向量召回先保证语义相关，这里再用关键词命中补一层精确信号。
    terms = _query_terms(question)
    if not terms:
        return 0, 0
    body_score = _match_term_count(terms, document)
    # 标题、章节和来源通常比正文更能说明片段归属，所以单独计算 context_score。
    context_text = " ".join(
        part for part in [
            str(metadata.get("title") or ""),
            str(metadata.get("heading_path") or ""),
            str(metadata.get("source") or ""),
        ] if part
    )
    context_score = _match_term_count(terms, context_text)
    return body_score, context_score


def _rerank_knowledge_hits(question: str, documents: list[str], metadatas: list[dict], top_k: int) -> list[tuple[str, dict]]:
    # RAG 第二阶段：轻量 rerank。
    # 第一阶段的向量召回负责“语义大方向相关”，但它不一定擅长文件名、函数名、命令、错误码等精确匹配。
    # 所以这里不直接相信向量相似度，而是在候选片段里再看问题关键词命中了多少。
    # 这不是重型 reranker：不跑 Cross-Encoder，也不调用 LLM，只做低成本的关键词重排。
    candidates = []
    for index, (document, metadata) in enumerate(zip(documents, metadatas)):
        body_score, context_score = _keyword_overlap_score(question, document, metadata)
        candidates.append({
            "document": document,
            "metadata": metadata,
            "vector_rank": index,
            "body_score": body_score,
            "context_score": context_score,
        })

    # 优先返回关键词更贴近问题的片段；分数相同时保留原始向量召回顺序。
    candidates.sort(key=lambda item: (-item["body_score"], -item["context_score"], item["vector_rank"]))
    return [(item["document"], item["metadata"]) for item in candidates[:top_k]]


def query_knowledge_base(question: str, top_k: int = 3) -> str:
    # 本地知识库查询是辅助 RAG 能力；需要先构建 rag/chroma_db 索引。
    #
    # 查询链路：
    # 1. 建库阶段：rag/build_index.py 读取 rag/docs -> 文档分块 -> 生成 embedding -> 存入 ChromaDB。
    # 2. 查询阶段：把用户问题也编码成 embedding，用 ChromaDB 先召回一批语义相关 chunk。
    # 3. 重排阶段：多取候选，然后用关键词命中做轻量 rerank，最后只返回 top_k。
    # 4. 输出阶段：返回片段正文 + source/title/heading/page 等 metadata，方便模型基于证据回答。
    #
    # 关键理解：embedding 负责语义召回，rerank 补充精确匹配信号，metadata 帮助模型判断来源。
    """查询本地知识库，返回与问题最相关的文档片段（需先运行 rag/build_index.py 构建索引）"""
    try:
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            return "知识库查询失败：top_k 必须是整数"
        if top_k <= 0:
            return "知识库中未找到相关内容"

        model, collection = _get_rag_components()
        # 先把用户问题编码成查询向量，用 ChromaDB 做第一阶段语义召回。
        # 这里使用和建库阶段相同的 embedding 模型，保证问题向量和文档向量在同一个向量空间。
        embedding = model.encode([question]).tolist()
        # 多取一些候选，给后面的轻量 rerank 留空间，再裁成最终 top_k。
        # 例：top_k=3 时先召回 9 个候选，再从里面挑出最相关的 3 个。
        candidate_count = max(top_k * 3, top_k)
        results = collection.query(query_embeddings=embedding, n_results=candidate_count)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if not documents:
            return "知识库中未找到相关内容"

        output = []
        # 第二阶段只在候选集合内排序，避免为了本地知识库查询引入重 reranker 的成本。
        # 如果问题里有 ToolPolicy、rm-rf、函数名、文件名这类精确词，关键词 rerank 往往能把更准确片段排前。
        ranked_hits = _rerank_knowledge_hits(question, documents, metadatas, top_k)
        for i, (doc, meta) in enumerate(ranked_hits, 1):
            source = meta.get("source", "未知来源")
            source_path = meta.get("source_path")
            file_type = meta.get("file_type")
            title = meta.get("title")
            heading_path = meta.get("heading_path")
            chunk_index = meta.get("chunk_index")
            page_start = meta.get("page_start")
            page_end = meta.get("page_end")

            header_parts = [f"来源：{source}"]
            if source_path and source_path != source:
                header_parts.append(f"路径：{source_path}")
            if file_type:
                header_parts.append(f"类型：{file_type}")
            if title:
                header_parts.append(f"标题：{title}")
            if heading_path:
                header_parts.append(f"章节：{heading_path}")
            if chunk_index is not None:
                header_parts.append(f"片段序号：{chunk_index}")
            if page_start is not None:
                page_label = f"页码：{page_start}" if page_end in (None, page_start) else f"页码：{page_start}-{page_end}"
                header_parts.append(page_label)

            output.append(f"【片段{i}】{'；'.join(header_parts)}\n{doc}")

        return "\n\n".join(output)

    except Exception as e:
        return f"知识库查询失败：{str(e)}（请先运行 uv run python rag/build_index.py 构建索引）"
