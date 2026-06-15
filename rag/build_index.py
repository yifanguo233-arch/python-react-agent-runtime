import os

import chromadb
from sentence_transformers import SentenceTransformer

try:
    from .document_pipeline import build_chunks, load_documents
except ImportError:
    from document_pipeline import build_chunks, load_documents

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
COLLECTION_NAME = "knowledge_base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def build_index():
    """读取文档、结构化分块、向量化，存入 ChromaDB"""
    # RAG 建库入口流程：
    # 1. 从 rag/docs 递归读取 .txt / .md / .docx / .pdf。
    # 2. document_pipeline.py 解析文档结构，尽量保留标题、章节、页码等 metadata。
    # 3. 按 chunk_size / overlap 切成 chunk，避免单个片段过长或语义被硬切断。
    # 4. 用 embedding 模型把 chunk 转成向量。
    # 5. 把 chunk 正文、向量、metadata 一起写入 ChromaDB，供 tools.py 的 query_knowledge_base 查询。
    # 面试短句：建库是“文档 -> chunk -> embedding -> 向量库”，查询是“问题 -> embedding -> 召回 -> rerank”。
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"已创建 docs 目录：{DOCS_DIR}")
        print("请将你的知识库文档（.txt / .md / .docx / .pdf）放入该目录后重新运行。")
        return

    documents = load_documents(DOCS_DIR)
    if not documents:
        print(f"docs/ 目录中没有找到可索引文档（支持 .txt / .md / .docx / .pdf），请添加后重新运行。")
        return

    print(f"递归加载了 {len(documents)} 个文档，开始结构化分块...")
    # 建库侧的模型要和查询侧保持一致，否则查询向量和文档向量不在同一语义空间。
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)

    chunks = build_chunks(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        print("未生成可用文本块，请检查文档内容是否为空或解析失败。")
        return

    all_chunks = [chunk["content"] for chunk in chunks]
    # embedding_text 会拼上标题、章节等上下文，让向量检索时不只看正文片段本身。
    embedding_inputs = [chunk["embedding_text"] for chunk in chunks]
    all_ids = [chunk["id"] for chunk in chunks]
    all_metas = [chunk["metadata"] for chunk in chunks]

    print(f"共 {len(all_chunks)} 个文本块，正在向量化（首次运行会下载模型，请稍等）...")
    # 这里生成的是文档侧向量，之后 query_knowledge_base 会生成问题侧向量来做相似度召回。
    embeddings = model.encode(embedding_inputs, show_progress_bar=True).tolist()

    # ChromaDB 持久化保存 chunk、embedding 和 metadata，查询时可以直接按向量召回。
    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        ids=all_ids,
        metadatas=all_metas,
    )

    print(f"\n✅ 索引构建完成！共导入 {len(documents)} 个文档，存入 {len(all_chunks)} 个文本块，保存至 {CHROMA_DIR}")


if __name__ == "__main__":
    build_index()
