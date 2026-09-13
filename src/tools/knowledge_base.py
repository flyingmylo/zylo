import uuid
from typing import Any

import chromadb
from chromadb import EmbeddingFunction
from chromadb.api.types import Documents, Embeddings

from src.embeddings.base import EmbeddingProvider
from src.embeddings.reranker_base import RerankerProvider


class ChromaEmbeddingAdapter(EmbeddingFunction):
    """把项目通用的 EmbeddingProvider 包装为 ChromaDB 原生兼容的 EmbeddingFunction"""

    def __init__(self, provider: EmbeddingProvider | None = None):
        self.provider = provider

    def __call__(self, input: Documents) -> Embeddings:
        if not self.provider:
            return []
        return self.provider.embed_documents(list(input))

    @classmethod
    def name(cls) -> str:
        return "zylo_custom_embedding"

    def get_config(self) -> dict[str, Any]:
        return {"name": self.name()}

    @classmethod
    def build_from_config(cls, config: dict[str, Any]) -> "ChromaEmbeddingAdapter":
        return cls(provider=None)


class KnowledgeBase:
    """
    文章级知识库管理器
    - 隔离每次写作任务的 Collection 生命周期
    - 封装中英双语扩展检索
    - 预留 Reranker 重排钩子
    """

    def __init__(
        self,
        collection_name: str | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        reranker_provider: RerankerProvider | None = None,
    ):
        self.collection_name = collection_name or f"writing_{uuid.uuid4().hex[:8]}"
        self.client = chromadb.Client()
        self.embedding_provider = embedding_provider
        self.reranker = reranker_provider

        embed_fn = (
            ChromaEmbeddingAdapter(self.embedding_provider)
            if self.embedding_provider
            else None
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: list[dict[str, str]]):
        """
        批量入库文档块
        documents: [{"text": "...", "source": "...", "page": "1"}, ...]
        """
        if not documents:
            return

        texts = [d["text"] for d in documents]
        metadatas = [
            {"source": d.get("source", "unknown"), "page": str(d.get("page", "1"))}
            for d in documents
        ]
        ids = [
            f"{self.collection_name}_{i}_{uuid.uuid4().hex[:6]}"
            for i in range(len(documents))
        ]

        batch_size = 64
        for i in range(0, len(texts), batch_size):
            self.collection.add(
                documents=texts[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
                ids=ids[i : i + batch_size],
            )

    def retrieve(
        self,
        query_zh: str,
        query_en: str = "",
        top_k: int = 4,
        candidate_pool: int = 6,
    ) -> list[dict[str, Any]]:
        """
        双语 Query 扩展检索 + 可选 Reranker 过滤
        返回 [{"text": "...", "source": "...", "page": "..."}, ...]
        """
        total_count = self.collection.count()
        if total_count == 0:
            return []

        n_fetch = min(candidate_pool, total_count)
        candidates_map: dict[str, dict[str, Any]] = {}

        # 1. 中文 Query 检索
        if query_zh.strip():
            res_zh = self.collection.query(
                query_texts=[query_zh.strip()],
                n_results=n_fetch,
            )
            if res_zh and res_zh.get("documents") and res_zh["documents"][0]:
                docs = res_zh["documents"][0]
                metas = (
                    res_zh["metadatas"][0]
                    if res_zh.get("metadatas") and res_zh["metadatas"][0]
                    else [{}] * len(docs)
                )
                for doc, meta in zip(docs, metas):
                    meta_dict = meta if isinstance(meta, dict) else {}
                    candidates_map[doc] = {
                        "text": doc,
                        "source": meta_dict.get("source", ""),
                        "page": meta_dict.get("page", "1"),
                    }

        # 2. 英文 Query 检索（同语言搜英文文献，彻底避免相似度折扣）
        if query_en.strip():
            res_en = self.collection.query(
                query_texts=[query_en.strip()],
                n_results=n_fetch,
            )
            if res_en and res_en.get("documents") and res_en["documents"][0]:
                docs = res_en["documents"][0]
                metas = (
                    res_en["metadatas"][0]
                    if res_en.get("metadatas") and res_en["metadatas"][0]
                    else [{}] * len(docs)
                )
                for doc, meta in zip(docs, metas):
                    if doc not in candidates_map:
                        meta_dict = meta if isinstance(meta, dict) else {}
                        candidates_map[doc] = {
                            "text": doc,
                            "source": meta_dict.get("source", ""),
                            "page": meta_dict.get("page", "1"),
                        }

        candidate_list = list(candidates_map.values())
        if not candidate_list:
            return []

        # 3. 如果插拔启用了 Reranker，做精排
        if self.reranker and len(candidate_list) > top_k:
            raw_docs = [c["text"] for c in candidate_list]
            reranked_texts = self.reranker.rerank(
                query=query_zh or query_en,
                documents=raw_docs,
                top_k=top_k,
            )
            text_to_candidate = {c["text"]: c for c in candidate_list}
            return [
                text_to_candidate[t] for t in reranked_texts if t in text_to_candidate
            ]

        # 4. 默认相对 Top-K 截断
        return candidate_list[:top_k]

    def count(self) -> int:
        return self.collection.count()
