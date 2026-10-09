import hashlib
import uuid
from collections.abc import Sequence
from typing import cast

import chromadb
from chromadb import EmbeddingFunction
from chromadb.api import ClientAPI
from chromadb.api.types import Documents, Embeddable, Embeddings, Metadata
from typing_extensions import override

from src.embeddings.base import EmbeddingProvider
from src.embeddings.reranker_base import RerankerProvider


class ChromaEmbeddingAdapter(EmbeddingFunction[Documents]):
    """把项目通用的 EmbeddingProvider 包装为 ChromaDB 原生兼容的 EmbeddingFunction"""

    provider: EmbeddingProvider | None

    def __init__(self, provider: EmbeddingProvider | None = None) -> None:
        # 有意不调用 super().__init__()：基类 Protocol 的 __init__ 会发出 DeprecationWarning
        self.provider = provider

    @override
    def __call__(self, input: Documents) -> Embeddings:
        if not self.provider:
            raise RuntimeError(
                "ChromaEmbeddingAdapter 未绑定 EmbeddingProvider；"
                "当前设计（进程内一次性知识库）下此分支不应被触达，"
                "若出现说明适配器被 build_from_config 异常重建"
            )
        return cast(Embeddings, self.provider.embed_documents(list(input)))

    @override
    @staticmethod
    def name() -> str:
        return "zylo_custom_embedding"

    @override
    def get_config(self) -> dict[str, object]:
        return {"name": self.name()}

    @override
    @staticmethod
    def build_from_config(config: dict[str, object]) -> "ChromaEmbeddingAdapter":
        return ChromaEmbeddingAdapter(provider=None)


class KnowledgeBase:
    """
    文章级知识库管理器
    - 隔离每次写作任务的 Collection 生命周期
    - 封装中英双语扩展检索
    - 预留 Reranker 重排钩子

    persist_dir 为空时是进程内内存库（CLI 一次性任务/单元测试）；
    指定目录时切换 PersistentClient：向量随目录落盘，resume 时以
    相同目录 + collection 名重开即可恢复检索，调研不重复执行。
    """

    def __init__(
        self,
        collection_name: str | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        reranker_provider: RerankerProvider | None = None,
        persist_dir: str | None = None,
    ):
        self.collection_name: str = collection_name or f"writing_{uuid.uuid4().hex[:8]}"
        self.client: ClientAPI = (
            chromadb.PersistentClient(path=persist_dir)
            if persist_dir
            else chromadb.Client()
        )
        self.embedding_provider: EmbeddingProvider | None = embedding_provider
        self.reranker: RerankerProvider | None = reranker_provider

        embed_fn = (
            ChromaEmbeddingAdapter(self.embedding_provider)
            if self.embedding_provider
            else None
        )

        self.collection: chromadb.Collection = self.client.get_or_create_collection(
            name=self.collection_name,
            # chromadb 参数声明为逆变的 EmbeddingFunction[Embeddable]，而适配器按官方
            # DefaultEmbeddingFunction 模式实现为 EmbeddingFunction[Documents]，需显式桥接
            embedding_function=cast("EmbeddingFunction[Embeddable] | None", embed_fn),
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _stable_chunk_id(doc: dict[str, str]) -> str:
        """内容哈希 ID：同文本+来源+页码永远同 ID，配合 upsert 实现幂等入库。

        重试与 resume 场景下重复 add 相同内容不会产生重复向量（随机 ID 时代的
        老问题），按 run 审计时 ID 也可反查内容。
        """
        digest = hashlib.sha256(
            f"{doc.get('text', '')}|{doc.get('source', '')}|{doc.get('page', '1')}".encode()
        ).hexdigest()
        return digest[:16]

    def add_documents(self, documents: list[dict[str, str]]):
        """
        批量入库文档块（幂等：稳定 ID + upsert，重复提交自动去重）
        documents: [{"text": "...", "source": "...", "page": "1"}, ...]
        """
        if not documents:
            return

        texts = [d["text"] for d in documents]
        metadatas: list[Metadata] = [
            {"source": d.get("source", "unknown"), "page": str(d.get("page", "1"))}
            for d in documents
        ]
        ids = [self._stable_chunk_id(d) for d in documents]

        batch_size = 64
        for i in range(0, len(texts), batch_size):
            self.collection.upsert(
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
    ) -> list[dict[str, str]]:
        """
        双语 Query 扩展检索 + 可选 Reranker 过滤
        返回 [{"text": "...", "source": "...", "page": "..."}, ...]
        """
        total_count = self.collection.count()
        if total_count == 0:
            return []

        n_fetch = min(candidate_pool, total_count)
        candidates_map: dict[str, dict[str, str]] = {}

        # 中英两个查询分支共用的局部变量：整个函数作用域内只声明一次，
        # 避免同名重复显式标注触发 Pyright reportRedeclaration
        metas: Sequence[Metadata]
        meta_dict: Metadata

        # 1. 中文 Query 检索
        if query_zh.strip():
            res_zh = self.collection.query(
                query_texts=[query_zh.strip()],
                n_results=n_fetch,
            )
            docs_batch = res_zh.get("documents")
            if docs_batch and docs_batch[0]:
                docs = docs_batch[0]
                metas_batch = res_zh.get("metadatas")
                if metas_batch and metas_batch[0]:
                    metas = metas_batch[0]
                else:
                    metas = [{} for _ in docs]
                for doc, meta in zip(docs, metas):
                    meta_dict = meta if isinstance(meta, dict) else {}
                    candidates_map[doc] = {
                        "text": doc,
                        "source": str(meta_dict.get("source", "")),
                        "page": str(meta_dict.get("page", "1")),
                    }

        # 2. 英文 Query 检索（同语言搜英文文献，彻底避免相似度折扣）
        if query_en.strip():
            res_en = self.collection.query(
                query_texts=[query_en.strip()],
                n_results=n_fetch,
            )
            docs_batch = res_en.get("documents")
            if docs_batch and docs_batch[0]:
                docs = docs_batch[0]
                metas_batch = res_en.get("metadatas")
                if metas_batch and metas_batch[0]:
                    metas = metas_batch[0]
                else:
                    metas = [{} for _ in docs]
                for doc, meta in zip(docs, metas):
                    if doc not in candidates_map:
                        meta_dict = meta if isinstance(meta, dict) else {}
                        candidates_map[doc] = {
                            "text": doc,
                            "source": str(meta_dict.get("source", "")),
                            "page": str(meta_dict.get("page", "1")),
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
