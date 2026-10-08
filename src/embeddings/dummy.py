"""固定向量 Dummy 嵌入：离线演示与测试专用，零依赖秒级可用。"""

import hashlib

from .base import EmbeddingProvider


class DummyEmbeddingProvider(EmbeddingProvider):
    """把文本哈希映射到稳定的 8 维向量。

    相同文本必然得到相同向量（可复现），不同文本大概率分散（检索有区分度），
    足以驱动离线演示走通全链路；语义检索请使用 BGE 实现。
    """

    DIM = 8

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in digest[: self.DIM]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
