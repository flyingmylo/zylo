from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """向量模型抽象基类，使应用与底层模型解耦"""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量对文档生成嵌入向量"""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """对检索词生成嵌入向量"""
        pass
