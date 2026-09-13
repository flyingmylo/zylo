from abc import ABC, abstractmethod


class RerankerProvider(ABC):
    """重排模型抽象接口（可选插拔钩子）"""

    @abstractmethod
    def rerank(self, query: str, documents: list[str], top_k: int = 3) -> list[str]:
        """
        利用 Cross-Encoder 全注意力机制对候选文本重排打分
        返回最相关的 top_k 个文本
        """
        pass
