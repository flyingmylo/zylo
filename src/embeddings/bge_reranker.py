import os

import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_default_cache = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".cache", "huggingface")
)
os.environ.setdefault("HF_HOME", _default_cache)

from sentence_transformers import CrossEncoder

from .reranker_base import RerankerProvider


class BGERerankerProvider(RerankerProvider):
    """
    BAAI/bge-reranker-v2-m3 本地重排模型
    阶段 4 可插拔开启，用于中英候选片段的精准排序
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(self, query: str, documents: list[str], top_k: int = 3) -> list[str]:
        if not documents:
            return []
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]
