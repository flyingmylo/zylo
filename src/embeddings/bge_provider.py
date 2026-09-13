import os

import torch

# 自动配置国内 Hugging Face 高速镜像源与本地独立可写缓存目录
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_default_cache = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".cache", "huggingface")
)
os.environ.setdefault("HF_HOME", _default_cache)

from sentence_transformers import SentenceTransformer

from .base import EmbeddingProvider


class BGEM3EmbeddingProvider(EmbeddingProvider):
    """
    BAAI/bge-m3 本地嵌入模型实现
    针对 Apple Silicon (M1/M2/M3/M4) 自动启用 MPS 硬件加速
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=16,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()
