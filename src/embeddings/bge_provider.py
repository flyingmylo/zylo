import torch
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
