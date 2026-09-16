import logging
import os

import torch

# 自动配置国内 Hugging Face 高速镜像源与本地独立可写缓存目录
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_default_cache = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".cache", "huggingface")
)
os.environ.setdefault("HF_HOME", _default_cache)
# 模型权重固定为公开 .bin 且已缓存，禁用 transformers 的后台 safetensors 转换探测：
# 它会向 Hub 查询转换 PR（bge-m3 为 refs/pr/130），是离线加载时仍产生
# 未认证请求警告的主要来源
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

from sentence_transformers import SentenceTransformer

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)


class BGEM3EmbeddingProvider(EmbeddingProvider):
    """
    BAAI/bge-m3 本地嵌入模型实现
    针对 Apple Silicon (M1/M2/M3/M4) 自动启用 MPS 硬件加速
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        # 模型是固定公开权重，优先离线加载本地缓存：跳过对 Hub 的 etag 复检，
        # 消除未认证请求警告并省去每次启动的联网往返；缓存缺失时回退联网下载
        try:
            self.model = SentenceTransformer(
                model_name, device=self.device, local_files_only=True
            )
        except Exception:
            logger.info("本地缓存未命中 %s，回退联网下载", model_name)
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
