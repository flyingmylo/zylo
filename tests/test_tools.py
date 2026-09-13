import os
import pytest
import pymupdf
from src.tools.pdf_reader import DocumentReader
from src.tools.knowledge_base import KnowledgeBase
from src.embeddings.base import EmbeddingProvider


class DummyEmbeddingProvider(EmbeddingProvider):
    """用于单元测试的轻量 Dummy 向量生成器，输出 8 维固定向量"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 8 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 8


def test_document_reader_text(tmp_path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text(
        "这是第一段较长的技术背景说明，用来详细描述传统大模型架构在长文本场景下的具体表现与瓶颈所在。\n\n"
        "这是第二段更长更深入的技术细节，剖析底层显存布局与注意力计算过程中的动态分配策略与碎片化损耗。"
    )

    reader = DocumentReader(chunk_size=30, chunk_overlap=5)
    chunks = reader.read_file(str(test_file))

    assert len(chunks) >= 2
    assert "第一段" in chunks[0]["text"]
    assert chunks[0]["source"] == "sample.txt"


def test_knowledge_base_bilingual_retrieve():
    dummy_embed = DummyEmbeddingProvider()
    kb = KnowledgeBase(collection_name="test_kb", embedding_provider=dummy_embed)

    docs = [
        {"text": "大语言模型的记忆机制包括短期工作记忆与长期情景记忆。", "source": "zh_doc.md", "page": "1"},
        {"text": "Large language model memory consists of working memory and episodic memory.", "source": "en_doc.pdf", "page": "3"},
        {"text": "KV-Cache 显存占用随上下文长度线性增长。", "source": "kv_doc.md", "page": "5"},
    ]
    kb.add_documents(docs)
    assert kb.count() == 3

    # 测试双语检索去重与召回
    results = kb.retrieve(
        query_zh="记忆机制",
        query_en="memory architecture",
        top_k=2,
    )

    assert len(results) <= 2
    assert len(results) > 0


def test_document_reader_pdf(tmp_path):
    pdf_path = tmp_path / "sample_paper.pdf"

    # 用 pymupdf 生成两页真实的测试 PDF
    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_text((50, 72), "Abstract: We present FlashAttention, a fast and memory-efficient exact attention algorithm.")
    page2 = doc.new_page()
    page2.insert_text((50, 72), "Section 2: Background on GPU memory hierarchy and tiling mechanisms.")
    doc.save(str(pdf_path))
    doc.close()

    reader = DocumentReader(chunk_size=100)
    chunks = reader.read_file(str(pdf_path))

    assert len(chunks) == 2
    assert "FlashAttention" in chunks[0]["text"]
    assert chunks[0]["page"] == "1"
    assert "GPU memory" in chunks[1]["text"]
    assert chunks[1]["page"] == "2"


@pytest.mark.asyncio
async def test_document_reader_read_source_local(tmp_path):
    txt_path = tmp_path / "note.md"
    txt_path.write_text("# 核心要点\n\n这是关于 FlashAttention 算子的深入技术解析。")

    reader = DocumentReader(chunk_size=100)
    chunks = await reader.read_source(str(txt_path))

    assert len(chunks) >= 1
    assert "FlashAttention" in chunks[0]["text"]


def test_extract_arxiv_id():
    # 测试各种格式的 arXiv 链接与输入变体
    cases = [
        ("https://arxiv.org/abs/2405.05254", "2405.05254"),
        ("https://arxiv.org/abs/2405.05254v1", "2405.05254v1"),
        ("https://arxiv.org/abs/2405.05254v2?context=cs", "2405.05254v2"),
        ("https://arxiv.org/pdf/2405.05254.pdf", "2405.05254"),
        ("https://arxiv.org/pdf/2405.05254", "2405.05254"),
        ("https://export.arxiv.org/abs/2405.05254", "2405.05254"),
        ("https://arxiv.org/html/2405.05254v1", "2405.05254v1"),
        ("https://arxiv.org/abs/math.PR/0501001", "math.PR/0501001"),
        ("https://arxiv.org/pdf/hep-th/9912012.pdf", "hep-th/9912012"),
        ("arxiv:2405.05254", "2405.05254"),
        ("2405.05254", "2405.05254"),
        ("https://example.com/not_arxiv", None),
        ("local_file.pdf", None),
    ]

    for raw, expected in cases:
        assert DocumentReader.extract_arxiv_id(raw) == expected, f"Failed on: {raw}"


@pytest.mark.asyncio
async def test_read_source_arxiv_cache_hit(tmp_path):
    cache_dir = tmp_path / "references"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = cache_dir / "arxiv_2405.05254.pdf"

    # 预先在本地缓存中创建该论文
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((50, 72), "YOCO: You Only Cache Once Architecture")
    doc.save(str(pdf_path))
    doc.close()

    reader = DocumentReader(chunk_size=100, cache_dir=str(cache_dir))
    # 传入 URL 时，应直接命中本地缓存文件，无需发起网络请求
    chunks = await reader.read_source("https://arxiv.org/abs/2405.05254")

    assert len(chunks) == 1
    assert "YOCO" in chunks[0]["text"]
    assert chunks[0]["source"] == "arxiv_2405.05254.pdf"


@pytest.mark.asyncio
async def test_read_source_arxiv_download_and_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "references"
    reader = DocumentReader(chunk_size=100, cache_dir=str(cache_dir))

    # 构建模拟的 PDF 二进制内容
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((50, 72), "Downloaded PDF content for DeepSeek-V3")
    pdf_bytes = doc.tobytes()
    doc.close()

    class MockResponse:
        status_code = 200
        headers = {"content-type": "application/pdf"}
        content = pdf_bytes

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, headers=None):
            return MockResponse()

    monkeypatch.setattr("httpx.AsyncClient", MockAsyncClient)

    # 执行读取 (传入纯 ID)
    chunks = await reader.read_source("2412.19437")

    # 验证本地 references 目录下是否成功自动持久化生成文件
    expected_cache_file = cache_dir / "arxiv_2412.19437.pdf"
    assert expected_cache_file.exists()
    assert expected_cache_file.stat().st_size > 0

    # 验证解析的内容
    assert len(chunks) == 1
    assert "DeepSeek-V3" in chunks[0]["text"]
    assert chunks[0]["source"] == "arxiv_2412.19437.pdf"

