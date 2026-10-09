import ipaddress
import socket

import httpx
import pymupdf
import pytest

from src.embeddings.base import EmbeddingProvider
from src.tools.knowledge_base import ChromaEmbeddingAdapter, KnowledgeBase
from src.tools.pdf_reader import DocumentReader
from src.tools.web_reader import WebReader


class DummyEmbeddingProvider(EmbeddingProvider):
    """用于单元测试的轻量 Dummy 向量生成器，输出 8 维固定向量"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 8 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 8


def _noop_url_validator():
    """绕过真实 DNS 解析的 SSRF 校验桩（测试环境无网络时必需）。"""

    async def _validate(url: str) -> None:
        return None

    return _validate


def _make_stream_stub(headers: dict, body: bytes, location: str | None = None):
    """构造可挂到 httpx.AsyncClient.stream 的桩：返回固定响应（或重定向）。"""

    class _Resp:
        def __init__(self):
            self.status_code = 302 if location else 200
            self.headers = dict(headers)
            if location:
                self.headers["location"] = location

        @property
        def is_redirect(self):
            return location is not None

        async def aiter_bytes(self):
            yield body

    class _CM:
        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, *args):
            return False

    def _stub(self, method, url, **kwargs):
        return _CM()

    return _stub


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://127.0.0.1/admin",
        "http://[::1]/admin",
        "http://169.254.169.254/latest/meta-data",
        "http://10.0.0.1/internal",
        "https://user:secret@example.com/",
    ],
)
async def test_web_reader_rejects_unsafe_urls(url):
    with pytest.raises(ValueError):
        await WebReader.validate_public_url(url)


@pytest.mark.asyncio
async def test_web_reader_accepts_public_ip_url():
    public_ip = str(ipaddress.ip_address("1.1.1.1"))
    await WebReader.validate_public_url(f"https://{public_ip}/docs")


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


def test_knowledge_base_add_is_idempotent():
    """稳定 ID + upsert：重复提交相同内容不产生重复向量（重试/恢复安全）。"""
    kb = KnowledgeBase(collection_name="test_idempotent", embedding_provider=DummyEmbeddingProvider())
    docs = [
        {"text": "KV-Cache 显存占用随上下文长度线性增长。", "source": "kv.md", "page": "1"},
        {"text": "PagedAttention 消除内存碎片。", "source": "kv.md", "page": "2"},
    ]

    kb.add_documents(docs)
    kb.add_documents(docs)  # 模拟重试重复提交

    assert kb.count() == 2


def test_knowledge_base_persists_across_instances(tmp_path):
    """PersistentClient：新实例同目录同名 collection，向量幸存可检索。"""
    persist_dir = str(tmp_path / "chroma" / "run_x")
    docs = [
        {"text": "大语言模型的记忆机制包括短期工作记忆。", "source": "zh.md", "page": "1"},
    ]

    first = KnowledgeBase(
        collection_name="writing_persist",
        embedding_provider=DummyEmbeddingProvider(),
        persist_dir=persist_dir,
    )
    first.add_documents(docs)
    assert first.count() == 1

    # 模拟进程重启：全新实例重开同一目录
    reopened = KnowledgeBase(
        collection_name="writing_persist",
        embedding_provider=DummyEmbeddingProvider(),
        persist_dir=persist_dir,
    )
    assert reopened.count() == 1
    results = reopened.retrieve(query_zh="记忆机制", top_k=1)
    assert len(results) == 1
    assert results[0]["source"] == "zh.md"


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

    monkeypatch.setattr(
        WebReader, "validate_public_url", _noop_url_validator()
    )
    monkeypatch.setattr(
        httpx.AsyncClient,
        "stream",
        _make_stream_stub({"content-type": "application/pdf"}, pdf_bytes),
    )

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


def test_chroma_embedding_adapter_unbound_raises_runtime_error():
    """未绑定 provider 的适配器必须响亮失败（契约翻转：静默空向量 → RuntimeError）。"""
    adapter = ChromaEmbeddingAdapter()
    with pytest.raises(RuntimeError, match="未绑定"):
        adapter(["某段文本"])

    # build_from_config 仍会产出未绑定实例：一旦被 chromadb 按配置重建，必须抛错而非静默空检索
    rebuilt = ChromaEmbeddingAdapter.build_from_config({})
    assert rebuilt.provider is None
    with pytest.raises(RuntimeError):
        rebuilt(["x"])

    bound = ChromaEmbeddingAdapter(DummyEmbeddingProvider())
    vectors = bound(["x"])
    assert len(vectors) == 1
    assert list(vectors[0]) == [0.1] * 8


@pytest.mark.asyncio
async def test_fetch_and_clean_swallows_socket_errors(monkeypatch):
    """DNS/socket 层故障（gaierror ⊂ OSError）应被吞掉返回空串，而非向上传播。"""

    async def raise_gaierror(url):
        raise socket.gaierror(8, "nodename nor servname provided")

    monkeypatch.setattr(WebReader, "validate_public_url", raise_gaierror)
    assert await WebReader.fetch_and_clean("https://dead-domain.invalid/") == ""


@pytest.mark.asyncio
async def test_fetch_and_clean_failure_paths_return_empty_string(monkeypatch):
    """httpx 层错误、重定向超额、SSRF 重定向、非文本响应都应返回空串。"""

    class _StubStream:
        def __init__(self, resp):
            self._resp = resp

        async def __aenter__(self):
            return self._resp

        async def __aexit__(self, *args):
            return False

    class _StubClient:
        def __init__(self, resp):
            self._resp = resp

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, *args, **kwargs):
            return _StubStream(self._resp)

    class _StubResponse:
        def __init__(self, redirect_to=None, status_code=200, content_type="text/html"):
            self.status_code = status_code
            self.headers = {"content-type": content_type}
            if redirect_to is not None:
                self.headers["location"] = redirect_to
            self._redirect_to = redirect_to

        @property
        def is_redirect(self):
            return self._redirect_to is not None

        async def aiter_bytes(self):
            yield b"<html><body>ok</body></html>"

    # 1) 连接层错误（httpx.ConnectError ⊂ HTTPError）
    class _ExplodingClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            raise httpx.ConnectError("连接失败")

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr("httpx.AsyncClient", _ExplodingClient)
    assert await WebReader.fetch_and_clean("https://1.1.1.1/docs") == ""

    # 2) 重定向超额（location 指向自身，超过 MAX_REDIRECTS 后放弃）
    loop_resp = _StubResponse(redirect_to="https://1.1.1.1/loop")
    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **kw: _StubClient(loop_resp))
    assert await WebReader.fetch_and_clean("https://1.1.1.1/loop") == ""

    # 3) 重定向到内网地址（SSRF 校验抛 ValueError，应被吞掉返回空串）
    ssrf_resp = _StubResponse(redirect_to="http://127.0.0.1/admin")
    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **kw: _StubClient(ssrf_resp))
    assert await WebReader.fetch_and_clean("https://1.1.1.1/redirect") == ""

    # 4) 非文本 content-type
    binary_resp = _StubResponse(
        status_code=200, content_type="application/octet-stream"
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **kw: _StubClient(binary_resp))
    assert await WebReader.fetch_and_clean("https://1.1.1.1/file") == ""


@pytest.mark.asyncio
async def test_read_url_corrupted_pdf_cache_falls_back_to_web(tmp_path, monkeypatch):
    """缓存 PDF 损坏（下载中断残留）时应回退网页抓取，FileDataError 不得逃逸。"""
    cache_dir = tmp_path / "references"
    reader = DocumentReader(chunk_size=100, cache_dir=str(cache_dir))

    monkeypatch.setattr(
        WebReader, "validate_public_url", _noop_url_validator()
    )
    monkeypatch.setattr(
        httpx.AsyncClient,
        "stream",
        _make_stream_stub(
            {"content-type": "application/pdf"},
            b"%PDF-1.4 truncated broken content",  # 带 PDF 头但内容损坏
        ),
    )

    async def fake_fetch(u):
        return "网页正文：FlashAttention 的核心思想是避免物化完整注意力矩阵。"

    monkeypatch.setattr(WebReader, "fetch_and_clean", fake_fetch)

    chunks = await reader.read_source("https://example.org/paper.pdf")

    assert len(chunks) == 1
    assert "FlashAttention" in chunks[0]["text"]
    assert chunks[0]["source"] == "https://example.org/paper.pdf"
    # 损坏缓存必须被清理，否则下次运行会在缓存命中分支反复崩溃
    expected_cache = cache_dir / "example_org_paper_pdf.pdf"
    assert not expected_cache.exists()


def test_cache_path_includes_host_path_and_query(tmp_path):
    """不同域名/路径/query 的 URL 不得互撞缓存文件名（会静默读到错误文献）。"""
    reader = DocumentReader(cache_dir=str(tmp_path))

    base = reader._get_cache_path("https://a.com/papers/flash.pdf")
    other_host = reader._get_cache_path("https://b.com/papers/flash.pdf")
    other_query = reader._get_cache_path("https://a.com/papers/flash.pdf?v=2")
    other_path = reader._get_cache_path("https://a.com/other/flash.pdf")

    assert len({base, other_host, other_query, other_path}) == 4


@pytest.mark.asyncio
async def test_corrupted_disk_cache_is_purged_and_refetched(tmp_path, monkeypatch):
    """二次运行命中损坏缓存：必须删除坏文件并重新下载，而非反复崩溃。"""
    cache_dir = tmp_path / "refs"
    cache_dir.mkdir()
    reader = DocumentReader(chunk_size=100, cache_dir=str(cache_dir))

    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((50, 72), "Refetched good PDF")
    good_bytes = doc.tobytes()
    doc.close()

    # 模拟上次运行残留的损坏缓存
    (cache_dir / "example_org_paper_pdf.pdf").write_bytes(b"%PDF-1.1 broken leftover")

    monkeypatch.setattr(WebReader, "validate_public_url", _noop_url_validator())
    monkeypatch.setattr(
        httpx.AsyncClient,
        "stream",
        _make_stream_stub({"content-type": "application/pdf"}, good_bytes),
    )

    chunks = await reader.read_source("https://example.org/paper.pdf")

    assert any("Refetched good PDF" in c["text"] for c in chunks)
    # 坏缓存已被重新下载的好版本覆盖
    assert (cache_dir / "example_org_paper_pdf.pdf").read_bytes() == good_bytes


@pytest.mark.asyncio
async def test_download_pdf_rejects_private_address(tmp_path):
    """下载路径与 WebReader 同标准：内网地址直接拒绝，不发请求不落盘。"""
    reader = DocumentReader(cache_dir=str(tmp_path))
    target = tmp_path / "leaked.pdf"

    ok = await reader._download_pdf("http://127.0.0.1:8080/secret.pdf", str(target))

    assert ok is False
    assert not target.exists()


@pytest.mark.asyncio
async def test_download_pdf_rejects_redirect_to_private_network(tmp_path, monkeypatch):
    """重定向落入内网：逐跳 SSRF 校验必须放弃下载。"""
    monkeypatch.setattr(WebReader, "validate_public_url", _noop_url_validator())
    monkeypatch.setattr(
        httpx.AsyncClient,
        "stream",
        _make_stream_stub({}, b"", location="http://10.0.0.5/inner.pdf"),
    )
    reader = DocumentReader(cache_dir=str(tmp_path))
    target = tmp_path / "leaked.pdf"

    ok = await reader._download_pdf("https://evil.example/paper.pdf", str(target))

    assert ok is False
    assert not target.exists()


@pytest.mark.asyncio
async def test_download_pdf_enforces_size_cap(tmp_path, monkeypatch):
    """超过大小上限的响应立即放弃，不落盘半成品。"""
    monkeypatch.setattr(WebReader, "validate_public_url", _noop_url_validator())
    monkeypatch.setattr(
        httpx.AsyncClient,
        "stream",
        _make_stream_stub({"content-type": "application/pdf"}, b"%PDF-" + b"x" * 100),
    )
    reader = DocumentReader(cache_dir=str(tmp_path))
    target = tmp_path / "big.pdf"

    ok = await reader._download_pdf(
        "https://example.org/big.pdf", str(target), max_bytes=10
    )

    assert ok is False
    assert not target.exists()


@pytest.mark.asyncio
async def test_download_pdf_rejects_fake_pdf_body(tmp_path, monkeypatch):
    """URL 以 .pdf 结尾但响应是 HTML：魔术字节校验必须拒绝落盘。"""
    monkeypatch.setattr(WebReader, "validate_public_url", _noop_url_validator())
    monkeypatch.setattr(
        httpx.AsyncClient,
        "stream",
        _make_stream_stub({"content-type": "application/pdf"}, b"<html>not a pdf</html>"),
    )
    reader = DocumentReader(cache_dir=str(tmp_path))
    target = tmp_path / "fake.pdf"

    ok = await reader._download_pdf("https://example.org/fake.pdf", str(target))

    assert ok is False
    assert not target.exists()
