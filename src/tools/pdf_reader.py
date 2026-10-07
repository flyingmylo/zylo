import asyncio
import hashlib
import os
import re
import urllib.parse
from pathlib import Path
from typing import cast

import httpx
import pymupdf
from rich.console import Console

from src.tools.web_reader import MAX_REDIRECTS, WebReader

console = Console()

# 与 WebReader 一致的浏览器 UA；对 arXiv 等学术站点伪装浏览器可显著降低 403 率
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko)"
)
# 论文 PDF 远大于网页正文，上限单独放宽；超出即放弃下载，防止内存被单响应耗尽
MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024

# 正则定义：全面覆盖现代 (YYMM.NNNNN) 与经典历史学科 (archive/YYMMNNN) 分类，兼容版本号
ARXIV_URL_PATTERN = re.compile(
    r"arxiv\.org/(?:abs|pdf|html)/([a-zA-Z\-]+(?:\.[a-zA-Z]{2})?/\d{7}(?:v\d+)?|\d{4}\.\d{4,5}(?:v\d+)?)",
    re.IGNORECASE,
)
ARXIV_PREFIX_PATTERN = re.compile(
    r"^arxiv:\s*([a-zA-Z\-]+(?:\.[a-zA-Z]{2})?/\d{7}(?:v\d+)?|\d{4}\.\d{4,5}(?:v\d+)?)$",
    re.IGNORECASE,
)
ARXIV_BARE_PATTERN = re.compile(
    r"^(\d{4}\.\d{4,5}(?:v\d+)?)$",
    re.IGNORECASE,
)


class DocumentReader:
    """
    智能多源资料读取与切片器：
    支持本地文件（PDF, Markdown, TXT）以及远程 URL（arXiv 论文、网页正文、PDF 直链）
    具备 references/ 自动持久化本地缓存机制与正则级 arXiv 解析
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        cache_dir: str = "references",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cache_dir = cache_dir

    @classmethod
    def extract_arxiv_id(cls, source: str) -> str | None:
        """
        从 URL、前缀或纯编号中精准提取 arXiv 论文 ID
        支持:
          - https://arxiv.org/abs/2405.05254 (以及 pdf, html, export 等)
          - https://arxiv.org/abs/hep-th/9912012 (老版格式)
          - arxiv:2405.05254
          - 2405.05254
        """
        s = source.strip()
        m = ARXIV_URL_PATTERN.search(s)
        if m:
            return m.group(1)
        m = ARXIV_PREFIX_PATTERN.match(s)
        if m:
            return m.group(1)
        m = ARXIV_BARE_PATTERN.match(s)
        if m:
            return m.group(1)
        return None

    def _get_cache_path(self, identifier: str, is_arxiv: bool = False) -> str:
        """生成规范的本地持久化缓存路径"""
        os.makedirs(self.cache_dir, exist_ok=True)
        if is_arxiv:
            safe_id = identifier.replace("/", "_")
            return os.path.join(self.cache_dir, f"arxiv_{safe_id}.pdf")

        # 域名与完整路径必须一起参与命名：不同站点（或同站点不同路径）的同名
        # 文件若互撞缓存，会静默加载错误文献——那是事实性错误而非性能问题。
        # query 串以短摘要区分版本；用 hostname 而非 netloc，避免凭据落进文件名。
        parsed = urllib.parse.urlparse(identifier)
        host_part = parsed.hostname or "unknown"
        if parsed.port:
            host_part = f"{host_part}_{parsed.port}"
        host = re.sub(r"[^a-zA-Z0-9\-]", "_", host_part)
        clean_path = re.sub(r"[^a-zA-Z0-9_\-]", "_", parsed.path.strip("/"))
        if parsed.query:
            clean_path += "_" + hashlib.md5(parsed.query.encode()).hexdigest()[:8]
        if not clean_path:
            clean_path = hashlib.md5(identifier.encode()).hexdigest()[:10]
        # 截断保护：部分文件系统对文件名长度有 255 字节硬限制
        filename = f"{host}_{clean_path[:120]}.pdf"
        return os.path.join(self.cache_dir, filename)

    def _load_cached_pdf(self, cache_path: str) -> list[dict[str, str]] | None:
        """读取缓存 PDF；文件损坏时删除缓存并返回 None，交由调用方重新获取。

        损坏缓存多为下载中断的残留：不清理的话，之后每次运行都会在
        缓存命中分支崩溃，该资料从此永久不可用。
        """
        if not (os.path.exists(cache_path) and os.path.getsize(cache_path) > 0):
            return None
        try:
            return self.read_file(cache_path)
        except pymupdf.FileDataError:
            console.print(
                f"[yellow]! 缓存文件已损坏，删除后重新获取: [cyan]{cache_path}[/cyan][/yellow]"
            )
            os.remove(cache_path)
            return None

    async def _download_pdf(
        self,
        url: str,
        cache_path: str,
        timeout: float = 30.0,
        max_bytes: int = MAX_DOWNLOAD_BYTES,
    ) -> bool:
        """流式下载 PDF 到缓存路径，成功返回 True。

        与 WebReader 保持同一套防护标准：
        - 逐跳 SSRF 校验：初始 URL 与每次重定向都不允许落入非公网段
        - 逐块接收并在超过 max_bytes 时立即放弃，不落盘半成品
        - 落盘前校验 %PDF 魔术字节，content-type 说谎的响应直接拒绝
        """
        current_url = url
        try:
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=False
            ) as client:
                for _ in range(MAX_REDIRECTS + 1):
                    await WebReader.validate_public_url(current_url)
                    async with client.stream(
                        "GET", current_url, headers={"User-Agent": BROWSER_UA}
                    ) as res:
                        if res.is_redirect:
                            location = res.headers.get("location")
                            if not location:
                                return False
                            current_url = urllib.parse.urljoin(current_url, location)
                            continue
                        if res.status_code != 200:
                            return False
                        content_type = res.headers.get("content-type", "").lower()
                        looks_like_pdf = (
                            "application/pdf" in content_type
                            or current_url.lower().endswith(".pdf")
                        )
                        if not looks_like_pdf:
                            return False
                        body = bytearray()
                        async for chunk in res.aiter_bytes():
                            body.extend(chunk)
                            if len(body) > max_bytes:
                                return False
                        if not bytes(body[:5]).startswith(b"%PDF"):
                            return False
                        await asyncio.to_thread(
                            Path(cache_path).write_bytes, bytes(body)
                        )
                        return True
        except (
            ValueError,
            LookupError,
            OSError,
            httpx.HTTPError,
            httpx.InvalidURL,
        ):
            return False
        return False

    async def read_source(self, source: str) -> list[dict[str, str]]:
        """
        异步读取任意源（URL、arXiv 编号或本地路径）并切块
        """
        source = source.strip()

        # 1. 本地已有文件优先读取 (当前目录或 cache_dir 目录)
        if os.path.exists(source) or os.path.exists(
            os.path.join(self.cache_dir, source)
        ):
            return self.read_file(source)

        # 2. 正则识别 arXiv 论文（支持 URL、arxiv:ID、纯数字 ID）
        arxiv_id = self.extract_arxiv_id(source)
        if arxiv_id:
            return await self._read_arxiv(arxiv_id, original_source=source)

        # 3. 其它 HTTP/HTTPS 网络链接
        if source.startswith(("http://", "https://")):
            return await self._read_url(source)

        # 4. 兜底尝试按本地路径读取
        return self.read_file(source)

    def read_file(self, file_path: str) -> list[dict[str, str]]:
        """
        读取本地文件并切分（支持当前目录与 references/ 目录自动探测）
        """
        actual_path = file_path
        if not os.path.exists(actual_path):
            alt_path = os.path.join(self.cache_dir, file_path)
            if os.path.exists(alt_path):
                actual_path = alt_path
            else:
                raise FileNotFoundError(f"未找到本地参考文件: {file_path}")

        ext = os.path.splitext(actual_path)[1].lower()
        if ext == ".pdf":
            return self._read_pdf(actual_path)
        else:
            return self._read_text(actual_path)

    async def _read_arxiv(
        self, arxiv_id: str, original_source: str
    ) -> list[dict[str, str]]:
        """
        处理 arXiv 论文：
        1. 检查 references/ 下是否已有 arxiv_<id>.pdf 本地持久化缓存
        2. 若无则流式下载（带 SSRF 校验与大小上限）并存入 references/，实现本地沉淀
        3. 优雅降级到 HTML Abstract 抓取
        """
        cache_path = self._get_cache_path(arxiv_id, is_arxiv=True)

        # 命中本地磁盘缓存（损坏缓存会被自动清理并返回 None，落入重下流程）
        cached = self._load_cached_pdf(cache_path)
        if cached is not None:
            console.print(
                f"[bold green]✓[/bold green] 检测到本地已缓存论文: [cyan]{cache_path}[/cyan]，直接加载"
            )
            return cached

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        fallback_url = f"https://export.arxiv.org/pdf/{arxiv_id}.pdf"

        console.print(
            f"[bold cyan]⬇ 正在从 arXiv 获取论文 PDF...[/bold cyan] [dim]({pdf_url})[/dim]"
        )

        for target in [pdf_url, fallback_url]:
            if await self._download_pdf(target, cache_path, timeout=45.0):
                console.print(
                    f"[bold green]✓[/bold green] 已下载论文并持久化缓存至: [cyan]{cache_path}[/cyan]"
                )
                chunks = self._load_cached_pdf(cache_path)
                if chunks is not None:
                    return chunks
                # 下载内容解析失败，缓存已被清理；不再换源，直接走 Abstract 降级
                break

        # 降级尝试拉取摘要页
        console.print(
            "[yellow]! arXiv PDF 下载未成功，尝试回退抓取 Abstract 网页内容...[/yellow]"
        )
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"
        text = await WebReader.fetch_and_clean(abs_url)
        if text:
            return self.chunk_text(text, source=original_source)

        raise RuntimeError(f"未能成功拉取 arXiv 论文内容: {arxiv_id}")

    async def _read_url(self, url: str) -> list[dict[str, str]]:
        """
        普通 URL 读取：
        - 若为 PDF 链接或响应返回 PDF，自动下载并缓存至 references/
        - 若为普通网页，提取正文清洗切块
        """
        cache_path = self._get_cache_path(url, is_arxiv=False)
        cached = self._load_cached_pdf(cache_path)
        if cached is not None:
            console.print(
                f"[bold green]✓[/bold green] 检测到本地已缓存文件: [cyan]{cache_path}[/cyan]，直接加载"
            )
            return cached

        if await self._download_pdf(url, cache_path):
            console.print(
                f"[bold green]✓[/bold green] 已下载文件并持久化缓存至: [cyan]{cache_path}[/cyan]"
            )
            chunks = self._load_cached_pdf(cache_path)
            if chunks is not None:
                return chunks
            # 下载内容损坏，缓存已被清理，降级网页抓取

        text = await WebReader.fetch_and_clean(url)
        if text:
            return self.chunk_text(text, source=url)

        return []

    def _read_pdf(self, file_path: str) -> list[dict[str, str]]:
        doc = pymupdf.open(file_path)
        chunks = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = cast(str, page.get_text("text")).strip()
            if not text:
                continue

            page_chunks = self._split_text(text)
            for c in page_chunks:
                chunks.append(
                    {
                        "text": c,
                        "source": os.path.basename(file_path),
                        "page": str(page_num + 1),
                    }
                )
        doc.close()
        return chunks

    def _read_text(self, file_path: str) -> list[dict[str, str]]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        text_chunks = self._split_text(text)
        return [
            {
                "text": c,
                "source": os.path.basename(file_path),
                "page": "1",
            }
            for c in text_chunks
        ]

    def chunk_text(self, text: str, source: str) -> list[dict[str, str]]:
        """通用文本切块入口：供搜索原文等外部抓取文本入库复用。"""
        return [
            {
                "text": c,
                "source": source,
                "page": "1",
            }
            for c in self._split_text(text)
        ]

    def _split_text(self, text: str) -> list[str]:
        """按段落与滑动窗口切片"""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_chunk = ""

        for p in paragraphs:
            if len(current_chunk) + len(p) <= self.chunk_size:
                current_chunk += ("\n\n" if current_chunk else "") + p
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(p) > self.chunk_size:
                    start = 0
                    while start < len(p):
                        end = start + self.chunk_size
                        chunks.append(p[start:end])
                        start += self.chunk_size - self.chunk_overlap
                    current_chunk = ""
                else:
                    current_chunk = p

        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [text]
