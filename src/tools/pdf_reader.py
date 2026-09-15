import hashlib
import os
import re
import urllib.parse

import httpx
import pymupdf
from rich.console import Console

from src.tools.web_reader import WebReader

console = Console()

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

        # 普通 URL，提取或推导安全的文件名
        parsed = urllib.parse.urlparse(identifier)
        filename = os.path.basename(parsed.path)
        if not filename.lower().endswith(".pdf"):
            clean_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", parsed.path.strip("/"))
            if not clean_name:
                clean_name = hashlib.md5(identifier.encode()).hexdigest()[:10]
            filename = f"{clean_name}.pdf"
        return os.path.join(self.cache_dir, filename)

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
        if source.startswith("http://") or source.startswith("https://"):
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
        2. 若无则流式下载并存入 references/，实现本地沉淀
        3. 优雅降级到 HTML Abstract 抓取
        """
        cache_path = self._get_cache_path(arxiv_id, is_arxiv=True)

        # 命中本地磁盘缓存
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            console.print(
                f"[bold green]✓[/bold green] 检测到本地已缓存论文: [cyan]{cache_path}[/cyan]，直接加载"
            )
            return self.read_file(cache_path)

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        fallback_url = f"https://export.arxiv.org/pdf/{arxiv_id}.pdf"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"
        }

        console.print(
            f"[bold cyan]⬇ 正在从 arXiv 获取论文 PDF...[/bold cyan] [dim]({pdf_url})[/dim]"
        )
        download_success = False

        for target in [pdf_url, fallback_url]:
            try:
                async with httpx.AsyncClient(
                    timeout=45.0, follow_redirects=True
                ) as client:
                    res = await client.get(target, headers=headers)
                    content_type = res.headers.get("content-type", "").lower()
                    if res.status_code == 200 and (
                        "application/pdf" in content_type
                        or res.content.startswith(b"%PDF")
                    ):
                        with open(cache_path, "wb") as f:
                            f.write(res.content)
                        console.print(
                            f"[bold green]✓[/bold green] 已下载论文并持久化缓存至: [cyan]{cache_path}[/cyan]"
                        )
                        download_success = True
                        break
            except Exception:
                continue

        if download_success:
            return self.read_file(cache_path)

        # 降级尝试拉取摘要页
        console.print(
            f"[yellow]! arXiv PDF 下载未成功，尝试回退抓取 Abstract 网页内容...[/yellow]"
        )
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"
        text = await WebReader.fetch_and_clean(abs_url)
        if text:
            chunks = self._split_text(text)
            return [
                {
                    "text": c,
                    "source": original_source,
                    "page": "1",
                }
                for c in chunks
            ]

        raise RuntimeError(f"未能成功拉取 arXiv 论文内容: {arxiv_id}")

    async def _read_url(self, url: str) -> list[dict[str, str]]:
        """
        普通 URL 读取：
        - 若为 PDF 链接或响应返回 PDF，自动下载并缓存至 references/
        - 若为普通网页，提取正文清洗切块
        """
        cache_path = self._get_cache_path(url, is_arxiv=False)
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            console.print(
                f"[bold green]✓[/bold green] 检测到本地已缓存文件: [cyan]{cache_path}[/cyan]，直接加载"
            )
            return self.read_file(cache_path)

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"
        }

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                res = await client.get(url, headers=headers)
                content_type = res.headers.get("content-type", "").lower()

                if res.status_code == 200 and (
                    "application/pdf" in content_type
                    or res.content.startswith(b"%PDF")
                    or url.lower().endswith(".pdf")
                ):
                    with open(cache_path, "wb") as f:
                        f.write(res.content)
                    console.print(
                        f"[bold green]✓[/bold green] 已下载文件并持久化缓存至: [cyan]{cache_path}[/cyan]"
                    )
                    return self.read_file(cache_path)

            # 否则作为普通网页抓取
            text = await WebReader.fetch_and_clean(url)
            if text:
                chunks = self._split_text(text)
                return [
                    {
                        "text": c,
                        "source": url,
                        "page": "1",
                    }
                    for c in chunks
                ]
        except Exception:
            text = await WebReader.fetch_and_clean(url)
            if text:
                chunks = self._split_text(text)
                return [{"text": c, "source": url, "page": "1"} for c in chunks]

        return []

    def _read_pdf(self, file_path: str) -> list[dict[str, str]]:
        doc = pymupdf.open(file_path)
        chunks = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
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
