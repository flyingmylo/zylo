import os
import re
import httpx
import pymupdf
from src.tools.web_reader import WebReader


class DocumentReader:
    """
    智能多源资料读取与切片器：
    支持本地文件（PDF, Markdown, TXT）以及远程 URL（arXiv 论文、网页正文、PDF 直链）
    自动分块并保留源路径与页码元数据
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def read_source(self, source: str) -> list[dict[str, str]]:
        """
        异步读取任意源（URL 或本地路径）并切块
        """
        source = source.strip()
        if source.startswith("http://") or source.startswith("https://"):
            return await self._read_url(source)
        return self.read_file(source)

    def read_file(self, file_path: str) -> list[dict[str, str]]:
        """
        读取本地文件并切分（支持当前目录与 references/ 目录自动探测）
        """
        actual_path = file_path
        if not os.path.exists(actual_path):
            alt_path = os.path.join("references", file_path)
            if os.path.exists(alt_path):
                actual_path = alt_path
            else:
                raise FileNotFoundError(f"未找到本地参考文件: {file_path}")

        ext = os.path.splitext(actual_path)[1].lower()
        if ext == ".pdf":
            return self._read_pdf(actual_path)
        else:
            return self._read_text(actual_path)

    async def _read_url(self, url: str) -> list[dict[str, str]]:
        """
        智能解析 URL：
        1. 若为 arXiv 页面 (arxiv.org/abs/xxx)，自动优先拉取其全文 PDF
        2. 若为 PDF 链接或返回 PDF 流，直接在内存中用 pymupdf 解析页码与段落
        3. 若为普通网页/博客，提取清洗后的正文并切分
        """
        target_pdf_url = url
        # 智能识别 arXiv 论文：自动将 abs 抽象页提升为 pdf 直链
        if "arxiv.org/abs/" in url:
            arxiv_id = url.split("arxiv.org/abs/")[-1].split("?")[0].strip("/")
            target_pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"
        }

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                res = await client.get(target_pdf_url, headers=headers)
                content_type = res.headers.get("content-type", "").lower()

                # 如果返回的是 PDF 或者是 PDF URL
                if "application/pdf" in content_type or target_pdf_url.lower().endswith(".pdf"):
                    return self._read_pdf_stream(res.content, source_name=url)

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
            # 若转 PDF 失败，降级回退到网页 HTML 抓取
            if target_pdf_url != url:
                text = await WebReader.fetch_and_clean(url)
                if text:
                    chunks = self._split_text(text)
                    return [{"text": c, "source": url, "page": "1"} for c in chunks]

        return []

    def _read_pdf_stream(self, pdf_bytes: bytes, source_name: str) -> list[dict[str, str]]:
        """直接从内存字节流解析 PDF"""
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
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
                        "source": source_name,
                        "page": str(page_num + 1),
                    }
                )
        doc.close()
        return chunks

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
