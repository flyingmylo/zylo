import os

import pymupdf


class DocumentReader:
    """
    文档读取与切片器，支持 PDF, Markdown, TXT
    将本地长论文或参考资料切分为语义段落块
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def read_file(self, file_path: str) -> list[dict[str, str]]:
        """
        读取文件并切分，返回:
        [{"text": "...", "source": file_path, "page": page_num}, ...]
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self._read_pdf(file_path)
        else:
            return self._read_text(file_path)

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
