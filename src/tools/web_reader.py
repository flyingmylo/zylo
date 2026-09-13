import httpx
from bs4 import BeautifulSoup


class WebReader:
    """网页抓取与清洗工具"""

    @staticmethod
    async def fetch_and_clean(url: str, timeout: float = 10.0) -> str:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"
        }
        try:
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=True
            ) as client:
                res = await client.get(url, headers=headers)
                if res.status_code != 200:
                    return ""
                html = res.text
                soup = BeautifulSoup(html, "html.parser")

                # 去除噪声标签
                for tag in soup(
                    ["script", "style", "nav", "footer", "header", "noscript", "aside"]
                ):
                    tag.decompose()

                text = soup.get_text(separator="\n")
                # 去除多余空行
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                return "\n".join(lines)
        except Exception:
            return ""
