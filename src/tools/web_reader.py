import asyncio
import ipaddress
import socket
from typing import cast
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 5
ALLOWED_CONTENT_TYPES = {
    "application/xhtml+xml",
    "application/xml",
}


class WebReader:
    """带 SSRF 与响应大小防护的网页抓取清洗工具。"""

    @staticmethod
    async def validate_public_url(url: str) -> None:
        """只允许解析到公网地址的 HTTP(S) URL；不安全时抛 ValueError。"""
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("仅支持带主机名的 HTTP/HTTPS URL")
        if parsed.username or parsed.password:
            raise ValueError("URL 不允许携带用户凭据")

        host = parsed.hostname.rstrip(".")
        try:
            addresses = [ipaddress.ip_address(host)]
        except ValueError:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            loop = asyncio.get_running_loop()
            infos = await loop.getaddrinfo(
                host,
                port,
                family=socket.AF_UNSPEC,
                type=socket.SOCK_STREAM,
            )
            addresses = {
                ipaddress.ip_address(cast(str, info[4][0]).split("%", 1)[0])
                for info in infos
            }

        if not addresses or any(not address.is_global for address in addresses):
            raise ValueError("URL 解析到非公网地址")

    @staticmethod
    async def fetch_and_clean(
        url: str,
        timeout: float = 10.0,
        max_bytes: int = MAX_RESPONSE_BYTES,
    ) -> str:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"
        }
        try:
            current_url = url
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=False
            ) as client:
                for redirect_count in range(MAX_REDIRECTS + 1):
                    # 每次跳转都重新校验，防止公开 URL 重定向到内网地址。
                    await WebReader.validate_public_url(current_url)
                    async with client.stream(
                        "GET", current_url, headers=headers
                    ) as res:
                        if res.is_redirect:
                            if redirect_count == MAX_REDIRECTS:
                                return ""
                            location = res.headers.get("location")
                            if not location:
                                return ""
                            current_url = urljoin(current_url, location)
                            continue

                        if res.status_code != 200:
                            return ""

                        content_type = (
                            res.headers.get("content-type", "")
                            .split(";", 1)[0]
                            .strip()
                            .lower()
                        )
                        if not (
                            content_type.startswith("text/")
                            or content_type in ALLOWED_CONTENT_TYPES
                        ):
                            return ""

                        declared_size = res.headers.get("content-length")
                        if declared_size:
                            try:
                                if int(declared_size) > max_bytes:
                                    return ""
                            except ValueError:
                                return ""

                        body = bytearray()
                        async for chunk in res.aiter_bytes():
                            body.extend(chunk)
                            if len(body) > max_bytes:
                                return ""

                        encoding = res.encoding or "utf-8"
                        html = bytes(body).decode(encoding, errors="replace")
                        soup = BeautifulSoup(html, "html.parser")

                        # 去除噪声标签
                        for tag in soup(
                            [
                                "script",
                                "style",
                                "nav",
                                "footer",
                                "header",
                                "noscript",
                                "aside",
                            ]
                        ):
                            tag.decompose()

                        text = soup.get_text(separator="\n")
                        lines = [
                            line.strip() for line in text.splitlines() if line.strip()
                        ]
                        return "\n".join(lines)
        except (
            ValueError,
            LookupError,
            OSError,
            httpx.HTTPError,
            httpx.InvalidURL,
        ):
            return ""

        return ""
