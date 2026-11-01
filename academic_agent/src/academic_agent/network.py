"""Bounded HTTP reads with explicit routing and credential-safe failures.

No environment proxies or TLS overrides are used. Host addresses are checked
before every hop. An injected resolver/transport permits fully offline tests.
"""

from __future__ import annotations

import ipaddress
import json
import socket
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import httpx

DOMESTIC_DOMAINS = frozenset(
    "126.com 163.com alibaba.com alicdn.com alipay.com aliyun.com baidu.com "
    "bdstatic.com bilibili.com bilivideo.com bytedance.com byteimg.com cnki.net "
    "coding.net csdn.net ctrip.com dianping.com douban.com douyin.com gitee.com "
    "huawei.com huaweicloud.com jd.com meituan.com mi.com pinduoduo.com qcloud.com "
    "qq.com sohu.com taobao.com tencent.com tmall.com trip.com weibo.com "
    "xiaohongshu.com zhihu.com zhimg.com".split()
)
_SENSITIVE = ("key", "token", "secret", "password", "signature", "credential", "authorization")


def redact_url(url: str) -> str:
    """Remove user info/fragments and redact common credential query fields."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        host = f"[{host}]" if ":" in host else host
        netloc = host + (f":{parts.port}" if parts.port else "")
        params = [(k, "[REDACTED]" if any(s in k.lower() for s in _SENSITIVE)
                   or k.lower() in {"auth", "code", "sig"} else v)
                  for k, v in parse_qsl(parts.query, keep_blank_values=True)]
        return urlunsplit((parts.scheme, netloc, parts.path, urlencode(params), ""))
    except ValueError:
        return "[invalid URL]"


def route_for_url(url: str) -> str:
    """Return direct for domestic/local destinations, otherwise proxy."""
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    try:
        if not ipaddress.ip_address(host).is_global:
            return "direct"
    except ValueError:
        pass
    if ("." not in host or host.endswith((".localhost", ".local", ".cn", ".中国", ".公司", ".网络",
                                         ".xn--fiqs8s", ".xn--55qx5d", ".xn--io0a7i"))
            or any(host == d or host.endswith("." + d) for d in DOMESTIC_DOMAINS)):
        return "direct"
    return "proxy"


class NetworkError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.rate_limited = status_code == 429


class HttpTransport:
    def __init__(
        self, proxy_url: str | None = None, *, timeout: float = 20,
        max_bytes: int = 8_000_000, retries: int = 2, max_redirects: int = 3,
        max_retry_after: float = 10, transport: httpx.BaseTransport | None = None,
        allow_private: bool = False, resolver: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if timeout <= 0 or max_bytes <= 0 or not 0 <= retries <= 2 or not 0 <= max_redirects <= 10:
            raise ValueError("Invalid network bounds")
        if not 0 <= max_retry_after <= 60:
            raise ValueError("Retry-After cap must be between zero and 60 seconds")
        self.proxy_url = proxy_url
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.retries = retries
        self.max_redirects = max_redirects
        self.max_retry_after = max_retry_after
        self.transport = transport
        self.allow_private = allow_private
        self.resolver = resolver or socket.getaddrinfo
        self.sleep = sleep

    def _validate_target(self, url: str) -> None:
        try:
            parts = urlsplit(url)
            if parts.scheme not in {"https", "http"} or not parts.hostname or parts.username or parts.password:
                raise ValueError
            host = parts.hostname.rstrip(".")
            port = parts.port or (443 if parts.scheme == "https" else 80)
        except ValueError:
            raise NetworkError("Invalid or credential-bearing target URL") from None
        if self.allow_private:
            return
        try:
            addresses = [ipaddress.ip_address(host)]
        except ValueError:
            if host.lower() == "localhost" or host.lower().endswith((".localhost", ".local")) or "." not in host:
                raise NetworkError("Private/local destination blocked")
            try:
                addresses = [ipaddress.ip_address(x[4][0]) for x in self.resolver(host, port, type=socket.SOCK_STREAM)]
            except (OSError, ValueError):
                raise NetworkError("Destination address could not be validated") from None
        if not addresses or any(not a.is_global for a in addresses):
            raise NetworkError("Private/non-public destination blocked")

    def _delay(self, value: str | None, attempt: int) -> float:
        delay = min(2 ** attempt, self.max_retry_after)
        if value:
            try:
                delay = float(value)
            except ValueError:
                try:
                    delay = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
                except (TypeError, ValueError, OverflowError):
                    pass
        return max(0, min(delay, self.max_retry_after))

    def get_bytes(self, url: str, params: Mapping[str, Any] | None = None) -> bytes:
        try:
            current = str(httpx.URL(url, params=params)) if params is not None else url
        except (ValueError, httpx.InvalidURL):
            raise NetworkError("Invalid target URL") from None
        redirects = attempts = 0
        while True:
            self._validate_target(current)
            route = route_for_url(current)
            if route == "proxy" and not self.proxy_url and self.transport is None:
                raise NetworkError("Foreign destination requires an explicit proxy")
            kwargs: dict[str, Any] = {"timeout": self.timeout, "trust_env": False, "follow_redirects": False}
            if self.transport is not None:
                kwargs["transport"] = self.transport
            elif route == "proxy":
                kwargs["proxy"] = self.proxy_url
            retry_delay = None
            try:
                with httpx.Client(**kwargs) as client:
                    with client.stream("GET", current, headers={"User-Agent": "academic-work-agent/0.1"}) as response:
                        code = response.status_code
                        if code in {301, 302, 303, 307, 308}:
                            if redirects >= self.max_redirects or not response.headers.get("location"):
                                raise NetworkError("Redirect limit exceeded or missing location", status_code=code)
                            next_url = urljoin(current, response.headers["location"])
                            if urlsplit(current).scheme == "https" and urlsplit(next_url).scheme != "https":
                                raise NetworkError("Insecure redirect blocked", status_code=code)
                            current = next_url
                            redirects += 1
                            continue
                        if code == 429 or code in {500, 502, 503, 504}:
                            if attempts < self.retries:
                                retry_delay = self._delay(response.headers.get("retry-after"), attempts)
                            else:
                                raise NetworkError(f"HTTP {code}; retry budget exhausted", status_code=code)
                        elif not 200 <= code < 300:
                            raise NetworkError(f"HTTP {code}", status_code=code)
                        else:
                            length = response.headers.get("content-length", "")
                            if length.isdigit() and int(length) > self.max_bytes:
                                raise NetworkError("Response exceeds byte limit")
                            data = bytearray()
                            for chunk in response.iter_bytes(chunk_size=min(65536, self.max_bytes + 1)):
                                data.extend(chunk)
                                if len(data) > self.max_bytes:
                                    raise NetworkError("Response exceeds byte limit")
                            return bytes(data)
            except httpx.HTTPError:
                if attempts >= self.retries:
                    raise NetworkError("HTTP transport failed; retry budget exhausted") from None
                retry_delay = self._delay(None, attempts)
            except (ValueError, OSError):
                raise NetworkError("HTTP transport configuration or connection failed") from None
            attempts += 1
            self.sleep(retry_delay or 0)

    def get_json(self, url: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        body = self.get_bytes(url, params)
        try:
            result = json.loads(body)
        except (ValueError, UnicodeError):
            raise NetworkError("Expected JSON response; invalid JSON or HTML error page") from None
        if not isinstance(result, dict):
            raise NetworkError("Expected JSON object response")
        return result
