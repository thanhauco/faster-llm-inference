"""Minimal stdlib HTTP helpers (no ``requests``/``aiohttp`` dependency).

Uses ``http.client`` directly so streaming responses can be read line by line
and so local inference servers are reached without going through any
environment-configured HTTP proxy.
"""

from __future__ import annotations

import http.client
import json
import ssl
from urllib.parse import urlsplit


def connect(url: str, timeout: float) -> tuple[http.client.HTTPConnection, str]:
    parts = urlsplit(url)
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    if parts.scheme == "https":
        conn = http.client.HTTPSConnection(parts.hostname, parts.port or 443, timeout=timeout,
                                           context=ssl.create_default_context())
    elif parts.scheme == "http":
        conn = http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=timeout)
    else:
        raise ValueError(f"unsupported URL scheme in {url!r}")
    return conn, path


def get_text(url: str, timeout: float = 10.0, headers: dict | None = None) -> str:
    conn, path = connect(url, timeout)
    try:
        conn.request("GET", path, headers=headers or {})
        resp = conn.getresponse()
        body = resp.read().decode("utf-8", "replace")
        if resp.status >= 400:
            raise RuntimeError(f"GET {url} -> HTTP {resp.status}: {body[:200]}")
        return body
    finally:
        conn.close()


def get_json(url: str, timeout: float = 10.0, headers: dict | None = None):
    return json.loads(get_text(url, timeout, headers))
