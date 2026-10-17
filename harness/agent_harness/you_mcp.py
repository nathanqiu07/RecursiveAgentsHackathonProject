"""Host-owned You.com MCP client; credentials never enter model tools or job VMs."""

import asyncio
import json

import httpx

from .clients import MAX_WEB_BYTES, ServiceError, _json, _key

KEYED_URL = "https://api.you.com/mcp?tools=you-search,you-contents"
FREE_URL = "https://api.you.com/mcp?profile=free"
ALLOWED_TOOLS = {"you-search", "you-contents"}


class YouMCP:
    def __init__(self, key=None, *, transport=None):
        self.key = _key(key)
        self.url = KEYED_URL if self.key else FREE_URL
        self.transport = transport  # MockTransport is an explicit protocol test seam.

    async def _operation(self, name=None, arguments=None, meta=None):
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        headers = {"Authorization": "Bearer " + self.key} if self.key else {}
        async with httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(20, connect=5),
            trust_env=False,
            follow_redirects=False,
            transport=self.transport,
        ) as http:
            async with streamable_http_client(self.url, http_client=http) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    available = {tool.name for tool in listed.tools}
                    if name is None:
                        return {
                            "transport": "streamable_http",
                            "provider": "you_mcp",
                            "profile": "keyed" if self.key else "free",
                            "tools": sorted(available.intersection(ALLOWED_TOOLS)),
                            "contents_available": "you-contents" in available,
                        }
                    if name not in ALLOWED_TOOLS or name not in available:
                        raise ServiceError("you_mcp_tool_unavailable")
                    result = await session.call_tool(name, arguments or {}, meta=meta)
                    if result.isError:
                        raise ServiceError("you_mcp_tool_failed")
                    value = result.structuredContent
                    if value is None:
                        texts = [item.text for item in result.content if item.type == "text"]
                        if len(texts) != 1:
                            raise ServiceError("you_mcp_invalid_response")
                        value = _json(texts[0])
                    rendered = json.dumps(value, allow_nan=False)
                    if len(rendered.encode()) > MAX_WEB_BYTES:
                        raise ServiceError("you_mcp_response_too_large")
                    if self.key:
                        value = _json(rendered.replace(self.key, "[redacted]"))
                    return value

    def _run(self, name=None, arguments=None, meta=None):
        try:
            return asyncio.run(asyncio.wait_for(self._operation(name, arguments, meta), timeout=30))
        except ServiceError:
            raise
        except Exception as error:
            # Async transport contexts wrap safe local errors in exception groups.
            pending = [error]
            while pending:
                current = pending.pop()
                if isinstance(current, ServiceError):
                    raise ServiceError(current.code) from None
                pending.extend(getattr(current, "exceptions", ()))
            # SDK/provider diagnostics may contain headers or page content.
            raise ServiceError("you_mcp_unavailable") from None

    def capabilities(self):
        return self._run()

    def search(self, query, domains):
        return self._run(
            "you-search",
            {"query": query, "count": 5},
            {"com.you.com/search": {"include_domains": list(domains), "count": 5}},
        )

    def contents(self, url):
        return self._run("you-contents", {"urls": [url], "formats": ["markdown", "metadata"]})
