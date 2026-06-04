"""
ArcUI Knowledge retrieval HTTP endpoint.

A minimal, local-first HTTP server that exposes ONE capability — grounded
passage retrieval — so an in-headset client (e.g. an XR assistant running on a
standalone device over the LAN) can ground its answers in the same Knowledge
Pack store the MCP tools index, without speaking MCP itself.

Why a separate, tiny server
---------------------------
The MCP server (``server.py``) talks to MCP hosts over stdio; the bridge
(``bridge.py``) is a *client* of the runtime. Neither exposes a plain HTTP
surface a device-side HTTP client can POST to. Rather than reimplement the
ChromaDB + Ollama retrieval on the client (duplicated logic, coupled to the
Chroma wire API), this endpoint keeps all retrieval in one place and hands back
``{matches: [{text, source, ...}]}``. The caller does its own answer synthesis
and citation.

Security & scope
----------------
* Off unless ``ARCUI_ENABLE_KNOWLEDGE_TOOLS=true`` — same gate as the tools.
* Binds to ``127.0.0.1`` by default. For device use over the LAN, set
  ``ARCUI_KNOWLEDGE_HTTP_HOST=0.0.0.0`` (or a specific LAN IP) AND set a bearer
  token via ``ARCUI_KNOWLEDGE_HTTP_TOKEN`` so the endpoint is not left open.
* Retrieval only. It never indexes, never writes, never generates — there is
  no path from this endpoint to mutate a store.

Run it
------
    arcui-knowledge-server            # console script (see pyproject)
    python -m arcui_mcp.knowledge_server
"""

from __future__ import annotations

import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import knowledge

logger = logging.getLogger(__name__)


def _host() -> str:
    return os.environ.get("ARCUI_KNOWLEDGE_HTTP_HOST", "127.0.0.1")


def _port() -> int:
    try:
        return int(os.environ.get("ARCUI_KNOWLEDGE_HTTP_PORT", "17900"))
    except ValueError:
        return 17900


def _token() -> str:
    return os.environ.get("ARCUI_KNOWLEDGE_HTTP_TOKEN", "")


class _Handler(BaseHTTPRequestHandler):
    server_version = "ArcUIKnowledge/1.0"

    # Route Python's default per-request stderr logging through the module
    # logger so it respects the configured level and format.
    def log_message(self, fmt, *args):  # noqa: A003
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        token = _token()
        if not token:
            return True
        return self.headers.get("Authorization", "") == f"Bearer {token}"

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/") == "/knowledge/health":
            self._send(200, {"ok": True, "enabled": knowledge.is_enabled()})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path.rstrip("/") != "/knowledge/retrieve":
            self._send(404, {"error": "not found"})
            return
        if not knowledge.is_enabled():
            self._send(
                503,
                {
                    "error": "Knowledge tools disabled. "
                    "Set ARCUI_ENABLE_KNOWLEDGE_TOOLS=true."
                },
            )
            return
        if not self._authorized():
            self._send(401, {"error": "unauthorized"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            req = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "body is not valid JSON"})
            return

        query = (req.get("query") or "").strip()
        if not query:
            self._send(400, {"error": "missing 'query'"})
            return

        store_name = req.get("store_name") or None
        try:
            n_results = int(req.get("n_results", 5))
        except (TypeError, ValueError):
            n_results = 5

        try:
            result = knowledge.retrieve_sync(
                query=query, store_name=store_name, n_results=n_results
            )
            self._send(200, result)
        except Exception as e:  # noqa: BLE001 — surface backend errors as 500 JSON
            logger.exception("retrieve failed")
            self._send(500, {"error": str(e)})


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    host, port = _host(), _port()

    if not knowledge.is_enabled():
        logger.warning(
            "ARCUI_ENABLE_KNOWLEDGE_TOOLS is not 'true' — /knowledge/retrieve "
            "will answer 503 until it is set. Starting anyway so /knowledge/health works."
        )
    if host not in ("127.0.0.1", "localhost") and not _token():
        logger.warning(
            "Binding to %s WITHOUT ARCUI_KNOWLEDGE_HTTP_TOKEN — the retrieval "
            "endpoint is reachable on the network with no auth. Set a token.",
            host,
        )

    httpd = ThreadingHTTPServer((host, port), _Handler)
    logger.info("ArcUI Knowledge retrieval endpoint on http://%s:%d", host, port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
