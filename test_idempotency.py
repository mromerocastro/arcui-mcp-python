"""
Offline conformance checks for the bridge's idempotency + retry behaviour.

No Unity and no network required: every request is served by an in-process
``httpx.MockTransport``. Run with ``python test_idempotency.py`` (mirrors the
style of ``test_bridge.py``); exits non-zero if any assertion fails.

What is pinned here:
  • POST calls attach an ``X-Idempotency-Key``; GET calls do not.
  • A dropped response is retried, reusing the SAME key, so the bridge can
    collapse the duplicate delivery into a single execution.
  • An HTTP 409 (identical request still in flight) is retried, not surfaced.
  • A genuine HTTP error (4xx) is surfaced immediately, never retried.
  • Exhausted transport retries raise a clear connection error.
"""

import asyncio

import httpx

import arcui_mcp.bridge as bridge_module
from arcui_mcp.bridge import ArcUIBridge, _MAX_ATTEMPTS


def _make_bridge(handler) -> ArcUIBridge:
    return ArcUIBridge(transport=httpx.MockTransport(handler))


async def test_post_attaches_key_get_does_not() -> None:
    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"ok": True})

    b = _make_bridge(handler)
    await b._post("/alarms/trigger", json_data={"tag": "x"})
    await b._get("/tags")

    assert "X-Idempotency-Key" in captured[0].headers, "POST must carry an idempotency key"
    assert "X-Idempotency-Key" not in captured[1].headers, "GET must not carry an idempotency key"
    print("[PASS] POST attaches X-Idempotency-Key; GET does not")


async def test_transport_error_retries_with_stable_key() -> None:
    keys = []
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        keys.append(request.headers.get("X-Idempotency-Key"))
        if state["calls"] < 3:
            raise httpx.ConnectError("connection dropped")
        return httpx.Response(200, json={"ok": True})

    b = _make_bridge(handler)
    result = await b._post("/session/inject", json_data={"tag_key": "x"})

    assert result == {"ok": True}
    assert state["calls"] == 3, "should retry until success"
    assert keys[0] is not None
    assert len(set(keys)) == 1, "the idempotency key must be identical across retries"
    print("[PASS] Transport errors retry with a stable idempotency key")


async def test_409_is_retried() -> None:
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] == 1:
            return httpx.Response(409, json={"error": "idempotency key already in progress"})
        return httpx.Response(200, json={"ok": True})

    b = _make_bridge(handler)
    result = await b._post("/session/start", json_data={"procedure": ""})

    assert result == {"ok": True}
    assert state["calls"] == 2, "409 should trigger exactly one retry here"
    print("[PASS] HTTP 409 (in-flight) is retried, not surfaced")


async def test_http_error_is_not_retried() -> None:
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    b = _make_bridge(handler)
    raised = False
    try:
        await b._post("/alarms/trigger", json_data={})
    except RuntimeError:
        raised = True

    assert raised, "a 4xx must raise"
    assert state["calls"] == 1, "genuine HTTP errors must not be retried"
    print("[PASS] Genuine HTTP errors surface immediately without retry")


async def test_exhausted_retries_raise_connection_error() -> None:
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        raise httpx.ConnectError("unreachable")

    b = _make_bridge(handler)
    raised_msg = ""
    try:
        await b._get("/tags")
    except RuntimeError as e:
        raised_msg = str(e)

    assert "Failed to connect" in raised_msg, "should surface a clear connection error"
    assert state["calls"] == _MAX_ATTEMPTS, "should exhaust all attempts before giving up"
    print("[PASS] Exhausted transport retries raise a clear connection error")


async def main() -> None:
    # Drop the inter-attempt backoff so the suite runs instantly.
    bridge_module._RETRY_BASE_DELAY_SECONDS = 0

    await test_post_attaches_key_get_does_not()
    await test_transport_error_retries_with_stable_key()
    await test_409_is_retried()
    await test_http_error_is_not_retried()
    await test_exhausted_retries_raise_connection_error()
    print("\nAll idempotency/retry checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
