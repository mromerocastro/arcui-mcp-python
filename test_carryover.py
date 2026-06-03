"""
Offline checks for the cross-session carryover (handover) client wrappers.

No Unity and no network: every request is served by an in-process
``httpx.MockTransport``. Run with ``python test_carryover.py``; exits non-zero
on any assertion failure (mirrors test_bridge.py / test_idempotency.py).

Pins: the read wrappers hit the right endpoints with the right query params and
carry no idempotency key, while confirm POSTs the full payload AND an
X-Idempotency-Key (so a retried confirm cannot duplicate a handover record).
"""

import asyncio
import json

import httpx

from arcui_mcp.bridge import ArcUIBridge


def _bridge(handler) -> ArcUIBridge:
    return ArcUIBridge(transport=httpx.MockTransport(handler))


async def test_get_carryover_hits_endpoint_with_params() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(200, json={"active": False})

    b = _bridge(handler)
    await b.get_carryover("wind-turbine-07", "startup")

    req = captured["request"]
    assert req.method == "GET"
    assert req.url.path.endswith("/carryover")
    assert req.url.params.get("equipment_id") == "wind-turbine-07"
    assert req.url.params.get("procedure") == "startup"
    assert "X-Idempotency-Key" not in req.headers, "reads must not carry an idempotency key"
    print("[PASS] get_carryover hits /carryover with equipment_id + procedure")


async def test_material_includes_session_id_only_when_given() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params.get("session_id"))
        return httpx.Response(200, json={"ok": True})

    b = _bridge(handler)
    await b.get_carryover_material("ward-3-infusion-pump", "priming")
    await b.get_carryover_material("ward-3-infusion-pump", "priming", session_id="sess-42")

    assert seen[0] is None, "session_id omitted when empty"
    assert seen[1] == "sess-42", "session_id forwarded when provided"
    print("[PASS] get_carryover_material forwards session_id only when given")


async def test_confirm_posts_payload_with_idempotency_key() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"ok": True, "continuity_key": "k", "version_ts": "t", "path": "p"})

    b = _bridge(handler)
    await b.confirm_carryover(
        equipment_id="wind-turbine-07",
        procedure="startup",
        summary="two items open",
        open_items=[{"id": "i1", "text": "recheck seal torque", "status": "open"}],
        author="marlon",
    )

    req = captured["request"]
    body = captured["body"]
    assert req.method == "POST"
    assert req.url.path.endswith("/carryover/confirm")
    assert "X-Idempotency-Key" in req.headers, "confirm must be replay-safe"
    assert body["equipment_id"] == "wind-turbine-07"
    assert body["author"] == "marlon"
    assert len(body["open_items"]) == 1
    assert body["open_items"][0]["text"] == "recheck seal torque"
    # Fields the caller omitted must still serialize as empty lists, not be absent.
    assert body["watch_items"] == []
    print("[PASS] confirm_carryover POSTs full payload with an idempotency key")


async def main() -> None:
    await test_get_carryover_hits_endpoint_with_params()
    await test_material_includes_session_id_only_when_given()
    await test_confirm_posts_payload_with_idempotency_key()
    print("\nAll carryover client checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
