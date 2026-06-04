import asyncio
import httpx
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Retry policy for transient failures. Mutable POSTs are made replay-safe by an
# X-Idempotency-Key header (the bridge executes the first delivery and replays
# its cached response for any retry carrying the same key), so a lost response
# can be retried without duplicating a command against the ArcUI runtime. GETs
# are naturally idempotent and reuse the same path.
_MAX_ATTEMPTS = 3
_RETRY_BASE_DELAY_SECONDS = 0.25

class ArcUIBridge:
    """
    HTTP client that bridges MCP Tool Calls to the ArcUI bridge endpoint.
    Assumes the ArcUI runtime is up and listening on localhost:17842 by default.
    Override with the ARCUI_BRIDGE_URL environment variable.
    """
    def __init__(
        self,
        base_url: str = "http://localhost:17842/mcp",
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.base_url = os.getenv("ARCUI_BRIDGE_URL", base_url).rstrip("/")
        token = os.getenv("ARCUI_BRIDGE_TOKEN", "")
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        # ``transport`` is an injection seam for tests (e.g. httpx.MockTransport);
        # it is None in normal use, leaving httpx to pick its default transport.
        self.client = httpx.AsyncClient(timeout=10.0, headers=headers, transport=transport)

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Issue a request to the bridge with idempotent, transient-failure retries.

        A POST carries one ``X-Idempotency-Key`` generated per logical call and
        reused across every retry, so the bridge collapses duplicate deliveries
        into a single execution. Retries fire only on transport-level failures
        (a lost/dropped response) and on HTTP 409 (an identical request still in
        flight on the bridge); genuine HTTP errors are surfaced, not retried.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        request_headers: Optional[Dict[str, str]] = None
        if method == "POST":
            request_headers = {"X-Idempotency-Key": str(uuid.uuid4())}

        last_error: Optional[Exception] = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = await self.client.request(
                    method, url, params=params, json=json_data, headers=request_headers
                )

                # 409 means an identical request (same idempotency key) is still
                # being processed. Back off and retry; the retry replays the
                # cached result once the in-flight call completes.
                if response.status_code == 409 and attempt < _MAX_ATTEMPTS:
                    await asyncio.sleep(_RETRY_BASE_DELAY_SECONDS * attempt)
                    continue

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                # The bridge answered with a real error (4xx/5xx). Not retried —
                # the decision is final and the body is meaningful to relay.
                logger.error(f"HTTP error from Unity Bridge: {e.response.text}")
                raise RuntimeError(f"ArcUI Bridge error: {e.response.text}")

            except httpx.TransportError as e:
                # The response was lost in transit (connection dropped/timeout).
                # Safe to retry: GETs are idempotent and POSTs carry a stable
                # idempotency key, so a re-send cannot duplicate a command.
                last_error = e
                if attempt < _MAX_ATTEMPTS:
                    logger.warning(
                        f"Transient bridge error on {method} {url} "
                        f"(attempt {attempt}/{_MAX_ATTEMPTS}): {e}. Retrying."
                    )
                    await asyncio.sleep(_RETRY_BASE_DELAY_SECONDS * attempt)
                    continue
                break

        logger.error(f"Connection error to Unity Bridge: {last_error}")
        raise RuntimeError(
            f"Failed to connect to ArcUI Unity Bridge at {url}. Is Unity running?"
        )

    async def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return await self._request("GET", endpoint, params=params)

    async def _post(self, endpoint: str, json_data: Optional[Dict[str, Any]] = None) -> Any:
        return await self._request("POST", endpoint, json_data=json_data)

    # --- Operations Tools ---
    async def get_tag(self, key: str) -> Dict[str, Any]:
        return await self._get("/tag", params={"key": key})

    async def list_tags(self) -> Dict[str, Any]:
        return await self._get("/tags")

    async def get_active_alarms(self) -> Dict[str, Any]:
        return await self._get("/alarms/active")

    async def get_alarm_history(self, limit: int = 50) -> Dict[str, Any]:
        return await self._get("/alarms/history", params={"limit": limit})

    async def trigger_alarm(self, tag: str, level: str = "warning", message: str = "", threshold: float = 0.0) -> Dict[str, Any]:
        payload = {"tag": tag, "level": level, "message": message, "threshold": threshold}
        return await self._post("/alarms/trigger", json_data=payload)

    async def get_system_health(self) -> Dict[str, Any]:
        return await self._get("/health")

    async def generate_report(self, report_type: str = "on-demand", requested_by: str = "mcp") -> Dict[str, Any]:
        payload = {"type": report_type, "requested_by": requested_by}
        return await self._post("/report", json_data=payload)

    async def get_provenance(self) -> Dict[str, Any]:
        return await self._get("/provenance")

    # --- TimeMachine Tools ---
    async def timemachine_play(self) -> Dict[str, Any]:
        return await self._post("/timemachine/play")

    async def timemachine_pause(self) -> Dict[str, Any]:
        return await self._post("/timemachine/pause")

    async def timemachine_seek(self, target_time: float) -> Dict[str, Any]:
        return await self._post("/timemachine/seek", json_data={"target_time": target_time})

    async def timemachine_forecast(self, tag: str, lookahead_seconds: float) -> Dict[str, Any]:
        return await self._post("/timemachine/forecast", json_data={"tag": tag, "lookahead_seconds": lookahead_seconds})

    async def timemachine_load_session(self, path: str) -> Dict[str, Any]:
        return await self._post("/timemachine/load", json_data={"path": path})

    async def timemachine_fork(self) -> Dict[str, Any]:
        return await self._post("/timemachine/fork")

    # --- Training Tools ---
    async def create_scenario(self, id: str, display_name: str, description: str, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload = {
            "id": id,
            "display_name": display_name,
            "description": description,
            "events": events
        }
        return await self._post("/scenario/create", json_data=payload)

    async def start_scenario(self, scenario_id: str) -> Dict[str, Any]:
        return await self._post("/scenario/start", json_data={"scenario_id": scenario_id})

    async def list_scenarios(self) -> Dict[str, Any]:
        return await self._get("/scenario/list")

    async def inject_event(self, tag_key: str, value_type: str, raw_value: str) -> Dict[str, Any]:
        payload = {"tag_key": tag_key, "value_type": value_type, "raw_value": raw_value}
        return await self._post("/session/inject", json_data=payload)

    async def evaluate_session(self) -> Dict[str, Any]:
        return await self._get("/session/evaluate")

    async def send_instructor_message(self, text: str, instructor_name: str = "AI Instructor") -> Dict[str, Any]:
        payload = {"text": text, "instructor_name": instructor_name}
        return await self._post("/session/instructor-message", json_data=payload)

    async def start_session(self, procedure: str = "") -> Dict[str, Any]:
        return await self._post("/session/start", json_data={"procedure": procedure})

    async def end_session(self) -> Dict[str, Any]:
        return await self._post("/session/end")

    async def annotate_session(self, label: str, note: str = "", author: str = "mcp_remote") -> Dict[str, Any]:
        payload = {"label": label, "note": note, "author": author}
        return await self._post("/session/annotate", json_data=payload)

    async def export_session_for_data_science(self, session_id: str = "") -> Dict[str, Any]:
        """
        Convert a closed session bundle into a CSV/JSON dataset package.

        Pass ``session_id`` to target a specific past session, or leave it empty
        to export the most recently closed bundle. The active session is never
        eligible — call :py:meth:`end_session` first so the journal stops
        appending to the bundle. The response carries the absolute paths of the
        emitted files plus the row counts for quick sanity checks; this includes
        ``debrief_path``, a human-readable ``debrief.html`` summary intended for
        an instructor or reviewer (the machine-readable CSV/JSON sit alongside).
        """
        payload = {"session_id": session_id or ""}
        return await self._post("/session/export", json_data=payload)

    def session_artifact_urls(self, session_id: str = "") -> Dict[str, Any]:
        """
        Build direct download URLs for a recorded session's human-readable
        ``debrief.html`` and full dataset ZIP, served by the ArcUI bridge.

        Pure URL construction — no network call. The URLs point at whatever
        bridge this client is configured to reach (``ARCUI_BRIDGE_URL``): the
        headset over ``adb forward`` / LAN, or a local editor. The endpoints
        themselves validate the session and return 404 if it is unknown, and
        require the same Authorization token as every other bridge call.
        """
        from urllib.parse import quote

        query = f"?session_id={quote(session_id)}" if session_id else ""
        return {
            "debrief_url": f"{self.base_url}/session/debrief{query}",
            "dataset_zip_url": f"{self.base_url}/session/export.zip{query}",
            "session_id": session_id or "(most recent closed session)",
            "note": (
                "Open debrief_url in a browser to view the visual report. "
                "Both URLs honour the bridge Authorization token when one is set."
            ),
        }

    # --- Continuity / Carryover Tools ---
    async def get_carryover(self, equipment_id: str, procedure: str) -> Dict[str, Any]:
        return await self._get(
            "/carryover",
            params={"equipment_id": equipment_id, "procedure": procedure},
        )

    async def get_carryover_material(
        self, equipment_id: str, procedure: str, session_id: str = ""
    ) -> Dict[str, Any]:
        params = {"equipment_id": equipment_id, "procedure": procedure}
        if session_id:
            params["session_id"] = session_id
        return await self._get("/carryover/material", params=params)

    async def confirm_carryover(
        self,
        equipment_id: str,
        procedure: str,
        summary: str,
        open_items: Optional[List[Dict[str, Any]]] = None,
        watch_items: Optional[List[Dict[str, Any]]] = None,
        key_annotations: Optional[List[str]] = None,
        unresolved_alarms: Optional[List[str]] = None,
        source_session_id: str = "",
        author: str = "",
    ) -> Dict[str, Any]:
        payload = {
            "equipment_id": equipment_id,
            "procedure": procedure,
            "summary": summary,
            "open_items": open_items or [],
            "watch_items": watch_items or [],
            "key_annotations": key_annotations or [],
            "unresolved_alarms": unresolved_alarms or [],
            "source_session_id": source_session_id,
            "author": author,
        }
        # _post attaches an X-Idempotency-Key, so a retried confirm cannot
        # write a duplicate handover record.
        return await self._post("/carryover/confirm", json_data=payload)

    # --- Builder Tools (stubs for now) ---
    async def get_protocol_config(self, industry: str, equipment: str) -> Dict[str, Any]:
        # Stubbbed implementation directly in python
        catalog = {
            "energy/wind-turbine":   {"protocol": "MQTT",   "broker": "mqtt://broker.local:1883", "tags": ["rotor_rpm", "pitch_angle", "wind_speed", "grid_power"]},
            "energy/solar-farm":     {"protocol": "MQTT",   "broker": "mqtt://broker.local:1883", "tags": ["dc_voltage", "inverter_temp", "ac_power", "irradiance"]},
            "medical/infusion-pump": {"protocol": "REST",   "baseUrl": "http://device.local/api", "tags": ["flow_rate", "volume_delivered", "occlusion"]},
            "industrial/reactor":    {"protocol": "OPC-UA", "endpoint": "opc.tcp://plc.local:4840", "tags": ["temperature", "pressure", "agitation_rpm"]},
            "defense/radar":         {"protocol": "WebSocket","url": "wss://radar.local/ws", "tags": ["track_count", "mode", "range_km"]},
        }
        key = f"{industry}/{equipment}".lower()
        return {
            "match": key,
            "recommendation": catalog.get(key, {"protocol": "REST", "note": "No reference entry — using REST default.", "tags": []}),
            "note": "Stub implementation — Month 4+ will read from the ArcUI reference context library."
        }

    async def validate_context_layer(self, json_str: str) -> Dict[str, Any]:
        import json
        errors = []
        try:
            parsed = json.loads(json_str)
            if not isinstance(parsed, dict):
                errors.append("Root must be a JSON object.")
                return {"valid": False, "errors": errors}

            # 1. Validate version (v1.0 or v1.1)
            version = parsed.get("arcui_context_version", "1.0")
            if version not in ("1.0", "1.1"):
                errors.append(f"Unsupported arcui_context_version: '{version}'. Must be '1.0' or '1.1'.")

            # 2. Validate 'system' block
            system = parsed.get("system")
            if system is not None and not isinstance(system, dict):
                errors.append("'system' must be a JSON object containing metadata.")
            elif system is not None and not system.get("name"):
                errors.append("Missing required field: 'system.name'.")

            # 3. Validate 'tags' dictionary
            tags = parsed.get("tags")
            if tags is None:
                errors.append("Missing required field: 'tags'.")
            elif not isinstance(tags, dict):
                errors.append("'tags' must be a JSON object (dictionary), not an array/list.")
            else:
                # Basic validation per tag
                for tag_key, tag_data in tags.items():
                    if not isinstance(tag_data, dict):
                        errors.append(f"Tag '{tag_key}' must be a JSON object.")
                        continue

                    # If v1.1, check minimum contract requirements
                    if version == "1.1":
                        tag_type = tag_data.get("type")
                        tag_role = tag_data.get("role")
                        tag_owner = tag_data.get("owner")

                        valid_types = {"float", "int", "bool", "string"}
                        valid_roles = {"measured", "setpoint", "state"}

                        if not tag_type:
                            errors.append(f"Tag '{tag_key}': missing required contract field 'type'.")
                        elif tag_type not in valid_types:
                            errors.append(f"Tag '{tag_key}': invalid type '{tag_type}'. Must be one of {valid_types}.")

                        if not tag_role:
                            errors.append(f"Tag '{tag_key}': missing required contract field 'role'.")
                        elif tag_role not in valid_roles:
                            errors.append(f"Tag '{tag_key}': invalid role '{tag_role}'. Must be one of {valid_roles}.")

                        if not tag_owner:
                            errors.append(f"Tag '{tag_key}': missing required contract field 'owner'.")

        except Exception as e:
            return {"valid": False, "errors": [f"JSON parse error: {str(e)}"]}

        return {"valid": len(errors) == 0, "errors": errors}

    async def generate_pilot_scope(self, vertical: str, timeline: str = "3 months") -> Dict[str, Any]:
        return {
            "vertical": vertical,
            "timeline": timeline,
            "scope": f"Proposed pilot for {vertical} over {timeline}.",
            "note": "Stub implementation."
        }

    async def list_available_tags(self, vertical: str) -> Dict[str, Any]:
        return {
            "vertical": vertical,
            "tags": [f"mock_{vertical}_tag_1", f"mock_{vertical}_tag_2"],
            "note": "Stub implementation."
        }

bridge = ArcUIBridge()
