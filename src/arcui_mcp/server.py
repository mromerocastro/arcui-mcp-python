import json
from mcp.server.fastmcp import FastMCP
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .bridge import bridge
from . import knowledge

# Create the FastMCP server
mcp = FastMCP("ArcUI Spatial Digital Twin Engine")

# ==========================================
# OPERATIONS TOOLS
# ==========================================

@mcp.tool()
async def get_sensor_value(tag: str) -> Dict[str, Any]:
    """
    Read the current live value of a DataStore tag (sensor reading, status flag, counter, etc.).
    Returns the value, its runtime type, and the tag key.
    Use this before making any claim about the state of the plant — never guess values.
    """
    return await bridge.get_tag(tag)

@mcp.tool()
async def list_sensors() -> Dict[str, Any]:
    """
    List every tag currently registered in the ArcUI DataStore, with current value and type.
    Useful as a first step when the user's query does not specify an exact tag name.
    """
    return await bridge.list_tags()

@mcp.tool()
async def get_active_alarms() -> Dict[str, Any]:
    """
    Return every alarm currently in Active or Acknowledged state, ordered by severity.
    Use this when the user asks about current problems, warnings, or system status.
    """
    return await bridge.get_active_alarms()

@mcp.tool()
async def get_alarm_history(limit: int = 50) -> Dict[str, Any]:
    """
    Return the most recent resolved alarms from the in-memory audit log.
    Use this when the user asks about past incidents, trends, or 'what happened earlier'.
    """
    return await bridge.get_alarm_history(limit=limit)

@mcp.tool()
async def trigger_alarm(
    tag: str,
    level: str = "warning",
    message: str = "Alarm on {tag}: value = {value}",
    threshold: float = 0.0
) -> Dict[str, Any]:
    """
    Register and activate a new alarm for a given DataStore tag.
    Use this when the AI agent detects a condition that warrants attention.
    This does NOT enact any physical change — it only raises a visible alert in the Digital Twin.
    """
    return await bridge.trigger_alarm(tag=tag, level=level, message=message, threshold=threshold)

@mcp.tool()
async def get_system_health() -> Dict[str, Any]:
    """
    Return the overall system health report: provider connectivity, uptime, tag count, warnings.
    Status is one of 'healthy' | 'degraded' | 'critical'.
    Use this to answer questions about the reliability or stability of the system.
    """
    return await bridge.get_system_health()

@mcp.tool()
async def generate_report(report_type: str = "on-demand", requested_by: str = "mcp") -> Dict[str, Any]:
    """
    Build a structured operational report snapshot (tag values, active alarms, recent history, health).
    Returns the data package plus a ready-to-use prompt string that an LLM can turn into a narrative report.
    Use this when the user asks for a shift summary, incident report, or operational brief.
    """
    return await bridge.generate_report(report_type=report_type, requested_by=requested_by)

@mcp.tool()
async def get_provenance() -> Dict[str, Any]:
    """
    Return the most recent DataStore writes captured by the live provenance buffer.
    Each record carries the timestamp, tag key, written value, and writer id.
    Use this to audit what wrote to a tag, in what order, and from which producer.
    """
    return await bridge.get_provenance()

# ==========================================
# TIMEMACHINE TOOLS
# ==========================================

@mcp.tool()
async def timemachine_play() -> Dict[str, Any]:
    """Resume playback of historical telemetry in the TimeMachine."""
    return await bridge.timemachine_play()

@mcp.tool()
async def timemachine_pause() -> Dict[str, Any]:
    """Pause playback of historical telemetry in the TimeMachine."""
    return await bridge.timemachine_pause()

@mcp.tool()
async def timemachine_seek(target_time: float) -> Dict[str, Any]:
    """Jump to a specific time (in seconds) in the TimeMachine simulation."""
    return await bridge.timemachine_seek(target_time)

@mcp.tool()
async def timemachine_forecast(tag: str, lookahead_seconds: float) -> Dict[str, Any]:
    """Predict the future value of a tag using the pre-loaded TimeMachine scenario data."""
    return await bridge.timemachine_forecast(tag, lookahead_seconds)

@mcp.tool()
async def timemachine_load_session(path: str) -> Dict[str, Any]:
    """Dynamically load a session bundle (.ndjson) into the TimeMachine for playback."""
    return await bridge.timemachine_load_session(path)

@mcp.tool()
async def timemachine_fork() -> Dict[str, Any]:
    """
    Branch a new live session from the current TimeMachine playback head.

    Pauses playback, switches the scene to Training mode, starts a fresh
    Session, and disconnects the TimeMachine provider so subsequent
    widget / inject_event / MCP writes are not overwritten by the next
    playback frame.

    The new session is marked as a fork with structured provenance:
    parent_session_id (the loaded bundle id) and fork_time_seconds (the
    playback head when the fork happened) are recorded both on the
    Session object and as a chronological event in its journal. The
    DataStore cue tags System.Session.IsForked and
    System.Session.ForkOrigin are also published so any consumer can
    surface a "forked session" indicator.

    Returns: { ok, new_session_id, parent_session_id, fork_time_seconds }
    on success, or { error } when no Session component is in the scene.

    Use when the user asks "what if we had intervened at minute X
    instead of Y?" — fork at minute X, then drive the system manually
    or via inject_event to capture the counterfactual run as a real,
    auditable, replayable session.
    """
    return await bridge.timemachine_fork()


# ==========================================
# TRAINING & SCENARIO TOOLS
# ==========================================

# We use Pydantic models for complex nested array structures
class EventModel(BaseModel):
    offset_seconds: float = Field(..., description="Seconds after playback starts when this event fires.")
    tag_key: str = Field(..., description="DataStore tag key to write.")
    value_type: str = Field("Float", description="Float, Int, Bool, String")
    raw_value: str = Field(..., description="Value as a string.")
    description: Optional[str] = Field(None, description="Optional human note for journals and inspectors.")

@mcp.tool()
async def create_scenario(
    id: str,
    display_name: str,
    description: str,
    events: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Author a Training scenario from a JSON description and register it in the ArcUI bridge.
    The scenario is a list of timed events; each event writes a value to a DataStore tag.
    Use this when the user asks to 'create a training case' or 'simulate a fault'.
    Note: Pass 'events' as a list of dictionaries with keys: offset_seconds, tag_key, value_type, raw_value.
    """
    return await bridge.create_scenario(id, display_name, description, events)

@mcp.tool()
async def start_scenario(scenario_id: str) -> Dict[str, Any]:
    """Begin playback of a registered scenario. Pair with 'list_scenarios' to discover available ids."""
    return await bridge.start_scenario(scenario_id)

@mcp.tool()
async def list_scenarios() -> Dict[str, Any]:
    """Enumerate every scenario currently registered in the ArcUI bridge."""
    return await bridge.list_scenarios()

@mcp.tool()
async def inject_event(tag_key: str, value_type: str, raw_value: str) -> Dict[str, Any]:
    """Write a single value to a DataStore tag during a live training session."""
    return await bridge.inject_event(tag_key, value_type, raw_value)

@mcp.tool()
async def evaluate_session() -> Dict[str, Any]:
    """Read the active TrainingSession's chronological record: alarm activations, acknowledgements, tag changes, instructor messages, and ARIA answer ratings (helpful / needs-work)."""
    return await bridge.evaluate_session()

@mcp.tool()
async def send_instructor_message(text: str, instructor_name: str = "AI Instructor") -> Dict[str, Any]:
    """Push a coaching message from the AI into the user's XR chat."""
    return await bridge.send_instructor_message(text, instructor_name)

@mcp.tool()
async def start_session(procedure: str = "") -> Dict[str, Any]:
    """Begin a new ArcUI TrainingSession on the running Unity scene."""
    return await bridge.start_session(procedure)

@mcp.tool()
async def end_session() -> Dict[str, Any]:
    """Stop the active ArcUI TrainingSession."""
    return await bridge.end_session()

@mcp.tool()
async def annotate_session(label: str, note: str = "", author: str = "mcp_remote") -> Dict[str, Any]:
    """Mark a meaningful moment on the active TrainingSession with a short label and optional free-form note."""
    return await bridge.annotate_session(label, note, author)

@mcp.tool()
async def export_session_for_data_science(session_id: str = "") -> Dict[str, Any]:
    """
    Export a closed session bundle as a CSV/JSON dataset package (timeseries.csv,
    events.csv, dataset_manifest.json, README.md) ready for pandas, Colab, or
    Edge Impulse. Also writes a human-readable debrief.html — a plain-language
    summary (answer ratings, alarms, timeline) for an instructor or reviewer to
    open in any browser; its path is returned as debrief_path. Pass session_id
    to target a specific past session, or omit it to export the most recently
    closed bundle. The currently active session is refused — call end_session
    first so the journal stops appending.
    Returns absolute paths of every emitted file plus row counts.
    Use when the user asks to "export this session for data science", "save the
    last run as CSV", "give me a dataset I can open in Colab", or "show me a
    readable summary / debrief of the session".
    """
    return await bridge.export_session_for_data_science(session_id)


# ==========================================
# BUILDER TOOLS
# ==========================================

@mcp.tool()
async def get_protocol_config(industry: str, equipment: str) -> Dict[str, Any]:
    """Return a recommended ArcUI data-provider configuration for a given industry and equipment class."""
    return await bridge.get_protocol_config(industry, equipment)

@mcp.tool()
async def validate_context_layer(json_str: str) -> Dict[str, Any]:
    """Validate a Context Layer JSON object against the minimum required schema."""
    return await bridge.validate_context_layer(json_str)

@mcp.tool()
async def generate_pilot_scope(vertical: str, timeline: str = "3 months") -> Dict[str, Any]:
    """Produce a scope outline for a pilot deployment."""
    return await bridge.generate_pilot_scope(vertical, timeline)

@mcp.tool()
async def list_available_tags(vertical: str) -> Dict[str, Any]:
    """List tag names typically available for a given vertical."""
    return await bridge.list_available_tags(vertical)


# ==========================================
# KNOWLEDGE PACK TOOLS (Local RAG)
# ==========================================
# Backend: ChromaDB (vector store) + Ollama (embeddings + generation).
# Local-first — no external API keys, documents never leave the host.
#
# Gated by ARCUI_ENABLE_KNOWLEDGE_TOOLS=true. knowledge_status is always
# exposed so MCP clients can discover how to turn the rest on; the
# indexing / search / grounded-generation tools register only when the
# flag is set. Install the chromadb + ollama Python clients with:
#     uv sync --extra knowledge

@mcp.tool()
async def knowledge_status() -> Dict[str, Any]:
    """
    Return ArcUI Knowledge Pack (Local RAG) configuration status.
    Use this before indexing or querying knowledge so setup issues
    surface explicitly. Always available, even when knowledge tools
    are otherwise disabled — that is its job.
    """
    return knowledge.status()


if knowledge.is_enabled():

    @mcp.tool()
    async def create_knowledge_store(
        display_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a Knowledge Pack vector store (ChromaDB collection).
        Use one store per system / equipment when possible, e.g.
        'MQTT_Turbine' or 'Compound_Library_2026'.
        """
        return await knowledge.create_store(display_name=display_name)

    @mcp.tool()
    async def list_knowledge_stores() -> Dict[str, Any]:
        """List Knowledge Pack stores available in the local ChromaDB."""
        return await knowledge.list_stores()

    @mcp.tool()
    async def list_knowledge_documents(
        store_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        List documents already indexed into a Knowledge Pack store.
        Defaults to ARCUI_KNOWLEDGE_STORE when store_name is omitted.
        """
        return await knowledge.list_documents(store_name=store_name)

    @mcp.tool()
    async def index_knowledge_file(
        path: str,
        store_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_tokens_per_chunk: int = 500,
        max_overlap_tokens: int = 50,
    ) -> Dict[str, Any]:
        """
        Upload and index one local UTF-8 text document into the configured
        Knowledge Pack store. Use for approved manuals, SOPs, protocols,
        contracts, published papers, and scenario references. Sandboxed
        to paths under ARCUI_KNOWLEDGE_ROOTS.
        """
        return await knowledge.index_file(
            path=path,
            store_name=store_name,
            metadata=metadata,
            max_tokens_per_chunk=max_tokens_per_chunk,
            max_overlap_tokens=max_overlap_tokens,
        )

    @mcp.tool()
    async def search_training_knowledge(
        query: str,
        store_name: Optional[str] = None,
        instruction: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ask a question grounded in the Knowledge Pack store. Returns the
        generated answer plus per-chunk metadata (citations) from the
        retrieved sources.
        """
        return await knowledge.search(
            query=query,
            store_name=store_name,
            instruction=instruction,
            model=model,
        )

    @mcp.tool()
    async def generate_grounded_scenario(
        request: str,
        constraints: Optional[str] = None,
        store_name: Optional[str] = None,
        model: Optional[str] = None,
        register: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a draft ArcUI scenario grounded in the Knowledge Pack
        and constrained to the live ArcUI tag vocabulary. By default
        returns a draft for review; set register=true to also push it
        to the running Unity bridge as a registered scenario.
        """
        # Pull the live tag vocabulary so the LLM stays inside it.
        try:
            tags_resp = await bridge.list_tags()
            tags = (
                tags_resp.get("tags", [])
                if isinstance(tags_resp, dict)
                else []
            )
        except Exception:  # noqa: BLE001 — best-effort enrichment
            tags = []

        result = await knowledge.generate_scenario(
            request=request,
            tags=tags,
            constraints=constraints,
            store_name=store_name,
            model=model,
        )

        if register and isinstance(result.get("scenario"), dict):
            scenario = result["scenario"]
            result["registration"] = await bridge.create_scenario(
                scenario.get("id", ""),
                scenario.get("display_name", ""),
                scenario.get("description", ""),
                scenario.get("events", []),
            )

        return result

    @mcp.tool()
    async def generate_training_debrief(
        request: str = "General performance",
        session_json: Optional[str] = None,
        store_name: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Write an AI-authored evaluation narrative for an ArcUI session,
        grounded in the Knowledge Pack (retrieved SOPs / manuals) and
        returned as { debrief, sources }. This is the analytical "how well
        did they do?" assessment.

        Distinct from the debrief.html produced by
        export_session_for_data_science, which is a deterministic, no-AI
        "what happened?" summary file. Use that for the factual record;
        use this for an expert, cited evaluation.

        Reads the live Unity session via evaluate_session unless
        session_json is provided (raw JSON string of an already captured
        session) — pass session_json when debriefing a session that has
        already ended, since evaluate_session reads the active one.
        """
        if session_json:
            try:
                session = json.loads(session_json)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"session_json is not valid JSON: {e}"
                ) from e
        else:
            session = await bridge.evaluate_session()

        return await knowledge.generate_debrief(
            request=request,
            session=session,
            store_name=store_name,
            model=model,
        )


if __name__ == "__main__":
    # Start the FastMCP server
    mcp.run()
