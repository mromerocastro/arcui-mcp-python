from mcp.server.fastmcp import FastMCP
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .bridge import bridge

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
    """Read the active TrainingSession's chronological record: alarm activations, acknowledgements, and tag changes."""
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

if __name__ == "__main__":
    # Start the FastMCP server
    mcp.run()
