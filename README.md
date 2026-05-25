# ArcUI MCP — Python Server

Model Context Protocol server that exposes ArcUI's Spatial Digital Twin engine to AI clients (Claude Desktop, Cursor, Gemini, Codex, Kiro, Windsurf, VS Code, custom Python scripts).

Use it to drive a running ArcUI Unity scene conversationally: read live tag values, trigger alarms, author and run experiment scenarios, replay historical sessions, and inject events on the fly — all from natural language or from your own Python pipeline.

---

## Quick Start

### 1. Install

Requires Python 3.13+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone <this-repo>
cd arcui-mcp-python
uv sync
```

### 2. Run the server

```bash
uv run arcui-mcp-server
```

The server speaks MCP over stdio and forwards every tool call to a running ArcUI Unity scene at `http://localhost:17842` by default. Override with `ARCUI_BRIDGE_URL`.

### 3. First experiment (5 lines of Python)

With Unity running and the McpBridge component playing, open a Python REPL or notebook:

```python
import asyncio
from arcui_mcp.bridge import bridge

async def first_experiment():
    print(await bridge.get_system_health())
    print(await bridge.list_tags())

asyncio.run(first_experiment())
```

If you see a health payload and a list of tags, your pipeline is live.

---

## Available Tools

The server registers ~25 MCP tools, grouped by domain:

### Operations

| Tool | Purpose |
|---|---|
| `get_sensor_value(tag)` | Read one DataStore tag's current value. |
| `list_sensors()` | Enumerate every registered tag with current value and type. |
| `get_active_alarms()` | Alarms currently Active or Acknowledged. |
| `get_alarm_history(limit)` | Recent resolved alarms from the audit log. |
| `trigger_alarm(tag, level, message, threshold)` | Raise a visible alert (no physical action). |
| `get_system_health()` | Overall status: providers, uptime, warnings. |
| `generate_report(report_type, requested_by)` | Structured snapshot + LLM-ready narrative prompt. |
| `get_provenance()` | Recent DataStore writes with timestamp, value, writer id. |

### TimeMachine (replay historical telemetry)

| Tool | Purpose |
|---|---|
| `timemachine_play()` / `timemachine_pause()` | Resume / pause replay. |
| `timemachine_seek(target_time)` | Jump to a time offset (seconds). |
| `timemachine_forecast(tag, lookahead_seconds)` | Predict the value of a tag using loaded data. |
| `timemachine_load_session(path)` | Load a session bundle (`.ndjson`) for playback. |

### Training & Scenarios (experiments)

| Tool | Purpose |
|---|---|
| `create_scenario(id, display_name, description, events)` | Author a scripted experiment as a list of timed events. |
| `start_scenario(scenario_id)` | Begin playback of a registered scenario. |
| `list_scenarios()` | Enumerate every registered scenario. |
| `inject_event(tag_key, value_type, raw_value)` | Write a single value to a tag during a live session. |
| `start_session(procedure)` / `end_session()` | Lifecycle of a recorded session. |
| `annotate_session(label, note, author)` | Mark a meaningful moment for later review. |
| `evaluate_session()` | Read the active session's chronological record. |
| `send_instructor_message(text, instructor_name)` | Push a coaching message into the operator's chat (training mode only). |
| `export_session_for_data_science(session_id)` | Convert a closed bundle into a CSV/JSON dataset package for pandas / Colab / Edge Impulse. Omit `session_id` to target the most recently closed bundle. |

### Builder

| Tool | Purpose |
|---|---|
| `get_protocol_config(industry, equipment)` | Recommended ArcUI provider configuration. |
| `validate_context_layer(json_str)` | Schema check on a Context Layer JSON. |
| `generate_pilot_scope(vertical, timeline)` | Scope outline for a pilot deployment. |
| `list_available_tags(vertical)` | Reference tag names for a vertical. |

---

## Configuration

The server reads two environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `ARCUI_BRIDGE_URL` | `http://localhost:17842/mcp` | Base URL of the ArcUI Unity HTTP bridge. |
| `ARCUI_BRIDGE_TOKEN` | (unset) | Bearer token when the bridge enables auth. |

Set them in your MCP client configuration (Claude Desktop, Cursor, etc.) or in your shell when running the server directly.

---

## Connect from your own Python pipeline

You don't need to go through MCP to use the bridge. Import `arcui_mcp.bridge` directly from any Python code:

```python
import asyncio
from arcui_mcp.bridge import ArcUIBridge

bridge = ArcUIBridge(base_url="http://localhost:17842/mcp")

async def run():
    await bridge.start_session(procedure="thermal_drift_study_01")
    await bridge.start_scenario("scenario_baseline")
    record = await bridge.evaluate_session()
    await bridge.end_session()
    return record

record = asyncio.run(run())
```

This pattern is what makes the Python server useful as a building block for Jupyter notebooks, batch analysis pipelines, and custom orchestration scripts — not only as an MCP endpoint.

---

## Offline analysis with `arcui_mcp.analyze`

After a session ends, the bundle directory on disk is the durable record. Load it from Python — no live Unity connection needed:

```python
from arcui_mcp.analyze import load_bundle, diff_bundles, plot_run

run = load_bundle("/path/to/SessionId")
print(run.procedure, run.session_id, run.tag_keys)

fig = plot_run(run)
fig.savefig("run_overview.png")

# Hypothesis test: compare baseline vs perturbed
report = diff_bundles("/path/to/baseline", "/path/to/perturbed")
print("Tags only in perturbed:", report.tags_only_in_b)
report.tag_summary
```

`analyze` is the offline counterpart to the live `bridge`: same data model, but reads from the on-disk artifacts that survive Unity restarts. Requires the `[science]` extras (`uv sync --extra science`).

See `examples/first_experiment.ipynb` for a full end-to-end walkthrough.

### Two ways to consume a finished session

There are two complementary paths once a session is closed — pick the one that fits the consumer:

- **For Python analysis (this package).** Call `load_bundle(path)` directly. It reads the raw NDJSON in the bundle directory and returns pandas DataFrames. No intermediate format, no separate export step.
- **For Colab notebooks or Edge Impulse uploads.** Call the `export_session_for_data_science` MCP tool — or click *Export for Data Science…* in the ArcUI Hub Recording panel — to derive a portable CSV/JSON package (`timeseries.csv`, `events.csv`, `dataset_manifest.json`, `README.md`) next to the bundle. The exporter never modifies the original bundle.

Both reads can coexist on the same bundle. The MCP tool also lets a Quest 3 session ask "export this session for data science" through ARIA without anyone returning to the Unity Editor.

---

## Extending the server with your own analytical tools

Any MCP-aware AI client can call tools registered against the FastMCP instance in `server.py`. Adding your own — for domain-specific analyses, custom heuristics, ML model invocations — is straightforward.

See `examples/custom_tools/drift_detector.py` for a working template that:

- Defines a pure analysis function (`compute_drift`) using numpy.
- Wraps it as an MCP tool via a `register(mcp)` helper.
- Doubles as a standalone CLI script when run directly.

**Two ways to wire it into the server:**

1. **Copy the function into the server (most reliable).** Open `src/arcui_mcp/server.py` and paste the `@mcp.tool()` block from `drift_detector.py` between the existing tool definitions. This works regardless of where the server is launched from.

2. **Import the example as a package** (only works when the server is launched from the repo root, since `examples/` is not installed with the wheel). Add to `src/arcui_mcp/server.py`:

   ```python
   from examples.custom_tools.drift_detector import register
   register(mcp)
   ```

   Then launch with `uv run arcui-mcp-server` from the repo root.

Either way, after restart the new `detect_drift` tool becomes discoverable from Claude Desktop, Cursor, and any other connected MCP client.

---

## Contributing

If you plan to modify the notebook or the source, install the dev hooks once:

```bash
uv sync --extra dev
uv run pre-commit install
```

This wires a pre-commit hook that strips Jupyter notebook outputs and runs basic whitespace checks on every `git commit` — so notebook execution outputs (paths, embedded data, plots) never leak into the repository's history.

---

## Where it fits

`arcui-mcp-python` is the Python bridge for the **ArcUI Spatial Digital Twin** engine. The Unity SDK runs the twin, owns the DataStore, executes scenarios, and records sessions. This package exposes those capabilities to any AI client or Python script that speaks MCP or HTTP.

For the Unity SDK itself, see the ArcUI project repository.
