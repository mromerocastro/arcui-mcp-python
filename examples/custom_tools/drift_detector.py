"""
drift_detector.py — template for extending ArcUI with a custom MCP tool.

This file teaches the pattern: take an offline analysis routine that uses the
scientific Python stack and expose it as a first-class MCP tool that
Claude Desktop, Cursor, Gemini, or any MCP-aware client can call by name.

Three ways to use this file
---------------------------

1. As a reference — read it, copy the ``@mcp.tool()`` block inside
   ``register()`` directly into ``src/arcui_mcp/server.py``. This is the most
   reliable wiring and works regardless of where the server is launched from.

2. As a registerable module — in ``src/arcui_mcp/server.py``, after the
   ``mcp = FastMCP(...)`` line, add::

       from examples.custom_tools.drift_detector import register
       register(mcp)

   Then launch the server from the repository root (so ``examples`` is
   importable). The ``examples/`` tree is not installed with the wheel, so
   this only works in a development checkout — not after a pip install.

3. As a standalone CLI — run the analysis offline without touching MCP::

       uv run --extra science python examples/custom_tools/drift_detector.py \\
           "<path-to-bundle-dir>" \\
           "<tag-key>" 0.01

   The bundle dir is the value returned by ``bridge.end_session()`` under the
   ``bundle_dir`` key.

Design notes
------------
The math lives in the plain ``compute_drift`` function so it is unit-testable
without MCP and reusable from other tools (e.g. a future
``detect_drift_all_tags`` that loops). The MCP wrapper is a thin shell that
just forwards a bundle path and tag name.

Dependencies
------------
``numpy`` (transitive via pandas in the ``[science]`` extras) and
``arcui_mcp.analyze``. No additional packages required.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd

from arcui_mcp.analyze import BundleData, load_bundle


# ─── Core analysis (pure function, no MCP, testable in isolation) ───

def compute_drift(
    bundle: BundleData,
    tag: str,
    slope_threshold_per_sec: float = 0.01,
) -> Dict[str, Any]:
    """
    Fit a linear trend to a tag's recorded writes and classify the drift.

    Parameters
    ----------
    bundle
        Loaded ``BundleData`` (see ``arcui_mcp.analyze.load_bundle``).
    tag
        DataStore tag key to analyze (e.g. ``"Research.ExperimentSetpoint"``).
    slope_threshold_per_sec
        Absolute slope below which the run is considered ``stable``. Units
        are tag-value per second. Default 0.01 is a sensible starting point
        for low-frequency setpoint tags; tune per tag.

    Returns
    -------
    dict
        Always contains ``ok: bool``. On success: ``tag``, ``sample_count``,
        ``slope_per_sec``, ``intercept``, ``r_squared``, ``verdict``
        (``stable`` | ``drifting_up`` | ``drifting_down``), ``threshold_per_sec``,
        ``bundle``. On failure: ``error`` and (when relevant) ``available_tags``.
    """
    writes = bundle.tag_writes
    if writes.empty or "tag" not in writes.columns:
        return {"ok": False, "error": "Bundle has no tag writes."}

    subset = writes[writes["tag"] == tag].copy()
    if subset.empty:
        return {
            "ok": False,
            "error": f"Tag '{tag}' not present in bundle.",
            "available_tags": bundle.tag_keys,
        }

    subset["ts_dt"]     = pd.to_datetime(subset["ts"], errors="coerce")
    subset["value_num"] = pd.to_numeric(subset["value"], errors="coerce")
    subset = subset.dropna(subset=["ts_dt", "value_num"])

    if len(subset) < 2:
        return {
            "ok": False,
            "error": f"Need at least 2 numeric writes for a linear fit; have {len(subset)}.",
        }

    # Time axis in seconds since the first write — keeps slope numerically
    # comparable across bundles regardless of their absolute timestamps.
    t0 = subset["ts_dt"].iloc[0]
    x = (subset["ts_dt"] - t0).dt.total_seconds().to_numpy()
    y = subset["value_num"].to_numpy()

    # Linear fit: y = slope * x + intercept.
    slope, intercept = np.polyfit(x, y, 1)

    # Coefficient of determination. Defined as 0 when the series is constant
    # so the response shape stays predictable.
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    abs_slope = abs(float(slope))
    if abs_slope < slope_threshold_per_sec:
        verdict = "stable"
    elif slope > 0:
        verdict = "drifting_up"
    else:
        verdict = "drifting_down"

    return {
        "ok":                 True,
        "tag":                tag,
        "sample_count":       int(len(subset)),
        "slope_per_sec":      float(slope),
        "intercept":          float(intercept),
        "r_squared":          float(r_squared),
        "verdict":            verdict,
        "threshold_per_sec":  float(slope_threshold_per_sec),
        "bundle":             str(bundle.bundle_dir),
    }


# ─── MCP tool registration ──────────────────────────────────────────

def register(mcp) -> None:
    """
    Attach ``detect_drift`` as an MCP tool against the supplied FastMCP instance.

    Importing this module does NOT register anything by itself — the user
    must call ``register(mcp)`` explicitly. This keeps the example
    side-effect-free for readers who only want to study the pattern.
    """

    @mcp.tool()
    async def detect_drift(
        bundle_path: str,
        tag: str,
        slope_threshold_per_sec: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Fit a linear trend to a DataStore tag's recorded writes in an ArcUI
        session bundle and classify the result as 'stable', 'drifting_up', or
        'drifting_down'.

        Use this when a researcher asks: "did the setpoint stay flat during
        the run?" or "is this measurement creeping upward across the
        session?". The session must already be ended — the analysis reads
        from the on-disk bundle, not live tag values.
        """
        bundle = load_bundle(bundle_path)
        return compute_drift(bundle, tag, slope_threshold_per_sec)


# ─── Standalone CLI ────────────────────────────────────────────────

def main() -> int:
    """Run the drift analysis from the command line and print JSON."""
    if len(sys.argv) < 3:
        print(
            "Usage: drift_detector.py <bundle_dir> <tag_key> [slope_threshold_per_sec]",
            file=sys.stderr,
        )
        return 2

    bundle_path = sys.argv[1]
    tag         = sys.argv[2]
    threshold   = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01

    bundle = load_bundle(bundle_path)
    result = compute_drift(bundle, tag, slope_threshold_per_sec=threshold)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
