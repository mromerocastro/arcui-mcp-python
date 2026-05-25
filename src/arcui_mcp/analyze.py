"""
arcui_mcp.analyze — load and compare ArcUI session bundles from Python.

A bundle is the on-disk artifact written by the ArcUI runtime when a session
ends. Its layout, as documented by the bridge API:

    {baseDir}/{sessionId}/
        manifest.json           index + SHA-256 hashes per file
        events.ndjson           chronological event stream
        tag_writes.ndjson       every authorized tag write with writer_id
        contract_snapshot.json  contract registry copy at session start

This module loads those files into pandas DataFrames and offers helpers for
the common scientific workflow: load a run, compare two runs, plot a run.

Dependencies
------------
Imports pandas and matplotlib at module load time. Install with the
``[science]`` extras::

    uv sync --extra science

Importing ``arcui_mcp.analyze`` without those packages will fail at import.
This is intentional — the module name declares the dependency, and the bare
MCP server install stays small for clients that never touch analysis code.

Tolerance
---------
- Missing artifacts are tolerated: a bundle that crashed before
  ``manifest.json`` was written still loads, with whatever NDJSON lines
  reached disk.
- Malformed NDJSON lines are skipped silently, mirroring the C# reader's
  behavior (`SessionBundleSerializer.ReadLines`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd


# ─── File-name constants (mirror SessionBundleSerializer.cs) ────────

_MANIFEST_FILE   = "manifest.json"
_EVENTS_FILE     = "events.ndjson"
_TAG_WRITES_FILE = "tag_writes.ndjson"
_CONTRACT_FILE   = "contract_snapshot.json"


# ─── Data classes ───────────────────────────────────────────────────

@dataclass
class BundleData:
    """Loaded contents of an ArcUI session bundle directory."""

    bundle_dir: Path
    manifest: dict           # parsed manifest.json (empty dict when absent)
    events: pd.DataFrame     # columns: ts, kind, subject, value, writer_id, severity, detail
    tag_writes: pd.DataFrame # columns: ts, tag, value, writer_id

    @property
    def session_id(self) -> Optional[str]:
        return self.manifest.get("session_id")

    @property
    def started_at(self) -> Optional[str]:
        return self.manifest.get("started_at")

    @property
    def ended_at(self) -> Optional[str]:
        return self.manifest.get("ended_at")

    @property
    def procedure(self) -> Optional[str]:
        return self.manifest.get("procedure")

    @property
    def tag_keys(self) -> List[str]:
        """Sorted list of distinct tag keys that received writes during the run."""
        if self.tag_writes.empty or "tag" not in self.tag_writes.columns:
            return []
        return sorted(self.tag_writes["tag"].dropna().unique().tolist())


@dataclass
class DiffReport:
    """Comparison of two bundles for hypothesis-vs-baseline analysis."""

    bundle_a: BundleData
    bundle_b: BundleData
    tags_only_in_a: List[str]
    tags_only_in_b: List[str]
    tags_in_both: List[str]
    # One row per shared tag with: tag, writes_a, writes_b, last_a, last_b,
    # mean_a, mean_b. mean_* is None when the tag holds non-numeric values.
    tag_summary: pd.DataFrame


# ─── Public API ─────────────────────────────────────────────────────

def load_bundle(path: Union[str, Path]) -> BundleData:
    """
    Read a bundle directory and return its contents.

    Parameters
    ----------
    path
        Absolute or relative path to the bundle directory (the one returned
        by ``bridge.end_session()`` under the ``bundle_dir`` key).

    Returns
    -------
    BundleData
        Wrap of the parsed manifest plus two pandas DataFrames.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist or is not a directory.
    """
    bundle_dir = Path(path).expanduser().resolve()
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")

    manifest   = _read_json(bundle_dir / _MANIFEST_FILE) or {}
    events     = _read_ndjson(bundle_dir / _EVENTS_FILE)
    tag_writes = _read_ndjson(bundle_dir / _TAG_WRITES_FILE)

    return BundleData(
        bundle_dir=bundle_dir,
        manifest=manifest,
        events=events,
        tag_writes=tag_writes,
    )


def diff_bundles(
    a: Union[str, Path, BundleData],
    b: Union[str, Path, BundleData],
) -> DiffReport:
    """
    Compare two bundles.

    Useful when a hypothesis is tested by running a baseline scenario and
    a perturbed scenario, then asking "what changed in the recorded record".

    Parameters
    ----------
    a, b
        Either a path to a bundle directory, or an already-loaded ``BundleData``.

    Returns
    -------
    DiffReport
        - ``tags_only_in_a`` / ``tags_only_in_b``: tags written in one run only.
        - ``tags_in_both``: tags written in both.
        - ``tag_summary``: per-shared-tag stats (write count, last value, mean
          when numeric). One row per tag in ``tags_in_both``.
    """
    bundle_a = a if isinstance(a, BundleData) else load_bundle(a)
    bundle_b = b if isinstance(b, BundleData) else load_bundle(b)

    tags_a = set(bundle_a.tag_keys)
    tags_b = set(bundle_b.tag_keys)
    only_a = sorted(tags_a - tags_b)
    only_b = sorted(tags_b - tags_a)
    both   = sorted(tags_a & tags_b)

    rows = []
    for tag in both:
        sub_a = bundle_a.tag_writes[bundle_a.tag_writes["tag"] == tag]
        sub_b = bundle_b.tag_writes[bundle_b.tag_writes["tag"] == tag]
        rows.append({
            "tag":       tag,
            "writes_a":  len(sub_a),
            "writes_b":  len(sub_b),
            "last_a":    sub_a["value"].iloc[-1] if not sub_a.empty else None,
            "last_b":    sub_b["value"].iloc[-1] if not sub_b.empty else None,
            "mean_a":    _safe_mean(sub_a["value"]),
            "mean_b":    _safe_mean(sub_b["value"]),
        })

    return DiffReport(
        bundle_a=bundle_a,
        bundle_b=bundle_b,
        tags_only_in_a=only_a,
        tags_only_in_b=only_b,
        tags_in_both=both,
        tag_summary=pd.DataFrame(rows),
    )


def plot_run(
    bundle: BundleData,
    tag: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    show_events: bool = True,
) -> plt.Figure:
    """
    Plot the tag-write timeseries of a bundle.

    Parameters
    ----------
    bundle
        Loaded ``BundleData``.
    tag
        When provided, plot only that single tag. When ``None``, plot one
        line per numeric tag in the bundle.
    ax
        Optional matplotlib Axes to draw into. A new Figure is created
        when ``None``.
    show_events
        When ``True``, overlay vertical dashed markers for non-``tag_changed``
        events (alarms, annotations, instructor messages). Useful as visual
        anchors during debrief.

    Returns
    -------
    matplotlib.figure.Figure
        The Figure containing the plot. Customize freely; call ``plt.show()``
        in interactive contexts.

    Raises
    ------
    ValueError
        If the bundle has no tag writes, or the requested ``tag`` is not
        present.
    """
    if bundle.tag_writes.empty:
        raise ValueError(f"Bundle {bundle.bundle_dir} has no tag writes to plot.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    writes = bundle.tag_writes.copy()
    writes["ts"]        = pd.to_datetime(writes["ts"], errors="coerce")
    writes["value_num"] = pd.to_numeric(writes["value"], errors="coerce")
    numeric_writes = writes.dropna(subset=["value_num", "ts"])

    if tag is not None:
        subset = numeric_writes[numeric_writes["tag"] == tag]
        if subset.empty:
            raise ValueError(f"Tag '{tag}' has no numeric writes in this bundle.")
        ax.plot(subset["ts"], subset["value_num"],
                marker="o", label=tag, drawstyle="steps-post")
    else:
        for t in sorted(numeric_writes["tag"].unique()):
            subset = numeric_writes[numeric_writes["tag"] == t]
            ax.plot(subset["ts"], subset["value_num"],
                    marker=".", label=t, drawstyle="steps-post")

    if show_events and not bundle.events.empty and "kind" in bundle.events.columns:
        ev = bundle.events.copy()
        ev["ts"] = pd.to_datetime(ev["ts"], errors="coerce")
        non_tag = ev[(ev["kind"] != "tag_changed") & ev["ts"].notna()]
        for _, row in non_tag.iterrows():
            ax.axvline(row["ts"], color="gray", alpha=0.3, linestyle="--")

    procedure = bundle.procedure or "(no procedure)"
    sid       = bundle.session_id or bundle.bundle_dir.name
    ax.set_title(f"Run: {sid}  ({procedure})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    return fig


# ─── Internals ──────────────────────────────────────────────────────

def _read_json(path: Path) -> Optional[dict]:
    """Parse a JSON file. Returns None on missing or malformed input."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_ndjson(path: Path) -> pd.DataFrame:
    """
    Parse an NDJSON file into a DataFrame.

    Mirrors the C# reader's tolerance: empty lines and malformed lines are
    skipped silently — a corrupt entry never blocks the rest of the file.
    """
    if not path.exists():
        return pd.DataFrame()

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                # Skip malformed line, keep going.
                pass
    return pd.DataFrame(records)


def _safe_mean(series: pd.Series) -> Optional[float]:
    """Mean of a string-typed value column, returning None when non-numeric."""
    try:
        nums = pd.to_numeric(series, errors="coerce").dropna()
        if nums.empty:
            return None
        return float(nums.mean())
    except Exception:
        return None
