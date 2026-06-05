"""
ArcUI Knowledge Pack — Local RAG over approved documents.

Implemented in Python so
the knowledge tools live in the same ecosystem the scientific Python
community already uses (chromadb, ollama, pandas, jupyter).

Backend
-------
* **ChromaDB** for vector storage. Connects to a local HTTP server
  (default ``http://localhost:8000``, override with ``CHROMA_URL``).
* **Ollama** for embeddings and generation. Assumes a local Ollama
  daemon with the configured embedding and generation models pulled.

Both backends are local-first: documents never leave the user's machine
and no third-party API keys are required. This is by design — the
Knowledge Pack indexes approved internal material (manuals, SOPs,
protocols, contracts), so keeping it on-host is part of the audit story.

Gating
------
All knowledge tools are off by default. Set
``ARCUI_ENABLE_KNOWLEDGE_TOOLS=true`` to expose them. The ``status()``
helper is always available so clients can discover how to turn the rest
on.

Sandboxing
----------
``ARCUI_KNOWLEDGE_ROOTS`` accepts a comma-separated list of directories.
``index_file`` refuses any path outside those roots. Default is ``/``
(no restriction).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


# ─── Environment-driven config ────────────────────────────────────────────────


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() == "true"


def _default_store() -> str:
    return os.environ.get("ARCUI_KNOWLEDGE_STORE", "arcui-knowledge")


def _default_model() -> str:
    return os.environ.get("ARCUI_KNOWLEDGE_MODEL", "gemma")


def _default_embedding_model() -> str:
    return os.environ.get("ARCUI_EMBEDDING_MODEL", "nomic-embed-text")


def _chroma_url() -> str:
    return os.environ.get("CHROMA_URL", "http://localhost:8000")


def is_enabled() -> bool:
    """Whether Knowledge Pack tools are exposed at all."""
    return _env_bool("ARCUI_ENABLE_KNOWLEDGE_TOOLS")


def is_indexing_enabled() -> bool:
    return is_enabled()


def is_configured() -> bool:
    """Local RAG needs no API keys — enabled is configured."""
    return is_enabled()


def allowed_roots() -> List[str]:
    raw = os.environ.get("ARCUI_KNOWLEDGE_ROOTS", "/")
    return [p for p in (raw or "").split(",") if p]


# ─── Lazy backend handles ─────────────────────────────────────────────────────
# Imports are kept lazy so a connector install without the ``knowledge`` extra
# still loads the rest of the server — only the tools that actually need
# chromadb / ollama will surface the missing-dependency error when called.


def _chroma():
    import chromadb  # type: ignore

    parsed = urlparse(_chroma_url())
    host = parsed.hostname or "localhost"
    port = parsed.port or 8000
    return chromadb.HttpClient(host=host, port=port)


def _ollama():
    import ollama  # type: ignore

    return ollama


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _chunk_text(text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
    """Word-based chunking."""
    words = text.split()
    chunks: List[str] = []
    step = max(1, max_tokens - overlap)
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + max_tokens]))
        i += step
    return chunks


def _embed(text: str) -> List[float]:
    response = _ollama().embeddings(
        model=_default_embedding_model(),
        prompt=text,
    )
    return response["embedding"]


def _resolve_within_roots(path: str) -> Path:
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    roots = allowed_roots()
    if roots and roots != ["/"]:
        in_allowed = any(
            str(file_path).startswith(str(Path(r).resolve())) for r in roots
        )
        if not in_allowed:
            raise PermissionError(
                f"Path {file_path} is outside ARCUI_KNOWLEDGE_ROOTS={roots}"
            )
    return file_path


# Common inline-LaTeX → Unicode, so academic-PDF manuals (papers, theses)
# read cleanly in the operator panel instead of showing raw "$\dot{Q}$" markup.
_LATEX_SYMBOLS = {
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\varepsilon": "ε", r"\epsilon": "ε", r"\zeta": "ζ", r"\eta": "η",
    r"\theta": "θ", r"\kappa": "κ", r"\lambda": "λ", r"\mu": "μ", r"\nu": "ν",
    r"\xi": "ξ", r"\rho": "ρ", r"\sigma": "σ", r"\tau": "τ", r"\varphi": "φ",
    r"\phi": "φ", r"\chi": "χ", r"\psi": "ψ", r"\omega": "ω", r"\pi": "π",
    r"\Delta": "Δ", r"\Sigma": "Σ", r"\Omega": "Ω", r"\Phi": "Φ",
    r"\partial": "∂", r"\nabla": "∇", r"\infty": "∞", r"\sqrt": "√",
    r"\times": "×", r"\cdot": "·", r"\pm": "±", r"\mp": "∓",
    r"\leq": "≤", r"\geq": "≥", r"\neq": "≠", r"\approx": "≈",
    r"\rightarrow": "→", r"\Rightarrow": "⇒", r"\circ": "°", r"\degree": "°",
}


def _clean_latex(text: str) -> str:
    """Best-effort conversion of inline LaTeX math (common in academic PDFs) to
    plain, operator-readable text. NOT a full LaTeX engine: it maps frequent
    Greek letters / symbols to Unicode, unwraps simple accents, \\text{} and
    sub/superscripts, then strips leftover $ delimiters, braces and
    backslash-commands. Goal is a readable manual citation, not typeset math.
    """
    if not text:
        return text

    # \text{...}, \mathrm{...}, \mathbf{...} → their content
    text = re.sub(r"\\(?:text|mathrm|mathbf|mathit|operatorname)\{([^{}]*)\}", r"\1", text)
    # accents \dot{Q} \hat{x} \bar{y} → inner symbol (accent dropped)
    text = re.sub(r"\\(?:dot|ddot|hat|bar|vec|tilde)\{([^{}]*)\}", r"\1", text)
    # degree: ^\circ → °
    text = re.sub(r"\^\s*\\circ", "°", text)
    # sub/superscripts → inline content, dropping the ^ _ markers and braces
    text = re.sub(r"[\^_]\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"[\^_]([A-Za-z0-9])", r"\1", text)
    # known symbol commands → Unicode (longest first, so \varepsilon beats \epsilon)
    for cmd in sorted(_LATEX_SYMBOLS, key=len, reverse=True):
        text = text.replace(cmd, _LATEX_SYMBOLS[cmd])
    # LaTeX line break "\\" → space
    text = text.replace("\\\\", " ")
    # strip any remaining \command tokens (\frac, \left, \right, …)
    text = re.sub(r"\\[A-Za-z]+", "", text)
    # drop the $ math delimiters and now-orphan braces
    text = text.replace("$", "").replace("{", "").replace("}", "")
    # tidy doubled spaces introduced by the removals
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def _extract_pdf_text(file_path: Path) -> str:
    """Extract text from a PDF with pypdf.

    Text-based PDFs only. Scanned / image-only PDFs yield little or no
    extractable text and would need OCR, which is intentionally out of scope —
    the function raises a clear error in that case rather than indexing empty
    pages.
    """
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as e:  # pragma: no cover - dependency hint
        raise RuntimeError(
            "Indexing a PDF needs the 'pypdf' package. Install the knowledge "
            "extras: uv sync --extra knowledge"
        ) from e

    reader = PdfReader(str(file_path))
    text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    if not text:
        raise RuntimeError(
            f"No extractable text found in {file_path.name}. It may be a scanned "
            "/ image-only PDF, which needs OCR (not supported)."
        )
    return _clean_latex(text)


def _read_document_text(file_path: Path) -> str:
    """Read a document as UTF-8 text. ``.pdf`` files are parsed with pypdf;
    everything else is read as UTF-8 text directly."""
    if file_path.suffix.lower() == ".pdf":
        return _extract_pdf_text(file_path)
    return file_path.read_text(encoding="utf-8")


# ─── Public API ───────────────────────────────────────────────────────────────


def status() -> Dict[str, Any]:
    return {
        "enabled": is_enabled(),
        "indexing_enabled": is_indexing_enabled(),
        "configured": is_configured(),
        "default_store": _default_store() if is_enabled() else None,
        "default_model": _default_model(),
        "embedding_model": _default_embedding_model(),
        "chroma_url": _chroma_url(),
        "allowed_roots": allowed_roots(),
        "note": (
            "Set ARCUI_ENABLE_KNOWLEDGE_TOOLS=true to enable Local RAG tools. "
            "Install the knowledge extras (uv sync --extra knowledge) for "
            "chromadb + ollama Python clients."
            if not is_enabled()
            else "Local RAG is ready. Ensure Ollama and ChromaDB are running. "
            "Set ARCUI_KNOWLEDGE_STORE or pass store_name to query / index."
        ),
    }


async def create_store(display_name: Optional[str] = None) -> Dict[str, Any]:
    store_name = display_name or _default_store()
    try:
        client = _chroma()
        client.create_collection(name=store_name)
        return {"success": True, "store_name": store_name}
    except Exception as e:  # noqa: BLE001 — surface backend errors verbatim
        return {"success": False, "error": str(e)}


async def list_stores() -> Dict[str, Any]:
    try:
        client = _chroma()
        collections = client.list_collections()
        return {"stores": [{"name": c.name} for c in collections]}
    except Exception as e:  # noqa: BLE001
        return {
            "stores": [],
            "error": str(e),
            "hint": f"Ensure ChromaDB server is running at {_chroma_url()}",
        }


async def list_documents(store_name: Optional[str] = None) -> Dict[str, Any]:
    name = store_name or _default_store()
    try:
        client = _chroma()
        collection = client.get_collection(name=name)
        results = collection.get()
        metadatas = results.get("metadatas") or []
        sources = {
            m.get("source")
            for m in metadatas
            if isinstance(m, dict) and m.get("source")
        }
        return {"documents": sorted(sources)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "documents": []}


async def index_file(
    path: str,
    store_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    max_tokens_per_chunk: int = 500,
    max_overlap_tokens: int = 50,
) -> Dict[str, Any]:
    name = store_name or _default_store()
    file_path = _resolve_within_roots(path)

    text = _read_document_text(file_path)
    chunks = _chunk_text(text, max_tokens_per_chunk, max_overlap_tokens)

    client = _chroma()
    collection = client.get_or_create_collection(name=name)

    source = str(file_path)
    base_meta: Dict[str, Any] = dict(metadata or {})
    base_meta["source"] = source

    # Make (re-)indexing idempotent: drop any chunks previously indexed from this
    # same source before inserting the new ones, so re-running index_file on a
    # document REPLACES it cleanly — no duplicates, and no stale chunks lingering
    # when the new version is shorter. Other documents are untouched (the filter
    # is scoped to this source). A no-op for a source that was never indexed.
    collection.delete(where={"source": source})

    # IDs are derived from the full source path (not just the file name), so two
    # different files that happen to share a name never collide.
    source_id = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]

    ids: List[str] = []
    documents: List[str] = []
    embeddings: List[List[float]] = []
    metadatas: List[Dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        ids.append(f"{source_id}-chunk-{i}")
        documents.append(chunk)
        embeddings.append(_embed(chunk))
        metadatas.append({**base_meta, "chunk_index": i})

    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )

    return {
        "success": True,
        "indexed_chunks": len(chunks),
        "file": str(file_path),
        "store": name,
    }


async def search(
    query: str,
    store_name: Optional[str] = None,
    instruction: Optional[str] = None,
    model: Optional[str] = None,
    n_results: int = 5,
) -> Dict[str, Any]:
    name = store_name or _default_store()
    client = _chroma()
    collection = client.get_collection(name=name)
    embedding = _embed(query)
    results = collection.query(query_embeddings=[embedding], n_results=n_results)

    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    context = "\n\n---\n\n".join(docs)

    instruction_line = f"{instruction}\n" if instruction else ""
    prompt = (
        "Context information is below:\n"
        "---------------------\n"
        f"{context}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, answer the "
        "query.\n"
        f"Query: {instruction_line}{query}\n"
        "Answer:\n"
    )

    response = _ollama().generate(
        model=model or _default_model(),
        prompt=prompt,
    )

    return {"text": response["response"], "grounding": metas}


def retrieve_sync(
    query: str,
    store_name: Optional[str] = None,
    n_results: int = 5,
) -> Dict[str, Any]:
    """Retrieve grounding passages WITHOUT LLM generation (synchronous).

    Returns the top matching chunks plus per-chunk citation metadata and a
    similarity distance, leaving answer synthesis to the caller. This is the
    path used by a caller that is itself the LLM voice (e.g. ARIA in Unity):
    it applies its own grounded-or-silent policy and cites the returned
    sources, so no second generation happens here.

    Distinct from :func:`search`, which also runs an Ollama generation pass
    and returns a finished answer.
    """
    name = store_name or _default_store()
    client = _chroma()
    collection = client.get_collection(name=name)
    embedding = _embed(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=max(1, n_results),
        include=["documents", "metadatas", "distances"],
    )

    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]

    matches: List[Dict[str, Any]] = []
    for i, text in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        if not isinstance(meta, dict):
            meta = {}
        matches.append(
            {
                "text": text,
                "source": meta.get("source"),
                "metadata": meta,
                "distance": dists[i] if i < len(dists) else None,
            }
        )

    return {"matches": matches, "store": name, "query": query}


async def retrieve(
    query: str,
    store_name: Optional[str] = None,
    n_results: int = 5,
) -> Dict[str, Any]:
    """Async wrapper over :func:`retrieve_sync` for MCP tool parity."""
    return retrieve_sync(query, store_name=store_name, n_results=n_results)


async def generate_scenario(
    request: str,
    tags: Optional[List[Any]] = None,
    constraints: Optional[str] = None,
    store_name: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    name = store_name or _default_store()
    client = _chroma()
    collection = client.get_collection(name=name)
    embedding = _embed(request)
    results = collection.query(query_embeddings=[embedding], n_results=5)

    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    context = "\n\n---\n\n".join(docs)

    prompt = (
        "You are an expert scenario designer for an ArcUI Spatial Digital "
        "Twin.\n"
        "Context information from approved knowledge documents is below:\n"
        "---------------------\n"
        f"{context}\n"
        "---------------------\n"
        f"Available live tags: {json.dumps(tags or [])}\n\n"
        "Create a scenario in valid JSON format. It must match this schema:\n"
        "{\n"
        '  "id": "scenario-id",\n'
        '  "display_name": "Scenario Title",\n'
        '  "description": "Scenario description",\n'
        '  "events": [\n'
        '    { "offset_seconds": 0, "tag_key": "some_tag", '
        '"value_type": "Float", "raw_value": "100.0", '
        '"description": "event description" }\n'
        "  ]\n"
        "}\n\n"
        f"Constraints: {constraints or 'None'}\n"
        f"Request: {request}\n\n"
        "Output ONLY valid JSON."
    )

    response = _ollama().generate(
        model=model or _default_model(),
        prompt=prompt,
        format="json",
    )

    try:
        scenario = json.loads(response["response"])
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse LLM output as JSON: {e}") from e

    return {"scenario": scenario, "grounded_sources": metas}


async def generate_debrief(
    request: str = "evaluation debrief",
    session: Optional[Dict[str, Any]] = None,
    store_name: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    name = store_name or _default_store()
    client = _chroma()
    collection = client.get_collection(name=name)
    embedding = _embed(request)
    results = collection.query(query_embeddings=[embedding], n_results=5)

    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    context = "\n\n---\n\n".join(docs)

    session_dump = json.dumps(session or {}, indent=2)

    prompt = (
        "You are an expert evaluator for an ArcUI Spatial Digital Twin.\n"
        "Context information from approved knowledge documents is below:\n"
        "---------------------\n"
        f"{context}\n"
        "---------------------\n"
        f"Session Data:\n{session_dump}\n\n"
        f"Write a comprehensive debrief focusing on: {request}"
    )

    response = _ollama().generate(
        model=model or _default_model(),
        prompt=prompt,
    )

    return {"debrief": response["response"], "sources": metas}
