"""MCP server: irys blackboard state primitives for Claude Code / Codex.

Exposes the stateful swarm's blackboard pattern as tools. Claude Code
provides the intelligence; irys provides structured state management,
signal propagation, contradiction detection, and convergence tracking.

No additional API keys required for core tools.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

_INSTRUCTIONS = """\
irys-state is a stateful blackboard for serious document analysis. Use it whenever \
the user asks you to analyze, compare, summarize, audit, extract obligations from, \
answer questions about, or synthesize across one or more documents — especially when \
accuracy, provenance, contradiction tracking, or multi-step analysis matters.

Default workflow:
1. Call irys_start_analysis with the task and docs_path. This creates a blackboard \
and returns initial context so you can begin immediately.
2. Read the returned document_sections, recent_entries, open_signals, and \
write_contract. These are your working context.
3. Read documents in passes. For large files, use irys_get_document_text with \
mark_read=true, and irys_search_documents for targeted follow-up.
4. After each reading or reasoning pass, call irys_add_entries. Record concise \
typed entries with source provenance:
   - observation: source-grounded fact
   - analysis: interpretation or conclusion
   - calculation: derived numeric/logical work
   - strategy: analysis plan or framing
   - gap: missing evidence, uncertainty, or unresolved work
5. Every entry that depends on a document should include source \
{document, section, evidence}. Use supports_entries, contradicts_entries, \
opens_questions, and addresses_signals to connect the analysis graph.
6. If you discover missing sources, unresolved contradictions, or required \
follow-up, add signals with irys_add_signal or opens_questions on entries.
7. Before final synthesis, call irys_convergence_report. Do not present a final \
answer as complete while critical blockers, disputed entries, or unread required \
documents remain.
8. When convergence is acceptable, call irys_synthesis_packet and base the final \
answer on must_include_entries, disputed_entries, open_signals, and document \
read status.
9. Use irys_save_snapshot at major milestones: after initial reading, after \
resolving major gaps, and before final synthesis.
10. Use irys_get_state to inspect accumulated evidence or resume an existing \
blackboard.

Do not treat irys-state as a storage API to use after the answer is written. \
Use it during analysis so the final answer is assembled from recorded, \
source-grounded state."""

mcp = FastMCP("irys-state", instructions=_INSTRUCTIONS)

_STORE_ROOT = Path(tempfile.gettempdir()) / "irys-state"
_blackboards: dict[str, Any] = {}
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _get_lock(bb_id: str) -> threading.Lock:
    with _locks_guard:
        if bb_id not in _locks:
            _locks[bb_id] = threading.Lock()
        return _locks[bb_id]


def _state_dir(bb_id: str) -> Path:
    d = _STORE_ROOT / bb_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_state(bb_id: str, bb: Any) -> None:
    d = _state_dir(bb_id)
    state = {
        "blackboard_id": bb_id,
        "task_instruction": bb.task_instruction,
        "iteration": bb.iteration,
        "documents": [_doc_status_dict(ds) for ds in bb.documents],
        "entries": [_entry_dict(e) for e in bb.entries],
        "signals": [_signal_dict(s) for s in bb.signals],
        "metadata": getattr(bb, "_mcp_metadata", {}),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = d / "state.tmp"
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    tmp.replace(d / "state.json")


def _save_doc_texts(bb_id: str, docs: list) -> None:
    """Write document text files once at ingestion time (not on every save)."""
    doc_dir = _state_dir(bb_id) / "documents"
    for ds in docs:
        if ds.text:
            doc_dir.mkdir(exist_ok=True)
            path = doc_dir / f"{ds.id}.txt"
            if not path.exists():
                path.write_text(ds.text, encoding="utf-8")


def _load_state(bb_id: str) -> Any:
    if bb_id in _blackboards:
        return _blackboards[bb_id]

    d = _state_dir(bb_id)
    state_file = d / "state.json"
    if not state_file.exists():
        raise FileNotFoundError(f"No blackboard found: {bb_id}")

    from .swarm.blackboard import Blackboard
    from .swarm.models import (
        DocumentStatus, Entry, EntrySource, EpistemicStatus, Signal,
        SectionIndex, WorkerRecord,
    )
    from .swarm.section_index import build_section_index

    data = json.loads(state_file.read_text(encoding="utf-8"))

    docs = []
    for dd in data.get("documents", []):
        doc_text_file = d / "documents" / f"{dd['id']}.txt"
        text = doc_text_file.read_text(encoding="utf-8") if doc_text_file.exists() else ""
        idx = build_section_index(text) if text else SectionIndex()
        docs.append(DocumentStatus(
            id=dd["id"], name=dd["name"], size_bytes=dd.get("size_bytes", 0),
            headings=dd.get("headings", []),
            structural_profile=dd.get("structural_profile"),
            read_status=dd.get("read_status", "unread"),
            sections_read=dd.get("sections_read", []),
            sections_unread=dd.get("sections_unread", []),
            section_index=idx, text=text,
        ))

    entries = []
    for ed in data.get("entries", []):
        src = None
        if ed.get("source"):
            src = EntrySource(
                document=ed["source"].get("document"),
                section=ed["source"].get("section"),
                evidence=ed["source"].get("evidence", ""),
            )
        epist = None
        if ed.get("epistemic"):
            epist = EpistemicStatus(
                classification=ed["epistemic"].get("classification", "inference"),
                source_credibility=ed["epistemic"].get("source_credibility", "unknown"),
                motivation=ed["epistemic"].get("motivation", ""),
            )
        cb = ed.get("created_by", {})
        entries.append(Entry(
            id=ed["id"], type=ed.get("type", "observation"),
            content=ed.get("content", ""),
            source=src, epistemic=epist,
            created_by=WorkerRecord(
                cb.get("worker_id", ""), cb.get("description", ""),
                cb.get("iteration", 0),
            ),
            confidence=ed.get("confidence", 0.5),
            verified=ed.get("verified"),
            tags=ed.get("tags", []),
            status=ed.get("status", "active"),
            opens_questions=ed.get("opens_questions", []),
            supports_entries=ed.get("supports_entries", []),
            contradicts_entries=ed.get("contradicts_entries", []),
            supersedes_entries=ed.get("supersedes_entries", []),
            addresses_signals=ed.get("addresses_signals", []),
        ))

    signals = []
    for sd in data.get("signals", []):
        signals.append(Signal(
            id=sd["id"], type=sd.get("type", "question"),
            content=sd.get("content", ""),
            origin_entry=sd.get("origin_entry", ""),
            priority=sd.get("priority", "medium"),
            status=sd.get("status", "open"),
            addressed_by=sd.get("addressed_by"),
            iteration_created=sd.get("iteration_created", 0),
        ))

    bb = Blackboard(
        task_instruction=data.get("task_instruction", ""),
        documents=docs, entries=entries, signals=signals,
        iteration=data.get("iteration", 0),
    )
    bb._mcp_metadata = data.get("metadata", {})
    _blackboards[bb_id] = bb

    _advance_id_counters(entries, signals)

    return bb


def _advance_id_counters(entries: list, signals: list) -> None:
    """Advance global ID counters past any IDs loaded from disk."""
    from .swarm.models import _id_lock, _entry_counter, _signal_counter
    import src.swarm.models as _models

    max_e = 0
    for e in entries:
        if e.id and e.id.startswith("e"):
            try:
                max_e = max(max_e, int(e.id[1:]))
            except ValueError:
                pass
    max_s = 0
    for s in signals:
        if s.id and s.id.startswith("s"):
            try:
                max_s = max(max_s, int(s.id[1:]))
            except ValueError:
                pass

    with _id_lock:
        if max_e > _models._entry_counter:
            _models._entry_counter = max_e
        if max_s > _models._signal_counter:
            _models._signal_counter = max_s


def _bb_summary(bb: Any) -> dict:
    active = [e for e in bb.entries if e.status == "active"]
    type_counts: dict[str, int] = {}
    for e in active:
        type_counts[e.type] = type_counts.get(e.type, 0) + 1
    open_sigs = [s for s in bb.signals if s.status == "open"]
    return {
        "iteration": bb.iteration,
        "total_entries": len(bb.entries),
        "active_entries": len(active),
        "entry_types": type_counts,
        "open_signals": len(open_sigs),
        "critical_signals": len([s for s in open_sigs if s.priority == "critical"]),
        "documents": len(bb.documents),
        "documents_unread": len([d for d in bb.documents if d.read_status == "unread"]),
    }


def _entry_dict(e: Any) -> dict:
    d: dict[str, Any] = {
        "id": e.id, "type": e.type, "content": e.content,
        "confidence": e.confidence, "status": e.status,
        "tags": e.tags,
    }
    if e.source:
        d["source"] = {
            "document": e.source.document,
            "section": e.source.section,
            "evidence": e.source.evidence,
        }
    if e.epistemic:
        d["epistemic"] = {
            "classification": e.epistemic.classification,
            "source_credibility": e.epistemic.source_credibility,
            "motivation": e.epistemic.motivation,
        }
    d["created_by"] = {
        "worker_id": e.created_by.worker_id,
        "description": e.created_by.description,
        "iteration": e.created_by.iteration,
    }
    if e.verified is not None:
        d["verified"] = e.verified
    if e.opens_questions:
        d["opens_questions"] = e.opens_questions
    if e.supports_entries:
        d["supports_entries"] = e.supports_entries
    if e.contradicts_entries:
        d["contradicts_entries"] = e.contradicts_entries
    if e.supersedes_entries:
        d["supersedes_entries"] = e.supersedes_entries
    if e.addresses_signals:
        d["addresses_signals"] = e.addresses_signals
    return d


def _signal_dict(s: Any) -> dict:
    return {
        "id": s.id, "type": s.type, "content": s.content,
        "origin_entry": s.origin_entry, "priority": s.priority,
        "status": s.status, "addressed_by": s.addressed_by,
        "iteration_created": s.iteration_created,
    }


def _doc_status_dict(ds: Any) -> dict:
    return {
        "id": ds.id, "name": ds.name, "size_bytes": ds.size_bytes,
        "headings": ds.headings, "read_status": ds.read_status,
        "sections_read": ds.sections_read,
        "sections_unread": ds.sections_unread,
    }


# ── Bootstrap ────────────────────────────────────────────────────────


@mcp.tool()
def irys_start_analysis(
    task_instruction: str,
    docs_path: str | None = None,
    metadata: str | None = None,
    max_chars: int = 24000,
) -> str:
    """Start the recommended irys document-analysis workflow.

    Creates a blackboard, ingests documents, and returns initial context
    with explicit next-step guidance. Use this as the FIRST call for any
    document analysis task.

    Args:
        task_instruction: The user's analysis question or deliverable request.
        docs_path: Path to a file or directory of documents to ingest.
        metadata: Optional JSON metadata string.
        max_chars: Initial context character budget (default: 24000).
    """
    create_result = json.loads(irys_create_blackboard(task_instruction, docs_path, metadata))
    if "error" in create_result:
        return json.dumps(create_result)

    bb_id = create_result["blackboard_id"]
    context_result = json.loads(irys_get_context(bb_id, max_chars=max_chars))

    # Mark documents returned in context as read so convergence doesn't block
    with _get_lock(bb_id):
        bb = _load_state(bb_id)
        for ds in context_result.get("document_sections", []):
            doc = next((d for d in bb.documents if d.id == ds["doc_id"]), None)
            if doc and doc.section_index:
                end_char = len(ds.get("text", ""))
                for sec in doc.section_index.sections:
                    if sec.start_char < end_char:
                        doc.mark_section_read(sec.name)
        _save_state(bb_id, bb)
    context_result["summary"] = _bb_summary(bb)

    return json.dumps({
        "blackboard_id": bb_id,
        "task_instruction": task_instruction,
        "documents": create_result.get("documents", []),
        "initial_context": context_result,
        "next_steps": [
            "Read the document_sections in initial_context. Identify source-grounded observations.",
            "Call irys_add_entries with typed entries (observation, analysis, gap, etc.) and source provenance.",
            "Use opens_questions on entries for unresolved issues — they become signals automatically.",
            "Call irys_get_document_text(mark_read=true) for remaining or truncated documents.",
            "Call irys_search_documents for targeted evidence lookup.",
            "Call irys_convergence_report before final synthesis — resolve critical blockers first.",
            "Call irys_synthesis_packet to assemble the final answer from blackboard state.",
        ],
        "entry_template": {
            "type": "observation",
            "content": "Concise source-grounded finding.",
            "source": {
                "document": "doc_id from documents list",
                "section": "section heading or location",
                "evidence": "short quote or source locator",
            },
            "confidence": 0.8,
            "tags": [],
            "opens_questions": [],
            "supports_entries": [],
            "contradicts_entries": [],
            "addresses_signals": [],
        },
    })


# ── Phase 1: Core Tools ──────────────────────────────────────────────


@mcp.tool()
def irys_create_blackboard(
    task_instruction: str,
    docs_path: str | None = None,
    metadata: str | None = None,
) -> str:
    """Create a new blackboard for document analysis.

    Args:
        task_instruction: What you want to analyze or answer about the documents.
        docs_path: Path to file or directory of documents to ingest.
        metadata: Optional JSON string of metadata to attach.
    """
    from .swarm.blackboard import Blackboard
    from .swarm.models import DocumentStatus
    from .swarm.section_index import build_section_index

    bb_id = str(uuid.uuid4())[:8]
    docs: list[DocumentStatus] = []

    if docs_path:
        from .ingestion import ingest_file, ingest_directory, SUPPORTED_EXTENSIONS
        path = Path(docs_path).resolve()
        if not path.exists():
            return json.dumps({"error": f"Path does not exist: {docs_path}"})
        try:
            if path.is_file():
                if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    return json.dumps({"error": f"Unsupported file type: {path.suffix}"})
                raw_docs = [ingest_file(path)]
            else:
                raw_docs = ingest_directory(path)
        except Exception as e:
            return json.dumps({"error": f"Ingestion error: {e}"})

        if not raw_docs:
            return json.dumps({"error": f"No supported documents found in {docs_path}"})

        for doc in raw_docs:
            idx = build_section_index(doc.text)
            docs.append(DocumentStatus(
                id=doc.id, name=doc.name, size_bytes=doc.size_bytes,
                headings=[s.name for s in idx.sections],
                sections_unread=[s.name for s in idx.sections],
                section_index=idx, text=doc.text,
            ))

    bb = Blackboard(
        task_instruction=task_instruction,
        documents=docs,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    meta: dict[str, Any] = {}
    if metadata:
        try:
            meta = json.loads(metadata)
        except json.JSONDecodeError:
            pass
    bb._mcp_metadata = meta  # type: ignore[attr-defined]
    _blackboards[bb_id] = bb
    _save_doc_texts(bb_id, docs)
    _save_state(bb_id, bb)

    return json.dumps({
        "blackboard_id": bb_id,
        "task_instruction": task_instruction,
        "documents": [_doc_status_dict(d) for d in docs],
        "summary": _bb_summary(bb),
    })


@mcp.tool()
def irys_get_context(
    blackboard_id: str,
    doc_ids: str | None = None,
    signal_ids: str | None = None,
    max_chars: int = 24000,
) -> str:
    """Get context for analysis: source text, current state, and open signals.

    Call this before doing your own analysis. Returns document text,
    current entries, and open signals so you know what to work on.

    Args:
        blackboard_id: The blackboard to get context from.
        doc_ids: Comma-separated doc IDs to focus on (default: all unread).
        signal_ids: Comma-separated signal IDs to focus on.
        max_chars: Max characters of document text to return (default: 24000).
    """
    with _get_lock(blackboard_id):
        try:
            bb = _load_state(blackboard_id)
        except FileNotFoundError:
            return json.dumps({"error": f"Blackboard not found: {blackboard_id}"})

        target_doc_ids = doc_ids.split(",") if doc_ids else None
        target_signal_ids = signal_ids.split(",") if signal_ids else None

        if target_doc_ids:
            target_docs = [d for d in bb.documents if d.id in target_doc_ids]
        else:
            unread = [d for d in bb.documents if d.read_status != "fully_read"]
            target_docs = unread if unread else bb.documents[:3]

        doc_sections = []
        chars_used = 0
        for ds in target_docs:
            if chars_used >= max_chars:
                break
            remaining = max_chars - chars_used
            text = ds.text[:remaining] if ds.text else ""
            doc_sections.append({
                "doc_id": ds.id,
                "name": ds.name,
                "text": text,
                "truncated": len(ds.text) > remaining if ds.text else False,
                "headings": ds.headings,
                "read_status": ds.read_status,
            })
            chars_used += len(text)

        open_sigs = [s for s in bb.signals if s.status == "open"]
        if target_signal_ids:
            open_sigs = [s for s in open_sigs if s.id in target_signal_ids]

        active = [e for e in bb.entries if e.status == "active"]
        relevant = active[-50:]

        return json.dumps({
            "blackboard_id": blackboard_id,
            "task_instruction": bb.task_instruction,
            "iteration": bb.iteration,
            "summary": _bb_summary(bb),
            "open_signals": [_signal_dict(s) for s in open_sigs],
            "recent_entries": [_entry_dict(e) for e in relevant],
            "document_sections": doc_sections,
            "write_contract": {
                "entry_types": ["observation", "analysis", "calculation", "strategy", "gap"],
                "signal_types": ["question", "convergence_gap", "contradiction_resolution", "source_gap"],
                "signal_priorities": ["low", "medium", "high", "critical"],
            },
        })


@mcp.tool()
def irys_add_entries(
    blackboard_id: str,
    entries: str,
    worker_id: str = "claude_code",
    worker_description: str = "",
) -> str:
    """Add structured findings to the blackboard.

    The blackboard automatically: creates signals from opens_questions,
    propagates confidence from supports/contradicts, marks contradictions
    as disputed, and tracks signal resolution.

    Args:
        blackboard_id: The blackboard to add entries to.
        entries: JSON array of entry objects. Each entry has:
            type: "observation"|"analysis"|"calculation"|"strategy"|"gap"
            content: The finding text.
            source: {document, section, evidence} or null.
            confidence: 0.0-1.0 (default 0.7).
            tags: string array.
            opens_questions: questions this finding raises (become signals).
            supports_entries: entry IDs this supports.
            contradicts_entries: entry IDs this contradicts.
            addresses_signals: signal IDs this resolves.
        worker_id: Identifier for who created these entries.
        worker_description: Description of the analysis pass.
    """
    with _get_lock(blackboard_id):
        try:
            bb = _load_state(blackboard_id)
        except FileNotFoundError:
            return json.dumps({"error": f"Blackboard not found: {blackboard_id}"})

        from .swarm.models import Entry, EntrySource, EpistemicStatus, WorkerRecord, gen_entry_id

        try:
            raw_entries = json.loads(entries)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid entries JSON: {e}"})

        if not isinstance(raw_entries, list):
            return json.dumps({"error": "entries must be a JSON array"})

        _VALID_TYPES = {"observation", "analysis", "calculation", "strategy", "gap"}

        new_entries = []
        for i, ed in enumerate(raw_entries):
            if not isinstance(ed, dict):
                return json.dumps({"error": f"Entry at index {i} must be an object, got {type(ed).__name__}"})
            conf = ed.get("confidence", 0.7)
            if not isinstance(conf, (int, float)):
                conf = 0.7
            conf = max(0.0, min(1.0, float(conf)))
            entry_type = ed.get("type", "observation")
            if entry_type not in _VALID_TYPES:
                entry_type = "observation"

            def _str_list(val: Any) -> list[str]:
                if not isinstance(val, list):
                    return []
                return [str(x) for x in val if isinstance(x, str)]

            src = None
            if isinstance(ed.get("source"), dict):
                s = ed["source"]
                src = EntrySource(
                    document=str(s.get("document", "") or ""),
                    section=str(s.get("section", "") or ""),
                    evidence=str(s.get("evidence", "") or ""),
                )
            epist = None
            if isinstance(ed.get("epistemic"), dict):
                ep = ed["epistemic"]
                epist = EpistemicStatus(
                    classification=str(ep.get("classification", "inference")),
                    source_credibility=str(ep.get("source_credibility", "unknown")),
                    motivation=str(ep.get("motivation", "")),
                )
            new_entries.append(Entry(
                id=gen_entry_id(),
                type=entry_type,
                content=str(ed.get("content", "")),
                source=src, epistemic=epist,
                created_by=WorkerRecord(worker_id, worker_description, bb.iteration),
                confidence=conf,
                tags=_str_list(ed.get("tags", [])),
                opens_questions=_str_list(ed.get("opens_questions", [])),
                supports_entries=_str_list(ed.get("supports_entries", [])),
                contradicts_entries=_str_list(ed.get("contradicts_entries", [])),
                supersedes_entries=_str_list(ed.get("supersedes_entries", [])),
                addresses_signals=_str_list(ed.get("addresses_signals", [])),
            ))

        signals_before = len(bb.signals)
        bb.add_entries_batch(new_entries)
        new_signals = bb.signals[signals_before:]

        _save_state(blackboard_id, bb)

    return json.dumps({
        "created_entries": [_entry_dict(e) for e in new_entries],
        "new_signals": [_signal_dict(s) for s in new_signals],
        "summary": _bb_summary(bb),
    })


@mcp.tool()
def irys_add_signal(
    blackboard_id: str,
    signal_type: str,
    content: str,
    priority: str = "medium",
    origin_entry: str = "",
) -> str:
    """Add a signal (question, gap, or issue) to the blackboard.

    Signals are automatically deduplicated against existing open signals.

    Args:
        blackboard_id: The blackboard to add the signal to.
        signal_type: "question"|"convergence_gap"|"contradiction_resolution"|"source_gap"
        content: Description of the question or gap.
        priority: "low"|"medium"|"high"|"critical"
        origin_entry: Entry ID that raised this signal (optional).
    """
    with _get_lock(blackboard_id):
        try:
            bb = _load_state(blackboard_id)
        except FileNotFoundError:
            return json.dumps({"error": f"Blackboard not found: {blackboard_id}"})

        from .swarm.models import Signal, gen_signal_id

        sig = Signal(
            id=gen_signal_id(), type=signal_type, content=content,
            origin_entry=origin_entry, priority=priority,
            status="open", iteration_created=bb.iteration,
        )
        count_before = len(bb.signals)
        bb.add_signal(sig)
        deduped = len(bb.signals) == count_before

        _save_state(blackboard_id, bb)

    return json.dumps({
        "signal": _signal_dict(sig),
        "deduped": deduped,
        "summary": _bb_summary(bb),
    })


@mcp.tool()
def irys_get_state(
    blackboard_id: str,
    entry_status: str = "active,disputed",
    signal_status: str = "open",
    max_entries: int = 100,
) -> str:
    """Inspect the current blackboard state.

    Args:
        blackboard_id: The blackboard to inspect.
        entry_status: Comma-separated entry statuses to include (default: active,disputed).
        signal_status: Comma-separated signal statuses to include (default: open).
        max_entries: Maximum entries to return (default: 100).
    """
    with _get_lock(blackboard_id):
        try:
            bb = _load_state(blackboard_id)
        except FileNotFoundError:
            return json.dumps({"error": f"Blackboard not found: {blackboard_id}"})

        statuses = set(entry_status.split(","))
        sig_statuses = set(signal_status.split(","))

        filtered_entries = [e for e in bb.entries if e.status in statuses][:max_entries]
        filtered_signals = [s for s in bb.signals if s.status in sig_statuses]

        return json.dumps({
            "blackboard_id": blackboard_id,
            "task_instruction": bb.task_instruction,
            "summary": _bb_summary(bb),
            "entries": [_entry_dict(e) for e in filtered_entries],
            "signals": [_signal_dict(s) for s in filtered_signals],
            "documents": [_doc_status_dict(d) for d in bb.documents],
        })


@mcp.tool()
def irys_get_document_text(
    blackboard_id: str,
    doc_id: str,
    start_char: int = 0,
    max_chars: int = 24000,
    mark_read: bool = False,
) -> str:
    """Read document text from the blackboard.

    Args:
        blackboard_id: The blackboard containing the document.
        doc_id: Document ID to read.
        start_char: Character offset to start reading from.
        max_chars: Maximum characters to return (default: 24000).
        mark_read: Mark sections covered as read.
    """
    with _get_lock(blackboard_id):
        try:
            bb = _load_state(blackboard_id)
        except FileNotFoundError:
            return json.dumps({"error": f"Blackboard not found: {blackboard_id}"})

        doc = next((d for d in bb.documents if d.id == doc_id), None)
        if not doc:
            return json.dumps({"error": f"Document not found: {doc_id}"})

        text = doc.text[start_char:start_char + max_chars]
        end_char = start_char + len(text)
        truncated = end_char < len(doc.text)

        if mark_read and doc.section_index:
            for sec in doc.section_index.sections:
                if sec.start_char < end_char and sec.end_char > start_char:
                    doc.mark_section_read(sec.name)
            _save_state(blackboard_id, bb)

    return json.dumps({
        "doc_id": doc.id, "name": doc.name,
        "start_char": start_char, "end_char": end_char,
        "text": text, "truncated": truncated,
        "total_chars": len(doc.text),
        "read_status": doc.read_status,
    })


@mcp.tool()
def irys_search_documents(
    blackboard_id: str,
    query: str,
    max_results: int = 20,
    context_chars: int = 500,
) -> str:
    """Search document text for a pattern (case-insensitive).

    Args:
        blackboard_id: The blackboard containing documents.
        query: Text pattern to search for.
        max_results: Maximum results to return.
        context_chars: Characters of context around each match.
    """
    with _get_lock(blackboard_id):
        try:
            bb = _load_state(blackboard_id)
        except FileNotFoundError:
            return json.dumps({"error": f"Blackboard not found: {blackboard_id}"})

        if not query or not query.strip():
            return json.dumps({"query": query, "results": [], "total": 0})

        results = []
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        for doc in bb.documents:
            if len(results) >= max_results:
                break
            for m in pattern.finditer(doc.text):
                if len(results) >= max_results:
                    break
                start = max(0, m.start() - context_chars // 2)
                end = min(len(doc.text), m.end() + context_chars // 2)
                results.append({
                    "doc_id": doc.id, "name": doc.name,
                    "start_char": start, "end_char": end,
                    "snippet": doc.text[start:end],
                })

    return json.dumps({"query": query, "results": results, "total": len(results)})


# ── Phase 2: Lifecycle & Convergence ─────────────────────────────────


@mcp.tool()
def irys_set_iteration(
    blackboard_id: str,
    increment: bool = True,
    expire_old_signals: bool = True,
) -> str:
    """Advance the blackboard iteration and optionally expire old signals.

    Args:
        blackboard_id: The blackboard to advance.
        increment: Increment iteration by 1 (default: true).
        expire_old_signals: Expire stale low/medium signals (default: true).
    """
    with _get_lock(blackboard_id):
        try:
            bb = _load_state(blackboard_id)
        except FileNotFoundError:
            return json.dumps({"error": f"Blackboard not found: {blackboard_id}"})

        if increment:
            bb.iteration += 1
        expired = []
        if expire_old_signals:
            before_ids = {s.id for s in bb.signals if s.status == "expired"}
            bb.expire_old_signals()
            expired = [s for s in bb.signals if s.status == "expired" and s.id not in before_ids]

        _save_state(blackboard_id, bb)

    return json.dumps({
        "iteration": bb.iteration,
        "expired_signals": [_signal_dict(s) for s in expired],
        "summary": _bb_summary(bb),
    })


@mcp.tool()
def irys_convergence_report(blackboard_id: str) -> str:
    """Check if the blackboard analysis is complete.

    Returns deterministic convergence blockers: open critical signals,
    unresolved contradictions, unread documents, and thin coverage.
    """
    with _get_lock(blackboard_id):
        try:
            bb = _load_state(blackboard_id)
        except FileNotFoundError:
            return json.dumps({"error": f"Blackboard not found: {blackboard_id}"})

        open_sigs = [s for s in bb.signals if s.status == "open"]
        critical = [s for s in open_sigs if s.priority == "critical"]
        high = [s for s in open_sigs if s.priority == "high"]
        disputed = [e for e in bb.entries if e.status == "disputed"]
        unread = [d for d in bb.documents if d.read_status == "unread"]
        partial = [d for d in bb.documents if d.read_status == "partially_read"]

        blockers = []
        if critical:
            blockers.append(f"{len(critical)} critical signal(s) unresolved")
        if disputed:
            blockers.append(f"{len(disputed)} disputed entry/entries")
        if unread:
            blockers.append(f"{len(unread)} document(s) completely unread")

        return json.dumps({
            "converged": len(blockers) == 0,
            "blockers": blockers,
            "critical_signals": [_signal_dict(s) for s in critical],
            "high_signals": [_signal_dict(s) for s in high],
            "disputed_entries": [_entry_dict(e) for e in disputed],
            "unread_documents": [_doc_status_dict(d) for d in unread],
            "partially_read_documents": [_doc_status_dict(d) for d in partial],
            "summary": _bb_summary(bb),
        })


@mcp.tool()
def irys_synthesis_packet(blackboard_id: str) -> str:
    """Get everything needed to synthesize a final answer.

    Returns the task instruction, must-include entries, open signals,
    and disputed entries for Claude Code to draft the deliverable.
    """
    with _get_lock(blackboard_id):
        try:
            bb = _load_state(blackboard_id)
        except FileNotFoundError:
            return json.dumps({"error": f"Blackboard not found: {blackboard_id}"})

        active = [e for e in bb.entries if e.status == "active"]
        high_conf = [e for e in active if e.confidence >= 0.6]
        disputed = [e for e in bb.entries if e.status == "disputed"]
        open_sigs = [s for s in bb.signals if s.status == "open"]

        return json.dumps({
            "task_instruction": bb.task_instruction,
            "must_include_entries": [_entry_dict(e) for e in high_conf],
            "disputed_entries": [_entry_dict(e) for e in disputed],
            "open_signals": [_signal_dict(s) for s in open_sigs],
            "documents": [{"name": d.name, "read_status": d.read_status} for d in bb.documents],
            "summary": _bb_summary(bb),
        })


@mcp.tool()
def irys_save_snapshot(blackboard_id: str, label: str = "") -> str:
    """Save a named snapshot of the current blackboard state.

    Args:
        blackboard_id: The blackboard to snapshot.
        label: Optional label for the snapshot.
    """
    with _get_lock(blackboard_id):
        try:
            bb = _load_state(blackboard_id)
        except FileNotFoundError:
            return json.dumps({"error": f"Blackboard not found: {blackboard_id}"})

        d = _state_dir(blackboard_id) / "snapshots"
        d.mkdir(exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_label = re.sub(r"[^\w\-]", "_", label)[:50] if label else ""
        suffix = f"_{safe_label}" if safe_label else ""
        path = d / f"{ts}{suffix}.json"
        state = json.loads((_state_dir(blackboard_id) / "state.json").read_text(encoding="utf-8"))
        path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

    return json.dumps({"path": str(path), "summary": _bb_summary(bb)})


@mcp.tool()
def irys_list_blackboards() -> str:
    """List all active blackboards in this session."""
    results = []
    if _STORE_ROOT.exists():
        for d in sorted(_STORE_ROOT.iterdir()):
            state_file = d / "state.json"
            if state_file.exists():
                try:
                    data = json.loads(state_file.read_text(encoding="utf-8"))
                    results.append({
                        "blackboard_id": d.name,
                        "task_instruction": data.get("task_instruction", "")[:100],
                        "documents": len(data.get("documents", [])),
                        "entries": len(data.get("entries", [])),
                        "updated_at": data.get("updated_at", ""),
                    })
                except Exception:
                    pass
    return json.dumps({"blackboards": results})


@mcp.tool()
def irys_supported_formats() -> str:
    """List document formats supported by irys for ingestion."""
    from .ingestion import SUPPORTED_EXTENSIONS
    return json.dumps({"formats": sorted(SUPPORTED_EXTENSIONS)})


# ── MCP Prompts (user-discoverable workflow templates) ───────────────


@mcp.prompt(
    name="analyze-documents",
    title="Irys Document Analysis",
    description="Analyze documents with source-grounded provenance tracking and convergence.",
)
def prompt_analyze_documents(task: str, docs_path: str = "") -> str:
    return (
        f"Use irys-state for this document analysis.\n\n"
        f"Task:\n{task}\n\n"
        f"Documents path:\n{docs_path or '(none — create empty blackboard)'}\n\n"
        "Workflow:\n"
        "1. Call irys_start_analysis with the task and docs_path above.\n"
        "2. Read the initial_context returned.\n"
        "3. Add source-grounded entries with irys_add_entries after each reading pass.\n"
        "4. Resolve open signals with more reading and irys_search_documents.\n"
        "5. Check irys_convergence_report — resolve blockers before finalizing.\n"
        "6. Call irys_synthesis_packet and write the final answer from blackboard state."
    )


@mcp.prompt(
    name="resume-blackboard",
    title="Irys Resume Blackboard",
    description="Resume work on an existing irys-state blackboard.",
)
def prompt_resume_blackboard(blackboard_id: str) -> str:
    return (
        f"Resume irys-state blackboard {blackboard_id}.\n\n"
        "Workflow:\n"
        "1. Call irys_get_state to see accumulated entries and signals.\n"
        "2. Call irys_get_context to get document text and open work.\n"
        "3. Address open signals and unread/partial documents.\n"
        "4. Add entries that resolve signals (use addresses_signals field).\n"
        "5. Call irys_convergence_report to check completion.\n"
        "6. Call irys_synthesis_packet when ready for the final answer."
    )


def main():
    mcp.run()


if __name__ == "__main__":
    main()
