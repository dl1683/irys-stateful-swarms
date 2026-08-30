"""Action executors — read, search, bind, analyze, verify.

Each executor is a bounded job for a cheap model: it receives exactly the
state slice it needs and writes claims back to the board. Workers may
propose new targets freely (discovery must never ask permission); the
ledger maintenance pass grooms proposals later.

Bind is an LLM call by design: mapping claims to targets is semantic
work, and rules would smuggle domain assumptions into the architecture.
"""
from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

from .hydration import build_evidence_context, source_claims_for_hydration
from .llm import call_json
from .state import CLAIM_KINDS, Board, Claim, Source, Target, Unit

# Smaller chunks = more parallel extraction calls, each with the full output
# budget — dense documents (policies, schedules, tables) lose their tail when
# one call must cover too much text.
_CHUNK_CHARS = int(os.getenv("LOOP_READ_CHUNK_CHARS", "20000"))
_MAX_PARALLEL = 8
_BIND_BATCH = 60

_ANALYZE_HYDRATE = os.getenv("LOOP_ANALYZE_HYDRATE", "0").strip().lower() in (
    "1", "true", "yes",
)

def execute_actions(actions: list[dict], board: Board, worker_caller,
                    smart_caller=None) -> dict:
    """Run an iteration's actions in parallel. Returns summary counts.

    worker_caller: cheap tier for read/search/bind/verify.
    smart_caller: judgment tier for analyze (falls back to worker_caller).
    """
    analyze_caller = smart_caller or worker_caller
    jobs = []
    for idx, action in enumerate(actions):
        action["_id"] = f"a{board.iteration}.{idx}"
        kind = action.get("kind", "")
        if kind == "read":
            jobs.extend(_read_jobs(action, board))
        elif kind == "search":
            jobs.append(("search", action))
        elif kind == "bind":
            jobs.extend(_bind_jobs(action, board))
        elif kind == "analyze":
            jobs.append(("analyze", action))
        elif kind == "verify":
            jobs.append(("verify", action))

    summary = {"claims": 0, "targets_proposed": 0, "bound": 0, "verified": 0,
               "jobs": len(jobs), "failed": 0}
    if not jobs:
        return summary

    with ThreadPoolExecutor(max_workers=_MAX_PARALLEL) as pool:
        futures = {}
        for kind, payload in jobs:
            fn = {
                "read_chunk": _run_read_chunk,
                "search": _run_search,
                "bind_batch": _run_bind_batch,
                "analyze": _run_analyze,
                "verify": _run_verify,
            }[kind]
            caller = analyze_caller if kind == "analyze" else worker_caller
            futures[pool.submit(fn, payload, board, caller)] = kind
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception as e:
                summary["failed"] += 1
                board.log("action_error", f"{futures[fut]} failed: {e}")
                continue
            for k, v in (result or {}).items():
                summary[k] = summary.get(k, 0) + v
    # Persist the iteration's structural action summary — admission
    # denominators and support paths must survive into artifacts, not just
    # the next controller prompt.
    board.log(
        "action_summary",
        f"iter {board.iteration}: {summary.get('claims', 0)} claims added, "
        f"{summary.get('claims_admitted', 0)} admitted of "
        f"{summary.get('claims_offered', 0)} offered",
        detail=dict(summary),
    )
    return summary


# --- READ ---

def _read_jobs(action: dict, board: Board) -> list[tuple[str, dict]]:
    source = board.find_source(str(action.get("source_id", "")))
    if source is None:
        return []
    text = source.text()
    if not _usable_source_text(text):  # whitespace-only = failed extraction
        board.log("read_skipped",
                  f"{source.id}: unusable source text (empty/whitespace)",
                  detail={"source_id": source.id,
                          "reason": "whitespace_only" if text else "empty"})
        return []
    source.read_status = "read"
    focus = str(action.get("focus", ""))
    target_ids = [str(t) for t in action.get("target_ids", [])]
    jobs = []
    for i in range(0, len(text), _CHUNK_CHARS):
        base = {
            "source": source,
            "chunk": text[i:i + _CHUNK_CHARS],
            "chunk_start": i,
            "chunk_end": min(i + _CHUNK_CHARS, len(text)),
            "chunk_no": i // _CHUNK_CHARS + 1,
            "chunks_total": (len(text) - 1) // _CHUNK_CHARS + 1,
            "focus": focus,
            "target_ids": target_ids,
            "action_id": action.get("_id", ""),
        }
        # Extraction depth is the controller's call: 'exhaustive' adds an
        # inventory lens (funnel analysis: 58% of failed criteria were never
        # extracted on completeness tasks), but the flood drowns drafting
        # tasks — so the lens is chosen per read, not fixed policy.
        jobs.append(("read_chunk", {**base, "mode": "guided"}))
        if str(action.get("depth", "")).lower() == "exhaustive":
            jobs.append(("read_chunk", {**base, "mode": "inventory"}))
    return jobs


def _run_read_chunk(job: dict, board: Board, caller) -> dict:
    source: Source = job["source"]
    chunk_note = (
        f" (part {job['chunk_no']}/{job['chunks_total']})"
        if job["chunks_total"] > 1 else ""
    )
    focus_note = f"\nFOCUS: {job['focus']}" if job["focus"] else ""

    if job.get("mode") == "inventory":
        # Breadth lens: no target framing — inventory everything citable.
        framing = """You are building a complete factual inventory of a document. Ignore any notion of relevance — extract EVERY specific, citable fact: every amount, date, deadline, party, defined term, obligation, condition, exception, threshold, percentage, cross-reference, schedule item, and named provision. Exact values, never paraphrased approximations. A fact you skip is a fact the system permanently lacks."""
    else:
        targets_text = _targets_brief(board, job["target_ids"])
        framing = f"""You are extracting evidence from a document for a research task. Extract every specific, citable fact: amounts, dates, parties, defined terms, obligations, conditions, numbers, named provisions. Exact values, never paraphrased approximations.

TASK CONTEXT:
{board.instruction}

QUESTIONS THIS READ SERVES:
{targets_text}{focus_note}"""

    set_valued = [o for o in board.obligations if o.set_valued and o.status == "open"]
    units_ask = ""
    units_schema = ""
    if set_valued:
        ob_list = "\n".join(f"  {o.id}: {o.text}" for o in set_valued)
        units_ask = f"""
COVERAGE OBLIGATIONS (the answer must account for every repeated item under these):
{ob_list}
If this text contains the repeated items an obligation tracks (numbered categories, named provisions, listed terms, schedule rows, enumerated issues), report each as a unit with its source anchor. Units are source-native names, never speculative."""
        units_schema = """,
 "units": [{"obligation_id": "...", "name": "<source-native item name>", "anchor": "<section/number/heading>"}]"""

    prompt = f"""{framing}
{units_ask}
DOCUMENT: {source.name}{chunk_note}
---
{job['chunk']}
---

Return JSON:
{{"claims": [{{"kind": "observation", "content": "<the fact, specific and self-contained>", "section": "<section/heading it came from>", "evidence": "<short exact quote copied verbatim from the document>", "confidence": 0.0-1.0}}],
 "proposed_targets": [{{"need": "<new question this document raises that the task must answer>", "materiality": "critical|high|medium|low"}}],
 "proposed_reads": [{{"source_hint": "<document/exhibit/schedule name, if stated>", "section_hint": "<section/page/clause to read next>", "reason": "<why this referenced material matters>", "target_ids": ["<target ids this would help, if known>"]}}]{units_schema}}}

Rules:
- kind is usually "observation". Use "contradiction" if this text conflicts with itself, "gap" if something expected is conspicuously absent, "issue" for a clear defect/risk stated in the text.
- kind "requirement" ONLY for constraints on the work product being created (the document this task will produce): its addressee and submission address, who signs/submits it, its length or format, its filing deadline, elements it must contain, references it must make, procedural requests it must include. Obligations that documents impose on parties (notice duties, filing duties, contractual obligations of the insured/permittee/borrower) are "observation", NEVER "requirement".
- Be exhaustive on facts relevant to the questions; include other clearly material facts too.
- Dense term-bearing text (policy declarations, schedules, fee tables, defined-term lists) demands EVERY term: every limit, sublimit, deductible, retention, date, exclusion, endorsement, and amount — completeness over brevity.
- proposed_targets only for genuinely new material questions, not restatements. A target must be a QUESTION answerable from the sources or web search — advice or actions for the client ("negotiate X", "obtain Y") are claims (recommendation/gap), never targets.
- evidence must be copied verbatim from the document. It is used to locate the source span; do not paraphrase it.
- proposed_reads are for explicit cross-references or clearly missing referenced materials, e.g. "Section 8.3", "Exhibit B", "Schedule 2.1". Do not propose generic extra research.
- proposed_reads should be few and high-value; omit them if no specific referenced source/section is visible."""

    parsed = call_json(caller, board, prompt, kind="read", max_tokens=32768)
    if job.get("mode") == "inventory" and isinstance(parsed, dict):
        parsed.pop("proposed_targets", None)  # breadth lens has no task context
    tag = "read_inv" if job.get("mode") == "inventory" else "read"
    result = _ingest_claims(
        parsed, board, source=source,
        created_by=f"{tag}:{job.get('action_id', '')}",
        span_text=job["chunk"],
        span_start=int(job.get("chunk_start", 0)),
    )

    if os.getenv("LOOP_READ_COMPLETENESS", "0").strip().lower() in ("1", "true", "yes"):
        result = _run_completeness_pass(
            job, board, caller, source, result,
            tag=tag, action_id=job.get("action_id", ""),
        )

    return result


def _run_completeness_pass(
    job: dict, board: Board, caller, source: Source, prior_result: dict,
    *, tag: str, action_id: str,
) -> dict:
    chunk = job["chunk"]
    chunk_start = int(job.get("chunk_start", 0))
    existing = []
    for claim in board.claims[-200:]:
        if claim.source_doc == source.name and claim.active:
            existing.append(f"- [{claim.kind}] {claim.content[:150]}")
    if not existing:
        return prior_result

    existing_text = "\n".join(existing[-80:])
    prompt = f"""You already extracted facts from this document section. Review the text again and find SPECIFIC facts that were MISSED.

ALREADY EXTRACTED ({len(existing)} claims):
{existing_text}

DOCUMENT: {source.name}
---
{chunk}
---

Return JSON with ONLY claims not already covered above. Same schema:
{{"claims": [{{"kind": "observation", "content": "<missed fact>", "section": "<section>", "evidence": "<verbatim quote>", "confidence": 0.0-1.0}}]}}

Rules:
- Only return facts genuinely missing from the list above.
- Focus on specific numbers, dates, amounts, parties, terms, conditions that were skipped.
- Do not rephrase or duplicate existing claims.
- evidence must be copied verbatim from the document."""

    parsed = call_json(caller, board, prompt, kind="read_completeness", max_tokens=16384)
    comp_result = _ingest_claims(
        parsed, board, source=source,
        created_by=f"completeness:{action_id}",
        span_text=chunk,
        span_start=chunk_start,
    )
    board.log(
        "completeness_pass",
        f"completeness:{action_id}: +{comp_result.get('claims', 0)} new claims",
        detail={"prior_claims": prior_result.get("claims", 0),
                "new_claims": comp_result.get("claims", 0)},
    )
    # Sum every numeric counter from both passes so admission denominators,
    # support paths, span stats, and rejection reasons all persist.
    merged = dict(prior_result)
    for k, v in comp_result.items():
        if isinstance(v, (int, float)):
            merged[k] = merged.get(k, 0) + v
    merged["completeness_added"] = comp_result.get("claims", 0)
    return merged


# --- SEARCH ---

def _run_search(action: dict, board: Board, caller) -> dict:
    from ..swarm.web_search import search_and_browse
    query = str(action.get("query", "")).strip()
    if not query:
        return {}
    results_text = search_and_browse(query)
    if not results_text:
        board.log("search", f"no results for: {query}")
        return {}

    src = Source(
        id=f"web_{hashlib.md5(query.encode(), usedforsecurity=False).hexdigest()[:8]}",
        name=f"web: {query[:60]}", kind="web",
        read_status="read", relevance="definite",
        relevance_reason="fetched for query",
        web_text=results_text[:60_000],
    )
    board.add_source(src)

    targets_text = _targets_brief(board, [str(t) for t in action.get("target_ids", [])])
    prompt = f"""You are extracting facts from web search results to answer specific questions. Only extract claims the results actually support — attribute each to its page.

QUESTIONS:
{targets_text}

SEARCH RESULTS:
---
{src.web_text}
---

Return JSON:
{{"claims": [{{"kind": "observation", "content": "<fact with attribution>", "section": "<page title or url>", "evidence": "<short quote>", "confidence": 0.0-1.0}}]}}

External claims need lower default confidence than primary documents unless from an authoritative source."""

    parsed = call_json(caller, board, prompt, kind="search", max_tokens=8192)
    out = _ingest_claims(
        parsed, board, source=src,
        created_by=f"search:{action.get('_id', '')}",
    )
    # Search results serve specific targets — bind directly.
    tids = [str(t) for t in action.get("target_ids", [])]
    if tids:
        for c in board.claims:
            if c.created_by.startswith("search") and c.source_doc == src.name and not c.target_refs:
                board.bind_claim(c.id, tids)
                out["bound"] = out.get("bound", 0) + 1
    return out


# --- BIND ---

def auto_bind(board: Board, caller, budget_stop_pct: float = 85.0) -> dict:
    """Bind all unbound claims to open targets. Returns structured counters:
    offered, bound, unbound_after, calls, failures, invalid."""
    unbound = board.unbound_claims()
    open_targets = board.open_targets()
    if not unbound or not open_targets:
        return {"offered": 0, "bound": 0,
                "unbound_after": len(unbound) if unbound else 0,
                "calls": 0, "failures": 0, "invalid": 0}
    offered = len(unbound)
    total_bound = 0
    calls = 0
    failures = 0
    invalid = 0
    for i in range(0, len(unbound), _BIND_BATCH):
        if board.budget_used_pct() >= budget_stop_pct:
            board.log("auto_bind_budget_stop",
                      f"stopped after {calls} calls at {board.budget_used_pct()}%")
            break
        batch = unbound[i:i + _BIND_BATCH]
        calls += 1
        try:
            result = _run_bind_batch({"claims": batch}, board, caller)
            if not result or not isinstance(result, dict):
                invalid += 1
            else:
                total_bound += result.get("bound", 0)
        except Exception as exc:
            failures += 1
            board.log("auto_bind_error", f"batch {calls} failed: {exc}")
    unbound_after = len(board.unbound_claims())
    if total_bound or failures or invalid:
        board.log("auto_bind",
                  f"offered={offered} bound={total_bound} "
                  f"unbound_after={unbound_after} calls={calls} "
                  f"failures={failures} invalid={invalid}")
    return {"offered": offered, "bound": total_bound,
            "unbound_after": unbound_after, "calls": calls,
            "failures": failures, "invalid": invalid}


def _bind_jobs(action: dict, board: Board) -> list[tuple[str, dict]]:
    unbound = board.unbound_claims()
    if not unbound:
        return []
    jobs = []
    for i in range(0, len(unbound), _BIND_BATCH):
        jobs.append(("bind_batch", {"claims": unbound[i:i + _BIND_BATCH]}))
    return jobs


def _run_bind_batch(job: dict, board: Board, caller) -> dict:
    claims = job["claims"]
    targets = board.open_targets()
    if not targets:
        return {}
    targets_text = "\n".join(f"{t.id} [{t.materiality}] {t.need}" for t in targets)
    claims_text = "\n".join(
        f"{c.id} [{c.kind}] {c.content}" for c in claims
    )
    active_units = [u for u in board.units if u.status != "waived"]
    units_text = ""
    units_schema = ""
    if active_units:
        units_text = "\nCOVERAGE UNITS (repeated items the answer must account for; attach claims that evidence a specific unit):\n" + "\n".join(
            f"{u.id} [{board.find_obligation(u.obligation_ref).text if board.find_obligation(u.obligation_ref) else ''}] {u.name}"
            for u in active_units
        )
        units_schema = ', "unit_ids": ["..."]'

    prompt = f"""You are connecting extracted evidence to the questions it helps answer. A claim can serve multiple questions. A claim that serves no current question gets an empty list — do NOT force-fit.

QUESTIONS (id, materiality, need):
{targets_text}
{units_text}

CLAIMS (id, kind, content):
{claims_text}

Return JSON:
{{"bindings": [{{"claim_id": "...", "target_ids": ["..."]{units_schema}}}]}}
Include every claim id. Bind on substance, not keyword overlap."""

    parsed = call_json(caller, board, prompt, kind="bind", max_tokens=16384)
    if not isinstance(parsed, dict) or "bindings" not in parsed:
        return {}
    bound = 0
    for b in parsed.get("bindings", []):
        if not isinstance(b, dict):
            continue
        cid = str(b.get("claim_id", ""))
        tids = [str(t) for t in b.get("target_ids", []) if t]
        if tids and board.bind_claim(cid, tids):
            bound += 1
        uids = [str(u) for u in b.get("unit_ids", []) if u]
        if uids:
            board.bind_claim_to_units(cid, uids)
    return {"bound": bound}


# --- ANALYZE ---

# Re-export for backward compat (tests import from actions)
_source_claims_for_hydration = source_claims_for_hydration


def _run_analyze(action: dict, board: Board, caller) -> dict:
    target = board.find_target(str(action.get("target_id", "")))
    if target is None:
        return {}
    bound = board.claims_for_target(target)
    if not bound:
        return {}
    instruction = str(action.get("instruction", ""))
    claims_text = "\n".join(
        f"{c.id} [{c.kind}, conf {c.confidence:.2f}] {c.content}"
        + (f" | evidence: {c.evidence}" if c.evidence else "")
        + (f" | source: {c.source_doc}" if c.source_doc else "")
        + (f" | supports: {', '.join(c.support_refs)}" if c.support_refs else "")
        for c in bound
    )

    hydrated_claim_ids: set[str] = set()

    if _ANALYZE_HYDRATE:
        evidence_context, hydrate_stats = build_evidence_context(board, bound)
        hydrated_claim_ids = set(hydrate_stats.get("hydrated_claim_ids", []))
        board.log(
            "analyze_hydrate",
            f"{target.id}: {hydrate_stats['merged_windows']} source spans, "
            f"{hydrate_stats['chars']} chars",
            detail={"target_id": target.id, **hydrate_stats},
        )

        prompt = f"""You are a top-tier expert doing the analytical work to close a specific question. Raw facts are inputs; your job is conclusions: calculations, comparisons, issue flags, recommendations, decisions. Show reasoning inside the claim content.

OVERALL TASK:
{board.instruction}

QUESTION TO CLOSE:
[{target.materiality}] {target.need}
{f'SPECIFIC INSTRUCTION: {instruction}' if instruction else ''}

EVIDENCE CLAIM CARDS BOUND TO THIS QUESTION:
{claims_text}

PRIMARY SOURCE TEXT FOR CLAIMS BOUND TO THIS QUESTION:
{evidence_context if evidence_context else '(no source spans available; rely on claim cards and evidence quotes)'}

Return JSON:
{{"claims": [{{"kind": "analysis|calculation|comparison|issue|recommendation|decision|gap|uncertainty|contradiction", "content": "<the conclusion, with reasoning and concrete numbers where applicable>", "support_refs": ["<ids of evidence claims used>"], "confidence": 0.0-1.0}}],
 "proposed_targets": [{{"need": "...", "materiality": "critical|high|medium|low"}}],
 "recommend_close": true/false,
 "close_reason": "<if recommend_close: why this question is now answerable>"}}

Rules:
- PRIMARY SOURCE TEXT is authoritative.
- Some source excerpts are included because they support prior derived claims bound to this question.
- Do not blindly trust prior derived claims. Check them against their underlying source text when source text is provided.
- Claim cards summarize what extraction or prior analysis believed; source text is the underlying evidence. If they conflict, trust the source text and emit a contradiction or uncertainty claim.
- support_refs may cite bound claim ids and source-backed claim ids shown in the excerpts.
- Every derived claim MUST cite support_refs from the evidence above.
- Calculations show the arithmetic. Comparisons name both sides. Issues state impact.
- If evidence is insufficient, emit a "gap" claim saying exactly what is missing.
- Advice for the client ("negotiate X", "request Y") is a "recommendation" claim, NOT a proposed target. Targets are questions answerable from sources or search.
- recommend_close only if the question is genuinely answerable from the derived claims."""
    else:
        prompt = f"""You are a top-tier expert doing the analytical work to close a specific question. Raw facts are inputs; your job is conclusions: calculations, comparisons, issue flags, recommendations, decisions. Show reasoning inside the claim content.

OVERALL TASK:
{board.instruction}

QUESTION TO CLOSE:
[{target.materiality}] {target.need}
{f'SPECIFIC INSTRUCTION: {instruction}' if instruction else ''}

EVIDENCE BOUND TO THIS QUESTION:
{claims_text}

Return JSON:
{{"claims": [{{"kind": "analysis|calculation|comparison|issue|recommendation|decision|gap|uncertainty|contradiction", "content": "<the conclusion, with reasoning and concrete numbers where applicable>", "support_refs": ["<ids of evidence claims used>"], "confidence": 0.0-1.0}}],
 "proposed_targets": [{{"need": "...", "materiality": "critical|high|medium|low"}}],
 "recommend_close": true/false,
 "close_reason": "<if recommend_close: why this question is now answerable>"}}

Rules:
- Every derived claim MUST cite support_refs from the evidence above.
- Calculations show the arithmetic. Comparisons name both sides. Issues state impact.
- If evidence is insufficient, emit a "gap" claim saying exactly what is missing.
- Advice for the client ("negotiate X", "request Y") is a "recommendation" claim, NOT a proposed target. Targets are questions answerable from sources or search.
- recommend_close only if the question is genuinely answerable from the derived claims."""

    parsed = call_json(caller, board, prompt, kind="analyze", max_tokens=16384)
    valid_support = {c.id for c in bound}
    if _ANALYZE_HYDRATE:
        valid_support |= hydrated_claim_ids
    out = _ingest_claims(
        parsed, board, source=None,
        created_by=f"analyze:{action.get('_id', '')}",
        bind_to=[target.id], valid_support=valid_support,
    )
    if isinstance(parsed, dict) and parsed.get("recommend_close"):
        board.log(
            "close_recommendation",
            f"{target.id}: {str(parsed.get('close_reason', ''))[:200]}",
            detail={"target_id": target.id},
        )
    return out


# --- VERIFY ---

def _run_verify(action: dict, board: Board, caller) -> dict:
    claim_ids = [str(c) for c in action.get("claim_ids", [])][:10]
    claims = [c for c in (board.find_claim(cid) for cid in claim_ids) if c]
    if not claims:
        return {}

    blocks = []
    for c in claims:
        evidence_context = ""
        src = next(
            (s for s in board.sources if s.name == c.source_doc), None,
        )
        if src is None and c.support_refs:
            sup = board.find_claim(c.support_refs[0])
            if sup is not None:
                src = next(
                    (s for s in board.sources if s.name == sup.source_doc), None,
                )
        if src is not None and src.kind == "document":
            from ..swarm.section_index import resolve_section_text
            section = c.source_section or ""
            evidence_context = resolve_section_text(
                src.text(), src.section_index(), section, max_chars=12_000,
            )
        support_text = "\n".join(
            f"  support {s.id}: {s.content} | evidence: {s.evidence}"
            for s in (board.find_claim(r) for r in c.support_refs) if s
        )
        blocks.append(
            f"CLAIM {c.id} [{c.kind}]: {c.content}\n{support_text}\n"
            f"SOURCE TEXT:\n{evidence_context[:10_000] if evidence_context else '(no source text located)'}"
        )

    prompt = f"""You are adversarially verifying claims against their cited sources. Try to refute each claim. A claim survives only if the source text actually supports it — including any arithmetic.

{chr(10).join(blocks)}

Return JSON:
{{"verdicts": [{{"claim_id": "...", "verified": true/false, "confidence": 0.0-1.0, "note": "<what the source shows>"}}]}}"""

    parsed = call_json(caller, board, prompt, kind="verify", max_tokens=8192)
    if not isinstance(parsed, dict):
        return {}
    verified = 0
    for v in parsed.get("verdicts", []):
        if not isinstance(v, dict):
            continue
        claim = board.find_claim(str(v.get("claim_id", "")))
        if claim is None:
            continue
        claim.verified = bool(v.get("verified"))
        try:
            claim.confidence = max(0.05, min(0.98, float(v.get("confidence", claim.confidence))))
        except (TypeError, ValueError):
            pass
        verified += 1
    return {"verified": verified}


# --- quote matching ---

def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    mapping: list[int] = []
    in_ws = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if chars and not in_ws:
                chars.append(" ")
                mapping.append(i)
            in_ws = True
        else:
            chars.append(ch)
            mapping.append(i)
            in_ws = False
    if chars and chars[-1] == " ":
        chars.pop()
        mapping.pop()
    return "".join(chars), mapping


def _find_quote_span(text: str, quote: str, base_offset: int = 0) -> tuple[int, int] | None:
    quote = quote.strip()
    if not text or not quote:
        return None

    pos = text.find(quote)
    if pos >= 0:
        return (base_offset + pos, base_offset + pos + len(quote))

    norm_text, text_map = _normalize_with_map(text)
    norm_quote, _ = _normalize_with_map(quote)
    if not norm_quote:
        return None

    pos = norm_text.find(norm_quote)
    if pos < 0:
        pos = norm_text.lower().find(norm_quote.lower())
    if pos < 0:
        return None

    start = text_map[pos]
    end_norm_idx = pos + len(norm_quote) - 1
    if end_norm_idx < 0 or end_norm_idx >= len(text_map):
        return None
    end = text_map[end_norm_idx] + 1
    return (base_offset + start, base_offset + end)


import math
import re as _re

_TFIDF_WINDOW = 600
_TFIDF_STRIDE = 200
_TFIDF_MIN_SCORE = 0.25
_TFIDF_PAD = 1200
_TOKENIZE_RE = _re.compile(r"[a-z0-9$%.,/:;'\"-]+", _re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKENIZE_RE.findall(text) if len(t) > 1]


def _tfidf_span(
    text: str, quote: str, base_offset: int = 0,
) -> tuple[int, int] | None:
    quote_tokens = _tokenize(quote)
    if not quote_tokens or not text:
        return None

    n_windows = max(1, (len(text) - _TFIDF_WINDOW) // _TFIDF_STRIDE + 1)
    doc_freq: dict[str, int] = {}
    windows: list[tuple[int, int, list[str]]] = []
    for i in range(n_windows):
        start = i * _TFIDF_STRIDE
        end = min(len(text), start + _TFIDF_WINDOW)
        wtokens = _tokenize(text[start:end])
        windows.append((start, end, wtokens))
        seen: set[str] = set()
        for t in wtokens:
            if t not in seen:
                doc_freq[t] = doc_freq.get(t, 0) + 1
                seen.add(t)

    quote_set = set(quote_tokens)
    idf: dict[str, float] = {}
    for t in quote_set:
        df = doc_freq.get(t, 0)
        idf[t] = math.log((n_windows + 1) / (df + 1)) + 1.0

    best_score = 0.0
    best_start = 0
    best_end = 0
    for start, end, wtokens in windows:
        if not wtokens:
            continue
        wset = set(wtokens)
        shared = quote_set & wset
        if not shared:
            continue
        score = sum(idf.get(t, 0) for t in shared) / (
            sum(idf.get(t, 0) for t in quote_set) + 1e-9
        )
        if score > best_score:
            best_score = score
            best_start = start
            best_end = end

    if best_score < _TFIDF_MIN_SCORE:
        return None

    span_start = max(0, best_start - _TFIDF_PAD)
    span_end = min(len(text), best_end + _TFIDF_PAD)
    return (base_offset + span_start, base_offset + span_end)


_FALLBACK_WINDOW = 3000


def _narrow_fallback_span(
    chunk_text: str, chunk_start: int, section_hint: str
) -> tuple[int, int]:
    if section_hint:
        sec_lower = section_hint.strip().lower()
        chunk_lower = chunk_text.lower()
        pos = chunk_lower.find(sec_lower)
        if pos >= 0:
            end = min(len(chunk_text), pos + _FALLBACK_WINDOW)
            return (chunk_start + pos, chunk_start + end)
    window = min(len(chunk_text), _FALLBACK_WINDOW)
    return (chunk_start, chunk_start + window)


# --- shared ingestion ---

# --- Deterministic source-evidence admission (cycle-4 treatment) ---
# Pure string checks between a claim, its quoted evidence, and the cited
# source slice. No model calls, no benchmark knowledge, no repair.

_MATCH_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_DIGIT_RUN_RE = re.compile(r"\d+")
_EVIDENCE_TOKEN_COVERAGE = 0.8  # fixed pre-smoke; never tuned against scores


def _usable_source_text(text) -> bool:
    """Whitespace-only extraction output is not a successful read input."""
    return bool(text) and bool(text.strip())


def _norm_match(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(s.split())


def _nontrivial_tokens(s: str) -> list[str]:
    return [t for t in _MATCH_TOKEN_RE.findall(_norm_match(s))
            if len(t) >= 3 or any(ch.isdigit() for ch in t)]


_SLICE_TOKEN_CAP = 2000  # bound matcher cost on oversized fallback slices


def _lcs_len(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length — order-preserving match that
    tolerates absent tokens anywhere, unlike a first-miss cursor."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def _ordered_token_coverage(evidence: str, slice_text: str) -> float:
    """Order-preserving coverage of the evidence's nontrivial tokens in the
    cited slice, guarded by distinct-token coverage so common structural
    tokens alone can never satisfy the threshold. Tolerates layout/OCR
    spacing, not invented prose."""
    ev = _nontrivial_tokens(evidence)
    if not ev:
        return 0.0
    sl = [t for t in _MATCH_TOKEN_RE.findall(_norm_match(slice_text))
          ][:_SLICE_TOKEN_CAP]
    ordered = _lcs_len(ev, sl) / len(ev)
    distinct_ev = set(ev)
    distinct = len(distinct_ev & set(sl)) / len(distinct_ev)
    return min(ordered, distinct)


def _evidence_supported(evidence: str, slice_text: str,
                        content: str = "") -> tuple[bool, str]:
    """Returns (supported, path) with path 'exact' | 'ordered' | ''.

    The ordered path carries an extra guard: every nontrivial token asserted
    in BOTH the claim content and the evidence must occur in the cited slice.
    The 80% tolerance absorbs peripheral quote/OCR noise, but an unmatched
    token cannot carry the claim's substantive assertion (one decisive
    noun/name/status differing from source while boilerplate matches)."""
    ne, ns = _norm_match(evidence), _norm_match(slice_text)
    if ne and ne in ns:
        return True, "exact"
    if _ordered_token_coverage(evidence, slice_text) < _EVIDENCE_TOKEN_COVERAGE:
        return False, ""
    if content:
        shared = set(_nontrivial_tokens(content)) & set(_nontrivial_tokens(evidence))
        slice_tokens = set(_MATCH_TOKEN_RE.findall(_norm_match(slice_text)))
        if shared - slice_tokens:
            return False, ""
    return True, "ordered"


def _digit_runs(s: str) -> set[str]:
    return set(_DIGIT_RUN_RE.findall(unicodedata.normalize("NFKC", s)))


def _digits_conserved(content: str, evidence: str, slice_text: str) -> bool:
    """Every digit run asserted in content must occur in the evidence; every
    digit run in the evidence must occur in the cited slice. No arithmetic
    equivalence, unit conversion, or inference."""
    return (_digit_runs(content) <= _digit_runs(evidence)
            and _digit_runs(evidence) <= _digit_runs(slice_text))


def _ingest_claims(parsed, board: Board, *, source: Source | None,
                   created_by: str, bind_to: list[str] | None = None,
                   valid_support: set[str] | None = None,
                   span_text: str | None = None,
                   span_start: int = 0) -> dict:
    if not isinstance(parsed, dict):
        return {"claims": 0}
    added = 0
    added_ids: list[str] = []
    span_hits = 0
    span_tfidf_hits = 0
    span_misses = 0
    rejected = {"empty_evidence": 0, "unsupported_quote": 0,
                "unsupported_digits": 0}
    offered = 0        # well-formed source-backed claim items seen
    admitted = 0       # passed the admission gate (before Board dedup)
    support_exact = 0
    support_ordered = 0
    numeric_checked = 0
    for item in parsed.get("claims") or []:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        kind = str(item.get("kind", "observation"))
        if kind not in CLAIM_KINDS:
            kind = "observation"
        support = [
            str(r) for r in (item.get("support_refs") or [])
            if valid_support is None or str(r) in valid_support
        ]
        try:
            conf = max(0.05, min(0.98, float(item.get("confidence", 0.6))))
        except (TypeError, ValueError):
            conf = 0.6
        evidence_raw = str(item.get("evidence", "")).strip()
        stored_evidence = evidence_raw[:500]
        source_span = None
        span_kind = ""
        if source is not None:
            # Deterministic evidence admission: a source-backed claim enters
            # the board only when its evidence is supportable by the cited
            # slice and its digits are conserved. No repair, no re-prompt.
            offered += 1
            if _digit_runs(content):
                numeric_checked += 1
            if not evidence_raw:
                rejected["empty_evidence"] += 1
                continue
            haystack = span_text if span_text is not None else source.text()
            source_span = _find_quote_span(haystack, evidence_raw, span_start)
            if source_span is not None:
                span_kind = "exact"
            else:
                # Every source-backed claim gets a bounded candidate slice —
                # search-style ingestion (span_text=None) uses the full
                # haystack for TF-IDF resolution, never an unbounded slice.
                source_span = _tfidf_span(haystack, evidence_raw, span_start)
                if source_span is not None:
                    span_kind = "tfidf"
                else:
                    span_kind = "miss"
                    if span_text is not None:
                        source_span = _narrow_fallback_span(
                            span_text, span_start,
                            str(item.get("section", "")),
                        )
            if source_span is not None:
                lo = max(0, source_span[0] - span_start)
                hi = max(0, source_span[1] - span_start)
                slice_text = haystack[lo:hi]
            else:
                slice_text = ""
            supported, support_path = (
                _evidence_supported(evidence_raw, slice_text, content)
                if slice_text else (False, "")
            )
            if not supported:
                rejected["unsupported_quote"] += 1
                continue
            if not _digits_conserved(content, evidence_raw, slice_text):
                rejected["unsupported_digits"] += 1
                continue
            admitted += 1
            if support_path == "exact":
                support_exact += 1
            else:
                support_ordered += 1
            if span_kind == "exact":
                span_hits += 1
            elif span_kind == "tfidf":
                span_tfidf_hits += 1
            else:
                span_misses += 1
        claim = Claim(
            kind=kind, content=content,
            source_doc=source.name if source else None,
            source_section=str(item.get("section", "")) or None,
            evidence=stored_evidence,
            source_span=source_span,
            support_refs=support,
            target_refs=list(bind_to or []),
            confidence=conf,
            iteration=board.iteration,
            created_by=created_by,
        )
        if board.add_claim(claim):
            added += 1
            added_ids.append(claim.id)

    # A response with any rejected source-backed claim loses its proposals:
    # the schema cannot prove which claim grounds each proposal, so this is
    # deterministic response bookkeeping, not a second semantic judgment.
    any_rejected = sum(rejected.values()) > 0
    if any_rejected:
        parsed = {k: v for k, v in parsed.items()
                  if k not in ("proposed_targets", "proposed_reads", "units")}

    proposed = 0
    for pt in parsed.get("proposed_targets") or []:
        if not isinstance(pt, dict):
            continue
        need = str(pt.get("need", "")).strip()
        if not need:
            continue
        materiality = str(pt.get("materiality", "medium"))
        if materiality not in ("critical", "high", "medium", "low"):
            materiality = "medium"
        board.add_target(Target(
            need=need, materiality=materiality,
            created_iteration=board.iteration, proposed_by=created_by,
        ))
        proposed += 1

    units_added = 0
    for un in parsed.get("units") or []:
        if not isinstance(un, dict):
            continue
        name = str(un.get("name", "")).strip()
        ob = board.find_obligation(str(un.get("obligation_id", "")))
        if not name or ob is None or not ob.set_valued:
            continue
        board.add_unit(Unit(
            name=name, obligation_ref=ob.id,
            anchor=str(un.get("anchor", ""))[:120],
        ))
        units_added += 1

    proposed_reads_count = 0
    for pr in parsed.get("proposed_reads") or []:
        if not isinstance(pr, dict):
            continue
        source_hint = str(pr.get("source_hint", "")).strip()[:160]
        section_hint = str(pr.get("section_hint", "")).strip()[:160]
        reason = str(pr.get("reason", "")).strip()[:240]
        target_ids_pr = [str(t) for t in (pr.get("target_ids") or []) if t]
        if not (source_hint or section_hint) or not reason:
            continue
        if proposed_reads_count == 0:
            proposed_reads_list: list[dict] = []
        proposed_reads_list.append({
            "source_hint": source_hint,
            "section_hint": section_hint,
            "reason": reason,
            "target_ids": target_ids_pr[:8],
            "created_by": created_by,
        })
        proposed_reads_count += 1
        if proposed_reads_count >= 10:
            break

    if proposed_reads_count > 0:
        board.log(
            "proposed_reads",
            f"{created_by}: {proposed_reads_count} proposed reads",
            detail={"items": proposed_reads_list},
        )

    if added or proposed or units_added:
        board.log(
            "action_output",
            f"{created_by}: {added} claims, {proposed} targets, {units_added} units",
            detail={"by": created_by, "claim_ids": added_ids},
        )

    if span_misses or span_tfidf_hits:
        board.log(
            "span_warning",
            f"{created_by}: {span_hits} exact, {span_tfidf_hits} tfidf, {span_misses} fallback",
            detail={"by": created_by, "span_hits": span_hits,
                    "span_tfidf_hits": span_tfidf_hits, "span_misses": span_misses},
        )

    if any_rejected:
        board.log(
            "claim_rejected",
            f"{created_by}: rejected {sum(rejected.values())} claims "
            f"(evidence admission)",
            detail={"by": created_by, **rejected},
        )

    return {
        "claims": added, "targets_proposed": proposed, "units": units_added,
        "span_hits": span_hits, "span_tfidf_hits": span_tfidf_hits,
        "span_misses": span_misses,
        "proposed_reads": proposed_reads_count,
        "claims_offered": offered,
        "claims_admitted": admitted,
        "support_exact": support_exact,
        "support_ordered": support_ordered,
        "numeric_claims_checked": numeric_checked,
        "rejected_empty_evidence": rejected["empty_evidence"],
        "rejected_unsupported_quote": rejected["unsupported_quote"],
        "rejected_unsupported_digits": rejected["unsupported_digits"],
    }


def _targets_brief(board: Board, target_ids: list[str]) -> str:
    targets = [t for t in (board.find_target(tid) for tid in target_ids) if t]
    if not targets:
        targets = board.material_open_targets()[:8]
    return "\n".join(f"- [{t.materiality}] {t.need}" for t in targets) or "- (general extraction)"


# --- BULK FRONTIER EXTRACTION (cycle-3 treatment) ---

_BULK_WAVE = 30           # max parallel calls per wave
_BULK_MAX_TOKENS = 16384  # bounded output budget per source call
_BULK_FRAMING_TOKENS = 200  # conservative allowance for request framing


def _bulk_prompt(board: Board, src: Source, text: str,
                 assoc: list[str], reasons: list[str]) -> str:
    return f"""You are extracting evidence from one document for an investigation. Metadata triage selected this document as decisive for specific questions.

TASK:
{board.instruction[:2000]}

QUESTIONS THIS DOCUMENT WAS SELECTED FOR:
{_targets_brief(board, assoc)}

WHY IT WAS SELECTED (metadata signals): {'; '.join(reasons) or '(none)'}

DOCUMENT: {src.name}
---
{text}
---

Return JSON:
{{"claims": [{{"kind": "observation", "content": "<the fact, specific and self-contained>", "section": "<section/heading it came from>", "evidence": "<short exact quote copied verbatim from the document>", "confidence": 0.0-1.0}}]}}

Rules:
- Extract every fact relevant to the questions above, plus clearly material facts for the task. Atomic claims — one fact each.
- evidence must be copied verbatim from the document; it locates the source span. Never paraphrase it, never include a number that does not appear in the quoted text.
- If the document holds nothing relevant, return {{"claims": []}} — an empty list is a valid answer.
- No proposals, no recommendations, no new questions — observations grounded in this document only."""


def _estimate_input_bound(prompt: str) -> int:
    """Tokenizer-independent hard upper bound on input tokens: one token per
    encoded byte (a byte-level tokenizer cannot emit more tokens than bytes)
    plus request framing."""
    return len(prompt.encode("utf-8")) + _BULK_FRAMING_TOKENS


def _estimate_call_tokens(prompt: str) -> int:
    """Hard worst-case bound for one call: input bound + worst-case output."""
    return _estimate_input_bound(prompt) + _BULK_MAX_TOKENS


def _valid_bulk_extraction(parsed) -> dict | None:
    """Strict response validation. Returns a cleaned payload or None.

    Contract: `claims` must be a list. An EMPTY list is a valid
    'read, no relevant evidence' result. A non-list, or a non-empty list
    containing zero well-formed claim objects, is a parse failure.
    """
    if not isinstance(parsed, dict) or not isinstance(parsed.get("claims"), list):
        return None
    raw = parsed["claims"]
    cleaned = [
        c for c in raw
        if isinstance(c, dict) and str(c.get("content", "")).strip()
    ]
    if raw and not cleaned:
        return None
    return {"claims": cleaned}


def bulk_extract_frontier(board: Board, worker_caller) -> dict:
    """Extract every retained definite frontier candidate in one target-guided
    call per canonical source, in bounded parallel waves, before the
    controller loop. Claims bind immediately to ALL retained target
    associations for the source. Activates only on a validated large-corpus
    frontier; every other path is untouched.

    Emits the start event only; run_loop applies the budget offset and emits
    the single completion event via finalize_bulk_extraction so the completion
    record can carry the adjusted budget. Returns structural stats
    ({} when inactive).
    """
    from .triage import _valid_frontier  # shared validator — no second copy

    doc_count = sum(1 for s in board.sources if s.kind == "document")
    frontier = board.metadata.get("retrieval_frontier")
    fallback = board.metadata.get("retrieval_fallback")
    if (doc_count <= 60
            or not board.metadata.get("retrieval_frontier_enabled")
            or not _valid_frontier(board, frontier, fallback)):
        return {}

    # Pass 1: canonical source ids retained with a definite record anywhere.
    ordered: list[str] = []
    for tid, lst in frontier.items():
        for c in lst:
            if c["priority"] == "definite" and c["source_id"] not in ordered:
                ordered.append(c["source_id"])
    # Pass 2: for selected ids, collect EVERY retained association and reason
    # across the full frontier, regardless of that record's priority.
    assoc: dict[str, list[str]] = {sid: [] for sid in ordered}
    reasons: dict[str, list[str]] = {sid: [] for sid in ordered}
    for tid, lst in frontier.items():
        for c in lst:
            sid = c["source_id"]
            if sid not in assoc:
                continue
            if tid not in assoc[sid]:
                assoc[sid].append(tid)
            if c["reason"] and c["reason"] not in reasons[sid]:
                reasons[sid].append(c["reason"])

    candidates = []
    for sid in ordered:
        src = board.find_source(sid)
        if src is not None and src.kind == "document" and src.read_status != "read":
            candidates.append(sid)
    if not candidates:
        return {}

    envelope = board.token_budget  # hard bulk envelope = original loop budget
    stats = {"candidates": len(candidates), "attempted": 0, "calls": 0,
             "succeeded": 0, "parse_failed": 0, "call_failed": 0,
             "text_load_failed": 0, "budget_skipped": 0, "claims_added": 0,
             "claims_bound": 0, "span_exact": 0, "span_fuzzy": 0,
             "span_fallback": 0, "sources_read": 0,
             "sources_with_accepted_claims": 0, "waves": 0,
             "max_parallelism": 0, "failed_call_reserved_tokens": 0,
             "claims_rejected_empty_evidence": 0,
             "claims_rejected_unsupported_quote": 0,
             "claims_rejected_unsupported_digits": 0,
             "claims_offered": 0, "claims_admitted": 0,
             "support_exact": 0, "support_ordered": 0,
             "numeric_claims_checked": 0,
             "integrity_failed_extractions": 0}

    import time as _time
    t0 = _time.time()  # includes preflight materialization time

    # Preflight: render every candidate's actual prompt once; keep the text
    # (documents cache their materialization) and hard per-call bounds.
    texts: dict[str, str] = {}
    input_bounds: dict[str, int] = {}
    estimates: dict[str, int] = {}
    render_failures = 0
    render_exceptions = 0
    render_whitespace_only = 0
    for sid in candidates:
        src = board.find_source(sid)
        try:
            text = src.text()
        except Exception:
            text = None
        if not _usable_source_text(text):
            if text is None:
                render_exceptions += 1
            elif text.strip() == "" and text != "":
                render_whitespace_only += 1
            texts[sid] = ""
            input_bounds[sid] = 0
            estimates[sid] = 0
            render_failures += 1
            continue
        texts[sid] = text
        prompt = _bulk_prompt(board, src, text, assoc[sid], reasons[sid])
        input_bounds[sid] = _estimate_input_bound(prompt)
        estimates[sid] = input_bounds[sid] + _BULK_MAX_TOKENS
    full_set_estimate = sum(estimates.values())
    source_bytes = sum(
        (board.find_source(sid).size_bytes or 0) for sid in candidates
    )

    tokens_before = board.total_tokens_used
    tin_before = board.tokens_input
    tout_before = board.tokens_output

    board.log(
        "bulk_extraction",
        f"start: {len(candidates)} definite candidates, full-set worst-case "
        f"{full_set_estimate} tokens vs envelope {envelope}",
        detail={
            "activation": "validated_frontier",
            "candidates": len(candidates),
            "target_associations": {sid: assoc[sid] for sid in candidates},
            "source_bytes": source_bytes,
            "estimated_input_tokens": sum(input_bounds.values()),
            "framing_tokens_per_call": _BULK_FRAMING_TOKENS,
            "worst_case_output_per_call": _BULK_MAX_TOKENS,
            "estimated_tokens_full_set": full_set_estimate,
            # unknown/false when any candidate failed to render
            "full_set_estimated_fit": (render_failures == 0
                                       and full_set_estimate <= envelope),
            "render_failures": render_failures,
            "render_exceptions": render_exceptions,
            "render_whitespace_only": render_whitespace_only,
            "envelope_tokens": envelope,
        },
    )

    def _one(sid: str) -> tuple[str, dict | None, str]:
        src = board.find_source(sid)
        text = texts[sid]
        prompt = _bulk_prompt(board, src, text, assoc[sid], reasons[sid])
        try:
            parsed = call_json(worker_caller, board, prompt,
                               kind="bulk_extract", max_tokens=_BULK_MAX_TOKENS)
        except Exception as exc:
            return sid, None, f"call: {exc}"
        cleaned = _valid_bulk_extraction(parsed)
        if cleaned is None:
            return sid, None, "parse"
        return sid, cleaned, ""

    idx = 0
    while idx < len(candidates):
        # Dynamic wave sizing against a HARD envelope: admit candidates while
        # actual-spend-so-far + summed worst-case bounds stay inside it. The
        # bound is tokenizer-independent (tokens <= encoded bytes), so a
        # launched wave can never breach the envelope; actual spend feeds back
        # between waves, so admission capacity regrows as reality undershoots
        # the bound.
        # A call that raised may have consumed tokens the caller never
        # reported; its worst-case reservation stays charged against the
        # envelope instead of being released for reuse.
        spent = (board.total_tokens_used - tokens_before
                 + stats["failed_call_reserved_tokens"])
        remaining = envelope - spent
        wave: list[str] = []
        wave_worst = 0
        j = idx
        while j < len(candidates) and len(wave) < _BULK_WAVE:
            sid = candidates[j]
            worst = estimates[sid] if texts[sid] else 0
            if texts[sid] and wave_worst + worst > remaining:
                break
            wave.append(sid)
            wave_worst += worst
            j += 1
        if not wave:
            stats["budget_skipped"] = len(candidates) - idx
            break
        stats["waves"] += 1
        launchable = [sid for sid in wave if texts[sid]]
        stats["max_parallelism"] = max(stats["max_parallelism"],
                                       len(launchable))
        for sid in wave:
            stats["attempted"] += 1
            if not texts[sid]:
                stats["text_load_failed"] += 1
        results: dict[str, tuple] = {}
        with ThreadPoolExecutor(max_workers=_BULK_WAVE) as pool:
            futures = {pool.submit(_one, sid): sid for sid in launchable}
            for fut in as_completed(futures):
                sid = futures[fut]
                _, parsed, err = fut.result()
                results[sid] = (parsed, err)
        stats["calls"] += len(launchable)
        # Deterministic state mutation: ingest in candidate order, not
        # completion order, so claim ids and board order are reproducible.
        for sid in wave:
            if sid not in results:
                continue
            parsed, err = results[sid]
            if parsed is None:
                if err.startswith("call"):
                    stats["call_failed"] += 1
                    stats["failed_call_reserved_tokens"] += estimates[sid]
                else:
                    stats["parse_failed"] += 1
                continue
            src = board.find_source(sid)
            out = _ingest_claims(
                parsed, board, source=src,
                created_by=f"bulk_extract:{sid}",
                bind_to=assoc[sid],
                span_text=texts[sid],
            )
            stats["succeeded"] += 1
            added = out.get("claims", 0)
            stats["claims_added"] += added
            stats["claims_bound"] += added  # bind_to applied at ingest
            if added:
                stats["sources_with_accepted_claims"] += 1
            stats["span_exact"] += out.get("span_hits", 0)
            stats["span_fuzzy"] += out.get("span_tfidf_hits", 0)
            stats["span_fallback"] += out.get("span_misses", 0)
            for key in ("empty_evidence", "unsupported_quote",
                        "unsupported_digits"):
                stats[f"claims_rejected_{key}"] += out.get(f"rejected_{key}", 0)
            for key in ("claims_offered", "claims_admitted", "support_exact",
                        "support_ordered", "numeric_claims_checked"):
                stats[key] += out.get(key, 0)
            if out.get("claims_offered", 0) and not out.get("claims_admitted", 0):
                # Every offered claim in a nonempty response failed the gate
                # (gate-admitted, not Board-added — dedup does not mislabel).
                stats["integrity_failed_extractions"] += 1
            src.read_status = "read"
            stats["sources_read"] += 1
        idx += len(wave)

    bulk_tokens = board.total_tokens_used - tokens_before
    n = len(candidates)
    stats.update({
        "bulk_tokens": bulk_tokens,
        "bulk_tokens_input": board.tokens_input - tin_before,
        "bulk_tokens_output": board.tokens_output - tout_before,
        "envelope_respected": bulk_tokens <= envelope,
        "all_candidates_attempted": stats["budget_skipped"] == 0,
        "attempt_rate": round(stats["attempted"] / n, 4),
        # Criterion-facing: share of retained candidates that produced a
        # valid parsed extraction (text-load failures count against it).
        "parse_success_rate": round(stats["succeeded"] / n, 4),
        "valid_response_rate_per_call": round(
            stats["succeeded"] / stats["calls"], 4) if stats["calls"] else 0.0,
        "evidence_conversion_rate": round(
            stats["sources_with_accepted_claims"] / n, 4),
        "wall_time_s": round(_time.time() - t0, 1),
        "original_budget": envelope,
    })
    return stats


def finalize_bulk_extraction(board: Board, stats: dict,
                             budget_stop_pct: float) -> None:
    """Apply the loop budget offset and emit the single completion event.

    Called once by run_loop directly after bulk_extract_frontier so the
    completion record carries the adjusted budget. The offset preserves the
    loop's pre-bulk stop-threshold headroom; total spend stays visible.
    """
    if not stats:
        return  # inactive treatment only — every active result completes
    original = board.token_budget
    board.token_budget = original + int(
        stats.get("bulk_tokens", 0) / (budget_stop_pct / 100.0)
    )
    detail = dict(stats)
    detail["adjusted_budget"] = board.token_budget
    board.log(
        "bulk_extraction",
        f"done: {stats['succeeded']}/{stats['candidates']} sources, "
        f"{stats['claims_added']} claims, {stats['bulk_tokens']} tokens, "
        f"{stats['waves']} waves"
        + ("" if stats["envelope_respected"] else " — ENVELOPE EXCEEDED")
        + ("" if stats["all_candidates_attempted"] else " — CANDIDATES SKIPPED"),
        detail=detail,
    )


