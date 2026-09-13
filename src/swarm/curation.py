from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .blackboard import Blackboard
from .models import Entry, ModelCaller
from .worker_dispatch import call_model, get_last_call_usage, set_last_call_usage

MIN_MUST_INCLUDE = 20
_MIN_CLUSTER_ENTRIES_FOR_DENSITY = 10
_MIN_CLUSTER_DENSITY = 0.15


def curate_entries(blackboard: Blackboard, caller: ModelCaller) -> tuple[list[dict], int]:
    total_tokens = 0
    active = [
        e for e in blackboard.entries
        if e.status == "active" and e.type != "entity_overview"
    ]
    usage_by_model: dict = {}

    clusters: dict[str, list[Entry]] = {}
    for e in active:
        doc = e.source.document if e.source else "cross_cutting"
        clusters.setdefault(doc or "cross_cutting", []).append(e)

    cluster_items = list(clusters.items())
    max_workers = min(
        len(cluster_items),
        max(1, int(os.getenv("SWARM_CURATION_WORKERS", "8"))),
    )

    cluster_results: list[tuple[str, list[Entry], list[dict]]] = []

    if max_workers <= 1:
        for index, (doc_name, entries) in enumerate(cluster_items):
            _, _, items, tokens, usage = _curate_cluster(
                blackboard, caller, index, doc_name, entries,
            )
            total_tokens += tokens
            _merge_usage(usage_by_model, usage)
            cluster_results.append((doc_name, entries, items))
            _write_curation_progress(
                blackboard, index + 1, len(cluster_items), doc_name, len(items),
            )
    else:
        results: list[tuple[int, str, list[dict], int, dict] | None] = [
            None for _ in cluster_items
        ]
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _curate_cluster, blackboard, caller, index, doc_name, entries,
                ): (index, doc_name)
                for index, (doc_name, entries) in enumerate(cluster_items)
            }
            for future in as_completed(futures):
                index, doc_name = futures[future]
                _, _, items, tokens, usage = future.result()
                results[index] = (index, doc_name, items, tokens, usage)
                completed += 1
                _write_curation_progress(
                    blackboard, completed, len(cluster_items), doc_name, len(items),
                )

        for i, result in enumerate(results):
            if result is None:
                continue
            _, doc_name_r, items, tokens, usage = result
            total_tokens += tokens
            _merge_usage(usage_by_model, usage)
            cluster_results.append((doc_name_r, cluster_items[i][1], items))

    sparse = _find_sparse_clusters(cluster_results)
    for doc_name, entries, _ in sparse:
        _, _, retry_items, retry_tokens, retry_usage = _curate_cluster(
            blackboard, caller, 0, doc_name, entries,
        )
        total_tokens += retry_tokens
        _merge_usage(usage_by_model, retry_usage)
        for cr in cluster_results:
            if cr[0] == doc_name:
                existing_ids = {
                    m.get("entry_id") for m in cr[2] if isinstance(m, dict) and m.get("entry_id")
                }
                for item in retry_items:
                    eid = item.get("entry_id") if isinstance(item, dict) else None
                    if not eid or eid not in existing_ids:
                        cr[2].append(item)
                        if eid:
                            existing_ids.add(eid)
                break

    must_include_all: list[dict] = []
    for _, _, items in cluster_results:
        must_include_all.extend(items)

    # Global density quality gate: if too few items, re-curate with stronger enforcement.
    if len(must_include_all) < MIN_MUST_INCLUDE and len(active) >= MIN_MUST_INCLUDE:
        fallback_items, fallback_tokens = _fallback_curate(blackboard, active, caller)
        total_tokens += fallback_tokens
        by_model, _, _, _ = get_last_call_usage()
        _merge_usage(usage_by_model, by_model)
        seen_ids = {m.get("entry_id") for m in must_include_all if m.get("entry_id")}
        for item in fallback_items:
            eid = item.get("entry_id")
            if eid and eid in seen_ids:
                continue
            must_include_all.append(item)
            if eid:
                seen_ids.add(eid)

    must_include_all = _coverage_safety_net(must_include_all, active)

    set_last_call_usage(usage_by_model)
    return must_include_all, total_tokens


def _find_sparse_clusters(
    cluster_results: list[tuple[str, list[Entry], list[dict]]],
) -> list[tuple[str, list[Entry], list[dict]]]:
    """Identify clusters that produced too few items relative to their entry count."""
    sparse = []
    for doc_name, entries, items in cluster_results:
        if len(entries) < _MIN_CLUSTER_ENTRIES_FOR_DENSITY:
            continue
        density = len(items) / len(entries) if entries else 1.0
        if density < _MIN_CLUSTER_DENSITY:
            sparse.append((doc_name, entries, items))
    return sparse


def _curate_cluster(
    blackboard: Blackboard,
    caller: ModelCaller,
    index: int,
    doc_name: str,
    entries: list[Entry],
) -> tuple[int, str, list[dict], int, dict]:
    summaries = "\n".join(
        f"[{e.id}] ({e.type}, conf={e.confidence:.1f}) {e.content[:400]}"
        for e in entries
    )
    prompt = f"""Organize findings from "{doc_name}" for the deliverable.

TASK: {blackboard.task_instruction}

FINDINGS ({len(entries)} total):
{summaries[:60000]}

EXHAUSTIVE ENUMERATION RULES:
1. You MUST include EVERY finding that contains a specific fact, number, date, party name, dollar amount, percentage, deadline, obligation, or legal conclusion.
2. One fact = one must_include item. Do NOT group or summarize multiple facts into one item.
3. Count the findings above. Your must_include list should have AT LEAST as many items as there are distinct facts. If there are 40 findings with facts, produce 40+ must_include items.
4. A must_include list with fewer than {MIN_MUST_INCLUDE} items is almost certainly incomplete - you are probably summarizing instead of enumerating.
5. Each must_include item must contain ONE specific, verifiable fact with exact numbers, names, dates, amounts.
6. Include ALL of these if present: dollar amounts, percentages, dates, deadlines, party names with legal designations, defined terms, obligations, conditions, restrictions, representations, warranties.
7. PRESERVE EXACT QUOTES: If a finding contains a direct quote, regulatory language, or specific wording, include that verbatim in the summary. Never paraphrase a quote into a concept.
8. PRESERVE CAUSAL DETAILS: If a finding explains WHY something happened, include the reason, not just the fact. "X ended" is incomplete; "X ended because Y relocated" is correct.
9. PRESERVE RECOMMENDATIONS: If a finding contains an explicit recommendation, include it as a separate must_include item with the specific action recommended.

Return: {{"must_include": [{{"entry_id": "e1", "importance": "critical|high|medium",
  "section": "section heading", "summary": "one specific fact with exact numbers/names"}}]}}"""

    payload, tokens = call_model(caller, prompt, max_tokens=16384)
    by_model, _, _, _ = get_last_call_usage()
    items = payload.get("must_include", [])
    if not isinstance(items, list):
        items = []
    return index, doc_name, items, tokens, by_model or {}


def _merge_usage(target: dict, source: dict | None) -> None:
    if not isinstance(source, dict):
        return
    for model, usage in source.items():
        if model not in target:
            target[model] = {"input": 0, "output": 0, "total": 0, "calls": 0}
        target[model]["input"] += usage.get("input", 0)
        target[model]["output"] += usage.get("output", 0)
        target[model]["total"] += usage.get("total", 0)
        target[model]["calls"] += usage.get("calls", 0)


def _write_curation_progress(
    blackboard: Blackboard,
    completed: int,
    total: int,
    doc_name: str,
    item_count: int,
) -> None:
    if not blackboard.output_dir:
        return
    swarm_dir = Path(blackboard.output_dir) / "swarm"
    swarm_dir.mkdir(parents=True, exist_ok=True)
    progress = {
        "completed_clusters": completed,
        "total_clusters": total,
        "latest_document": doc_name,
        "latest_item_count": item_count,
    }
    (swarm_dir / "curation_progress.json").write_text(
        json.dumps(progress, indent=2),
        encoding="utf-8",
    )


def _fallback_curate(
    blackboard: Blackboard,
    active: list[Entry],
    caller: ModelCaller,
) -> tuple[list[dict], int]:
    all_summaries = "\n".join(
        f"[{e.id}] ({e.type}) {e.content[:300]}"
        for e in active
    )
    prompt = f"""The curation step produced too few items. You must now enumerate EVERY fact from these findings.

TASK: {blackboard.task_instruction}

ALL FINDINGS ({len(active)} total):
{all_summaries[:80000]}

MANDATORY: Produce one must_include item for EVERY finding that contains a specific fact.
Target: at least {len(active) // 2} items (half the findings should have extractable facts).

Return: {{"must_include": [{{"entry_id": "e1", "importance": "critical|high|medium",
  "section": "section heading", "summary": "the specific fact"}}]}}"""

    payload, tokens = call_model(caller, prompt, max_tokens=16384)
    items = payload.get("must_include", [])
    if not isinstance(items, list):
        items = []
    return items, tokens


COVERAGE_CONFIDENCE_THRESHOLD = 0.6
MAX_AUTOINCLUDES_PER_DOC = 3


def _coverage_safety_net(
    must_include: list[dict],
    active: list[Entry],
) -> list[dict]:
    """Auto-include high-confidence entries from documents with zero curation coverage."""
    curated_ids = set()
    for m in must_include:
        if isinstance(m, dict):
            raw_ids = m.get("entry_ids")
            if isinstance(raw_ids, list):
                for v in raw_ids:
                    s = str(v).strip()
                    if s:
                        curated_ids.add(s)
            eid = m.get("entry_id", "")
            if eid:
                for part in str(eid).split(","):
                    curated_ids.add(part.strip())

    by_doc: dict[str, list[Entry]] = {}
    for e in active:
        if e.id in curated_ids:
            continue
        doc = e.source.document if e.source else "cross_cutting"
        by_doc.setdefault(doc or "cross_cutting", []).append(e)

    curated_docs = set()
    for e in active:
        if e.id in curated_ids:
            doc = e.source.document if e.source else "cross_cutting"
            curated_docs.add(doc or "cross_cutting")

    for doc, entries in by_doc.items():
        if doc in curated_docs:
            continue
        best = sorted(
            [e for e in entries if e.confidence >= COVERAGE_CONFIDENCE_THRESHOLD and e.type != "gap"],
            key=lambda e: (-e.confidence, e.id),
        )
        for e in best[:MAX_AUTOINCLUDES_PER_DOC]:
            must_include.append({
                "entry_id": e.id,
                "importance": "high",
                "section": e.source.section if e.source and e.source.section else "Additional Findings",
                "summary": (e.content or "")[:500],
                "source": "coverage_safety_net",
            })

    return must_include
