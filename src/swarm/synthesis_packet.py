from __future__ import annotations

import json
from pathlib import Path

from .blackboard import Blackboard
from .models import Entry


EVIDENCE_TYPES = frozenset({"observation", "analysis", "calculation"})
OPEN_ISSUE_TYPES = frozenset({"strategy", "gap"})


def build_synthesis_packet(
    must_include: list[dict],
    blackboard: Blackboard,
) -> list[dict]:
    """Normalize must_include items into structured packet rows.

    Strategy/gap entries are marked open_issue_only=True so synthesis
    renders them as explicit open issues rather than asserting them as facts.
    """
    by_id = {e.id: e for e in blackboard.entries if e.type != "entity_overview"}
    packet: list[dict] = []
    seen_keys: set[str] = set()

    for item in must_include:
        if not isinstance(item, dict):
            continue

        row = _normalize_item(item, by_id)
        key = _dedup_key(row)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        packet.append(row)

    return packet


def _normalize_item(item: dict, by_id: dict[str, Entry]) -> dict:
    entry_ids = _extract_entry_ids(item)
    entries = [by_id[eid] for eid in entry_ids if eid in by_id]

    source_refs = []
    for e in entries:
        if e.source and e.source.document:
            ref = e.source.document
            if e.source.section:
                ref += f" / {e.source.section}"
            if ref not in source_refs:
                source_refs.append(ref)

    open_issue_only = False
    if entries:
        open_issue_only = all(e.type in OPEN_ISSUE_TYPES for e in entries)
    elif entry_ids:
        # All referenced entries missing from blackboard — treat as ungrounded
        open_issue_only = True
    elif item.get("source") == "artifact_contract":
        open_issue_only = True

    return {
        "entry_ids": entry_ids,
        "summary": item.get("summary", ""),
        "section": item.get("section", "General"),
        "importance": item.get("importance", "medium"),
        "target_file": item.get("target_file", ""),
        "native_form": item.get("native_form", ""),
        "source": item.get("source", "curation"),
        "obligation_type": item.get("obligation_type", ""),
        "verification_terms": item.get("verification_terms", ""),
        "required_source_refs": source_refs,
        "open_issue_only": open_issue_only,
        "artifact_function": item.get("artifact_function", ""),
        "satisfaction_conditions": item.get("satisfaction_conditions", []),
        "evidence_entry_ids": item.get("evidence_entry_ids", []),
    }


def _extract_entry_ids(item: dict) -> list[str]:
    raw = item.get("entry_ids")
    if isinstance(raw, list):
        return [str(e).strip() for e in raw if str(e).strip()]
    raw = item.get("entry_id", "")
    if not raw:
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _dedup_key(row: dict) -> str:
    ids = ",".join(sorted(row.get("entry_ids", [])))
    target = row.get("target_file", "")
    return f"{ids}|{target}|{row.get('summary', '')[:60].lower().strip()}"


def packet_items_for_file(
    packet: list[dict],
    filename: str,
) -> list[dict]:
    """Return packet rows targeted at a specific file, plus untargeted rows."""
    return [
        row for row in packet
        if not row.get("target_file") or row["target_file"] == filename
    ]


def filter_evidence_entries(
    packet: list[dict],
    active: list[Entry],
) -> tuple[list[Entry], list[dict]]:
    """Split active entries into evidence vs open-issue-only based on packet.

    Returns (evidence_entries, open_issue_items) where evidence_entries
    are entries that can be cited as factual support, and open_issue_items
    are packet rows that should be rendered as explicit open issues.
    """
    open_issue_entry_ids: set[str] = set()
    open_issue_items: list[dict] = []

    for row in packet:
        if row.get("open_issue_only"):
            open_issue_entry_ids.update(row.get("entry_ids", []))
            open_issue_items.append(row)

    evidence_entry_ids: set[str] = set()
    for row in packet:
        if not row.get("open_issue_only"):
            evidence_entry_ids.update(row.get("entry_ids", []))

    pure_open_issue_ids = open_issue_entry_ids - evidence_entry_ids

    by_id = {e.id: e for e in active}
    evidence_entries = [
        e for e in active
        if e.type != "entity_overview" and e.id not in pure_open_issue_ids
    ]

    return evidence_entries, open_issue_items


def write_synthesis_packet_report(
    packet: list[dict],
    output_dir: str | None,
) -> None:
    """Write diagnostic report of the synthesis packet."""
    if not output_dir:
        return
    swarm_dir = Path(output_dir) / "swarm"
    swarm_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "total_items": len(packet),
        "evidence_items": sum(1 for r in packet if not r.get("open_issue_only")),
        "open_issue_items": sum(1 for r in packet if r.get("open_issue_only")),
        "by_source": _count_by_key(packet, "source"),
        "by_importance": _count_by_key(packet, "importance"),
        "items": packet,
    }
    (swarm_dir / "synthesis_packet.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )


def _count_by_key(items: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        val = str(item.get(key, "unknown"))
        counts[val] = counts.get(val, 0) + 1
    return counts


_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or",
    "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
    "this", "that", "these", "those", "it", "its", "their", "our", "your",
})
_IMPORTANCE_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def _significant_words(text: str) -> set[str]:
    words = set(text.lower().split())
    return {w for w in words if len(w) >= 3 and w not in _STOP_WORDS}


def _partition_key(item: dict) -> str:
    sec = item.get("section", "General")
    target = item.get("target_file", "")
    source = item.get("source", "")
    native = item.get("native_form", "")
    return f"{sec}|{target}|{source}|{native}"


_HIGH_IMPORTANCE = frozenset({"critical", "high"})
_MAX_ITEM_DROP_RATIO = 0.70
_MAX_HIGH_DROP_RATIO = 0.80


def consolidate_items(items: list[dict], similarity_threshold: float = 0.85) -> list[dict]:
    """Merge near-duplicate items within same partition via word-set overlap.

    Includes a regression guard: if consolidation drops more than 30% of total
    items or 20% of critical/high items, the original list is returned unchanged.
    """
    by_partition: dict[str, list[dict]] = {}
    for item in items:
        key = _partition_key(item)
        by_partition.setdefault(key, []).append(item)

    result = []
    for _key, group in by_partition.items():
        group.sort(key=lambda x: len(x.get("summary", "")), reverse=True)
        fingerprints = [_significant_words(item.get("summary", "")) for item in group]
        absorbed = [False] * len(group)

        for i in range(len(group)):
            if absorbed[i]:
                continue
            merged = dict(group[i])
            merged_ids = set(_extract_entry_ids(merged))
            for j in range(i + 1, len(group)):
                if absorbed[j]:
                    continue
                fi, fj = fingerprints[i], fingerprints[j]
                union = fi | fj
                if not union:
                    continue
                overlap = len(fi & fj) / len(union)
                if overlap >= similarity_threshold:
                    absorbed[j] = True
                    merged_ids.update(_extract_entry_ids(group[j]))
                    j_imp = _IMPORTANCE_RANK.get(group[j].get("importance", "medium"), 2)
                    m_imp = _IMPORTANCE_RANK.get(merged.get("importance", "medium"), 2)
                    if j_imp < m_imp:
                        merged["importance"] = group[j]["importance"]
            merged["entry_ids"] = sorted(merged_ids)
            if "entry_id" in merged:
                merged["entry_id"] = ",".join(merged["entry_ids"])
            result.append(merged)

    if len(items) < 15:
        return result

    if len(result) / len(items) < _MAX_ITEM_DROP_RATIO:
        return list(items)

    orig_high = sum(1 for i in items if i.get("importance") in _HIGH_IMPORTANCE)
    if orig_high > 4:
        new_high = sum(1 for i in result if i.get("importance") in _HIGH_IMPORTANCE)
        if new_high / orig_high < _MAX_HIGH_DROP_RATIO:
            return list(items)

    return result
