"""Deterministic cross-document entity resolution for the blackboard.

Detects near-duplicate entity names in blackboard entries using pure
string-similarity algorithms — no LLM calls, no embeddings, no external
dependencies beyond Python stdlib.

Env gating:
  SWARM_ENABLE_ENTITY_RESOLUTION=1 — enables the full resolution pipeline
  SWARM_ENTITY_RESOLUTION_THRESHOLD=0.85 — similarity threshold (default 0.85)
  SWARM_ENTITY_RESOLUTION_EDIT_WEIGHT=0.5 — weight for Levenshtein score (default 0.5)
  SWARM_ENTITY_RESOLUTION_JARO_WEIGHT=0.3 — weight for Jaro-Winkler score (default 0.3)
  SWARM_ENTITY_RESOLUTION_TRIGRAM_WEIGHT=0.2 — weight for trigram score (default 0.2)

Design:
  1. Extract candidate entity names from active entries (capitalised phrases,
     company names, defined terms)
  2. Normalise names by stripping legal suffixes for better matching
  3. Compute pairwise similarity using a weighted combination of:
     - Normalized Levenshtein distance
     - Jaro-Winkler similarity
     - Jaccard trigram overlap (consistent with signals_similar in blackboard.py)
     - Token overlap fraction
  4. When similarity >= threshold, emit a contradiction entry linking the
     near-duplicates and update supersedes_entries to reconcile them.
  5. Write entity_resolution_report.json to the swarm output directory.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

from .blackboard import Blackboard
from .models import (
    Entry,
    EntrySource,
    Signal,
    WorkerRecord,
    gen_entry_id,
    gen_signal_id,
)

# --- Env gating ---

ENTITY_MIN_NAME_LENGTH = 4
"""Minimum character length for a candidate entity name."""

ENTITY_MAX_EDIT_RATIO = 0.4
"""Maximum normalized Levenshtein distance to even consider a pair."""

_WEIGHT_EDIT_KEY = "SWARM_ENTITY_RESOLUTION_EDIT_WEIGHT"
_WEIGHT_JARO_KEY = "SWARM_ENTITY_RESOLUTION_JARO_WEIGHT"
_WEIGHT_TRIGRAM_KEY = "SWARM_ENTITY_RESOLUTION_TRIGRAM_WEIGHT"
_THRESHOLD_KEY = "SWARM_ENTITY_RESOLUTION_THRESHOLD"


def entity_resolution_enabled() -> bool:
    return _env_on("SWARM_ENABLE_ENTITY_RESOLUTION")


def _resolution_threshold() -> float:
    raw = os.getenv(_THRESHOLD_KEY, "0.85").strip()
    try:
        return max(0.0, min(1.0, float(raw)))
    except (ValueError, TypeError):
        return 0.85


def _edit_weight() -> float:
    raw = os.getenv(_WEIGHT_EDIT_KEY, "0.3").strip()
    try:
        return max(0.0, min(1.0, float(raw)))
    except (ValueError, TypeError):
        return 0.3


def _jaro_weight() -> float:
    raw = os.getenv(_WEIGHT_JARO_KEY, "0.5").strip()
    try:
        return max(0.0, min(1.0, float(raw)))
    except (ValueError, TypeError):
        return 0.5


def _trigram_weight() -> float:
    raw = os.getenv(_WEIGHT_TRIGRAM_KEY, "0.2").strip()
    try:
        return max(0.0, min(1.0, float(raw)))
    except (ValueError, TypeError):
        return 0.2


# --- String similarity functions ---


def _normalized_levenshtein(a: str, b: str) -> float:
    """Normalized Levenshtein distance (0 = identical, 1 = completely different)."""
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    # Make a the shorter string
    if len(a) > len(b):
        a, b = b, a
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(
                curr[-1] + 1,       # insertion
                prev[j] + 1,        # deletion
                prev[j - 1] + cost, # substitution
            ))
        prev = curr
    return prev[n] / max(m, n, 1)


def _jaro_winkler(a: str, b: str) -> float:
    """Jaro-Winkler similarity (0 = completely different, 1 = identical)."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    # Jaro common characters
    max_dist = max(len(a), len(b)) // 2 - 1
    if max_dist < 0:
        max_dist = 0
    a_match = [False] * len(a)
    b_match = [False] * len(b)
    common = 0
    for i, ca in enumerate(a):
        start = max(0, i - max_dist)
        end = min(len(b), i + max_dist + 1)
        for j in range(start, end):
            if not b_match[j] and ca == b[j]:
                a_match[i] = True
                b_match[j] = True
                common += 1
                break
    if common == 0:
        return 0.0
    # Transpositions
    k = 0
    transpositions = 0
    for i, ca in enumerate(a):
        if not a_match[i]:
            continue
        while not b_match[k]:
            k += 1
        if ca != b[k]:
            transpositions += 1
        k += 1
    jaro = (common / len(a) + common / len(b) + (common - transpositions / 2) / common) / 3
    # Winkler boost for common prefix
    prefix = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            prefix += 1
        else:
            break
    prefix = min(prefix, 4)
    return jaro + prefix * 0.1 * (1 - jaro)


def _trigram_similarity(a: str, b: str) -> float:
    """Jaccard similarity over character trigrams."""
    def trigrams(s: str) -> set[str]:
        # Normalize: lower-case, collapse whitespace
        clean = re.sub(r"[^a-z0-9\s]", "", s.lower())
        tokens = re.sub(r"\s+", " ", clean).strip().split()
        text = " ".join(tokens)
        if len(text) < 3:
            return {text}
        return {text[i:i+3] for i in range(len(text) - 2)}
    set_a = trigrams(a)
    set_b = trigrams(b)
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    return len(intersection) / max(len(set_a | set_b), 1)


def _token_overlap(a: str, b: str) -> float:
    """Fraction of common significant tokens."""
    def tokens(s: str) -> set[str]:
        # Extract words, skip stop-like short tokens
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9.#&\-]+", s.lower())
        return {w for w in words if len(w) > 1 and w not in _ENTITY_STOP_WORDS}
    tokens_a = tokens(a)
    tokens_b = tokens(b)
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = tokens_a & tokens_b
    return len(overlap) / max(len(tokens_a | tokens_b), 1)


_ENTITY_STOP_WORDS = frozenset({
    "the", "and", "for", "that", "with", "from", "this", "are", "was",
    "were", "has", "have", "been", "its", "their", "section", "paragraph",
    "clause", "exhibit", "schedule", "agreement", "document", "number",
    "dated", "between", "company", "party", "parties",
})


# --- Name normalisation ---

_LEGAL_SUFFIXES = re.compile(
    r"(?:[,\s]+(?:Corporation|Corp\.?|Inc\.?|LLC|Ltd\.?|LLP|LP|PLC|GmbH|AG|SA"
    r"|GmbH\s*&\s*Co\.?\s*KG|Incorporated|Limited|Company|Co\.?"
    r"|Corp|Inc|Ltd|N\.?\s*V\.?|P\.?\s*L\.?\s*C\.?))[,.]?\s*$",
    re.IGNORECASE,
)

# Words that, while not legal suffixes, are common trailing business terms
# that can be stripped for a core name variant
_CORE_NAME_STRIP = re.compile(
    r"\s+(?:Industries|Solutions|Services|Systems|Technologies|Enterprises|"
    r"Holdings|Holding|International|Group|Ventures|Partners|Associates|"
    r"Properties|Markets|Capital|Management|Consulting|Logistics|Global)[,.]?\s*$",
    re.IGNORECASE,
)


def _normalize_name(raw: str) -> str:
    """Normalize a company/entity name for better matching.

    Strips legal suffixes, trailing commas/periods, and common business
    designators so that "Zenith Petrochemical Industries LLC" becomes
    "Zenith Petrochemical".
    """
    name = raw.strip()
    name = _LEGAL_SUFFIXES.sub("", name).strip()
    name = re.sub(r"[,.#]$", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


# --- Composite similarity ---


def compute_entity_similarity(a: str, b: str) -> float:
    """Weighted composite similarity between two entity names.

    Returns a value in [0, 1] where 1 = identical and 0 = completely
    different. Combines Levenshtein distance (as similarity), Jaro-Winkler,
    trigram Jaccard, and token overlap. Names are normalised first.
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    # Normalise before comparison
    norm_a = _normalize_name(a)
    norm_b = _normalize_name(b)

    if norm_a == norm_b:
        return 1.0
    if not norm_a or not norm_b:
        # One was entirely suffix — fall back to raw
        norm_a, norm_b = a, b

    # Quick reject: if edit ratio is too high, skip expensive checks
    if _normalized_levenshtein(norm_a, norm_b) > ENTITY_MAX_EDIT_RATIO:
        return 0.0

    lev_sim = 1.0 - _normalized_levenshtein(norm_a, norm_b)
    jaro = _jaro_winkler(norm_a, norm_b)
    trigram = _trigram_similarity(norm_a, norm_b)

    w_edit = _edit_weight()
    w_jaro = _jaro_weight()
    w_trigram = _trigram_weight()
    w_total = w_edit + w_jaro + w_trigram
    if w_total == 0:
        w_edit = w_jaro = w_trigram = 1.0
        w_total = 3.0

    score = (w_edit * lev_sim + w_jaro * jaro + w_trigram * trigram) / w_total

    # Substring boost: if one normalised name is a significant substring
    # of the other (e.g. "Petrochem" in "Petrochemical"), add a boost
    short, long = (norm_a, norm_b) if len(norm_a) < len(norm_b) else (norm_b, norm_a)
    if short and len(short) >= 3 and long.startswith(short):
        containment = len(short) / max(len(long), 1)
        if containment > 0.3:
            score = max(score, containment * jaro)

    # Boost by token overlap
    token = _token_overlap(norm_a, norm_b)
    if token > 0.5:
        score = min(1.0, score + 0.05 * token)

    return score


# --- Entity extraction ---


def _extract_entity_names(content: str) -> list[str]:
    """Extract candidate entity names from a piece of text.

    For each extracted name, also generates:
    - The name with legal suffix stripped (e.g. "Inc.", "LLC")
    - The name with trailing business terms stripped (e.g. "Industries", "Solutions")

    This multi-resolution approach improves matching across different
    representation styles in the blackboard.
    """
    names: list[str] = []
    seen: set[str] = set()

    # Pattern 1: Capitalized multi-word phrase (2-8 words)
    for match in re.finditer(
        r"(?:[A-Z][a-z]+[']?)+(?:\s+(?:[A-Z][a-z]+[']?|and|of|the|for|in|&)){1,7}",
        content,
    ):
        name = match.group().strip()
        # Remove leading stop words
        name = re.sub(r"^(?:and|of|the|for|in)\s+", "", name)
        if len(name) >= ENTITY_MIN_NAME_LENGTH:
            _add_name_variants(name, names, seen)

    # Pattern 2: ALL-CAPS acronyms (3+ letters)
    for match in re.finditer(r"\b[A-Z]{3,}\b", content):
        name = match.group()
        if name not in seen:
            seen.add(name)
            names.append(name)

    # Pattern 3: "X Corporation" / "X LLC" / "X Inc." / "X Ltd." style
    for match in re.finditer(
        r"(?:[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:Corporation|Corp\.?|Inc\.?|LLC|Ltd\.?|LLP|LP|PLC|GmbH|AG|SA|GmbH\s*&\s*Co\.?\s*KG)\b",
        content,
    ):
        name = match.group().strip()
        if len(name) >= ENTITY_MIN_NAME_LENGTH:
            _add_name_variants(name, names, seen)

    return names


def _add_name_variants(name: str, names: list[str], seen: set[str]) -> None:
    """Add a name and its normalised variants to the names list."""
    if name not in seen:
        seen.add(name)
        names.append(name)

    # Add variant with legal suffix stripped
    stripped_suffix = _LEGAL_SUFFIXES.sub("", name).strip()
    stripped_suffix = re.sub(r"[,.#]$", "", stripped_suffix).strip()
    if stripped_suffix and stripped_suffix != name and stripped_suffix not in seen:
        if len(stripped_suffix) >= ENTITY_MIN_NAME_LENGTH:
            seen.add(stripped_suffix)
            names.append(stripped_suffix)

    # Add variant with business terms stripped too (for the core name)
    core = _CORE_NAME_STRIP.sub("", stripped_suffix).strip()
    core = re.sub(r"[,.#]$", "", core).strip()
    if core and core != stripped_suffix and core not in seen:
        if len(core) >= ENTITY_MIN_NAME_LENGTH:
            seen.add(core)
            names.append(core)


def _extract_all_entity_names(
    entries: list[Entry],
) -> dict[str, set[str]]:
    """Extract candidate entity names from active entries.

    Returns a dict: {entry_id -> set of entity names found in that entry}
    """
    result: dict[str, set[str]] = {}
    for entry in entries:
        if entry.status != "active":
            continue
        if not entry.content or len(entry.content.strip()) < 20:
            continue
        names = _extract_entity_names(entry.content)
        if names:
            result[entry.id] = set(names)
    return result


# --- Pairwise comparison ---


def _find_near_duplicates(
    entity_map: dict[str, set[str]],
    threshold: float,
) -> list[dict]:
    """Find near-duplicate entity names across entries.

    Returns a list of resolution dicts:
      {
        "entry_a": "e1", "entry_b": "e5",
        "name_a": "Zenith Petrochem",
        "name_b": "Zenith Petrochemical",
        "similarity": 0.91,
        "method": "weighted_composite"
      }
    """
    # Build a reverse index: entity_name -> list of entry_ids
    name_to_entries: dict[str, list[str]] = {}
    for entry_id, names in entity_map.items():
        for name in names:
            name_to_entries.setdefault(name, []).append(entry_id)

    all_names = list(name_to_entries.keys())
    duplicates: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()

    # Phase 1: Near-duplicate name matching (different entity names that
    # refer to the same thing, e.g. "Zenith Petrochem" vs "Zenith Petrochemical")
    for i in range(len(all_names)):
        for j in range(i + 1, len(all_names)):
            name_a = all_names[i]
            name_b = all_names[j]

            if name_a == name_b:
                continue

            score = compute_entity_similarity(name_a, name_b)
            if score < threshold:
                continue

            entries_a = set(name_to_entries[name_a])
            entries_b = set(name_to_entries[name_b])

            for ea in entries_a:
                for eb in entries_b:
                    if ea == eb:
                        continue
                    pair_key = (ea, eb) if ea < eb else (eb, ea)
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    duplicates.append({
                        "entry_a": ea,
                        "entry_b": eb,
                        "name_a": name_a,
                        "name_b": name_b,
                        "similarity": round(score, 4),
                        "method": "weighted_composite",
                    })

    # Phase 2: Exact name cross-entry linking — entries from different
    # documents that mention the same entity name should be linked.
    for name, entry_ids in name_to_entries.items():
        ids = sorted(set(entry_ids))
        if len(ids) < 2:
            continue
        # Create links between every pair of entries sharing this name
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pair_key = (ids[i], ids[j])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                duplicates.append({
                    "entry_a": ids[i],
                    "entry_b": ids[j],
                    "name_a": name,
                    "name_b": name,
                    "similarity": 1.0,
                    "method": "exact_name_share",
                })

    # Sort by similarity descending
    duplicates.sort(key=lambda d: -d["similarity"])
    return duplicates


# --- Blackboard integration ---


def apply_resolutions(
    blackboard: Blackboard,
    duplicates: list[dict],
    *,
    max_resolutions: int = 50,
) -> tuple[list[Entry], list[Signal]]:
    """Apply entity resolution proposals to the blackboard.

    For each near-duplicate pair:
    1. Create a contradiction entry linking the two
    2. Emit an entity_resolution signal

    Returns (new_entries, new_signals).
    """
    new_entries: list[Entry] = []
    new_signals: list[Signal] = []
    applied = 0
    applied_pairs: set[tuple[str, str]] = set()

    for dup in duplicates:
        if applied >= max_resolutions:
            break

        ea, eb = dup["entry_a"], dup["entry_b"]
        pair_key = (ea, eb) if ea < eb else (eb, ea)
        if pair_key in applied_pairs:
            continue
        applied_pairs.add(pair_key)

        name_a, name_b = dup["name_a"], dup["name_b"]
        sim = dup["similarity"]

        # Create contradiction entry
        contradiction_entry = Entry(
            id=gen_entry_id(),
            type="contradiction",
            content=(
                f"ENTITY RESOLUTION: '{name_a}' (in {ea}) and "
                f"'{name_b}' (in {eb}) appear to refer to the same "
                f"entity (similarity={sim:.2f}). "
                f"Recommend reconciliation."
            ),
            source=None,
            created_by=WorkerRecord(
                "entity_resolution",
                f"automated_entity_resolution_sim={sim:.2f}",
                blackboard.iteration,
            ),
            confidence=min(0.95, sim),
            tags=["entity_resolution", "automated"],
            status="active",
            supports_entries=[ea, eb],
        )
        new_entries.append(contradiction_entry)

        # Emit resolution signal
        resolution_signal = Signal(
            id=gen_signal_id(),
            type="contradiction_resolution",
            content=(
                f"Resolve near-duplicate entities: '{name_a}' (in {ea}) "
                f"vs '{name_b}' (in {eb}) — similarity={sim:.2f}"
            ),
            origin_entry=contradiction_entry.id,
            priority="medium",
            status="open",
            iteration_created=blackboard.iteration,
        )
        new_signals.append(resolution_signal)

        applied += 1

    return new_entries, new_signals


def run_entity_resolution(
    blackboard: Blackboard,
) -> tuple[list[Entry], int]:
    """Run the full entity resolution pipeline.

    Steps:
    1. Extract entity names from all active entries
    2. Find near-duplicate pairs using composite string similarity
    3. Apply resolutions (create entries + signals)
    4. Write report to swarm output directory

    Returns (new_entries, tokens_used=0 — no LLM calls).
    """
    if not entity_resolution_enabled():
        return [], 0

    active = [e for e in blackboard.entries if e.status == "active"]
    if len(active) < 2:
        return [], 0

    threshold = _resolution_threshold()

    # Step 1: Extract entities
    entity_map = _extract_all_entity_names(active)
    if not entity_map:
        return [], 0

    # Step 2: Find near-duplicates
    duplicates = _find_near_duplicates(entity_map, threshold)

    # Step 3: Apply resolutions
    new_entries, new_signals = apply_resolutions(blackboard, duplicates)

    # Step 4: Write report
    report = {
        "schema_version": 1,
        "enabled": True,
        "threshold": threshold,
        "entries_scanned": len(active),
        "names_extracted": sum(len(names) for names in entity_map.values()),
        "near_duplicate_pairs_found": len(duplicates),
        "resolutions_applied": len(new_entries),
        "signals_emitted": len(new_signals),
        "duplicates": duplicates[:100],  # cap report size
        "summary": {
            "entries_checked": len(entity_map),
            "total_names": sum(len(n) for n in entity_map.values()),
            "resolutions": len(new_entries),
            "signals": len(new_signals),
        },
    }

    _write_report(blackboard.output_dir, report)

    return new_entries, 0  # zero token cost — pure computation


def _write_report(output_dir: str, report: dict) -> None:
    if not output_dir:
        return
    swarm_dir = Path(output_dir) / "swarm"
    swarm_dir.mkdir(parents=True, exist_ok=True)
    (swarm_dir / "entity_resolution_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


# --- Deterministic test helpers ---


def _env_on(key: str) -> bool:
    return os.getenv(key, "").strip().lower() in ("1", "true", "yes", "on")
