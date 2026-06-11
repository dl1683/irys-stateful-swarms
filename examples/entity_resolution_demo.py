#!/usr/bin/env python3
"""Demo: deterministic entity resolution on the README's documented failure cases.

Run:  python examples/entity_resolution_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.swarm.blackboard import Blackboard
from src.swarm.entity_resolution import resolve_entities, run_entity_resolution
from src.swarm.models import Entry, EntrySource, WorkerRecord, gen_entry_id


def mk(content: str, doc: str) -> Entry:
    return Entry(
        id=gen_entry_id(), type="observation", content=content,
        source=EntrySource(document=doc, section="Full Document"),
        created_by=WorkerRecord("reader_worker", "initial_reading", 0),
        confidence=0.9,
    )


def demo(name: str, entries: list[tuple[str, str]]):
    bb = Blackboard()
    for content, doc in entries:
        bb.add_entry(mk(content, doc))

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Entries: {len(bb.entries)}")

    result = resolve_entities(bb, threshold=0.65)

    print(f"  Clusters: {len(result.clusters)}")
    print(f"  Match pairs: {len(result.match_pairs)}")
    print(f"  Resolution entries created: {len(result.entries_created)}")
    print(f"  LLM tokens used: {result.tokens_used}")

    if result.match_pairs:
        print("\n  Fuzzy matches:")
        for m in result.match_pairs:
            print(f"    \"{m.raw_name_a}\" ↔ \"{m.raw_name_b}\"")
            print(f"      composite={m.composite:.3f}  "
                  f"jw={m.jaro_winkler:.3f}  lev={m.levenshtein:.3f}  "
                  f"tri={m.trigram_jaccard:.3f}  tok={m.token_overlap:.3f}  "
                  f"prefix={m.is_prefix}")

    if result.clusters:
        print("\n  Clusters:")
        for cluster in result.clusters:
            canonical = max(cluster, key=lambda c: len(c["norm"]))
            variants = [c for c in cluster if c["raw"] != canonical["raw"]]
            print(f"    Canonical: \"{canonical['raw']}\" ({canonical['doc']})")
            for v in variants:
                print(f"      Variant: \"{v['raw']}\" ({v['doc']})")

    if result.entries_created:
        print("\n  Created blackboard entries:")
        for entry in result.entries_created:
            print(f"    [{entry.id}] conf={entry.confidence:.2f}")
            for line in entry.content.split("\n")[:4]:
                print(f"      {line}")

    return result


# --- Case 1: Zenith Petrochem (from README failure analysis) ---
demo("Zenith Petrochem — sanctions entity extraction near-miss", [
    (
        "The exporter is Zenith Petrochem Industries LLC, located in "
        "Jebel Ali Free Zone, UAE.",
        "credit-agreement.docx",
    ),
    (
        "Zenith Petrochemical Industries LLC, Jebel Ali Free Zone, Dubai, UAE",
        "sanctions-screening.xlsx",
    ),
])

# --- Case 2: Pinnacle Industrial Solutions (from README failure analysis) ---
demo("Pinnacle Industrial Solutions — UCC lien extraction near-miss", [
    (
        "Debtor: Pinnacle Industrial Solutions, Inc., a corporation "
        "organized in Ohio, Charter No. 2187650",
        "ucc-filing-1.pdf",
    ),
    (
        "Filing OH-2019-0178443 (Tristate Capital Equipment Corp.) "
        "against Pinnacle Industrial Solutions is LAPSED as of May 15, 2024.",
        "ucc-filing-2.pdf",
    ),
])

# --- Case 3: Northbrook Capital Markets (from README perfect-score example) ---
demo("Northbrook Capital Markets — credit agreement comparison", [
    (
        "Northbrook Capital Markets, LLC commits to provide a first lien "
        "senior secured term loan B facility in an aggregate principal "
        "amount of $350,000,000.",
        "commitment-letter.docx",
    ),
    (
        "Northbrook Capital Markets LLC shall serve as Administrative Agent "
        "under the Credit Agreement.",
        "credit-agreement.docx",
    ),
])

# --- Case 4: No false positives ---
demo("Different entities — should NOT match", [
    ("Alpha Corp Holdings Inc is the buyer.", "purchase-agreement.docx"),
    ("Beta Industries LLC is the seller.", "purchase-agreement.docx"),
    ("Gamma Partners LLP provided financing.", "loan-agreement.docx"),
])

print("\nDone. Zero LLM tokens used across all demos.")
