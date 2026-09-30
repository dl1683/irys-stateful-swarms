from src.swarm.blackboard import Blackboard
from src.swarm.models import Entry, EntrySource, ModelResult
from src.swarm.synthesis_packet import (
    build_synthesis_packet,
    consolidate_items,
    filter_evidence_entries,
    packet_items_for_file,
    project_overview_statements,
)
from src.swarm.synthesis import _draft_synthesis


def _bb_with_entries(entries):
    return Blackboard(task_instruction="Test task", entries=entries)


def test_normalizes_curation_items():
    entries = [
        Entry(id="e1", type="observation", content="Revenue is $10M",
              source=EntrySource("10k.pdf", "Financials", "Revenue $10M"),
              confidence=0.9),
        Entry(id="e2", type="analysis", content="Growth rate is 15%",
              source=EntrySource("10k.pdf", "Analysis", "15% growth"),
              confidence=0.85),
    ]
    bb = _bb_with_entries(entries)
    items = [
        {"entry_id": "e1", "importance": "critical", "section": "Revenue",
         "summary": "Revenue is $10M"},
        {"entry_id": "e2", "importance": "high", "section": "Growth",
         "summary": "Growth rate is 15%"},
    ]
    packet = build_synthesis_packet(items, bb)
    assert len(packet) == 2
    assert packet[0]["entry_ids"] == ["e1"]
    assert packet[0]["required_source_refs"] == ["10k.pdf / Financials"]
    assert packet[0]["open_issue_only"] is False


def test_strategy_entries_marked_open_issue():
    entries = [
        Entry(id="e1", type="strategy", content="Consider alternative approach",
              confidence=0.7),
        Entry(id="e2", type="gap", content="Missing vendor backup plan",
              confidence=0.6),
        Entry(id="e3", type="observation", content="Contract expires June 2027",
              source=EntrySource("msa.pdf", "Terms", "June 2027"),
              confidence=0.95),
    ]
    bb = _bb_with_entries(entries)
    items = [
        {"entry_id": "e1", "importance": "medium", "section": "Strategy",
         "summary": "Alternative approach"},
        {"entry_id": "e2", "importance": "high", "section": "Gaps",
         "summary": "Missing vendor backup"},
        {"entry_id": "e3", "importance": "critical", "section": "Terms",
         "summary": "Contract expires June 2027"},
    ]
    packet = build_synthesis_packet(items, bb)
    by_eid = {r["entry_ids"][0]: r for r in packet}
    assert by_eid["e1"]["open_issue_only"] is True
    assert by_eid["e2"]["open_issue_only"] is True
    assert by_eid["e3"]["open_issue_only"] is False


def test_mixed_entry_ids_not_marked_open_issue():
    entries = [
        Entry(id="e1", type="observation", content="Fact A",
              source=EntrySource("doc.pdf", "S1", "ev"), confidence=0.9),
        Entry(id="e2", type="gap", content="Gap B", confidence=0.5),
    ]
    bb = _bb_with_entries(entries)
    items = [
        {"entry_ids": ["e1", "e2"], "importance": "high", "section": "Mixed",
         "summary": "Fact with gap"},
    ]
    packet = build_synthesis_packet(items, bb)
    assert len(packet) == 1
    assert packet[0]["open_issue_only"] is False


def test_deduplication():
    entries = [
        Entry(id="e1", type="observation", content="X",
              source=EntrySource("d.pdf", "S", "ev"), confidence=0.9),
    ]
    bb = _bb_with_entries(entries)
    items = [
        {"entry_id": "e1", "section": "A", "summary": "same summary here"},
        {"entry_id": "e1", "section": "B", "summary": "same summary here"},
    ]
    packet = build_synthesis_packet(items, bb)
    assert len(packet) == 1


def test_packet_items_for_file():
    packet = [
        {"target_file": "memo.docx", "summary": "A"},
        {"target_file": "sheet.xlsx", "summary": "B"},
        {"target_file": "", "summary": "C"},
    ]
    memo_items = packet_items_for_file(packet, "memo.docx")
    assert len(memo_items) == 2
    assert {r["summary"] for r in memo_items} == {"A", "C"}


def test_filter_evidence_entries():
    entries = [
        Entry(id="e1", type="observation", content="Fact",
              source=EntrySource("d.pdf", "S", "ev"), confidence=0.9),
        Entry(id="e2", type="gap", content="Missing info", confidence=0.5),
        Entry(id="e3", type="analysis", content="Analysis",
              source=EntrySource("d.pdf", "S2", "ev2"), confidence=0.8),
    ]
    bb = _bb_with_entries(entries)
    active = bb.entries

    packet = [
        {"entry_ids": ["e1"], "open_issue_only": False, "summary": "Fact"},
        {"entry_ids": ["e2"], "open_issue_only": True, "summary": "Gap"},
        {"entry_ids": ["e3"], "open_issue_only": False, "summary": "Analysis"},
    ]

    evidence, open_issues = filter_evidence_entries(packet, active)
    evidence_ids = {e.id for e in evidence}
    assert "e1" in evidence_ids
    assert "e3" in evidence_ids
    assert "e2" not in evidence_ids
    assert len(open_issues) == 1
    assert open_issues[0]["summary"] == "Gap"


def test_entity_overviews_do_not_enter_final_evidence():
    entries = [
        Entry(id="e1", type="entity_overview", content="Long derived overview", confidence=0.9),
        Entry(id="e2", type="analysis", content="Supported new analysis", confidence=0.8),
    ]
    bb = _bb_with_entries(entries)
    packet = build_synthesis_packet([
        {"entry_id": "e1", "summary": "Overview"},
        {"entry_id": "e2", "summary": "Analysis"},
    ], bb)
    evidence, _ = filter_evidence_entries(packet, entries)
    assert [entry.id for entry in evidence] == ["e2"]


def test_overview_projection_preserves_original_sources_and_open_issues():
    entries = [
        Entry(id="e1", type="observation", content="Crestmoor Ltd is incorporated in Cyprus.",
              source=EntrySource("kyc.pdf", "Entity details", "Cyprus"), confidence=0.9),
        Entry(id="e2", type="observation", content="Crestmoor Trade & Supply GmbH is a screening candidate.",
              source=EntrySource("screening.pdf", "Candidate", "74% name match"), confidence=0.8),
        Entry(id="e3", type="analysis", content="The candidate was cleared after identifier review.",
              source=EntrySource("screening.pdf", "Disposition", "Cleared"), confidence=0.8),
        Entry(id="e4", type="observation", content="Petrov has conflicting dates of birth.",
              source=EntrySource("case-notes.pdf", "Identity", "1968 and 1975"), confidence=0.7),
        Entry(id="ov1", type="entity_overview", content="Derived overview", confidence=0.9),
    ]
    bb = _bb_with_entries(entries)
    bb.entity_overview_state = {"overviews": {
        "crestmoor": {
            "entry_id": "ov1",
            "structured": {
                "entity_profiles": [{"label": "Crestmoor Ltd", "statements": [{
                    "text": "Crestmoor Ltd is incorporated in Cyprus.",
                    "supports_entries": ["e1"], "kind": "fact",
                }]}],
                "relationships_and_distinctions": [{"label": "Screening", "statements": [{
                    "text": "Crestmoor Trade & Supply GmbH is a separate screening candidate cleared after identifier review.",
                    "supports_entries": ["e2", "e3"], "kind": "derived",
                }]}],
                "unresolved_or_conflicting_evidence": [{"label": "Petrov", "statements": [{
                    "text": "Petrov's 1968 and 1975 dates of birth remain unresolved.",
                    "supports_entries": ["e4"], "kind": "unresolved",
                }]}],
            },
        },
    }}

    projected = project_overview_statements(bb)
    assert [item["overview_group_label"] for item in projected] == [
        "Crestmoor Ltd", "Screening", "Petrov",
    ]
    # Ordinary curation may select the same underlying card; packet deduplication
    # should preserve one report row rather than duplicate the overview statement.
    packet = build_synthesis_packet(projected + [{
        "entry_id": "e1", "summary": "Crestmoor Ltd is incorporated in Cyprus.",
        "section": "Entity Profiles",
    }], bb)
    assert {row["section"] for row in packet} == {
        "Entity Profiles", "Entity Relationships and Distinctions", "Open Issues",
    }
    assert all("ov1" not in row["entry_ids"] for row in packet)
    assert all(row["overview_lineage"] == ["ov1"] for row in packet)
    assert {ref for row in packet for ref in row["required_source_refs"]} == {
        "kyc.pdf / Entity details", "screening.pdf / Candidate",
        "screening.pdf / Disposition", "case-notes.pdf / Identity",
    }
    unresolved = next(row for row in packet if row["section"] == "Open Issues")
    assert unresolved["open_issue_only"] is True
    evidence, open_issues = filter_evidence_entries(packet, bb.entries)
    assert "ov1" not in {entry.id for entry in evidence}
    assert [row["summary"] for row in open_issues] == [unresolved["summary"]]

    class DraftCaller:
        def __init__(self):
            self.prompt = ""

        def complete(self, prompt, **_kwargs):
            self.prompt = prompt
            return ModelResult("assembled report", 1, 1, 2, "fake", 1)

    caller = DraftCaller()
    draft, _ = _draft_synthesis(bb, packet, bb.entries, caller)
    assert draft == "assembled report"
    assert "[Entity Profiles] [Subject: Crestmoor Ltd] Crestmoor Ltd is incorporated in Cyprus." in caller.prompt
    assert "[Subject: Screening] Crestmoor Trade & Supply GmbH" in caller.prompt
    assert "[OPEN ISSUE] [Subject: Petrov]" in caller.prompt
    assert "[Entity Relationships and Distinctions] [Subject: Screening] Crestmoor Trade & Supply GmbH" in caller.prompt
    assert "[OPEN ISSUE] [Subject: Petrov] Petrov's 1968 and 1975 dates of birth remain unresolved." in caller.prompt
    assert "[ov1]" not in caller.prompt


def test_consolidate_does_not_merge_overview_projection_rows():
    items = [
        {"entry_id": "e1", "section": "Entity Profiles", "source": "entity_overview_projection",
         "summary": "Petrov date of birth is 1968", "overview_kind": "fact"},
        {"entry_id": "e2", "section": "Entity Profiles", "source": "entity_overview_projection",
         "summary": "Petrov date of birth is 1975", "overview_kind": "fact"},
    ]
    assert len(consolidate_items(items, similarity_threshold=0.1)) == 2


def test_artifact_contract_items_preserved():
    entries = [
        Entry(id="e1", type="observation", content="X",
              source=EntrySource("d.pdf", "S", "ev"), confidence=0.9),
    ]
    bb = _bb_with_entries(entries)
    items = [
        {"section": "LP Terms", "native_form": "table", "summary": "Compare LP",
         "importance": "critical", "target_file": "memo.docx",
         "source": "artifact_contract"},
        {"entry_id": "e1", "importance": "high", "section": "Facts",
         "summary": "Fact X"},
    ]
    packet = build_synthesis_packet(items, bb)
    contract_rows = [r for r in packet if r["source"] == "artifact_contract"]
    assert len(contract_rows) == 1
    assert contract_rows[0]["native_form"] == "table"
    assert contract_rows[0]["target_file"] == "memo.docx"


def test_consolidate_merges_similar_items_in_same_section():
    items = [
        {"entry_id": "e1", "section": "Revenue", "importance": "high",
         "summary": "Revenue increased from $10M to $15M in fiscal year 2024"},
        {"entry_id": "e2", "section": "Revenue", "importance": "critical",
         "summary": "Revenue increased from $10M to $15M during fiscal year 2024 period"},
        {"entry_id": "e3", "section": "Costs", "importance": "medium",
         "summary": "Operating costs decreased by 12% due to automation"},
    ]
    result = consolidate_items(items)
    revenue_items = [r for r in result if r["section"] == "Revenue"]
    cost_items = [r for r in result if r["section"] == "Costs"]
    assert len(revenue_items) == 1, f"Expected 1 merged revenue item, got {len(revenue_items)}"
    assert len(cost_items) == 1
    assert "e1" in revenue_items[0].get("entry_ids", [])
    assert "e2" in revenue_items[0].get("entry_ids", [])
    assert revenue_items[0]["importance"] == "critical"


def test_consolidate_keeps_distinct_items_separate():
    items = [
        {"entry_id": "e1", "section": "Analysis", "importance": "high",
         "summary": "The merger creates significant antitrust concerns in the retail sector"},
        {"entry_id": "e2", "section": "Analysis", "importance": "high",
         "summary": "Environmental compliance costs exceed $2M annually for the facility"},
    ]
    result = consolidate_items(items)
    assert len(result) == 2


def test_consolidate_does_not_merge_across_sections():
    items = [
        {"entry_id": "e1", "section": "Revenue", "importance": "high",
         "summary": "Revenue increased from $10M to $15M in fiscal year 2024"},
        {"entry_id": "e2", "section": "Costs", "importance": "high",
         "summary": "Revenue increased from $10M to $15M in fiscal year 2024"},
    ]
    result = consolidate_items(items)
    assert len(result) == 2


def test_consolidate_does_not_merge_across_target_files():
    items = [
        {"entry_id": "e1", "section": "Revenue", "importance": "high",
         "target_file": "memo.docx",
         "summary": "Revenue increased from $10M to $15M in fiscal year 2024"},
        {"entry_id": "e2", "section": "Revenue", "importance": "high",
         "target_file": "spreadsheet.xlsx",
         "summary": "Revenue increased from $10M to $15M in fiscal year 2024"},
    ]
    result = consolidate_items(items)
    assert len(result) == 2, "Items targeting different files must not merge"


def test_artifact_contract_without_entries_marked_open_issue():
    entries = [
        Entry(id="e1", type="observation", content="Fact A",
              source=EntrySource("doc.pdf", "S1", "ev"), confidence=0.9),
    ]
    bb = _bb_with_entries(entries)
    items = [
        {"section": "Summary", "native_form": "section", "summary": "Executive summary",
         "importance": "critical", "target_file": "memo.docx",
         "source": "artifact_contract"},
    ]
    packet = build_synthesis_packet(items, bb)
    assert len(packet) == 1
    assert packet[0]["open_issue_only"] is True, (
        "Artifact contract items without evidence must be open_issue_only"
    )


def test_consolidate_regression_guard_rollback_on_excessive_drop():
    """If consolidation would drop >30% of items, roll back to original."""
    words = [
        "revenue growth quarterly earnings fiscal",
        "revenue growth quarterly earnings annual",
        "revenue growth quarterly profit fiscal",
        "revenue growth quarterly profit annual",
        "revenue growth monthly earnings fiscal",
        "revenue growth monthly earnings annual",
        "revenue growth monthly profit fiscal",
        "revenue growth monthly profit annual",
        "operating expenses quarterly reduction plan",
        "operating expenses quarterly reduction budget",
        "operating expenses annual reduction plan",
        "operating expenses annual reduction budget",
        "capital expenditure quarterly investment plan",
        "capital expenditure quarterly investment budget",
        "capital expenditure annual investment plan",
        "capital expenditure annual investment budget",
        "debt restructuring quarterly payment schedule",
        "debt restructuring quarterly payment timeline",
        "debt restructuring annual payment schedule",
        "debt restructuring annual payment timeline",
    ]
    items = [
        {"entry_id": f"e{i}", "section": "Finance", "importance": "high",
         "summary": w, "source": "curation"}
        for i, w in enumerate(words)
    ]
    result = consolidate_items(items, similarity_threshold=0.3)
    assert len(result) == len(items), (
        "Regression guard should roll back when >30% of items would be lost"
    )


def test_consolidate_regression_guard_rollback_on_high_importance_drop():
    """If consolidation would lose >20% of critical/high items, roll back."""
    items = []
    for i in range(8):
        items.append({
            "entry_id": f"e{i}", "section": "A", "importance": "critical",
            "summary": f"critical obligation payment deadline thirty days term {i}",
            "source": "curation",
        })
    for i in range(8, 18):
        items.append({
            "entry_id": f"e{i}", "section": "A", "importance": "medium",
            "summary": f"background context general information item {i}",
            "source": "curation",
        })
    result = consolidate_items(items, similarity_threshold=0.5)
    high_orig = sum(1 for i in items if i["importance"] in ("critical", "high"))
    high_result = sum(1 for i in result if i["importance"] in ("critical", "high"))
    assert high_result / high_orig >= 0.8, (
        "Regression guard should prevent losing >20% of critical/high items"
    )


def test_missing_entry_ids_marked_open_issue():
    entries = [
        Entry(id="e1", type="observation", content="X",
              source=EntrySource("d.pdf", "S", "ev"), confidence=0.9),
    ]
    bb = _bb_with_entries(entries)
    items = [
        {"entry_id": "e999", "importance": "high", "section": "Facts",
         "summary": "Claim with dangling entry reference"},
    ]
    packet = build_synthesis_packet(items, bb)
    assert len(packet) == 1
    assert packet[0]["open_issue_only"] is True, (
        "Items with all entry_ids missing must be open_issue_only"
    )
