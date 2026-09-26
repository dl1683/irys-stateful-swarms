import json

from src.swarm.blackboard import Blackboard
from src.swarm.entity_overview import (
    NameCatalogue, attach_relevant_overviews, automatic_overview_tasks, overview_inventory,
    defer_overview_dependent_tasks, entry_state, parse_discovered_names, prepare_overview_tasks,
    resolve_overview_request, run_entity_overview,
    discover_pending_names,
)
from src.swarm.models import Entry, EntrySource, ModelResult, WorkerRecord
from src.swarm.orchestrator import run_orchestrator
from src.swarm.worker_dispatch import execute_workers_parallel, parse_worker_output


class FakeCaller:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def complete(self, prompt, *, max_tokens=8192, temperature=0.05, json_mode=True):
        self.prompts.append(prompt)
        return ModelResult(next(self.responses), 10, 5, 15, "fake", 1)


def structured(*statements):
    """Compact valid overview payloads for the local integration checks."""
    return {"entity_profiles": [{"label": "Candidate profile", "statements": [
        {"text": text, "supports_entries": refs, "kind": kind}
        for text, refs, kind in statements
    ]}], "relationships_and_distinctions": [], "unresolved_or_conflicting_evidence": []}


def test_discovery_links_explicit_alias_once_and_preserves_existing_overview():
    card = Entry(id="e126", content="Full Legal Name: Crestmoor Trading AG; Short Name / Trade Name: Crestmoor")
    alias = {"name": "Crestmoor Trading AG", "short_form": "Crestmoor", "source_card_id": "e126"}
    payload = {"names": ["Crestmoor Trading AG", "Crestmoor"], "aliases": [alias]}
    for existing in ("crestmoor trading", "crestmoor"):
        record = {"entry_id": "e642", "input_ids": ["old"]}
        state = {"variants": {"crestmoor trading": ["Crestmoor Trading AG"], "crestmoor": ["Crestmoor"]},
                 "overviews": {existing: record}}
        caller = FakeCaller([json.dumps(payload), json.dumps({"names": ["Crestmoor"]})])
        discover_pending_names(state, [card], caller)
        assert len(caller.prompts) == 1
        assert state["variants"] == {existing: ["Crestmoor", "Crestmoor Trading AG"]}
        assert state["overviews"] == {existing: record}
        discover_pending_names(state, [card, Entry(id="later", content="Crestmoor is the applicant.")], caller)
        assert len(state["variants"]) == 1
        [item] = overview_inventory(state, [card])
        assert item["overview_id"] == "e642"
        assert item["matched_card_ids"] == ["e126"]


def test_discovery_does_not_merge_unsupported_or_ambiguous_aliases():
    card = Entry(id="e1", content="Crestmoor Trading AG; Crestmoor Logistics AG; Crestmoor")
    names = ["Crestmoor Trading AG", "Crestmoor Logistics AG", "Crestmoor"]
    first = {"name": names[0], "short_form": "Crestmoor", "source_card_id": "e1"}
    for aliases in ([], [dict(first, source_card_id="missing")],
                    [dict(first, short_form="Invented")],
                    [first, dict(first, name=names[1])]):
        state = {}
        caller = FakeCaller([json.dumps({"names": names, "aliases": aliases})])
        discover_pending_names(state, [card], caller)
        assert len(state["variants"]) == 3

    # A later conflicting association cannot join the two full names through a shared alias.
    state = {}
    caller = FakeCaller([json.dumps({"names": names, "aliases": [first]}),
                         json.dumps({"names": names, "aliases": [dict(first, name=names[1], source_card_id="e2")]})])
    discover_pending_names(state, [card], caller)
    discover_pending_names(state, [Entry(id="e2", content=card.content)], caller)
    assert len(state["variants"]) == 2


def test_name_parser_accepts_a_json_array_response():
    assert parse_discovered_names({"value": ["Crestmoor Trading AG"]}) == ["Crestmoor Trading AG"]


def test_generic_worker_parser_accepts_a_top_level_findings_array():
    entries = parse_worker_output({"value": [{
        "type": "analysis", "content": "The supplied overview identifies a screening follow-up.",
    }]}, 1, "w1", "Assess the overview.")
    assert [entry.type for entry in entries] == ["analysis"]


def test_overview_inventory_suggests_initial_then_refresh_without_duplicates():
    state = {"variants": {"crestmoor trading": ["Crestmoor Trading AG"]}}
    entries = [
        Entry(id=f"e{i}", content=f"Crestmoor Trading AG fact {i}.", source=EntrySource(f"d{i}", "s"))
        for i in range(1, 7)
    ]
    [initial] = overview_inventory(state, entries)
    assert initial["suggestion"] == "initial"

    state["overviews"] = {"crestmoor trading": {"entry_id": "e7", "input_ids": [f"e{i}" for i in range(1, 7)]}}
    entries.extend([
        Entry(id=f"e{i}", content=f"Crestmoor Trading AG update {i}.", source=EntrySource(f"d{i}", "s"))
        for i in range(8, 12)
    ])
    [refresh] = overview_inventory(state, entries)
    assert refresh["suggestion"] == "refresh"

    state["pending"] = ["crestmoor trading"]
    assert overview_inventory(state, entries)[0]["suggestion"] == ""


def test_automatic_coverage_drains_all_qualifying_groups_in_bounded_batches():
    state = {"variants": {f"entity {i}": [f"Entity {i}"] for i in range(50)}}
    entries = [
        Entry(id=f"e{entity}_{card}", content=f"Entity {entity} source fact {card}.",
              source=EntrySource(f"d{entity}_{card}", "s"))
        for entity in range(50) for card in range(6)
    ]
    inventory = overview_inventory(state, entries)
    served = []
    while True:
        tasks = automatic_overview_tasks(inventory, state, iteration=1)
        if not tasks:
            break
        assert len(tasks) <= 3
        for task in tasks:
            entity_id = task["entity_overview_id"]
            for group_id in task["entity_overview_group_ids"]:
                served.append(group_id)
                state["pending"].remove(group_id)
                state["initial_queue"].remove(group_id)
                item = next(item for item in inventory if item["entity_id"] == group_id)
                state.setdefault("overviews", {})[group_id] = {
                    "entry_id": f"overview-{entity_id}", "input_ids": item["source_card_ids"],
                }
        inventory = overview_inventory(state, entries)
    assert len(served) == 50
    assert len(set(served)) == 50


def test_exactly_overlapping_groups_share_a_job_without_merging_variants():
    state = {"variants": {
        "crestmoor": ["Crestmoor Trading AG"],
        "crestmoor candidate": ["Crestmoor Trade & Supply GmbH"],
    }}
    entries = [
        Entry(id=f"e{i}", content="Crestmoor Trading AG and Crestmoor Trade & Supply GmbH screening context.",
              source=EntrySource(f"d{i}", "s"))
        for i in range(6)
    ]
    inventory = overview_inventory(state, entries)
    tasks = automatic_overview_tasks(inventory, state, iteration=1)
    assert len(tasks) == 1
    assert tasks[0]["entity_overview_group_ids"] == ["crestmoor", "crestmoor candidate"]
    assert tasks[0]["entity_variants"] == ["Crestmoor Trade & Supply GmbH", "Crestmoor Trading AG"]


def test_snapshot_writes_a_compact_entity_overview_report(tmp_path):
    board = Blackboard(task_instruction="Assess sanctions", output_dir=str(tmp_path))
    board.entity_overview_state = {
        "variants": {"crestmoor trading": ["Crestmoor Trading AG"]},
        "discovered_card_ids": ["e1", "e2", "e3"],
        "overviews": {"crestmoor trading": {"entry_id": "e4", "input_ids": ["e1", "e2", "e3"]}},
        "usage": {"overview_calls": 1, "overview_tokens": 123},
        "recipients": [{"worker_id": "w1", "overview_ids": ["e4"]}],
        "failed": [],
        "rejected_requests": [{"entity_id": "", "reason": "missing or unavailable overview suggestion"}],
    }

    board.save_snapshot("test")

    report = json.loads((tmp_path / "swarm" / "entity_overview_report.json").read_text())
    assert report["catalogue_groups"] == 1
    assert report["processed_card_count"] == 3
    assert report["usage"]["overview_tokens"] == 123
    assert report["recipients"][0]["overview_ids"] == ["e4"]
    assert report["referencing_recipient_count"] == 0


def test_overview_request_resolves_current_old_and_unambiguous_name_but_rejects_ambiguity():
    inventory = [
        {"entity_id": "petrov", "variants": ["Alexei Petrov", "A. Petrov"]},
        {"entity_id": "other petrov", "variants": ["A. Petrov"]},
    ]
    state = {"overviews": {"petrov": {
        "entry_id": "overview-current", "superseded_entry_ids": ["overview-old"],
    }}}

    assert resolve_overview_request("petrov", inventory, state) == ("petrov", "")
    assert resolve_overview_request("overview-current", inventory, state) == ("petrov", "")
    assert resolve_overview_request("overview-old", inventory, state) == ("petrov", "")
    assert resolve_overview_request("Alexei Petrov", inventory, state) == ("petrov", "")
    assert resolve_overview_request("A. Petrov", inventory, state) == ("", "ambiguous overview identifier")
