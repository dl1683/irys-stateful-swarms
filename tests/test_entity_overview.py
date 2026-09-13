import json

from src.swarm.blackboard import Blackboard
from src.swarm.entity_overview import (
    NameCatalogue, overview_inventory, parse_discovered_names, prepare_overview_tasks,
    run_entity_overview,
    discover_pending_names,
)
from src.swarm.models import Entry, EntrySource, ModelResult
from src.swarm.orchestrator import run_orchestrator
from src.swarm.worker_dispatch import execute_workers_parallel, parse_worker_output


class FakeCaller:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def complete(self, prompt, *, max_tokens=8192, temperature=0.05, json_mode=True):
        self.prompts.append(prompt)
        return ModelResult(next(self.responses), 10, 5, 15, "fake", 1)


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


def test_entity_overview_slice_preserves_candidate_matches_and_full_context():
    names = parse_discovered_names({"names": ["Müller AG", "Muller AG", ""]})
    catalogue = NameCatalogue()
    [entity_id] = catalogue.add(names)
    cards = [
        Entry(id="e1", content="Müller AG paid CHF 2m on 12 June 2025.", source=EntrySource("a", "s")),
        Entry(id="e2", content="Muller signed the amendment; the person may be unrelated to Müller AG.", source=EntrySource("b", "s")),
        Entry(id="e3", content="Muler AG appears in a separate contract and may be a typo.", source=EntrySource("c", "s")),
        Entry(id="e4", content="Mullerian analysis is unrelated to the named entity.", source=EntrySource("d", "s")),
    ]
    overview = "Müller AG is a company. " + "detail " * 80 + "[e1] supports the payment."
    caller = FakeCaller([
        json.dumps({"findings": [
            {"type": "entity_overview", "content": overview, "confidence": 0.8, "supports_entries": ["e1", "e2"]},
            {"type": "gap", "content": "Determine whether the individual signer is related to the company.", "confidence": 0.7, "supports_entries": ["e2"]},
        ]}),
        json.dumps({"findings": [{"type": "analysis", "content": "The supplied dossier identifies a payment and an identity ambiguity.", "confidence": 0.8}]}),
    ])

    output, matched_ids, _ = run_entity_overview(catalogue, cards, entity_id, caller)
    assert matched_ids == ["e1", "e2", "e3"]
    assert output[0].type == "entity_overview"
    assert output[1].type == "gap" and output[1].supports_entries == ["e2"]
    assert "[e1] match=" in caller.prompts[0] and "[e2] match=" in caller.prompts[0]
    assert "close spelling candidate: muler" in caller.prompts[0]
    assert "e4" not in caller.prompts[0]

    board = Blackboard(task_instruction="Assess the matter", entries=output)
    downstream = execute_workers_parallel([{
        "description": "Use the entity context.", "reads_from_blackboard": [output[0].id],
        "reads_from_documents": [], "expected_output_type": "analysis", "priority": "high",
    }], board, caller)
    assert len(downstream[0].entries) == 1
    assert overview in caller.prompts[1] and len(overview) > 300


def test_name_parser_accepts_a_json_array_response():
    assert parse_discovered_names({"value": ["Crestmoor Trading AG"]}) == ["Crestmoor Trading AG"]


def test_entity_overview_drops_unsupported_follow_up_types():
    catalogue = NameCatalogue()
    [entity_id] = catalogue.add(["Crestmoor Trading AG"])
    caller = FakeCaller([json.dumps({"findings": [
        {"type": "entity_overview", "content": "A detailed overview with original-card support.", "supports_entries": ["e1"]},
        {"type": "actionable_question", "content": "This must not become an observation."},
    ]})])
    output, _, _ = run_entity_overview(
        catalogue, [Entry(id="e1", content="Crestmoor Trading AG is the applicant.")],
        entity_id, caller,
    )
    assert [entry.type for entry in output] == ["entity_overview"]


def test_entity_overview_accepts_a_top_level_findings_array():
    catalogue = NameCatalogue()
    [entity_id] = catalogue.add(["Crestmoor Trading AG"])
    caller = FakeCaller([json.dumps([{
        "type": "entity_overview", "content": "A detailed overview returned as an array.",
    }])])
    output, _, _ = run_entity_overview(
        catalogue, [Entry(id="e1", content="Crestmoor Trading AG is the applicant.")],
        entity_id, caller,
    )
    assert [entry.type for entry in output] == ["entity_overview"]


def test_generic_worker_parser_accepts_a_top_level_findings_array():
    entries = parse_worker_output({"value": [{
        "type": "analysis", "content": "The supplied overview identifies a screening follow-up.",
    }]}, 1, "w1", "Assess the overview.")
    assert [entry.type for entry in entries] == ["analysis"]


def test_overview_inventory_suggests_initial_then_refresh_without_duplicates():
    state = {"variants": {"crestmoor trading": ["Crestmoor Trading AG"]}}
    entries = [
        Entry(id=f"e{i}", content=f"Crestmoor Trading AG fact {i}.")
        for i in range(1, 4)
    ]
    [initial] = overview_inventory(state, entries)
    assert initial["suggestion"] == "initial"

    state["overviews"] = {"crestmoor trading": {"entry_id": "e4", "input_ids": ["e1", "e2", "e3"]}}
    entries.extend([
        Entry(id="e5", content="Crestmoor Trading AG update one."),
        Entry(id="e6", content="Crestmoor Trading AG update two."),
    ])
    [refresh] = overview_inventory(state, entries)
    assert refresh["suggestion"] == "refresh"

    state["pending"] = ["crestmoor trading"]
    assert overview_inventory(state, entries)[0]["suggestion"] == ""


def test_overview_tasks_reject_missing_or_unscheduled_entity_ids():
    state = {}
    inventory = [{
        "entity_id": "crestmoor trading", "variants": ["Crestmoor Trading AG"],
        "suggestion": "initial",
    }]
    tasks = [
        {"expected_output_type": "entity_overview", "entity_overview_id": ""},
        {"expected_output_type": "entity_overview", "entity_overview_id": "crestmoor trading"},
        {"expected_output_type": "analysis", "description": "Continue analysis."},
    ]

    accepted = prepare_overview_tasks(tasks, inventory, state, iteration=4)

    assert [task["expected_output_type"] for task in accepted] == ["entity_overview", "analysis"]
    assert accepted[0]["entity_variants"] == ["Crestmoor Trading AG"]
    assert state["pending"] == ["crestmoor trading"]
    assert state["rejected_requests"] == [{
        "iteration": 4, "entity_id": "",
        "reason": "missing or unavailable overview suggestion",
    }]


def test_orchestrated_overview_reaches_a_downstream_worker_without_live_calls():
    board = Blackboard(task_instruction="Assess sanctions exposure", entries=[
        Entry(id=f"e{i}", content=f"Crestmoor Trading AG fact {i}.")
        for i in range(1, 4)
    ])
    inventory = [{
        "entity_id": "crestmoor trading", "variants": ["Crestmoor Trading AG"],
        "matched_card_ids": ["e1", "e2", "e3"], "overview_id": "",
        "new_card_count": 3, "suggestion": "initial",
    }]
    overview_text = "Crestmoor Trading AG is the buyer in each supplied card."
    caller = FakeCaller([
        json.dumps({"workers": [{
            "description": "Build the entity overview.",
            "expected_output_type": "entity_overview",
            "entity_overview_id": "crestmoor trading",
            "reads_from_blackboard": [], "reads_from_documents": [],
            "priority": "high",
        }]}),
        json.dumps({"findings": [{
            "type": "entity_overview", "content": overview_text,
            "supports_entries": ["e1", "e2", "e3"],
        }]}),
        json.dumps({"findings": [{
            "type": "analysis", "content": "The buyer should be screened.",
            "supports_entries": [],
        }]}),
    ])

    orchestration, _ = run_orchestrator(board, caller, entity_overviews=inventory)
    overview_task = prepare_overview_tasks(
        orchestration["workers"], inventory, board.entity_overview_state, iteration=1,
    )
    overview_output = execute_workers_parallel(overview_task, board, caller)[0]
    board.add_entries_batch(overview_output.entries)
    overview_id = overview_output.entries[0].id
    downstream = execute_workers_parallel([{
        "description": "Assess the overview.", "expected_output_type": "analysis",
        "reads_from_blackboard": [overview_id], "reads_from_documents": [],
        "priority": "high",
    }], board, caller)

    assert "crestmoor trading: overview=missing" in caller.prompts[0]
    assert overview_output.task["entity_input_ids"] == ["e1", "e2", "e3"]
    assert downstream[0].entries[0].type == "analysis"
    assert overview_text in caller.prompts[2]


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


def test_overview_worker_task_records_input_ids_and_failure_stays_visible():
    board = Blackboard(task_instruction="Assess sanctions", entries=[
        Entry(id="e1", content="Crestmoor Trading AG is the applicant."),
    ])
    caller = FakeCaller([json.dumps({"findings": [{
        "type": "entity_overview", "content": "Crestmoor Trading AG is the applicant [e1].",
        "supports_entries": ["e1"],
    }]})])
    output = execute_workers_parallel([{
        "expected_output_type": "entity_overview", "entity_overview_id": "crestmoor trading",
        "entity_variants": ["Crestmoor Trading AG"], "description": "Create overview.",
    }], board, caller)[0]
    assert output.entries[0].type == "entity_overview"
    assert output.task["entity_input_ids"] == ["e1"]
