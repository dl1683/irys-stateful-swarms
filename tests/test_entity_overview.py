import json

from src.swarm.blackboard import Blackboard
from src.swarm.entity_overview import (
    NameCatalogue, attach_relevant_overviews, automatic_overview_tasks, overview_inventory,
    defer_overview_dependent_tasks, entry_state, parse_discovered_names, prepare_overview_tasks,
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
            {"type": "entity_overview", "content": overview, "confidence": 0.8,
             "structured_overview": structured(
                 ("Müller AG paid CHF 2m on 12 June 2025.", ["e1"], "fact"),
                 ("Muller the signer may be unrelated to Müller AG.", ["e2"], "unresolved"),
                 ("Muler AG may be a separate typo candidate.", ["e3"], "derived"),
             )},
            {"type": "gap", "content": "Determine whether the individual signer is related to the company.", "confidence": 0.7, "supports_entries": ["e2"]},
        ]}),
        json.dumps({"findings": [{"type": "analysis", "content": "The supplied dossier identifies a payment and an identity ambiguity.", "confidence": 0.8}]}),
    ])

    output, matched_ids, _, state = run_entity_overview(catalogue, cards, entity_id, caller)
    assert matched_ids == ["e1", "e2", "e3"]
    assert output[0].type == "entity_overview"
    assert output[1].type == "gap" and output[1].supports_entries == ["e2"]
    assert "[e1] match=" in caller.prompts[0] and "[e2] match=" in caller.prompts[0]
    assert "close spelling candidate: muler" in caller.prompts[0]
    assert "e4" not in caller.prompts[0]
    assert output[0].supports_entries == ["e1", "e2", "e3"]
    assert state["entity_profiles"][0]["statements"][2]["kind"] == "derived"

    board = Blackboard(task_instruction="Assess the matter", entries=output)
    downstream = execute_workers_parallel([{
        "description": "Use the entity context.", "reads_from_blackboard": [output[0].id],
        "reads_from_documents": [], "expected_output_type": "analysis", "priority": "high",
    }], board, caller)
    assert len(downstream[0].entries) == 1
    assert "Müller AG paid CHF 2m" in caller.prompts[1]
    assert "supports: e1" in caller.prompts[1]


def test_name_parser_accepts_a_json_array_response():
    assert parse_discovered_names({"value": ["Crestmoor Trading AG"]}) == ["Crestmoor Trading AG"]


def test_entity_overview_drops_unsupported_follow_up_types():
    catalogue = NameCatalogue()
    [entity_id] = catalogue.add(["Crestmoor Trading AG"])
    caller = FakeCaller([json.dumps({"findings": [
        {"type": "entity_overview", "content": "A detailed overview with original-card support.",
         "structured_overview": structured(("Crestmoor Trading AG is the applicant.", ["e1"], "fact"))},
        {"type": "actionable_question", "content": "This must not become an observation."},
    ]})])
    output, _, _, _ = run_entity_overview(
        catalogue, [Entry(id="e1", content="Crestmoor Trading AG is the applicant.")],
        entity_id, caller,
    )
    assert [entry.type for entry in output] == ["entity_overview"]


def test_entity_overview_accepts_a_top_level_findings_array():
    catalogue = NameCatalogue()
    [entity_id] = catalogue.add(["Crestmoor Trading AG"])
    caller = FakeCaller([json.dumps([{
        "type": "entity_overview",
        "structured_overview": structured(("Crestmoor Trading AG is named in the card.", ["e1"], "fact")),
    }])])
    output, _, _, _ = run_entity_overview(
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


def test_automatic_and_requested_overviews_deduplicate_and_attach_relevant_context():
    state = {
        "variants": {"crestmoor": ["Crestmoor Trading AG"]},
    }
    entries = [
        Entry(id=f"e{i}", content=f"Crestmoor Trading AG fact {i}.", source=EntrySource(f"d{i}", "s"))
        for i in range(1, 7)
    ] + [Entry(id="overview", type="entity_overview", content="Structured context")]
    inventory = overview_inventory(state, entries)
    automatic = automatic_overview_tasks(inventory, state, iteration=2)
    requested = prepare_overview_tasks([{
        "expected_output_type": "entity_overview", "entity_overview_id": "crestmoor",
    }], inventory, state, iteration=2)
    assert len(automatic) == 1
    assert requested == []

    state["overviews"] = {"crestmoor": {"entry_id": "overview", "input_ids": ["e1", "e2", "e3"]}}
    tasks = attach_relevant_overviews([{
        "expected_output_type": "analysis", "reads_from_blackboard": ["e1", "e2"],
    }], entries, state)
    assert tasks[0]["reads_from_blackboard"] == ["e1", "e2", "overview"]
    assert tasks[0]["entity_overview_attachments"][0]["overlap"] == 2


def test_pending_overview_defers_only_dependent_substantive_work():
    state = {"pending": ["crestmoor"]}
    inventory = [{"entity_id": "crestmoor", "matched_card_ids": ["e1", "e2"]}]
    tasks = defer_overview_dependent_tasks([
        {"expected_output_type": "analysis", "reads_from_blackboard": ["e1", "e2"], "description": "Compare entities."},
        {"expected_output_type": "observation", "reads_from_blackboard": ["e1", "e2"], "description": "Read facts."},
        {"expected_output_type": "analysis", "reads_from_blackboard": ["e1"], "description": "Unrelated analysis."},
    ], inventory, state, iteration=3)
    assert [task["description"] for task in tasks] == ["Read facts.", "Unrelated analysis."]
    assert state["deferred"][0]["entity_id"] == "crestmoor"


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


def test_orchestrator_can_request_a_supported_low_frequency_overview():
    state = {}
    inventory = [{
        "entity_id": "crestmoor trading", "variants": ["Crestmoor Trading AG"],
        "matched_card_ids": ["e1"], "suggestion": "",
    }]
    accepted = prepare_overview_tasks([{
        "expected_output_type": "entity_overview", "entity_overview_id": "crestmoor trading",
    }], inventory, state, iteration=1)
    assert accepted[0]["requested_overview"] is True
    assert state["pending"] == ["crestmoor trading"]


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
            "structured_overview": structured(("Crestmoor Trading AG is the buyer in each supplied card.", ["e1", "e2", "e3"], "fact")),
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
        "structured_overview": structured(("Crestmoor Trading AG is the applicant.", ["e1"], "fact")),
    }]})])
    output = execute_workers_parallel([{
        "expected_output_type": "entity_overview", "entity_overview_id": "crestmoor trading",
        "entity_variants": ["Crestmoor Trading AG"], "description": "Create overview.",
    }], board, caller)[0]
    assert output.entries[0].type == "entity_overview"
    assert output.task["entity_input_ids"] == ["e1"]
    assert output.entries[0].supports_entries == ["e1"]
    assert output.task["entity_structured_overview"]["entity_profiles"][0]["statements"][0]["supports_entries"] == ["e1"]


def test_entity_overview_keeps_valid_statements_and_rejects_bad_references():
    catalogue = NameCatalogue()
    [entity_id] = catalogue.add(["Crestmoor Trading AG"])
    cards = [
        Entry(id="e1", content="Crestmoor Trading AG is the applicant."),
        Entry(id="e2", content="Old overview", type="entity_overview"),
        Entry(id="e3", content="Inactive source", status="inactive"),
    ]
    caller = FakeCaller([json.dumps({"findings": [{
        "type": "entity_overview", "content": "ignored once structured",
        "structured_overview": structured(
            ("The company is the applicant.", ["e1"], "fact"),
            ("This must be discarded.", ["e2", "e3", "missing"], "fact"),
        ),
    }]})])

    output, _, _, state = run_entity_overview(catalogue, cards, entity_id, caller)

    assert output[0].supports_entries == ["e1"]
    assert "discarded" not in output[0].content
    assert len(state["entity_profiles"][0]["statements"]) == 1


def test_refresh_uses_changed_source_state_without_derived_trigger_and_keeps_old_state_in_prompt():
    original = Entry(id="e1", content="Crestmoor Trading AG address is Zürich.", source=EntrySource("kyc", "s"))
    state = {"variants": {"crestmoor trading": ["Crestmoor Trading AG"]}, "overviews": {
        "crestmoor trading": {
            "entry_id": "old", "input_ids": ["e1"], "source_states": {"e1": entry_state(original)},
            "last_success_iteration": 1, "structured": structured(("Old address statement.", ["e1"], "fact")),
        },
    }}
    changed = Entry(id="e1", content="Crestmoor Trading AG address is Geneva.", source=EntrySource("kyc", "s"))
    derived = Entry(id="d1", content="Crestmoor Trading AG analysis changed.", type="analysis")
    [item] = overview_inventory(state, [changed, derived], iteration=3)
    assert item["suggestion"] == "refresh"
    assert item["changed_source_ids"] == ["e1"]

    catalogue = NameCatalogue({"crestmoor trading": {"Crestmoor Trading AG"}})
    caller = FakeCaller([json.dumps({"findings": [{
        "type": "entity_overview", "structured_overview": structured(("Address is Geneva.", ["e1"], "fact")),
    }]})])
    run_entity_overview(catalogue, [changed, derived], "crestmoor trading", caller,
                        iteration=3, previous_state=state["overviews"]["crestmoor trading"]["structured"])
    assert "REFRESH: Previous structured state follows" in caller.prompts[0]
    assert "Old address statement" in caller.prompts[0]
