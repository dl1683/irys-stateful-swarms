"""Bounded pointer maintenance without a provider."""

import json

from src.swarm import _after_document_round, _run_due_deltas, _sync_entity_state
from src.swarm.blackboard import Blackboard
from src.swarm.entity_overview import entry_state, render_pointer_card, select_delta_tasks, overview_inventory
from src.swarm.models import DocumentStatus, Entry, EntrySource, ModelResult, WorkerOutput


class Caller:
    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    def complete(self, prompt, **_kwargs):
        self.prompts.append(prompt)
        return ModelResult(json.dumps(self.payload), 12, 8, 20, "fixture-delta", 1,
                           provider_attempts=2)


def board_with_card(tmp_path, *, new_count=4):
    originals = [Entry(id=f"src{i}", content=f"Meridian AG original fact {i}.",
                       source=EntrySource("matter.txt", "s")) for i in range(1, 7 + new_count)]
    card = {"id": "ov1", "name": "Meridian AG", "identity_source_ids": ["src1", "src2"],
            "candidate_source_ids": [],
            "facts": [{"text": "Meridian AG original fact 1.", "source_ids": ["src1"]}],
            "identity_clues": []}
    overview = Entry(id="ov1", type="entity_overview", content=render_pointer_card(card),
                     supports_entries=["src1", "src2"])
    board = Blackboard(task_instruction="Review Meridian AG", entries=originals + [overview],
                       documents=[DocumentStatus(id="d1", name="matter.txt", text="source text")],
                       iteration=3, output_dir=str(tmp_path))
    board.entity_overview_state = {"variants": {"meridian": ["Meridian AG"]},
        "overviews": {"meridian": {"entry_id": "ov1", "card_ids": ["ov1"],
            "input_ids": [f"src{i}" for i in range(1, 7)], "cards": [card],
            "source_states": {f"src{i}": entry_state(originals[i-1]) for i in range(1, 7)}}}}
    return board


def test_third_document_round_runs_one_delta_and_resume_does_not_repeat(tmp_path):
    board = board_with_card(tmp_path)
    caller = Caller({"operations": [{"op": "add", "text": "Meridian AG new fact 7.",
                                      "source_ids": ["src7"]}]})
    analysis = WorkerOutput([], 0, 0, 0, "", "w", {}, [])
    reading = WorkerOutput([], 0, 0, 0, "", "w", {}, [("matter.txt", "s")])
    _after_document_round(board, [analysis], caller)
    assert board.entity_overview_state.get("document_rounds", 0) == 0
    _after_document_round(board, [reading], caller)
    _after_document_round(board, [reading], caller)
    assert not caller.prompts
    _after_document_round(board, [reading], caller)
    assert len(caller.prompts) == 1
    state = board.entity_overview_state
    card = state["overviews"]["meridian"]["cards"][0]
    assert card["facts"][-1]["source_ids"] == ["src7"]
    assert "src7" in card["identity_source_ids"]
    assert state["document_rounds"] == 3
    assert state["jobs"][-1]["provider_attempts"] == 2
    assert state["delta_backlog"] == []
    checkpoint = board.save_checkpoint({}, [], loop_ended=False)
    restored, *_ = Blackboard.load_checkpoint(checkpoint)
    _run_due_deltas(restored, caller)
    assert len(caller.prompts) == 1


def test_broken_reference_hidden_and_unsupported_patch_rejected(tmp_path):
    board = board_with_card(tmp_path, new_count=1)
    board.entries[0].status = "inactive"
    board.entries[-2].content = "Meridian AG original fact 1."  # candidate replacement only
    _sync_entity_state(board)
    card = board.entity_overview_state["overviews"]["meridian"]["cards"][0]
    assert card["facts"] == []
    assert "original fact 1" not in board.entries[-1].content
    error = board.entity_overview_state["reference_errors"][0]
    assert error["affected_card_ids"] == ["src1"]
    assert error["candidate_id"] == "src7"
    caller = Caller({"operations": [{"op": "add", "text": "Unsupported", "source_ids": ["made-up"]}]})
    _run_due_deltas(board, caller, final=True)
    assert card["facts"] == []
    assert board.entity_overview_state["delta_errors"]
    assert board.entity_overview_state["jobs"][-1]["failed"]
    assert board.entries[-1].status == "active"
    _run_due_deltas(board, caller, final=True)
    assert len(caller.prompts) == 1


def test_delta_selection_caps_wave_at_three_and_keeps_backlog(tmp_path):
    board = board_with_card(tmp_path)
    state = board.entity_overview_state
    for n in range(2, 5):
        group = f"meridian{n}"
        state["variants"][group] = [f"Meridian{n} AG"]
        base = [Entry(id=f"{group}-{i}", content=f"Meridian{n} AG fact {i}.",
                      source=EntrySource("matter.txt", "s")) for i in range(10)]
        board.entries.extend(base)
        card = {"id": f"ov{n}", "name": f"Meridian{n} AG",
                "identity_source_ids": [base[0].id], "candidate_source_ids": [],
                "facts": [], "identity_clues": []}
        board.entries.append(Entry(id=card["id"], type="entity_overview",
                                   content=render_pointer_card(card)))
        state["overviews"][group] = {"entry_id": card["id"], "card_ids": [card["id"]],
            "input_ids": [entry.id for entry in base[:6]], "cards": [card],
            "source_states": {entry.id: entry_state(entry) for entry in base[:6]}}
    inventory = overview_inventory(state, board.entries, 3, {"matter.txt"})
    tasks = select_delta_tasks(state, inventory, board.entries)
    assert len(tasks) == 3
    assert len(state["delta_backlog"]) == 1
    caller = Caller({"operations": []})
    _run_due_deltas(board, caller, final=True)
    assert len(caller.prompts) == 3
    assert len(state["delta_backlog"]) == 1


def test_worker_requests_only_attached_cards_and_routes_identity_separately(tmp_path):
    from src.swarm import _commit_entity_worker_outputs
    from src.swarm.worker_dispatch import execute_workers_parallel

    board = board_with_card(tmp_path, new_count=0)
    caller = Caller({"findings": [], "overview_review_requests": [
        {"overview_id": "ov1", "reason": "material address correction"},
        {"overview_id": "ov1", "reason": "identity may be duplicate"},
        {"overview_id": "not-attached", "reason": "material correction"},
    ]})
    task = {"description": "Review the original evidence", "expected_output_type": "analysis",
            "reads_from_blackboard": ["src1", "ov1"], "reads_from_documents": [],
            "entity_overview_attachments": [{"overview_id": "ov1", "entity_id": "meridian"}]}
    output = execute_workers_parallel([task], board, caller)
    _commit_entity_worker_outputs(board, output, caller, 3)
    state = board.entity_overview_state
    assert "overview_review_requests" in caller.prompts[0]
    assert state["overview_review_requests"] == [
        {"overview_id": "ov1", "reason": "material address correction"}]
    assert state["identity_review_requests"] == [
        {"overview_id": "ov1", "reason": "identity may be duplicate"}]
    assert "not-attached" not in str(state)

    caller.prompts.clear()
    execute_workers_parallel([{"description": "Read without overview", "expected_output_type": "analysis",
                               "reads_from_blackboard": [], "reads_from_documents": []}], board, caller)
    assert "overview_review_requests" not in caller.prompts[0]


def test_only_registered_direct_readings_advance_card_threshold(tmp_path):
    board = board_with_card(tmp_path, new_count=0)
    state = board.entity_overview_state
    board.entries.extend([
        Entry(id=f"derived{i}", type="analysis", content="Meridian AG derived view.",
              source=EntrySource("matter.txt", "s")) for i in range(4)
    ])
    board.entries.extend([
        Entry(id=f"outside{i}", content="Meridian AG unknown file claim.",
              source=EntrySource("unregistered.txt", "s")) for i in range(4)
    ])
    [item] = overview_inventory(state, board.entries, 3, {"matter.txt"})
    assert item["new_card_count"] == 0
    assert not select_delta_tasks(state, [item], board.entries)


def test_budget_limited_delta_is_backlogged_without_call(tmp_path):
    board = board_with_card(tmp_path)
    board.token_budget = board.total_tokens_used + 10
    caller = Caller({"operations": []})
    _run_due_deltas(board, caller, final=True)
    assert not caller.prompts
    assert board.entity_overview_state["budget_limited"] == ["ov1"]
    assert board.entity_overview_state["delta_backlog"] == ["ov1"]


def test_zero_token_stage_does_not_reuse_prior_model_usage():
    from src.swarm.worker_dispatch import set_last_call_usage

    board = Blackboard(task_instruction="No provider stage")
    set_last_call_usage({"previous": {"input": 12, "output": 8, "total": 20, "calls": 1}})
    board.add_tokens_from_last_call(0)
    assert (board.total_tokens_used, board.tokens_input, board.tokens_output) == (0, 0, 0)
    assert board.cost_by_model == {}
    set_last_call_usage(None)


def test_shared_retrieval_groups_schedule_one_card_delta(tmp_path):
    board = board_with_card(tmp_path)
    state = board.entity_overview_state
    state["variants"]["meridian alias"] = ["Meridian AG"]
    state["overviews"]["meridian alias"] = json.loads(json.dumps(state["overviews"]["meridian"]))
    _sync_entity_state(board)
    inventory = overview_inventory(state, board.entries, 3, {"matter.txt"})
    tasks = select_delta_tasks(state, inventory, board.entries)
    assert len(tasks) == 1
    assert (state["overviews"]["meridian"]["cards"][0]
            is state["overviews"]["meridian alias"]["cards"][0])


def test_failed_initial_validation_keeps_model_usage(tmp_path):
    from src.swarm import _run_initial_overviews

    board = board_with_card(tmp_path, new_count=0)
    board.entries = [entry for entry in board.entries if entry.type != "entity_overview"]
    board.entity_overview_state = {"variants": {"meridian": ["Meridian AG"]}}
    smart = Caller({"cards": []})
    _run_initial_overviews(board, Caller({}), smart)
    state = board.entity_overview_state
    assert state["failed"] == ["meridian"]
    assert state["jobs"][0]["model"] == "fixture-delta"
    assert state["jobs"][0]["input_tokens"] == 12
    assert state["jobs"][0]["output_tokens"] == 8
    assert state["jobs"][0]["failed"]
    assert state["usage"]["initial_tokens"] == 20


def test_repair_hint_finds_exact_identifier_in_existing_original(tmp_path):
    board = board_with_card(tmp_path, new_count=0)
    original, lost = board.entries[0], board.entries[1]
    original.content = "Buyer Meridian AG, register CHE-198.765.432."
    lost.content = "Meridian AG Swiss Commercial Register No.: CHE-198.765.432"
    card = board.entity_overview_state["overviews"]["meridian"]["cards"][0]
    card["facts"] = [{"text": "Swiss Commercial Register No. for Meridian AG: CHE-198.765.432",
                      "source_ids": [lost.id]}]
    board.entity_overview_state["overviews"]["meridian"]["source_states"] = {
        original.id: entry_state(original), lost.id: entry_state(lost)}
    lost.status = "inactive"
    _sync_entity_state(board)
    error = board.entity_overview_state["reference_errors"][0]
    assert error["candidate_id"] == original.id
    assert card["facts"] == []
