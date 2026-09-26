"""Conservative identity review and reversible pointer behavior."""

import json
from copy import deepcopy

from src.swarm import _run_identity_reviews, _sync_entity_state
from src.swarm.blackboard import Blackboard
from src.swarm.entity_overview import (
    attach_relevant_overviews, entry_state, render_pointer_card, reverse_identity_merge,
    select_identity_reviews,
)
from src.swarm.models import DocumentStatus, Entry, EntrySource, ModelResult


class Caller:
    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.prompts = []

    def complete(self, prompt, **_kwargs):
        self.prompts.append(prompt)
        return ModelResult(json.dumps(self.payloads.pop(0)), 10, 5, 15, "fixture-smart", 1)


def board(tmp_path, *, names=("Alex Rowan", "Alex Rowan"),
          clues=(("birth_date", "1 Jan 1980"), ("address", "8 Oak Road")),
          second_clues=None, shared_card=False):
    second_clues = clues if second_clues is None else second_clues
    originals = []
    cards = []
    for n, values in enumerate((clues, second_clues)):
        ids = [f"s{n}{i}" for i in range(2)]
        for i, (kind, value) in enumerate(values):
            originals.append(Entry(id=ids[i], content=f"{names[n]} {kind}: {value}",
                                   source=EntrySource("matter.txt", "s")))
        cards.append({"id": f"ov{n}", "name": names[n], "identity_source_ids": ids,
                      "candidate_source_ids": [], "facts": [{"text": f"private fact {n}",
                      "source_ids": [ids[0]]}],
                      "identity_clues": [{"type": kind, "value": value, "source_ids": [ids[i]]}
                                         for i, (kind, value) in enumerate(values)]})
    if shared_card:
        cards[1]["identity_clues"][0]["source_ids"] = ["s00"]
    overviews = [Entry(id=c["id"], type="entity_overview", content=render_pointer_card(c),
                       supports_entries=c["identity_source_ids"][:]) for c in cards]
    result = Blackboard(task_instruction="Check identities", entries=originals + overviews,
                        documents=[DocumentStatus(id="d1", name="matter.txt", text="source")],
                        output_dir=str(tmp_path), iteration=4, token_budget=100000)
    result.entity_overview_state = {"variants": {"alex": ["Alex Rowan"]}, "overviews": {
        "alex": {"cards": cards, "card_ids": [c["id"] for c in cards], "entry_id": "ov0",
                 "input_ids": [e.id for e in originals],
                 "source_states": {e.id: entry_state(e) for e in originals}}}}
    return result


def test_proposal_requires_two_distinct_supported_clues_or_worker_flag(tmp_path):
    b = board(tmp_path, second_clues=(("birth_date", "2 Feb 1990"), ("address", "8 Oak Road")))
    assert select_identity_reviews(b.entity_overview_state, b.entries) == []
    assert not b.entity_overview_state["identity_review_backlog"]
    b.entity_overview_state["identity_review_requests"] = [
        {"overview_id": "ov0", "reason": "Identity ambiguity with the other Alex Rowan"}]
    assert len(select_identity_reviews(b.entity_overview_state, b.entries)) == 1


def test_two_clues_propose_once_and_shared_source_does_not(tmp_path):
    b = board(tmp_path)
    tasks = select_identity_reviews(b.entity_overview_state, b.entries)
    assert len(tasks) == 1
    assert tasks[0]["pair"] == ["ov0", "ov1"]
    assert b.entity_overview_state["overviews"]["alex"]["cards"][0]["possibly_same_as"][0]["status"] == "pending"
    shared = board(tmp_path, shared_card=True)
    assert select_identity_reviews(shared.entity_overview_state, shared.entries) == []


def test_unresolved_is_visible_and_not_repeated_without_new_evidence(tmp_path):
    b = board(tmp_path)
    smart = Caller({"outcome": "unresolved", "source_ids": [], "basis": "uncertain"})
    _run_identity_reviews(b, Caller(), smart)
    assert len(smart.prompts) == 1
    assert b.entity_overview_state["identity_reviews"]["ov0|ov1"]["outcome"] == "unresolved"
    _run_identity_reviews(b, Caller(), smart)
    assert len(smart.prompts) == 1
    assert all(e.status == "active" for e in b.entries[-2:])


def test_budget_limited_pairs_stay_in_backlog(tmp_path):
    b = board(tmp_path)
    b.token_budget = 1
    smart = Caller()
    _run_identity_reviews(b, Caller(), smart)
    assert not smart.prompts
    assert b.entity_overview_state["identity_review_backlog"] == [["ov0", "ov1"]]
    assert "ov0|ov1" not in b.entity_overview_state["identity_reviews"]


def test_shared_identifier_merges_without_copying_facts_and_can_reverse(tmp_path):
    clues = (("registration", "CHE-123.456.789"), ("address", "8 Oak Road"))
    b = board(tmp_path, clues=clues)
    before = [(e.id, e.content, e.status, e.source) for e in b.entries[:4]]
    smart = Caller({"outcome": "same", "basis": "shared_identifier",
                    "source_ids": ["s00", "s10"], "reason": "same registration"})
    _run_identity_reviews(b, Caller(), smart)
    cards = b.entity_overview_state["overviews"]["alex"]["cards"]
    assert cards[1]["redirect_to"] == "ov0"
    assert set(cards[0]["identity_source_ids"]) == {"s00", "s01", "s10", "s11"}
    assert cards[0]["facts"] == [{"text": "private fact 0", "source_ids": ["s00"]}]
    assert next(e for e in b.entries if e.id == "ov1").status == "inactive"
    task = {"expected_output_type": "analysis", "reads_from_blackboard": ["ov1", "s10"]}
    attach_relevant_overviews([task], b.entries, b.entity_overview_state)
    assert "ov0" in task["reads_from_blackboard"] and "ov1" not in task["reads_from_blackboard"]
    finding = next(e for e in b.entries if e.type == "analysis")
    assert finding.supports_entries == ["s00", "s10"]
    assert [(e.id, e.content, e.status, e.source) for e in b.entries[:4]] == before
    _sync_entity_state(b)
    assert set(cards[0]["identity_source_ids"]) == {"s00", "s01", "s10", "s11"}
    assert reverse_identity_merge(b.entity_overview_state, b.entries, "ov1")
    assert next(e for e in b.entries if e.id == "ov1").status == "active"
    assert finding.status == "inactive"


def test_conflicting_identifier_rejects_same(tmp_path):
    b = board(tmp_path, clues=(("registration", "CHE-123.456.789"), ("address", "8 Oak Road")),
              second_clues=(("registration", "CHE-987.654.321"), ("address", "8 Oak Road")))
    b.entity_overview_state["identity_review_requests"] = [{"overview_id": "ov0", "reason": "identity with ov1"}]
    smart = Caller({"outcome": "same", "basis": "shared_identifier", "source_ids": ["s00", "s10"]})
    _run_identity_reviews(b, Caller(), smart)
    assert b.entity_overview_state["jobs"][-1]["failed"]
    assert all(e.status == "active" for e in b.entries[-2:])


def test_confirmed_split_rebuilds_with_smart_caller(tmp_path):
    b = board(tmp_path, second_clues=(("birth_date", "2 Feb 1990"), ("address", "9 Elm Road")))
    b.entity_overview_state["identity_review_requests"] = [
        {"overview_id": "ov0", "reason": "Possible conflation of two identities in this card"}]
    smart = Caller(
        {"outcome": "split", "basis": "conflation", "split_overview_id": "ov0",
         "source_ids": ["s00", "s01"], "reason": "two people"},
        {"cards": [
            {"name": "Alex Rowan A", "identity_source_ids": ["s00"],
             "candidate_source_ids": [], "facts": [], "identity_clues": []},
            {"name": "Alex Rowan B", "identity_source_ids": ["s01"],
             "candidate_source_ids": [], "facts": [], "identity_clues": []},
        ]},
    )
    _run_identity_reviews(b, Caller(), smart)
    assert len(smart.prompts) == 2
    assert next(e for e in b.entries if e.id == "ov0").status == "inactive"
    cards = b.entity_overview_state["overviews"]["alex"]["cards"]
    assert len(cards) == 3
    assert all(c.get("split_from") == "ov0" for c in cards[1:])
    assert b.entity_overview_state["identity_history"][-1]["kind"] == "split"
    _sync_entity_state(b)
    assert all(c["identity_source_ids"] for c in cards[1:])
