import dataclasses

from src.swarm.blackboard import Blackboard
from src.swarm.models import Entry


def make_entry(entry_id: str, **kwargs) -> Entry:
    return Entry(id=entry_id, content=f"content for {entry_id}", **kwargs)


def test_empty_blackboard_has_entry_index():
    bb = Blackboard()
    assert hasattr(bb, "_entry_index")
    assert bb._entry_index == {}


def test_construction_with_entries_indexes_them():
    e1 = make_entry("e1")
    e2 = make_entry("e2")
    bb = Blackboard(entries=[e1, e2])
    assert bb.find_entry("e1") is e1
    assert bb.find_entry("e2") is e2


def test_find_entry_returns_none_for_unknown_id():
    bb = Blackboard(entries=[make_entry("e1")])
    assert bb.find_entry("does-not-exist") is None


def test_add_entry_keeps_index_consistent():
    bb = Blackboard()
    e1 = make_entry("e1")
    bb.add_entry(e1)
    assert bb.find_entry("e1") is e1


def test_add_entries_batch_keeps_index_consistent():
    bb = Blackboard()
    entries = [make_entry(f"e{i}") for i in range(1, 6)]
    bb.add_entries_batch(entries)
    for e in entries:
        assert bb.find_entry(e.id) is e


def test_staged_contradiction_entries_are_findable():
    bb = Blackboard()
    e1 = make_entry("e1", confidence=0.5)
    bb.add_entry(e1)

    e2 = make_entry("e2", confidence=0.5, contradicts_entries=["e1"])
    bb.add_entry(e2)

    contradiction_entries = [e for e in bb.entries if e.type == "contradiction"]
    assert len(contradiction_entries) == 1
    contradiction = contradiction_entries[0]

    assert bb.find_entry(contradiction.id) is contradiction


def test_ensure_unique_id_resolves_collisions_via_index():
    bb = Blackboard()
    e1 = make_entry("dup")
    bb.add_entry(e1)

    e2 = make_entry("dup")
    bb._ensure_unique_id(e2)

    assert e2.id != "dup"
    assert e2.id not in bb._entry_index or bb._entry_index.get(e2.id) is not e1


def test_index_consistent_after_many_sequential_adds():
    bb = Blackboard()
    entries = [make_entry(f"e{i}") for i in range(500)]
    for e in entries:
        bb.add_entry(e)

    for e in entries:
        assert bb.find_entry(e.id) is e
    assert len(bb._entry_index) == len(entries)


def test_entry_index_excluded_from_repr():
    bb = Blackboard(entries=[make_entry("e1")])
    assert "_entry_index" not in repr(bb)


def test_entry_index_excluded_from_equality():
    e1 = make_entry("e1")
    e2 = make_entry("e1")
    bb1 = Blackboard(entries=[e1])
    bb2 = Blackboard(entries=[e2])

    assert bb1 == bb2
    assert bb1._entry_index is not bb2._entry_index
