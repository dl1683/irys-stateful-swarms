import json

from src.swarm.blackboard import Blackboard
from src.swarm.entity_maintenance_store import EntityMaintenanceState
from src.swarm.entity_profiles import (
    ProfileRefreshSummary,
    project_entity_information_cards,
    refresh_entity_profiles,
)
from src.swarm.models import Entry, EntrySource


def direct_company(entry_id: str, name: str, registration: str) -> Entry:
    evidence = f"{name} {registration}"
    return Entry(
        id=entry_id,
        content=evidence,
        source=EntrySource("registry.pdf", "Registry", evidence),
        direct_document_context=True,
        entities=[{
            "entity_type": "company",
            "name": name,
            "attributes": [{"kind": "registration_number", "value": registration}],
        }],
        entity_annotation_provenance=[{"method": "worker"}],
    )


def direct_person(entry_id: str, content: str, evidence: str) -> Entry:
    return Entry(
        id=entry_id,
        content=content,
        source=EntrySource("agreement.pdf", "Signature", evidence),
        direct_document_context=True,
        entities=[{"entity_type": "person", "name": "Maria Chen", "attributes": []}],
        entity_annotation_provenance=[{"method": "worker"}],
    )


def clone_direct(entry: Entry, entry_id: str) -> Entry:
    return Entry(
        id=entry_id,
        content=entry.content,
        source=entry.source,
        direct_document_context=True,
        entities=entry.entities,
        entity_annotation_provenance=entry.entity_annotation_provenance,
    )


def blackboard_with_profile(name: str) -> tuple[Blackboard, EntityMaintenanceState]:
    blackboard = Blackboard(iteration=4)
    state = EntityMaintenanceState(profiles={
        "company:northwind-limited": {
            "profile_id": "company:northwind-limited",
            "entity_type": "company",
            "primary_name": name,
            "aliases": [],
            "source_card_ids": ["e1", "e2"],
            "facts": [],
            "revision": 1,
            "fingerprint": "profile_fp_0123456789abcdef",
        },
    })
    return blackboard, state


def changed(profile_id: str) -> ProfileRefreshSummary:
    return ProfileRefreshSummary(dirty_profile_ids=(profile_id,))


def unchanged() -> ProfileRefreshSummary:
    return ProfileRefreshSummary()


def test_profile_requires_two_direct_cards_and_keeps_conflicting_values():
    first = direct_company("e1", "Northwind Limited", "CH-7788")
    second = direct_company("e2", "Northwind Limited", "CH-9999")
    state = EntityMaintenanceState()

    assert refresh_entity_profiles(state, [first], min_card_count=2).created_profile_ids == ()
    summary = refresh_entity_profiles(state, [first, second], min_card_count=2)
    profile = state.profiles[summary.created_profile_ids[0]]

    assert {fact["value"] for fact in profile["facts"] if fact["field"] == "registration_number"} == {"CH-7788", "CH-9999"}


def test_profile_fact_records_direct_worker_context_without_quote():
    entry = direct_person("e1", "Maria Chen signed for Northwind.", evidence="")
    entry.entities = [{"entity_type": "person", "name": "Maria Chen", "attributes": []}]
    state = EntityMaintenanceState()
    summary = refresh_entity_profiles(state, [entry, clone_direct(entry, "e2")], min_card_count=2)

    assert state.profiles[summary.created_profile_ids[0]]["facts"][0]["provenance_quality"] == "direct_worker_context"


def test_profile_preserves_verified_attribute_qualifiers_deterministically():
    first = direct_person("e2", "Maria Chen presented her passport.", "Maria Chen passport P-123")
    first.entities[0]["attributes"] = [{
        "kind": "government_id",
        "value": "P-123",
        "verified": True,
        "qualifiers": {"issuer": "CH", "identifier_type": "passport"},
    }]
    second = clone_direct(first, "e1")
    state = EntityMaintenanceState()

    created = refresh_entity_profiles(state, [first, second], min_card_count=2)
    profile = state.profiles[created.created_profile_ids[0]]
    facts = [fact for fact in profile["facts"] if fact["field"] == "government_id"]
    repeated = refresh_entity_profiles(state, [second, first], min_card_count=2)

    assert facts == [{
        "field": "government_id",
        "value": "P-123",
        "normalized_value": "p123",
        "source_card_id": "e1",
        "source_document": "agreement.pdf",
        "source_section": "Signature",
        "quote": "Maria Chen passport P-123",
        "provenance_quality": "quoted_direct",
        "status": "observed",
        "verified": True,
        "qualifiers": {"identifier_type": "passport", "issuer": "CH"},
    }, {
        "field": "government_id",
        "value": "P-123",
        "normalized_value": "p123",
        "source_card_id": "e2",
        "source_document": "agreement.pdf",
        "source_section": "Signature",
        "quote": "Maria Chen passport P-123",
        "provenance_quality": "quoted_direct",
        "status": "observed",
        "verified": True,
        "qualifiers": {"identifier_type": "passport", "issuer": "CH"},
    }]
    assert repeated == ProfileRefreshSummary()


def test_project_updates_one_current_entity_information_card_only_on_change():
    blackboard, state = blackboard_with_profile("Northwind Limited")
    first_ids = project_entity_information_cards(blackboard, state, changed("company:northwind-limited"))
    second_ids = project_entity_information_cards(blackboard, state, unchanged())

    assert first_ids == ("entity-info-company-northwind-limited",)
    assert second_ids == ()
    assert len([entry for entry in blackboard.entries if entry.type == "entity_information"]) == 1


def test_refresh_is_deterministic_and_only_revisions_changed_payloads():
    first = direct_company("e2", "Northwind Limited", "CH-9999")
    second = direct_company("e1", "Northwind Limited", "CH-7788")
    state = EntityMaintenanceState()

    created = refresh_entity_profiles(state, [first, second], min_card_count=2)
    profile = state.profiles[created.created_profile_ids[0]]
    unchanged_summary = refresh_entity_profiles(state, [second, first], min_card_count=2)

    assert profile["source_card_ids"] == ["e1", "e2"]
    assert profile["revision"] == 1
    assert unchanged_summary == ProfileRefreshSummary()


def test_projected_card_is_compact_json_and_replaces_existing_current_card():
    blackboard, state = blackboard_with_profile("Northwind Limited")
    state.profiles["company:northwind-limited"]["aliases"] = ["Northwind Ltd"]
    state.profiles["company:northwind-limited"]["facts"] = [{
        "field": "registration_number",
        "value": "CH-7788",
        "normalized_value": "ch7788",
        "source_card_id": "e1",
        "source_document": "registry.pdf",
        "source_section": "Registry",
        "quote": "Northwind Limited CH-7788",
        "provenance_quality": "quoted_direct",
        "status": "observed",
    }]

    project_entity_information_cards(blackboard, state, changed("company:northwind-limited"))
    projected = blackboard.find_entry("entity-info-company-northwind-limited")

    assert projected is not None
    assert projected.source is None
    assert projected.created_by.worker_id == "entity_maintenance"
    assert json.loads(projected.content) == {
        "aliases": ["Northwind Ltd"],
        "facts": {"registration_number": ["CH-7788"]},
        "primary_name": "Northwind Limited",
        "profile_id": "company:northwind-limited",
        "revision": 1,
    }
