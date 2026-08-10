from src.swarm.entity_mention_repair import (
    build_entity_catalogue,
    repair_entity_mentions,
)
from src.swarm.models import Entry, EntrySource


def direct_entry(entry_id: str, content: str, *, evidence: str | None = None) -> Entry:
    return Entry(
        id=entry_id,
        content=content,
        source=EntrySource(
            document="agreement.pdf",
            section="Signature",
            evidence=content if evidence is None else evidence,
        ),
        direct_document_context=True,
    )


def profile(entity_type: str, name: str) -> dict[str, object]:
    return {"entity_type": entity_type, "primary_name": name, "aliases": []}


def test_repair_includes_unquoted_direct_card_and_excludes_derived_card():
    direct = direct_entry("e1", "Maria Chen signed for Northwind Limited.", evidence="")
    derived = Entry(id="e2", type="analysis", content="Maria Chen is material.")

    result = repair_entity_mentions(
        [direct, derived],
        {"company:northwind-limited": profile("company", "Northwind Limited")},
        max_mentions_per_card=20,
        run_id="run-1",
    )

    assert result.changed_card_ids == ("e1",)
    assert direct.entities[-1]["name"] == "Northwind Limited"
    assert derived.entities == []


def test_repair_uses_literal_boundaries_not_substrings():
    card = direct_entry("e1", "The annual filing is complete.")

    repair_entity_mentions([card], {"person:ann": profile("person", "Ann")}, 20, "run-1")

    assert card.entities == []


def test_catalogue_includes_typed_worker_name_from_another_new_card():
    named = direct_entry("e1", "Northwind Limited signed.")
    named.entities = [{"entity_type": "company", "name": "Northwind Limited", "attributes": []}]
    named.entity_annotation_provenance = [{"method": "worker"}]
    omitted = direct_entry("e2", "Northwind Limited amended the agreement.")

    catalogue = build_entity_catalogue([named, omitted], {})
    repair_entity_mentions([omitted], catalogue, 20, "run-1")

    assert omitted.entities == [{"entity_type": "company", "name": "Northwind Limited", "attributes": []}]


def test_catalogue_ignores_annotation_without_worker_or_repair_provenance():
    untrusted = direct_entry("e1", "Northwind Limited signed.")
    untrusted.entities = [{"entity_type": "company", "name": "Northwind Limited", "attributes": []}]
    untrusted.entity_annotation_provenance = [{"method": "duplicate_resolution"}]
    omitted = direct_entry("e2", "Northwind Limited amended the agreement.")

    catalogue = build_entity_catalogue([untrusted], {})
    repair_entity_mentions([omitted], catalogue, 20, "run-1")

    assert catalogue == {}
    assert omitted.entities == []


def test_repair_sorts_eligible_cards_by_id_for_deterministic_summary():
    later = direct_entry("z-card", "Northwind Limited signed.")
    earlier = direct_entry("a-card", "Northwind Limited amended the agreement.")

    result = repair_entity_mentions(
        [later, earlier],
        {"company:northwind-limited": profile("company", "Northwind Limited")},
        20,
        "run-1",
    )

    assert result.changed_card_ids == ("a-card", "z-card")


def test_repeat_repair_is_idempotent_and_preserves_worker_annotation():
    card = direct_entry("e1", "Maria Chen signed for Northwind Limited.")
    card.entities = [{"entity_type": "person", "name": "Maria Chen", "attributes": []}]
    card.entity_annotation_provenance = [{"method": "worker"}]
    catalogue = {"company:northwind-limited": profile("company", "Northwind Limited")}

    repair_entity_mentions([card], catalogue, 20, "run-1")
    repair_entity_mentions([card], catalogue, 20, "run-2")

    assert [item["name"] for item in card.entities] == ["Maria Chen", "Northwind Limited"]
    assert card.entity_annotation_provenance[0] == {"method": "worker"}
    assert card.entity_annotation_provenance[1]["method"] == "deterministic_repair"
