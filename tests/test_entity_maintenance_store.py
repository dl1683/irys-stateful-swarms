import json

from src.swarm.blackboard import Blackboard
from src.swarm.entity_maintenance_store import (
    EntityMaintenanceConfig,
    EntityMaintenanceState,
)
from src.swarm.models import Entry, EntrySource


def direct_entry(entry_id: str, content: str) -> Entry:
    return Entry(
        id=entry_id,
        content=content,
        source=EntrySource(document="agreement.pdf", section="Signature", evidence=content),
        direct_document_context=True,
        entities=[{"entity_type": "company", "name": "Northwind Limited"}],
        entity_annotation_provenance=[{"method": "worker"}],
    )


def test_config_uses_documented_defaults_and_metadata_override():
    assert EntityMaintenanceConfig.from_metadata({}) == EntityMaintenanceConfig(
        entity_resolution_interval_iterations=3,
        entity_profile_min_card_count=2,
        duplicate_review_threshold=0.85,
        entity_repair_max_mentions_per_card=20,
    )
    assert EntityMaintenanceConfig.from_metadata({
        "entity_resolution_interval_iterations": 5,
    }).entity_resolution_interval_iterations == 5


def test_sidecar_round_trip_and_card_revision_watermark(tmp_path):
    first = direct_entry("e1", "Northwind Limited signed.")
    state = EntityMaintenanceState()
    assert state.card_is_dirty(first)
    state.mark_cards_processed([first])
    state.write(str(tmp_path))

    restored = EntityMaintenanceState.load(str(tmp_path))
    assert restored.card_is_dirty(first) is False
    first.content = "Northwind Limited signed the amended agreement."
    assert restored.card_is_dirty(first) is True


def test_sidecar_json_and_fingerprint_are_deterministic(tmp_path):
    state = EntityMaintenanceState()
    assert state.fingerprint({"b": [2, 1], "a": "x"}) == state.fingerprint({"a": "x", "b": [2, 1]})

    state.profiles["northwind"] = {"name": "Northwind Limited"}
    path = state.write(str(tmp_path))
    first_write = path.read_bytes()
    state.write(str(tmp_path))
    assert path.read_bytes() == first_write


def test_blackboard_snapshot_exposes_only_maintenance_status(tmp_path):
    blackboard = Blackboard(output_dir=str(tmp_path))
    blackboard.entity_maintenance_state.profiles["northwind"] = {"name": "Northwind Limited"}
    blackboard.save_snapshot()

    snapshot = json.loads((tmp_path / "swarm" / "blackboard_iter_0.json").read_text())
    assert snapshot["entity_maintenance"] == {
        "processed_cards": 0,
        "profiles": 1,
        "decisions": 0,
    }
