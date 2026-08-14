from pathlib import Path


def test_readme_documents_periodic_entity_maintenance_contract():
    readme = Path("README.md").read_text(encoding="utf-8")
    for required in (
        "entity_resolution_interval_iterations",
        "duplicate_review_threshold",
        "entity_profile_min_card_count",
        "entity_repair_max_mentions_per_card",
        "swarm/entity_resolution/state.json",
        "duplicate_name_resolution",
        "direct-document-worker",
    ):
        assert required in readme
