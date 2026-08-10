from src.swarm.worker_dispatch import parse_worker_output


def test_direct_document_context_keeps_unquoted_worker_entity():
    entries = parse_worker_output(
        {"findings": [{
            "type": "observation",
            "content": "Northwind Limited signed the agreement.",
            "source_document": "agreement.pdf",
            "source_section": "Signature",
            "entities": [{"entity_type": "company", "name": "Northwind Limited", "attributes": []}],
        }]},
        1, "w1", "read", {"agreement.pdf"}, direct_document_context=True,
    )

    assert entries[0].direct_document_context is True
    assert entries[0].entities[0]["name"] == "Northwind Limited"
    assert entries[0].entity_annotation_provenance == [{"method": "worker"}]


def test_blackboard_only_worker_is_not_direct_document_context():
    entries = parse_worker_output(
        {"findings": [{"type": "analysis", "content": "Northwind Limited is material."}]},
        2, "w2", "analysis", set(), direct_document_context=False,
    )

    assert entries[0].direct_document_context is False


def test_invalid_entity_annotation_is_audited_without_dropping_card():
    entries = parse_worker_output(
        {"findings": [{
            "type": "observation", "content": "Northwind signed the agreement.",
            "source_document": "agreement.pdf", "source_section": "Signature",
            "evidence": "Northwind signed the agreement.",
            "entities": [{"entity_type": "company", "name": "Invented Co", "attributes": []}],
        }]},
        1, "w1", "read", {"agreement.pdf"}, direct_document_context=True,
    )

    assert len(entries) == 1
    assert entries[0].entities == []
    assert entries[0].entity_annotation_rejections == ["entity_name_not_in_evidence"]


def test_direct_document_context_validates_nonempty_evidence_without_source_document():
    entries = parse_worker_output(
        {"findings": [{
            "type": "observation", "content": "Northwind signed the agreement.",
            "evidence": "Northwind signed the agreement.",
            "entities": [{"entity_type": "company", "name": "Invented Co", "attributes": []}],
        }]},
        1, "w1", "read", {"agreement.pdf"}, direct_document_context=True,
    )

    assert entries[0].entities == []
    assert entries[0].entity_annotation_rejections == ["entity_name_not_in_evidence"]
