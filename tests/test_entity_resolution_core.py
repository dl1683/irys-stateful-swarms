from src.entity_resolution import (
    CompanyRecord,
    PersonAttribute,
    PersonRecord,
    resolve_companies,
    resolve_people,
)


def test_company_registration_match_merges_while_conflict_stays_reviewable():
    result = resolve_companies([
        CompanyRecord(
            "company-a",
            "Zenith Petrochem Industries LLC",
            "invoice.docx",
            metadata={"registration_number": "AE-123", "address": "Jebel Ali"},
        ),
        CompanyRecord(
            "company-b",
            "Zenith Petrochemical Industries LLC",
            "kyc.docx",
            metadata={"registration_number": "AE-123", "address": "Jebel Ali"},
        ),
        CompanyRecord(
            "company-c",
            "Zenith Petroleum Industries FZE",
            "screening.docx",
            metadata={"registration_number": "AE-999"},
        ),
    ])

    assert {report.status for report in result.duplicate_name_reports} == {
        "auto_merged",
        "review_required",
    }
    assert not any(
        "company-c" in cluster.member_record_ids for cluster in result.clusters
    )


def test_person_matching_verified_issuer_qualified_id_merges_without_conflicts():
    identifier = {
        "kind": "government_id",
        "value": "CH-123",
        "verified": True,
        "qualifiers": (("identifier_type", "passport"), ("issuer", "CH")),
    }
    result = resolve_people([
        PersonRecord(
            "person-a",
            "Maria Chen",
            "passport.pdf",
            attributes=(PersonAttribute(source_document_id="passport.pdf", **identifier),),
        ),
        PersonRecord(
            "person-b",
            "Chen, Maria",
            "kyc.pdf",
            attributes=(PersonAttribute(source_document_id="kyc.pdf", **identifier),),
        ),
    ])

    assert len(result.confirmed_profiles) == 1
    assert result.confirmed_profiles[0].member_record_ids == ("person-a", "person-b")
    assert result.uncertain_connections == ()
