from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


ENTITY_TYPES = frozenset({"company", "person"})
COMPANY_ATTRIBUTE_KINDS = frozenset({
    "registration_number", "tax_id", "domain", "website", "email_domain",
    "address", "alias",
})
PERSON_ATTRIBUTE_KINDS = frozenset({
    "government_id", "birth_date", "email", "phone", "residence", "birthplace",
    "workplace", "profession", "nationality", "gender", "family_member", "alias",
})
MAX_ENTITIES_PER_ENTRY = 20
MAX_ATTRIBUTES_PER_ENTITY = 20
VERIFIED_COMPANY_IDENTIFIER_KINDS = frozenset({"registration_number", "tax_id"})


def _literally_supported(value: str, evidence: str) -> bool:
    """Require the worker's exact source-local spelling in its evidence quote."""
    stripped = value.strip()
    return bool(stripped and stripped in evidence)


@dataclass(frozen=True)
class EntityAnnotationValidation:
    annotations: list[dict[str, Any]]
    rejections: Sequence[str]


def validate_entity_annotation_result(
    value: object,
    evidence: str,
    *,
    require_literal_evidence: bool = True,
) -> EntityAnnotationValidation:
    if not isinstance(value, list):
        return EntityAnnotationValidation([], ("entities_not_a_list",))

    validated: list[dict[str, Any]] = []
    rejections: list[str] = []
    for raw_entity in value[:MAX_ENTITIES_PER_ENTRY]:
        if not isinstance(raw_entity, dict):
            rejections.append("entity_not_object")
            continue

        entity_type = raw_entity.get("entity_type")
        name = raw_entity.get("name")
        if entity_type not in ENTITY_TYPES or not isinstance(name, str) or not name.strip():
            rejections.append("invalid_entity_shape")
            continue
        name = name.strip()
        if require_literal_evidence and not _literally_supported(name, evidence):
            rejections.append("entity_name_not_in_evidence")
            continue

        allowed_kinds = (
            COMPANY_ATTRIBUTE_KINDS if entity_type == "company" else PERSON_ATTRIBUTE_KINDS
        )
        attributes: list[dict[str, Any]] = []
        raw_attributes = raw_entity.get("attributes", [])
        if not isinstance(raw_attributes, list):
            rejections.append("attributes_not_a_list")
            raw_attributes = []
        for raw_attribute in raw_attributes[:MAX_ATTRIBUTES_PER_ENTITY]:
            if not isinstance(raw_attribute, dict):
                rejections.append("attribute_not_object")
                continue
            kind, attribute_value = raw_attribute.get("kind"), raw_attribute.get("value")
            if kind not in allowed_kinds or not isinstance(attribute_value, str) or not attribute_value.strip():
                rejections.append("invalid_attribute_shape")
                continue
            attribute_value = attribute_value.strip()
            if require_literal_evidence and not _literally_supported(attribute_value, evidence):
                rejections.append("attribute_value_not_in_evidence")
                continue
            raw_qualifiers = raw_attribute.get("qualifiers", {})
            qualifiers = {
                key.strip(): item.strip()
                for key, item in raw_qualifiers.items()
                if isinstance(key, str) and key.strip() and isinstance(item, str) and item.strip()
            } if isinstance(raw_qualifiers, dict) else {}
            requested_verified = raw_attribute.get("verified") is True
            verification_allowed = (
                entity_type == "company" and kind in VERIFIED_COMPANY_IDENTIFIER_KINDS
            ) or (
                entity_type == "person" and kind in {"government_id", "birth_date"}
            )
            if kind == "government_id":
                verification_allowed = verification_allowed and bool(
                    qualifiers.get("issuer") and qualifiers.get("identifier_type")
                )
            attributes.append({
                "kind": kind,
                "value": attribute_value,
                "verified": requested_verified and verification_allowed,
                "qualifiers": dict(sorted(qualifiers.items())),
            })
        validated.append({
            "entity_type": entity_type,
            "name": name,
            "attributes": attributes,
        })
    return EntityAnnotationValidation(validated, tuple(sorted(set(rejections))))


def validate_entity_annotations(value: object, evidence: str) -> list[dict[str, Any]]:
    """Keep only source-local entity annotations supported by this entry's evidence."""
    return validate_entity_annotation_result(value, evidence).annotations
