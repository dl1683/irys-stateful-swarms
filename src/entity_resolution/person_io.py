from __future__ import annotations

"""Validated JSON boundaries for person connection screening."""

import json

from .person_models import PersonAttribute, PersonRecord, PersonResolutionResult
from .person_normalization import normalize_person_name


_MISSING = object()


def _reject_duplicate_object_keys(pairs: list[tuple[object, object]]) -> dict[object, object]:
    result: dict[object, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"{field_name} must be a string or null")


def load_person_records_json(payload: str) -> list[PersonRecord]:
    data = json.loads(payload, object_pairs_hook=_reject_duplicate_object_keys)
    if not isinstance(data, list):
        raise ValueError("input must be a JSON array")
    records: list[PersonRecord] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"record at index {index} must be an object")
        raw_record_id = item.get("record_id", _MISSING)
        if raw_record_id is _MISSING:
            raise ValueError(f"record at index {index} is missing required field record_id")
        record_id = _required_string(raw_record_id, f"record at index {index} record_id").strip()
        if not record_id:
            raise ValueError(f"record at index {index} is missing required field record_id")
        if record_id in seen_ids:
            raise ValueError(f"duplicate record_id: {record_id}")
        seen_ids.add(record_id)
        raw_name = item.get("name", _MISSING)
        if raw_name is _MISSING:
            raise ValueError(f"record at index {index} is missing required field name")
        name = _required_string(raw_name, f"record at index {index} name").strip()
        if not name or not normalize_person_name(name).normalized:
            raise ValueError(f"record at index {index} is missing required field name")
        snippets = item.get("snippets", [])
        if not isinstance(snippets, list):
            raise ValueError(f"record {record_id} snippets must be a list")
        normalized_snippets: list[str] = []
        for snippet_index, snippet in enumerate(snippets):
            if not isinstance(snippet, str):
                raise ValueError(f"record {record_id} snippet {snippet_index} must be a string")
            normalized_snippets.append(snippet)
        raw_attributes = item.get("attributes", [])
        if not isinstance(raw_attributes, list):
            raise ValueError(f"record {record_id} attributes must be a list")
        attributes: list[PersonAttribute] = []
        for attribute_index, raw in enumerate(raw_attributes):
            if not isinstance(raw, dict):
                raise ValueError(f"record {record_id} attribute {attribute_index} must be an object")
            raw_kind = raw.get("kind", _MISSING)
            if raw_kind is _MISSING:
                raise ValueError(f"record {record_id} attribute kind is required")
            kind = _required_string(
                raw_kind,
                f"record {record_id} attribute kind",
            ).strip().casefold()
            raw_value = raw.get("value", _MISSING)
            if raw_value is _MISSING:
                raise ValueError(f"record {record_id} attribute value is required")
            value = _required_string(
                raw_value,
                f"record {record_id} attribute value",
            ).strip()
            if not kind:
                raise ValueError(f"record {record_id} attribute kind is required")
            if not value:
                raise ValueError(f"record {record_id} attribute value is required")
            verified = raw.get("verified", False)
            if not isinstance(verified, bool):
                raise ValueError(f"record {record_id} attribute verified must be a boolean")
            qualifiers = raw.get("qualifiers", {})
            if not isinstance(qualifiers, dict):
                raise ValueError(f"record {record_id} attribute qualifiers must be an object")
            normalized_qualifiers: list[tuple[str, str]] = []
            for key, qualifier_value in qualifiers.items():
                if not isinstance(key, str):
                    raise ValueError(f"record {record_id} attribute qualifier key must be a string")
                if not isinstance(qualifier_value, str):
                    raise ValueError(
                        f"record {record_id} attribute qualifier {key!r} must be a string"
                    )
                normalized_qualifiers.append((key, qualifier_value))
            attributes.append(
                PersonAttribute(
                    kind=kind,
                    value=value,
                    source_document_id=_optional_string(
                        raw.get("source_document_id"), "source_document_id"
                    ),
                    effective_from=_optional_string(raw.get("effective_from"), "effective_from"),
                    effective_to=_optional_string(raw.get("effective_to"), "effective_to"),
                    verified=verified,
                    notes=_optional_string(raw.get("notes"), "notes"),
                    qualifiers=tuple(sorted(normalized_qualifiers)),
                )
            )
        records.append(
            PersonRecord(
                record_id=record_id,
                name=name,
                document_id=_optional_string(item.get("document_id"), "document_id"),
                snippets=tuple(normalized_snippets),
                attributes=tuple(attributes),
            )
        )
    return records


def write_person_result_json(result: PersonResolutionResult) -> str:
    return json.dumps(result.to_dict(), indent=2, sort_keys=True)
