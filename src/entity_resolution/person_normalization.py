from __future__ import annotations

"""Conservative normalization for person names and sourced attributes."""

from datetime import date
from difflib import SequenceMatcher
import json
import re
import unicodedata

from .person_models import PersonAttribute, PersonNameForms


RECOGNIZED_ATTRIBUTE_KINDS = frozenset(
    {
        "government_id",
        "birth_date",
        "email",
        "phone",
        "residence",
        "birthplace",
        "workplace",
        "profession",
        "nationality",
        "gender",
        "family_member",
        "alias",
    }
)


def canonicalize_person_attribute_kind(kind: str) -> str:
    return kind.strip().casefold()


def _fold_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value).casefold()
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(
        re.sub(r"[^\w]+", " ", without_marks, flags=re.UNICODE).replace("_", " ").split()
    )


def _compact_alphanumeric(value: str) -> str:
    return "".join(character for character in _fold_text(value) if character.isalnum())


def _normalize_government_id_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    return "".join(
        character
        for character in normalized
        if character.isalnum() or unicodedata.category(character).startswith("M")
    )


def normalize_person_name(name: str) -> PersonNameForms:
    normalized = _fold_text(name)
    tokens = tuple(normalized.split())
    return PersonNameForms(
        raw=name,
        normalized=normalized,
        tokens=tokens,
        sorted_tokens=" ".join(sorted(tokens)),
    )


def normalize_person_attribute(attribute: PersonAttribute) -> str:
    kind = canonicalize_person_attribute_kind(attribute.kind)
    value = attribute.value.strip()
    if kind == "email":
        return value.casefold()
    if kind == "phone":
        return _compact_alphanumeric(value)
    if kind == "government_id":
        return _normalize_government_id_text(value)
    if kind == "birth_date":
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError:
            return _fold_text(value)
    if kind == "alias":
        return normalize_person_name(value).normalized
    return _fold_text(value)


def government_id_components(attribute: PersonAttribute) -> tuple[str, str, str] | None:
    if canonicalize_person_attribute_kind(attribute.kind) != "government_id" or not attribute.verified:
        return None
    qualifiers: dict[str, str] = {}
    for key, value in attribute.qualifiers:
        canonical_key = key.strip().casefold()
        canonical_value = value.strip().casefold()
        if canonical_key in {"issuer", "identifier_type"}:
            if canonical_key in qualifiers and qualifiers[canonical_key] != canonical_value:
                return None
        qualifiers[canonical_key] = canonical_value
    issuer = qualifiers.get("issuer")
    identifier_type = qualifiers.get("identifier_type")
    identifier = normalize_person_attribute(attribute)
    if not issuer or not identifier_type or not identifier:
        return None
    return issuer, identifier_type, identifier


def government_id_key(attribute: PersonAttribute) -> str | None:
    if components := government_id_components(attribute):
        return json.dumps(components, ensure_ascii=False, separators=(",", ":"))
    return None


def person_name_similarity(left: str, right: str) -> float:
    left_forms = normalize_person_name(left)
    right_forms = normalize_person_name(right)
    if not left_forms.normalized or not right_forms.normalized:
        return 0.0
    direct = SequenceMatcher(None, left_forms.normalized, right_forms.normalized).ratio()
    reordered = SequenceMatcher(None, left_forms.sorted_tokens, right_forms.sorted_tokens).ratio()
    return max(direct, reordered)
