from __future__ import annotations

"""Normalized access to metadata used by blocking, scoring, and clustering."""

from .models import CompanyRecord
from .normalization import normalize_domain, normalize_identifier


def record_domain(record: CompanyRecord) -> str | None:
    for key in ("domain", "website", "email_domain"):
        if domain := normalize_domain(record.metadata.get(key)):
            return domain
    return None


def record_registration(record: CompanyRecord) -> str | None:
    for key in ("registration_number", "tax_id"):
        if identifier := normalize_identifier(record.metadata.get(key)):
            return identifier
    return None


def record_address(record: CompanyRecord) -> str | None:
    return normalize_identifier(record.metadata.get("address"))
