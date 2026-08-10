from __future__ import annotations

"""JSON boundary functions for validated records and stable results."""

import json
from typing import Any

from .models import CompanyRecord, ResolutionResult
from .normalization import normalize_company_name


def load_records_json(payload: str) -> list[CompanyRecord]:
    data = json.loads(payload)
    if not isinstance(data, list):
        raise ValueError("input must be a JSON array")
    records: list[CompanyRecord] = []
    seen_record_ids: set[str] = set()
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"record at index {index} must be an object")
        record_id = str(item.get("record_id", "")).strip()
        if not record_id:
            raise ValueError(f"record at index {index} is missing required field record_id")
        if record_id in seen_record_ids:
            raise ValueError(f"duplicate record_id: {record_id}")
        seen_record_ids.add(record_id)
        name = str(item.get("name", "")).strip()
        if not name or not normalize_company_name(name).core:
            raise ValueError(f"record at index {index} is missing required field name")
        metadata = item.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError(f"record {item['record_id']} metadata must be an object")
        snippets = item.get("snippets", [])
        if snippets is None:
            snippets = []
        if not isinstance(snippets, list):
            raise ValueError(f"record {item['record_id']} snippets must be a list")
        records.append(
            CompanyRecord(
                record_id=record_id,
                name=name,
                document_id=item.get("document_id"),
                snippets=[str(snippet) for snippet in snippets],
                metadata={str(key): value for key, value in metadata.items()},
            )
        )
    return records


def write_result_json(result: ResolutionResult) -> str:
    return json.dumps(result.to_dict(), indent=2, sort_keys=True)
