from __future__ import annotations

"""Deterministic normalization helpers for names and matching metadata."""

import re
import unicodedata
from urllib.parse import urlparse

from .models import CompanyNameForms


LEGAL_SUFFIXES = {
    "co",
    "company",
    "corp",
    "corporation",
    "gmbh",
    "inc",
    "incorporated",
    "limited",
    "llc",
    "llp",
    "ltd",
    "plc",
    "sa",
}

CONNECTORS = {"and"}

SINGULARS = {
    "chemicals": "chemical",
    "petrochemicals": "petrochemical",
    "technologies": "technology",
    "solutions": "solution",
    "energies": "energy",
}


def _ascii_fold(value: str) -> str:
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")


def _fold_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value).casefold()
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def _clean_tokens(value: str) -> list[str]:
    folded = _fold_name(value).replace("&", " and ")
    cleaned = re.sub(r"[^\w]+", " ", folded, flags=re.UNICODE).replace("_", " ")
    return [SINGULARS.get(token, token) for token in cleaned.split()]


def normalize_company_name(name: str) -> CompanyNameForms:
    tokens = _clean_tokens(name)
    normalized_tokens = [token for token in tokens if token not in CONNECTORS]
    core_tokens = [token for token in normalized_tokens if token not in LEGAL_SUFFIXES]
    normalized = " ".join(normalized_tokens)
    core = " ".join(core_tokens)
    return CompanyNameForms(
        raw=name,
        normalized=normalized,
        core=core,
        tokens=tuple(core_tokens),
        sorted_tokens=" ".join(sorted(core_tokens)),
    )


def company_initials(tokens: tuple[str, ...]) -> str | None:
    if len(tokens) == 1 and 3 <= len(tokens[0]) <= 8:
        return tokens[0] if tokens[0].isalpha() else None
    if len(tokens) >= 3:
        if any(not token.isalpha() for token in tokens):
            return None
        initials = "".join(token[0] for token in tokens if token)
        return initials if 3 <= len(initials) <= 8 else None
    return None


def normalize_identifier(value: object) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"[^a-z0-9]+", "", _ascii_fold(str(value)).lower())
    return cleaned or None


def normalize_domain(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if "@" in text and not text.startswith(("http://", "https://")):
        text = text.rsplit("@", 1)[-1]
    if "://" not in text:
        text = "http://" + text
    try:
        parsed = urlparse(text)
        host = parsed.hostname or parsed.path.split("/", 1)[0]
    except ValueError:
        return None
    host = host.removeprefix("www.").rstrip(".")
    return host or None
