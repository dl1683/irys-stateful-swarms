"""Public API for deterministic company and person duplicate resolution."""

from .io import load_records_json, write_result_json
from .models import (
    CompanyCandidate,
    CompanyCluster,
    CompanyNameForms,
    CompanyRecord,
    DuplicateNameReport,
    ResolutionResult,
    ResolverConfig,
)
from .normalization import normalize_company_name
from .resolver import resolve_companies
from .review import generate_review_prompt
from .person_io import load_person_records_json, write_person_result_json
from .person_models import (
    PersonAttribute,
    PersonAutoMatch,
    PersonEvidence,
    PersonNameForms,
    PersonProfile,
    PersonRecord,
    PersonResolutionResult,
    UncertainConnection,
)
from .person_normalization import normalize_person_name
from .person_resolver import resolve_people
from .person_review import generate_person_review_prompt
from .token_stats import token_weights, weighted_token_overlap

__all__ = [
    "CompanyCandidate",
    "CompanyCluster",
    "CompanyNameForms",
    "CompanyRecord",
    "DuplicateNameReport",
    "PersonAttribute",
    "PersonAutoMatch",
    "PersonEvidence",
    "PersonNameForms",
    "PersonProfile",
    "PersonRecord",
    "PersonResolutionResult",
    "ResolutionResult",
    "ResolverConfig",
    "UncertainConnection",
    "generate_person_review_prompt",
    "generate_review_prompt",
    "load_person_records_json",
    "load_records_json",
    "normalize_company_name",
    "normalize_person_name",
    "resolve_companies",
    "resolve_people",
    "token_weights",
    "weighted_token_overlap",
    "write_person_result_json",
    "write_result_json",
]
