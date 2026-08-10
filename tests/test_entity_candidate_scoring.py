from src.swarm.entity_candidate_scoring import refresh_candidates
from src.swarm.entity_maintenance_store import EntityMaintenanceConfig, EntityMaintenanceState


def company_profile(name: str, registration: str | None = None, verified: bool = False) -> dict[str, object]:
    profile_id = "company:" + "-".join(name.casefold().split())
    facts: list[dict[str, object]] = [{
        "field": "name",
        "value": name,
        "normalized_value": "".join(character for character in name.casefold() if character.isalnum()),
        "source_card_id": profile_id + "-card",
    }]
    if registration:
        facts.append({
            "field": "registration_number",
            "value": registration,
            "normalized_value": "".join(character for character in registration.casefold() if character.isalnum()),
            "source_card_id": profile_id + "-card",
            "verified": verified,
        })
    return {
        "profile_id": profile_id,
        "entity_type": "company",
        "primary_name": name,
        "aliases": [],
        "source_card_ids": [profile_id + "-card"],
        "facts": facts,
        "revision": 1,
        "fingerprint": profile_id + "-fingerprint",
    }


def state_with_profiles(*profiles: str | dict[str, object]) -> EntityMaintenanceState:
    payloads = [company_profile(profile) if isinstance(profile, str) else profile for profile in profiles]
    return EntityMaintenanceState(profiles={str(profile["profile_id"]): profile for profile in payloads})


def person_profile(name: str, birth_date: str) -> dict[str, object]:
    profile_id = "person:" + "-".join(name.casefold().split())
    return {
        "profile_id": profile_id,
        "entity_type": "person",
        "primary_name": name,
        "aliases": [],
        "source_card_ids": [profile_id + "-card"],
        "facts": [{
            "field": "birth_date",
            "value": birth_date,
            "normalized_value": birth_date,
            "source_card_id": profile_id + "-card",
            "verified": True,
        }],
        "revision": 1,
        "fingerprint": profile_id + "-fingerprint",
    }


def test_dirty_profile_is_compared_to_relevant_old_profile_but_not_unrelated_profile():
    state = state_with_profiles("Northwind Ltd", "Northwind Limited", "Bluewater PLC")

    summary = refresh_candidates(state, {"company:northwind-ltd"}, EntityMaintenanceConfig())

    pairs = {tuple(candidate["profile_ids"]) for candidate in state.candidates.values()}
    assert ("company:northwind-limited", "company:northwind-ltd") in pairs
    assert all("company:bluewater-plc" not in pair for pair in pairs)
    assert summary.new_candidate_ids


def test_weak_lexical_candidate_stays_sidecar_only():
    state = state_with_profiles("Alpha Logistics", "Alpine Law Office")

    summary = refresh_candidates(state, set(state.profiles), EntityMaintenanceConfig())

    assert summary.review_candidate_ids == ()
    assert all(candidate["status"] == "ignored" for candidate in state.candidates.values())


def test_verified_registration_conflict_is_review_eligible_even_below_threshold():
    state = state_with_profiles(
        company_profile("Zenith Trading", registration="CH-111", verified=True),
        company_profile("Zenith Trading Limited", registration="CH-222", verified=True),
    )

    summary = refresh_candidates(state, set(state.profiles), EntityMaintenanceConfig())

    candidate = state.candidates[summary.review_candidate_ids[0]]
    assert "verified_registration_number_conflict" in candidate["conflicts"]
    assert candidate["status"] == "pending_review"


def test_unchanged_candidate_is_reused_without_another_review():
    state = state_with_profiles("Northwind Ltd", "Northwind Limited")
    first = refresh_candidates(state, set(state.profiles), EntityMaintenanceConfig())

    second = refresh_candidates(state, {"company:northwind-ltd"}, EntityMaintenanceConfig())

    assert first.new_candidate_ids
    assert second.reused_candidate_ids == first.new_candidate_ids
    assert second.review_candidate_ids == ()


def test_conflicting_verified_values_within_one_profile_stay_pending_review():
    profile = company_profile("Zenith Trading", registration="CH-111", verified=True)
    profile["facts"].append({
        "field": "registration_number",
        "value": "CH-222",
        "normalized_value": "ch222",
        "source_card_id": "zenith-second-card",
        "verified": True,
    })
    state = state_with_profiles(profile)

    summary = refresh_candidates(state, set(state.profiles), EntityMaintenanceConfig())

    candidate = state.candidates[summary.review_candidate_ids[0]]
    assert candidate["profile_ids"] == ["company:zenith-trading"]
    assert candidate["source_card_groups"] == [["company:zenith-trading-card"], ["zenith-second-card"]]


def test_screened_out_blocked_person_pair_with_verified_birth_conflict_is_reviewed():
    state = state_with_profiles(
        person_profile("Alice Smith", "1980-01-01"),
        person_profile("Alice Jones", "1990-02-02"),
    )

    summary = refresh_candidates(state, {"person:alice-smith"}, EntityMaintenanceConfig())

    candidate = state.candidates[summary.review_candidate_ids[0]]
    assert candidate["profile_ids"] == ["person:alice-jones", "person:alice-smith"]
    assert candidate["status"] == "pending_review"
    assert "verified_birth_date_conflict" in candidate["conflicts"]
