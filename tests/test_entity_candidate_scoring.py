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


def test_unrelated_person_pair_is_not_reviewed_for_birth_conflict_alone():
    state = state_with_profiles(
        person_profile("Alice Smith", "1980-01-01"),
        person_profile("Alice Jones", "1990-02-02"),
    )

    summary = refresh_candidates(state, {"person:alice-smith"}, EntityMaintenanceConfig())

    assert summary.review_candidate_ids == ()
    candidate = next(iter(state.candidates.values()))
    assert candidate["status"] == "ignored"
    assert "verified_birth_date_conflict" in candidate["conflicts"]


def test_person_spelling_variant_name_only_reaches_specialist_review():
    dimitri = person_profile("Dimitri Volkov", "1980-01-01")
    dymitri = person_profile("Dymitri Volkov", "1980-01-01")
    dimitri["facts"] = []
    dymitri["facts"] = []
    state = state_with_profiles(dimitri, dymitri)

    summary = refresh_candidates(state, set(state.profiles), EntityMaintenanceConfig())

    candidate = state.candidates[summary.review_candidate_ids[0]]
    assert candidate["status"] == "pending_review"
    assert candidate["evidence"] == ["name:similar"]


def test_reordered_abbreviated_person_name_reaches_specialist_review():
    state = state_with_profiles(
        person_profile("Dmitri K. Volkov", "1980-01-01"),
        person_profile("VOLKOV, Dmitriy Konstantinovich", "1980-01-01"),
    )

    summary = refresh_candidates(state, set(state.profiles), EntityMaintenanceConfig())

    assert len(summary.review_candidate_ids) == 1


def test_confirmed_pair_is_not_reviewed_again_when_only_profile_revision_changes():
    state = state_with_profiles(
        person_profile("Dimitri Volkov", "1980-01-01"),
        person_profile("Dymitri Volkov", "1980-01-01"),
    )
    first = refresh_candidates(state, set(state.profiles), EntityMaintenanceConfig())
    candidate_id = first.review_candidate_ids[0]
    candidate = state.candidates[candidate_id]
    candidate["status"] = "reviewed"
    state.decisions["decision-confirmed"] = {
        "semantic_key": candidate["semantic_key"],
        "outcome": "same_entity",
        "conflicts": candidate["conflicts"],
    }
    changed_profile = state.profiles["person:dymitri-volkov"]
    changed_profile["revision"] = 2
    changed_profile["fingerprint"] = "updated-fingerprint"

    second = refresh_candidates(
        state, {"person:dymitri-volkov"}, EntityMaintenanceConfig(),
    )

    assert second.review_candidate_ids == ()
    assert second.reused_candidate_ids == (candidate_id,)
    assert state.candidates[candidate_id]["status"] == "reviewed"
