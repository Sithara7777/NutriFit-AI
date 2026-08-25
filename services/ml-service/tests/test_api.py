"""Integration tests for the ML microservice API."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SERVICE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVICE_ROOT))
sys.path.insert(0, str(SERVICE_ROOT.parent.parent / "ml"))

from app.main import app  # noqa: E402

VALID_PROFILE = {
    "age": 28,
    "gender": "Male",
    "height_cm": 178.0,
    "weight_kg": 82.0,
    "fitness_goal": "muscle_gain",
    "workout_frequency": 4,
    "session_duration_h": 1.25,
    "experience_level": 2,
    "body_fat_pct": 18.0,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in {"ok", "degraded"}
        assert "models_loaded" in body

    def test_root_lists_endpoints(self, client):
        body = client.get("/").json()
        assert "/predict" in body["endpoints"]

    def test_timing_header_present(self, client):
        assert "X-Response-Time-ms" in client.get("/health").headers


class TestPredict:
    def test_returns_plausible_targets(self, client):
        response = client.post("/predict", json=VALID_PROFILE)
        assert response.status_code == 200
        body = response.json()
        assert 1500 < body["calorie_target"] < 5000
        assert 80 < body["protein_target"] < 300
        assert body["source"] in {"model", "formula"}

    def test_reports_derived_physiology(self, client):
        body = client.post("/predict", json=VALID_PROFILE).json()
        assert body["bmi"] == pytest.approx(25.88, abs=0.05)
        assert body["bmi_category"] == "overweight"
        assert body["bmr_equation"] == "Katch-McArdle"
        assert body["body_fat_source"] == "measured"

    def test_estimates_body_fat_when_absent(self, client):
        payload = {k: v for k, v in VALID_PROFILE.items() if k != "body_fat_pct"}
        body = client.post("/predict", json=payload).json()
        assert body["body_fat_source"] == "estimated_deurenberg"
        assert body["bmr_equation"] == "Mifflin-St Jeor"
        assert 3 <= body["body_fat_pct"] <= 60

    def test_always_returns_formula_reference(self, client):
        body = client.post("/predict", json=VALID_PROFILE).json()
        assert body["formula_reference"]["calorie_target"] > 0

    @pytest.mark.parametrize("goal", ["fat_loss", "maintenance", "muscle_gain"])
    def test_all_goals_accepted(self, client, goal):
        payload = {**VALID_PROFILE, "fitness_goal": goal}
        assert client.post("/predict", json=payload).status_code == 200

    def test_goal_ordering_is_respected(self, client):
        targets = {}
        for goal in ("fat_loss", "maintenance", "muscle_gain"):
            payload = {**VALID_PROFILE, "fitness_goal": goal}
            targets[goal] = client.post("/predict", json=payload).json()["calorie_target"]
        assert targets["fat_loss"] < targets["maintenance"] < targets["muscle_gain"]

    def test_gender_tokens_normalised(self, client):
        payload = {**VALID_PROFILE, "gender": "male"}
        assert client.post("/predict", json=payload).status_code == 200

    @pytest.mark.parametrize(
        "field,value",
        [("age", 5), ("age", 200), ("weight_kg", 5), ("weight_kg", 900),
         ("height_cm", 20), ("workout_frequency", 20), ("experience_level", 9),
         ("body_fat_pct", 95), ("fitness_goal", "get_swole"), ("gender", "banana")],
    )
    def test_rejects_out_of_range_input(self, client, field, value):
        payload = {**VALID_PROFILE, field: value}
        assert client.post("/predict", json=payload).status_code == 422

    def test_rejects_unknown_field(self, client):
        payload = {**VALID_PROFILE, "sneaky": "value"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_rejects_missing_required_field(self, client):
        payload = {k: v for k, v in VALID_PROFILE.items() if k != "weight_kg"}
        assert client.post("/predict", json=payload).status_code == 422


class TestRecommend:
    def test_slot_specific_suggestions(self, client):
        response = client.post("/recommend", json={
            "calorie_target": 2400, "protein_target": 150,
            "fitness_goal": "muscle_gain", "meal_slot": "breakfast", "top_n": 3,
        })
        assert response.status_code == 200
        body = response.json()
        assert len(body["suggestions"]) <= 3
        first = body["suggestions"][0]
        # The UI must be able to show a macro breakdown, not just a name.
        for key in ("name", "calories", "protein_g", "carbs_g", "fat_g", "servings"):
            assert key in first

    def test_full_day_composition(self, client):
        response = client.post("/recommend", json={
            "calorie_target": 2400, "protein_target": 150, "fitness_goal": "maintenance",
        })
        assert response.status_code == 200
        body = response.json()
        assert set(body["slots"]) == {"breakfast", "lunch", "dinner", "snack"}
        assert body["totals"]["calories"] == pytest.approx(2400, rel=0.15)

    def test_exclusions_respected(self, client):
        first = client.post("/recommend", json={
            "calorie_target": 2400, "protein_target": 150, "meal_slot": "lunch", "top_n": 1,
        }).json()["suggestions"][0]["food_id"]

        second = client.post("/recommend", json={
            "calorie_target": 2400, "protein_target": 150, "meal_slot": "lunch",
            "top_n": 1, "exclude": [first],
        }).json()["suggestions"][0]["food_id"]
        assert second != first

    def test_rejects_invalid_slot(self, client):
        response = client.post("/recommend", json={
            "calorie_target": 2400, "protein_target": 150, "meal_slot": "brunch",
        })
        assert response.status_code == 422

    def test_rejects_absurd_targets(self, client):
        response = client.post("/recommend", json={
            "calorie_target": 99999, "protein_target": 150,
        })
        assert response.status_code == 422


class TestMealPlan:
    def test_generates_two_month_plan(self, client):
        response = client.post("/mealplan", json={
            "calorie_target": 2400, "protein_target": 150,
            "fitness_goal": "muscle_gain", "weeks": 8, "seed": 42,
        })
        assert response.status_code == 200
        body = response.json()
        assert body["weeks"] == 8
        assert len(body["days"]) == 56
        assert len(body["weekly_summary"]) == 8

    def test_weekly_macros_within_tolerance(self, client):
        body = client.post("/mealplan", json={
            "calorie_target": 2400, "protein_target": 150,
            "fitness_goal": "maintenance", "weeks": 4, "seed": 7,
        }).json()
        for week in body["weekly_summary"]:
            assert abs(week["calorie_error_pct"]) <= 5.0
            assert abs(week["protein_error_pct"]) <= 5.0

    def test_accepts_start_date(self, client):
        body = client.post("/mealplan", json={
            "calorie_target": 2400, "protein_target": 150,
            "weeks": 1, "start_date": "2026-09-01",
        }).json()
        assert body["days"][0]["date"] == "2026-09-01"

    def test_rejects_malformed_start_date(self, client):
        response = client.post("/mealplan", json={
            "calorie_target": 2400, "protein_target": 150,
            "weeks": 1, "start_date": "01/09/2026",
        })
        assert response.status_code == 422

    def test_is_deterministic_for_a_seed(self, client):
        payload = {"calorie_target": 2400, "protein_target": 150, "weeks": 1, "seed": 99}
        first = client.post("/mealplan", json=payload).json()
        second = client.post("/mealplan", json=payload).json()
        assert first["days"] == second["days"]

    def test_rejects_too_many_weeks(self, client):
        response = client.post("/mealplan", json={
            "calorie_target": 2400, "protein_target": 150, "weeks": 100,
        })
        assert response.status_code == 422


class TestDriftCheck:
    def test_no_regeneration_for_small_drift(self, client):
        body = client.post("/drift-check", json={
            "old_calorie_target": 2400, "new_calorie_target": 2450,
            "old_protein_target": 150, "new_protein_target": 152,
        }).json()
        assert body["needs_regeneration"] is False

    def test_regeneration_for_large_drift(self, client):
        body = client.post("/drift-check", json={
            "old_calorie_target": 2400, "new_calorie_target": 2800,
            "old_protein_target": 150, "new_protein_target": 150,
        }).json()
        assert body["needs_regeneration"] is True
        assert body["calorie_drift_pct"] == pytest.approx(16.67, abs=0.01)
