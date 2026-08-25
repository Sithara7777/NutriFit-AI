"""Pydantic request/response models for the ML microservice.

Validation lives here rather than in handler bodies so that malformed input
produces a 422 with a precise field-level message, never a 500 from deep inside
scikit-learn.  The numeric bounds are physiological, not arbitrary -- they are
the same ranges used to clean the training data.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Goal = Literal["fat_loss", "maintenance", "muscle_gain"]
Gender = Literal["Male", "Female"]
MealSlot = Literal["breakfast", "lunch", "dinner", "snack"]


class UserProfile(BaseModel):
    """The inputs the prediction models need."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(..., ge=16, le=80, description="Years")
    gender: Gender
    height_cm: float = Field(..., ge=120, le=230)
    weight_kg: float = Field(..., ge=30, le=250)
    fitness_goal: Goal
    workout_frequency: int = Field(..., ge=0, le=7, description="Training days per week")
    session_duration_h: float = Field(1.25, ge=0.1, le=5.0)
    experience_level: int = Field(2, ge=1, le=3, description="1 novice .. 3 advanced")
    body_fat_pct: float | None = Field(
        None, ge=3, le=60,
        description="Measured body fat %. Deurenberg-estimated when omitted.",
    )

    @field_validator("gender", mode="before")
    @classmethod
    def _normalise_gender(cls, value: object) -> object:
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"male", "m", "man"}:
                return "Male"
            if token in {"female", "f", "woman"}:
                return "Female"
        return value


class PredictionResponse(BaseModel):
    calorie_target: float
    protein_target: float
    bmi: float
    bmi_category: str
    bmr: float
    bmr_equation: str
    tdee: float
    body_fat_pct: float
    body_fat_source: str
    activity_level: str
    activity_multiplier: float
    model_version: str
    #: "model" when a trained pipeline served the request, "formula" when the
    #: service fell back to the deterministic equations.
    source: Literal["model", "formula"]
    formula_reference: dict[str, float] = Field(default_factory=dict)


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calorie_target: float = Field(..., ge=800, le=8000)
    protein_target: float = Field(..., ge=20, le=400)
    fitness_goal: Goal = "maintenance"
    meal_slot: MealSlot | None = None
    top_n: int = Field(5, ge=1, le=25)
    exclude: list[str] = Field(default_factory=list)


class MealPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calorie_target: float = Field(..., ge=800, le=8000)
    protein_target: float = Field(..., ge=20, le=400)
    fitness_goal: Goal = "maintenance"
    weeks: int = Field(8, ge=1, le=12)
    start_date: str | None = Field(None, description="ISO date, e.g. 2026-09-01")
    seed: int = Field(42, ge=0, le=2**31 - 1)


class DriftCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    old_calorie_target: float = Field(..., gt=0)
    new_calorie_target: float = Field(..., gt=0)
    old_protein_target: float = Field(..., gt=0)
    new_protein_target: float = Field(..., gt=0)
    threshold: float = Field(0.07, gt=0, lt=1)


class DriftCheckResponse(BaseModel):
    needs_regeneration: bool
    calorie_drift_pct: float
    protein_drift_pct: float
    threshold_pct: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_version: str
    models_loaded: dict[str, bool]
    catalogue_items: int
    detail: str | None = None
