"""NutriFit-AI ML microservice.

A small stateless FastAPI application that serves the trained scikit-learn
pipelines, the content-based recommendation engine and the eight-week meal-plan
generator.

Architectural note (for the System Design chapter): the proposal specifies that
"the trained model will be integrated with the Node.js backend through APIs".
This service is that integration point.  Keeping the models behind their own
HTTP boundary means the ML code stays in Python (where scikit-learn actually
runs) while the web backend stays in Node, and the two scale independently.

**Security:** this service is never exposed to the internet.  It listens on an
internal address only and is reached exclusively by the Node backend, which
performs Supabase JWT authentication first.  The service therefore holds no
user credentials and stores no PII -- it receives the minimum physiological
fields needed for one prediction and keeps nothing.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

# Make the shared `nutrifit` package importable when running from source.
# In the Docker image the package is installed, so this is a no-op.
_ML_DIR = Path(__file__).resolve().parents[3] / "ml"
if _ML_DIR.exists() and str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

from fastapi import FastAPI, HTTPException, Request, status  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from nutrifit import planner  # noqa: E402
from nutrifit.nutrition import MEAL_SLOTS  # noqa: E402

from .model_store import store  # noqa: E402
from .schemas import (  # noqa: E402
    DriftCheckRequest,
    DriftCheckResponse,
    HealthResponse,
    MealPlanRequest,
    PredictionResponse,
    RecommendationRequest,
    UserProfile,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")
logger = logging.getLogger("ml-service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.load()
    yield
    logger.info("ML service shutting down")


app = FastAPI(
    title="NutriFit-AI ML Service",
    description=(
        "Calorie/protein prediction, content-based meal recommendation and "
        "eight-week meal-plan generation for the NutriFit-AI system."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Only the Node backend calls this service. CORS is restrictive by default;
# localhost origins are allowed so the interactive /docs page works in dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Expose per-request latency -- used as evidence for the Performance NFR."""
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.2f}"
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Consistent error shape; never leak a stack trace to the caller."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_error", "detail": "An unexpected error occurred."},
    )


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> dict:
    """Liveness/readiness probe.

    Returns 200 with ``status="degraded"`` rather than failing when an artefact
    is missing -- the service is still able to answer using the formula
    fallback, and an operator needs to see *why* it is degraded.
    """
    return store.status()


@app.get("/", tags=["ops"])
def root() -> dict:
    return {
        "service": "NutriFit-AI ML Service",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/health", "/predict", "/recommend", "/mealplan", "/drift-check"],
    }


# --------------------------------------------------------------------------
# Prediction  (FR4, FR5)
# --------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(profile: UserProfile) -> dict:
    """Predict daily calorie and protein requirements for one user."""
    return store.predict(profile.model_dump())


# --------------------------------------------------------------------------
# Recommendation  (FR6)
# --------------------------------------------------------------------------
@app.post("/recommend", tags=["recommendation"])
def recommend(request: RecommendationRequest) -> dict:
    """Meal suggestions for one slot, or a full composed day.

    With ``meal_slot`` set, returns a ranked menu for that slot.  Without it,
    returns a composed plan for all four slots, which is what the dashboard
    shows as "today's plan".
    """
    if store.recommender is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation engine unavailable. Run ml/scripts/export_models.py.",
        )

    if request.meal_slot:
        suggestions = store.recommender.recommend(
            slot=request.meal_slot,
            calorie_target=request.calorie_target,
            protein_target=request.protein_target,
            goal=request.fitness_goal,
            top_n=request.top_n,
            exclude=request.exclude,
        )
        return {
            "meal_slot": request.meal_slot,
            "fitness_goal": request.fitness_goal,
            "suggestions": [s.to_dict() for s in suggestions],
        }

    day = store.recommender.daily_plan(
        calorie_target=request.calorie_target,
        protein_target=request.protein_target,
        goal=request.fitness_goal,
        exclude=request.exclude,
    )
    return {
        "fitness_goal": request.fitness_goal,
        "targets": {
            "calories": request.calorie_target,
            "protein_g": request.protein_target,
        },
        "totals": {
            "calories": round(sum(p.calories for p in day.values()), 1),
            "protein_g": round(sum(p.protein_g for p in day.values()), 1),
            "carbs_g": round(sum(p.carbs_g for p in day.values()), 1),
            "fat_g": round(sum(p.fat_g for p in day.values()), 1),
        },
        "slots": {slot: plan.to_dict() for slot, plan in day.items()},
    }


@app.get("/meal-slots", tags=["recommendation"])
def meal_slots() -> dict:
    available = store.recommender.available_slots() if store.recommender else []
    return {"slots": list(MEAL_SLOTS), "available": available}


# --------------------------------------------------------------------------
# Meal plan  (FR7)
# --------------------------------------------------------------------------
@app.post("/mealplan", tags=["meal-plan"])
def mealplan(request: MealPlanRequest) -> dict:
    """Generate a multi-week meal plan (default: the two-month, 8-week plan)."""
    if store.recommender is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation engine unavailable. Run ml/scripts/export_models.py.",
        )

    start = None
    if request.start_date:
        try:
            start = date.fromisoformat(request.start_date)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"start_date must be an ISO date (YYYY-MM-DD): {error}",
            ) from error

    plan = planner.generate_meal_plan(
        recommender=store.recommender,
        calorie_target=request.calorie_target,
        protein_target=request.protein_target,
        goal=request.fitness_goal,
        weeks=request.weeks,
        start_date=start,
        seed=request.seed,
    )
    payload = plan.to_dict()
    payload["within_tolerance"] = plan.within_tolerance()
    return payload


# --------------------------------------------------------------------------
# Progress-driven recalculation  (FR9)
# --------------------------------------------------------------------------
@app.post("/drift-check", response_model=DriftCheckResponse, tags=["progress"])
def drift_check(request: DriftCheckRequest) -> dict:
    """Has the user's requirement moved enough to warrant a new plan?

    The backend uses this after a weight log to decide whether to *prompt* the
    user.  It never silently overwrites an active plan.
    """
    needed, detail = planner.needs_regeneration(
        old_calories=request.old_calorie_target,
        new_calories=request.new_calorie_target,
        old_protein=request.old_protein_target,
        new_protein=request.new_protein_target,
        threshold=request.threshold,
    )
    return {"needs_regeneration": needed, **detail}
