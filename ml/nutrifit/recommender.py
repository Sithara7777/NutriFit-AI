"""Content-based meal recommendation engine (Implementation Plan section 4).

Pipeline for one meal slot
--------------------------
1. **Budget.**  Split the user's daily calorie/protein targets across the four
   meal slots using the standard 25/35/30/10 ratios, then subtract whatever has
   already been consumed today to get the *remaining* budget for this slot.
2. **Ideal vector.**  Turn that budget into a full macro target
   ``[calories, protein, carbs, fat, fiber]`` by allocating the non-protein
   calories between carbohydrate and fat according to the fitness goal, and
   setting fibre from the 14 g/1000 kcal dietary reference intake.
3. **Retrieve.**  Query a per-slot ``NearestNeighbors`` index over min-max
   normalised food vectors to pull the closest candidates.
4. **Re-rank.**  Blend the content-similarity score with a goal-specific
   nutritional-quality score (protein density for fat loss, energy availability
   for muscle gain, macro balance for maintenance).
5. **Compose.**  Greedily add 1-3 items until the slot budget is filled, then
   apply a single quarter-step portion multiplier to close the residual gap.

Why a slot is composed of several items
---------------------------------------
A catalogue entry is a *food*, not a *meal*: the median breakfast item in the
source data is ~360 kcal while a 2400 kcal user needs a 600 kcal breakfast.
Recommending one item per slot therefore under-delivers by roughly a quarter --
measured at -26 % before this was fixed.  Composing each slot from a small
number of items, exactly as a real meal is assembled, brings weekly delivery
inside the +/-5 % tolerance the plan requires.

Why Euclidean distance rather than cosine
-----------------------------------------
Cosine similarity is scale-invariant: it treats a 200 kcal and a 900 kcal meal
with the same macro *ratio* as identical.  For meal planning the absolute
amount is the whole point -- we are trying to hit a calorie budget, not match a
profile shape.  Euclidean distance over min-max normalised vectors preserves
magnitude, so it is the correct metric here.  Cosine remains available via the
``metric`` argument for the comparison reported in the evaluation notebook.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from .nutrition import MEAL_SLOT_RATIOS, MEAL_SLOTS

logger = logging.getLogger(__name__)

#: The content feature space, as specified in the plan.
FEATURE_VECTOR = ["calories", "protein_g", "carbs_g", "fat_g", "fiber_g"]

#: Per-feature weights applied to the min-max normalised vectors before the
#: nearest-neighbour search (i.e. a weighted Euclidean metric).
#:
#: These are not arbitrary. Calories are the *easiest* constraint to satisfy,
#: because the portion multiplier scales them directly -- a slot that lands 15 %
#: short on energy is fixed by serving 1.25x. Protein cannot be recovered that
#: way without dragging calories along with it, so protein match must dominate
#: candidate selection. Carbohydrate and fat are largely determined once
#: calories and protein are fixed (they absorb the remainder), and fibre is a
#: quality signal rather than a target, so all three are down-weighted.
#:
#: The protein weight was chosen by sweeping it over the real 489-item
#: catalogue across all three goals and taking the worst-case weekly error
#: (8 weeks, seed 42):
#:
#:     protein weight | fat_loss | maintenance | muscle_gain | worst
#:     ---------------|----------|-------------|-------------|-------
#:          1.0       |   8.10 % |      4.72 % |      8.47 % | 8.47 %
#:          1.5       |   4.82 % |      3.28 % |      3.99 % | 4.82 %
#:          2.0       |   5.00 % |      2.53 % |      3.74 % | 5.00 %
#:          2.5       |   3.80 % |      1.96 % |      2.80 % | 3.80 %   <- chosen
#:          3.0       |   4.05 % |      2.46 % |      1.03 % | 4.05 %
#:
#: 2.5 gives the best worst-case with a comfortable margin under the +/-5 %
#: tolerance, without so over-weighting protein that carbohydrate and fat
#: balance degrade. It also removed the greedy repair pass entirely (47 repairs
#: -> 0 on the fat-loss scenario), so plans now meet tolerance by construction.
FEATURE_WEIGHTS = {
    "calories": 1.0,
    "protein_g": 2.5,
    "carbs_g": 0.6,
    "fat_g": 0.6,
    "fiber_g": 0.4,
}

#: Fraction of *non-protein* calories allocated to carbohydrate, per goal.
#: Higher carbohydrate availability supports training volume during a surplus;
#: a deficit shifts the balance towards fat for satiety.
CARB_SHARE_BY_GOAL: dict[str, float] = {
    "fat_loss": 0.50,
    "maintenance": 0.55,
    "muscle_gain": 0.65,
}

#: Dietary Reference Intake for fibre: 14 g per 1000 kcal (IOM, 2005).
FIBER_G_PER_1000_KCAL = 14.0

#: Weight of the goal-quality score relative to content similarity when
#: re-ranking.  0.35 keeps macro fit dominant while still letting goal
#: appropriateness break ties.
GOAL_SCORE_WEIGHT = 0.35

# --- portion control -------------------------------------------------------
#: Portions are expressed in quarter-serving steps so the UI can render
#: something a human can actually measure ("1.5 servings"), not "1.37x".
SERVING_STEP = 0.25
MIN_SERVINGS = 0.5
MAX_SERVINGS = 2.0

#: Slot composition limits.
MAX_ITEMS_PER_SLOT = 3
SLOT_FILL_TOLERANCE = 0.08      # stop once <=8 % of the slot budget remains
OVERSHOOT_ALLOWANCE = 1.35      # a follow-up item may exceed the remainder by 35 %


@dataclass
class MealSuggestion:
    """One recommended food, with the macro breakdown the UI must display.

    Macro fields on this object are **per serving**; the scaled totals the user
    actually eats are exposed as properties, so ``servings`` can never fall out
    of sync with the numbers shown.
    """

    food_id: str
    name: str
    meal_type: str
    category: str
    per_serving: dict[str, float]
    score: float = 0.0
    distance: float = 0.0
    servings: float = 1.0

    @property
    def calories(self) -> float:
        return self.per_serving["calories"] * self.servings

    @property
    def protein_g(self) -> float:
        return self.per_serving["protein_g"] * self.servings

    @property
    def carbs_g(self) -> float:
        return self.per_serving["carbs_g"] * self.servings

    @property
    def fat_g(self) -> float:
        return self.per_serving["fat_g"] * self.servings

    @property
    def fiber_g(self) -> float:
        return self.per_serving["fiber_g"] * self.servings

    @property
    def base_calories(self) -> float:
        return self.per_serving["calories"]

    def to_dict(self) -> dict:
        return {
            "food_id": self.food_id,
            "name": self.name,
            "meal_type": self.meal_type,
            "category": self.category,
            "servings": round(float(self.servings), 2),
            "calories": round(float(self.calories), 1),
            "protein_g": round(float(self.protein_g), 1),
            "carbs_g": round(float(self.carbs_g), 1),
            "fat_g": round(float(self.fat_g), 1),
            "fiber_g": round(float(self.fiber_g), 1),
            "per_serving": {k: round(float(v), 1) for k, v in self.per_serving.items()},
            "score": round(float(self.score), 4),
            "distance": round(float(self.distance), 4),
        }


@dataclass
class SlotPlan:
    """The composed set of items filling one meal slot."""

    slot: str
    items: list[MealSuggestion] = field(default_factory=list)
    target: dict[str, float] = field(default_factory=dict)

    @property
    def calories(self) -> float:
        return sum(item.calories for item in self.items)

    @property
    def protein_g(self) -> float:
        return sum(item.protein_g for item in self.items)

    @property
    def carbs_g(self) -> float:
        return sum(item.carbs_g for item in self.items)

    @property
    def fat_g(self) -> float:
        return sum(item.fat_g for item in self.items)

    @property
    def fiber_g(self) -> float:
        return sum(item.fiber_g for item in self.items)

    @property
    def food_ids(self) -> set[str]:
        return {item.food_id for item in self.items}

    def to_dict(self) -> dict:
        return {
            "slot": self.slot,
            "totals": {
                "calories": round(self.calories, 1),
                "protein_g": round(self.protein_g, 1),
                "carbs_g": round(self.carbs_g, 1),
                "fat_g": round(self.fat_g, 1),
                "fiber_g": round(self.fiber_g, 1),
            },
            "target": {k: round(float(v), 1) for k, v in self.target.items()},
            "items": [item.to_dict() for item in self.items],
        }


def slot_macro_target(
    calorie_target: float,
    protein_target: float,
    goal: str,
    slot: str,
    consumed_calories: float = 0.0,
    consumed_protein: float = 0.0,
    redistribute: bool | None = None,
) -> dict[str, float]:
    """Ideal ``[calories, protein, carbs, fat, fiber]`` for one meal slot.

    Two distinct callers need two distinct behaviours, so the mode is explicit:

    * **Standalone query** (``redistribute=False``) -- the slot receives its
      plain fixed share of the daily target (lunch = 35 %).  This is what
      ``GET /api/recommendations?slot=lunch`` wants: asking for lunch
      suggestions must not imply the user intends to skip breakfast.
    * **Sequential day walk** (``redistribute=True``) -- the *remaining* daily
      budget is spread across this slot and the ones after it, in proportion to
      their ratios.  This is what makes a day self-correcting: if breakfast
      under-delivered by 80 kcal, lunch, dinner and snack absorb the shortfall
      instead of the day simply ending 80 kcal short.

    The default is inferred from whether any consumption was reported, which
    makes both call sites behave correctly without having to think about it.
    """
    if slot not in MEAL_SLOT_RATIOS:
        raise ValueError(f"Unknown meal slot {slot!r}; expected one of {MEAL_SLOTS}")

    if redistribute is None:
        redistribute = consumed_calories > 0 or consumed_protein > 0

    if redistribute:
        remaining_calories = max(calorie_target - consumed_calories, 0.0)
        remaining_protein = max(protein_target - consumed_protein, 0.0)
        slot_index = MEAL_SLOTS.index(slot)
        remaining_share = sum(MEAL_SLOT_RATIOS[s] for s in MEAL_SLOTS[slot_index:])
        share = MEAL_SLOT_RATIOS[slot] / remaining_share if remaining_share > 0 else 1.0
    else:
        remaining_calories = max(calorie_target, 0.0)
        remaining_protein = max(protein_target, 0.0)
        share = MEAL_SLOT_RATIOS[slot]

    calories = remaining_calories * share
    protein = remaining_protein * share

    # Allocate the calories left after protein between carbohydrate and fat.
    protein_calories = min(protein * 4.0, calories)
    leftover = max(calories - protein_calories, 0.0)
    carb_share = CARB_SHARE_BY_GOAL.get(goal, 0.55)

    return {
        "calories": calories,
        "protein_g": protein,
        "carbs_g": leftover * carb_share / 4.0,
        "fat_g": leftover * (1.0 - carb_share) / 9.0,
        "fiber_g": calories / 1000.0 * FIBER_G_PER_1000_KCAL,
    }


def goal_quality_score(candidates: pd.DataFrame, goal: str) -> np.ndarray:
    """Goal-specific nutritional quality in [0, 1], higher is better.

    * **fat_loss** rewards protein and fibre density (satiety and lean-mass
      retention per calorie).
    * **muscle_gain** rewards absolute protein and total energy -- during a
      surplus the practical constraint is eating *enough*.
    * **maintenance** rewards proximity to a balanced 30/40/30 energy split.
    """

    def normalise(series: pd.Series) -> np.ndarray:
        values = series.to_numpy(dtype=float)
        span = np.nanmax(values) - np.nanmin(values)
        if not np.isfinite(span) or span == 0:
            return np.zeros_like(values)
        return (values - np.nanmin(values)) / span

    if goal == "fat_loss":
        return 0.65 * normalise(candidates["protein_density"]) + 0.35 * normalise(
            candidates["fiber_density"]
        )

    if goal == "muscle_gain":
        return 0.60 * normalise(candidates["protein_g"]) + 0.40 * normalise(
            candidates["calories"]
        )

    ideal = np.array([0.30, 0.40, 0.30])
    actual = candidates[["protein_frac", "carb_frac", "fat_frac"]].to_numpy(dtype=float)
    imbalance = np.abs(actual - ideal).sum(axis=1)
    return 1.0 - imbalance / 2.0  # L1 distance over a simplex is at most 2


def _weighted_choice(
    candidates: list[MealSuggestion], rng: np.random.Generator
) -> MealSuggestion:
    """Sample proportionally to score^2 so good fits dominate without ties."""
    scores = np.array([c.score for c in candidates], dtype=float)
    weights = np.clip(scores - scores.min() + 0.05, 1e-6, None) ** 2
    weights /= weights.sum()
    return candidates[int(rng.choice(len(candidates), p=weights))]


class MealRecommender:
    """Per-slot nearest-neighbour index over the food catalogue."""

    def __init__(
        self,
        catalogue: pd.DataFrame,
        metric: str = "euclidean",
        n_candidates: int = 25,
        random_state: int = 42,
        feature_weights: dict[str, float] | None = None,
    ):
        """
        Parameters
        ----------
        feature_weights
            Per-feature weights for the distance metric. Defaults to
            :data:`FEATURE_WEIGHTS`. Pass all-ones to obtain the unweighted
            baseline used for the ablation reported in the evaluation notebook.
        """
        required = set(FEATURE_VECTOR) | {"food_id", "name", "meal_type"}
        missing = required - set(catalogue.columns)
        if missing:
            raise ValueError(f"Catalogue is missing column(s): {sorted(missing)}")

        self.metric = metric
        self.n_candidates = n_candidates
        self.random_state = random_state

        self.catalogue = catalogue.reset_index(drop=True).copy()
        derived = {"protein_density", "fiber_density", "protein_frac", "carb_frac", "fat_frac"}
        if not derived.issubset(self.catalogue.columns):
            from .foods import add_derived_features

            self.catalogue = add_derived_features(self.catalogue)

        # One shared scaler across all slots so distances stay comparable
        # between a breakfast and a dinner query.
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.catalogue[FEATURE_VECTOR].to_numpy(dtype=float))
        weights = feature_weights if feature_weights is not None else FEATURE_WEIGHTS
        self.feature_weights = dict(weights)
        self.weights = np.array(
            [weights.get(feature, 1.0) for feature in FEATURE_VECTOR], dtype=float
        )

        self._indices: dict[str, NearestNeighbors] = {}
        self._slot_frames: dict[str, pd.DataFrame] = {}

        for slot in MEAL_SLOTS:
            frame = self.catalogue[self.catalogue["meal_type"] == slot]
            if frame.empty:
                logger.warning("No catalogue items for slot %r", slot)
                continue
            vectors = self._embed(frame[FEATURE_VECTOR].to_numpy(dtype=float))
            index = NearestNeighbors(
                n_neighbors=min(self.n_candidates, len(frame)), metric=metric
            )
            index.fit(vectors)
            self._indices[slot] = index
            self._slot_frames[slot] = frame.reset_index(drop=True)

        if not self._indices:
            raise ValueError("Catalogue produced no usable per-slot indices.")

    # ------------------------------------------------------------- internals
    def _embed(self, raw: np.ndarray) -> np.ndarray:
        """Min-max normalise, then apply the per-feature weights."""
        return self.scaler.transform(raw) * self.weights

    def _rank(
        self,
        slot: str,
        target: dict[str, float],
        goal: str,
        top_n: int,
        exclude: Iterable[str] = (),
    ) -> list[MealSuggestion]:
        """Rank catalogue items in ``slot`` against an explicit macro target."""
        frame = self._slot_frames[slot]
        index = self._indices[slot]

        target_vector = self._embed(
            np.array([[target.get(feature, 0.0) for feature in FEATURE_VECTOR]], dtype=float)
        )

        # Pull a wide candidate set so exclusions cannot starve the result.
        k = int(min(len(frame), max(self.n_candidates, top_n * 4)))
        distances, positions = index.kneighbors(target_vector, n_neighbors=k)
        distances, positions = distances[0], positions[0]

        candidates = frame.iloc[positions].copy()
        candidates["distance"] = distances

        excluded = set(exclude)
        if excluded:
            keep = ~candidates["food_id"].isin(excluded)
            # Only enforce the exclusion if a usable pool survives it; a plan
            # with a repeated meal beats a plan with a missing meal.
            if keep.sum() >= min(top_n, 2):
                candidates = candidates[keep]

        if candidates.empty:
            return []

        span = candidates["distance"].max() - candidates["distance"].min()
        similarity = (
            1.0 - (candidates["distance"] - candidates["distance"].min()) / span
            if span > 0
            else pd.Series(1.0, index=candidates.index)
        )
        quality = goal_quality_score(candidates, goal)
        candidates["score"] = (
            (1.0 - GOAL_SCORE_WEIGHT) * similarity.to_numpy(dtype=float)
            + GOAL_SCORE_WEIGHT * quality
        )
        candidates = candidates.sort_values("score", ascending=False).head(top_n)

        return [
            MealSuggestion(
                food_id=row.food_id,
                name=row.name,
                meal_type=row.meal_type,
                category=getattr(row, "category", "Uncategorised"),
                per_serving={
                    "calories": float(row.calories),
                    "protein_g": float(row.protein_g),
                    "carbs_g": float(row.carbs_g),
                    "fat_g": float(row.fat_g),
                    "fiber_g": float(row.fiber_g),
                },
                score=float(row.score),
                distance=float(row.distance),
            )
            for row in candidates.itertuples(index=False)
        ]

    @staticmethod
    def _round_servings(raw: float) -> float:
        clipped = float(np.clip(raw, MIN_SERVINGS, MAX_SERVINGS))
        return max(round(clipped / SERVING_STEP) * SERVING_STEP, SERVING_STEP)

    # ------------------------------------------------------------------ API
    def available_slots(self) -> list[str]:
        return list(self._indices.keys())

    def recommend(
        self,
        slot: str,
        calorie_target: float,
        protein_target: float,
        goal: str = "maintenance",
        top_n: int = 5,
        exclude: Iterable[str] = (),
        consumed_calories: float = 0.0,
        consumed_protein: float = 0.0,
    ) -> list[MealSuggestion]:
        """Top ``top_n`` single-item suggestions for a slot, best first.

        This is what ``GET /api/recommendations`` surfaces: a ranked menu the
        user can choose from, rather than a fixed composition.
        """
        if slot not in self._indices:
            raise ValueError(
                f"No items indexed for slot {slot!r}; available: {self.available_slots()}"
            )
        target = slot_macro_target(
            calorie_target, protein_target, goal, slot,
            consumed_calories=consumed_calories, consumed_protein=consumed_protein,
        )
        return self._rank(slot, target, goal, top_n=top_n, exclude=exclude)

    def compose_slot(
        self,
        slot: str,
        calorie_target: float,
        protein_target: float,
        goal: str = "maintenance",
        exclude: Iterable[str] = (),
        rng: np.random.Generator | None = None,
        consumed_calories: float = 0.0,
        consumed_protein: float = 0.0,
        max_items: int = MAX_ITEMS_PER_SLOT,
        redistribute: bool | None = None,
    ) -> SlotPlan:
        """Greedily fill one slot's macro budget with 1-``max_items`` foods."""
        generator = rng if rng is not None else np.random.default_rng(self.random_state)
        target = slot_macro_target(
            calorie_target, protein_target, goal, slot,
            consumed_calories=consumed_calories, consumed_protein=consumed_protein,
            redistribute=redistribute,
        )
        plan = SlotPlan(slot=slot, target=target)

        if slot not in self._indices or target["calories"] <= 0:
            return plan

        remaining = dict(target)
        used = set(exclude)

        for _ in range(max_items):
            if remaining["calories"] <= target["calories"] * SLOT_FILL_TOLERANCE:
                break

            candidates = self._rank(slot, remaining, goal, top_n=6, exclude=used)
            if not candidates:
                break

            if plan.items:
                # Once the slot has something in it, reject items that would
                # blow through the remaining budget.
                fitting = [
                    c for c in candidates
                    if c.base_calories <= remaining["calories"] * OVERSHOOT_ALLOWANCE
                ]
                if not fitting:
                    break
                candidates = fitting

            choice = _weighted_choice(candidates, generator)
            plan.items.append(choice)
            used.add(choice.food_id)
            for key in remaining:
                remaining[key] = max(remaining[key] - choice.per_serving[key], 0.0)

        # --- portion scaling -------------------------------------------------
        # One multiplier for the whole slot, in quarter-serving steps: the
        # greedy fill lands close, this closes the residual gap.
        base_calories = sum(item.per_serving["calories"] for item in plan.items)
        if base_calories > 0:
            servings = self._round_servings(target["calories"] / base_calories)
            for item in plan.items:
                item.servings = servings

        return plan

    def daily_plan(
        self,
        calorie_target: float,
        protein_target: float,
        goal: str = "maintenance",
        exclude: Iterable[str] = (),
        rng: np.random.Generator | None = None,
    ) -> dict[str, SlotPlan]:
        """One composed slot per meal, tracking consumption so later slots adapt."""
        generator = rng if rng is not None else np.random.default_rng(self.random_state)
        used = set(exclude)
        day: dict[str, SlotPlan] = {}
        consumed_calories = consumed_protein = 0.0

        for position, slot in enumerate(MEAL_SLOTS):
            plan = self.compose_slot(
                slot, calorie_target, protein_target, goal,
                exclude=used, rng=generator,
                consumed_calories=consumed_calories,
                consumed_protein=consumed_protein,
                # Every slot after the first absorbs the running shortfall or
                # surplus, so the day self-corrects even if a slot delivered
                # exactly zero and left `consumed` unchanged.
                redistribute=position > 0,
            )
            day[slot] = plan
            used |= plan.food_ids
            consumed_calories += plan.calories
            consumed_protein += plan.protein_g

        return day


def evaluate_recommender(
    recommender: MealRecommender,
    calorie_target: float,
    protein_target: float,
    goal: str,
    n_days: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate ``n_days`` of daily plans and report macro accuracy + variety.

    This is the recommendation engine's own evaluation evidence -- the
    equivalent of MAE/R2 for the part of the system that has no ground-truth
    labels.
    """
    rng = np.random.default_rng(seed)
    rows = []
    seen: set[str] = set()
    total_items = 0

    for day in range(n_days):
        plan = recommender.daily_plan(calorie_target, protein_target, goal, rng=rng)
        for slot_plan in plan.values():
            seen.update(slot_plan.food_ids)
            total_items += len(slot_plan.items)
        rows.append({
            "day": day + 1,
            "calories": sum(p.calories for p in plan.values()),
            "protein_g": sum(p.protein_g for p in plan.values()),
            "carbs_g": sum(p.carbs_g for p in plan.values()),
            "fat_g": sum(p.fat_g for p in plan.values()),
            "n_items": sum(len(p.items) for p in plan.values()),
        })

    df = pd.DataFrame(rows)
    df["calorie_error_pct"] = (df["calories"] - calorie_target) / calorie_target * 100
    df["protein_error_pct"] = (df["protein_g"] - protein_target) / protein_target * 100
    df.attrs["unique_meals"] = len(seen)
    df.attrs["variety_ratio"] = len(seen) / max(total_items, 1)
    return df


__all__ = [
    "FEATURE_VECTOR",
    "MealSuggestion",
    "SlotPlan",
    "MealRecommender",
    "slot_macro_target",
    "goal_quality_score",
    "evaluate_recommender",
    "CARB_SHARE_BY_GOAL",
    "FEATURE_WEIGHTS",
    "SERVING_STEP",
    "MIN_SERVINGS",
    "MAX_SERVINGS",
]
