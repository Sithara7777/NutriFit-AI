"""Two-month (eight-week) meal plan generator -- Implementation Plan section 5.

Generates 8 weeks x 7 days x 4 slots of composed meals under two constraints:

**Variety.**  A food may not repeat inside a rolling five-day window.  Without
this the score-weighted sampler still converges on a handful of high-scoring
foods and the plan reads as copy-pasted.

**Weekly macro accuracy.**  After each week the delivered daily average is
compared against the target.  If either calories or protein drift beyond +/-5 %,
a greedy repair pass rebuilds the worst-fitting days with a corrected budget.
This is deliberately greedy rather than a global optimisation: a full
constraint solve over 224 slots would be far slower and is unnecessary to hold
+/-5 %.  The trade-off is stated here and belongs in the Limitations chapter.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from .nutrition import MEAL_SLOTS
from .recommender import MealRecommender, SlotPlan

logger = logging.getLogger(__name__)

DAYS_OF_WEEK = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

DEFAULT_WEEKS = 8
VARIETY_WINDOW_DAYS = 5
MACRO_TOLERANCE = 0.05      # +/-5 % weekly drift budget
MAX_REPAIR_PASSES = 6


@dataclass
class PlannedDay:
    week: int
    day_index: int              # 0-6 within the week
    day_of_week: str
    plan_date: date | None
    slots: dict[str, SlotPlan] = field(default_factory=dict)

    @property
    def calories(self) -> float:
        return sum(plan.calories for plan in self.slots.values())

    @property
    def protein_g(self) -> float:
        return sum(plan.protein_g for plan in self.slots.values())

    @property
    def carbs_g(self) -> float:
        return sum(plan.carbs_g for plan in self.slots.values())

    @property
    def fat_g(self) -> float:
        return sum(plan.fat_g for plan in self.slots.values())

    @property
    def food_ids(self) -> set[str]:
        ids: set[str] = set()
        for plan in self.slots.values():
            ids |= plan.food_ids
        return ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "week": self.week,
            "day_index": self.day_index,
            "day_of_week": self.day_of_week,
            "date": self.plan_date.isoformat() if self.plan_date else None,
            "totals": {
                "calories": round(self.calories, 1),
                "protein_g": round(self.protein_g, 1),
                "carbs_g": round(self.carbs_g, 1),
                "fat_g": round(self.fat_g, 1),
            },
            "slots": {slot: plan.to_dict() for slot, plan in self.slots.items()},
        }


@dataclass
class MealPlan:
    weeks: int
    goal: str
    calorie_target: float
    protein_target: float
    days: list[PlannedDay]
    start_date: date | None = None
    repairs_applied: int = 0
    seed: int = 42

    # ------------------------------------------------------------- summary
    def weekly_summary(self) -> pd.DataFrame:
        rows = []
        for week in range(1, self.weeks + 1):
            week_days = [d for d in self.days if d.week == week]
            if not week_days:
                continue
            mean_calories = float(np.mean([d.calories for d in week_days]))
            mean_protein = float(np.mean([d.protein_g for d in week_days]))
            rows.append({
                "week": week,
                "mean_daily_calories": round(mean_calories, 1),
                "calorie_target": round(self.calorie_target, 1),
                "calorie_error_pct": round(
                    (mean_calories - self.calorie_target) / self.calorie_target * 100, 2
                ),
                "mean_daily_protein_g": round(mean_protein, 1),
                "protein_target_g": round(self.protein_target, 1),
                "protein_error_pct": round(
                    (mean_protein - self.protein_target) / self.protein_target * 100, 2
                ),
                "unique_foods": len({fid for d in week_days for fid in d.food_ids}),
            })
        return pd.DataFrame(rows)

    def variety_report(self) -> dict[str, Any]:
        all_ids = [
            item.food_id
            for day in self.days
            for plan in day.slots.values()
            for item in plan.items
        ]
        return {
            "total_items": len(all_ids),
            "unique_foods": len(set(all_ids)),
            "variety_ratio": round(len(set(all_ids)) / max(len(all_ids), 1), 4),
            "max_repeats_of_one_food": (
                int(pd.Series(all_ids).value_counts().max()) if all_ids else 0
            ),
            "violations_of_5_day_rule": self.count_variety_violations(),
        }

    def count_variety_violations(self, window: int = VARIETY_WINDOW_DAYS) -> int:
        """Number of times a food repeats inside the rolling window."""
        violations = 0
        history: deque[set[str]] = deque(maxlen=window)
        for day in self.days:
            recent: set[str] = set()
            for previous in history:
                recent |= previous
            violations += len(day.food_ids & recent)
            history.append(day.food_ids)
        return violations

    def within_tolerance(self, tolerance: float = MACRO_TOLERANCE) -> bool:
        weekly = self.weekly_summary()
        if weekly.empty:
            return False
        return bool(
            (weekly["calorie_error_pct"].abs().max() <= tolerance * 100)
            and (weekly["protein_error_pct"].abs().max() <= tolerance * 100)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "weeks": self.weeks,
            "goal": self.goal,
            "calorie_target": round(self.calorie_target, 1),
            "protein_target": round(self.protein_target, 1),
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "repairs_applied": self.repairs_applied,
            "seed": self.seed,
            "variety": self.variety_report(),
            "weekly_summary": self.weekly_summary().to_dict(orient="records"),
            "days": [day.to_dict() for day in self.days],
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Flat (week, day, slot, food) table -- the DB insert shape."""
        rows = []
        for day in self.days:
            for slot in MEAL_SLOTS:
                plan = day.slots.get(slot)
                if plan is None:
                    continue
                for position, item in enumerate(plan.items):
                    rows.append({
                        "week": day.week,
                        "day_index": day.day_index,
                        "day_of_week": day.day_of_week,
                        "date": day.plan_date,
                        "meal_slot": slot,
                        "position": position,
                        "food_id": item.food_id,
                        "name": item.name,
                        "servings": item.servings,
                        "calories": round(item.calories, 1),
                        "protein_g": round(item.protein_g, 1),
                        "carbs_g": round(item.carbs_g, 1),
                        "fat_g": round(item.fat_g, 1),
                        "fiber_g": round(item.fiber_g, 1),
                    })
        return pd.DataFrame(rows)


def _build_day(
    recommender: MealRecommender,
    week: int,
    day_index: int,
    plan_date: date | None,
    calorie_target: float,
    protein_target: float,
    goal: str,
    excluded: set[str],
    rng: np.random.Generator,
) -> PlannedDay:
    slots = recommender.daily_plan(
        calorie_target=calorie_target,
        protein_target=protein_target,
        goal=goal,
        exclude=excluded,
        rng=rng,
    )
    return PlannedDay(
        week=week,
        day_index=day_index,
        day_of_week=DAYS_OF_WEEK[day_index % 7],
        plan_date=plan_date,
        slots=slots,
    )


def _window_exclusions(history: "deque[set[str]]") -> set[str]:
    excluded: set[str] = set()
    for previous in history:
        excluded |= previous
    return excluded


def generate_meal_plan(
    recommender: MealRecommender,
    calorie_target: float,
    protein_target: float,
    goal: str = "maintenance",
    weeks: int = DEFAULT_WEEKS,
    start_date: date | None = None,
    seed: int = 42,
    tolerance: float = MACRO_TOLERANCE,
    max_repair_passes: int = MAX_REPAIR_PASSES,
) -> MealPlan:
    """Generate a ``weeks``-week plan meeting the variety and macro constraints."""
    rng = np.random.default_rng(seed)
    history: deque[set[str]] = deque(maxlen=VARIETY_WINDOW_DAYS)
    days: list[PlannedDay] = []
    repairs = 0

    for week in range(1, weeks + 1):
        week_days: list[PlannedDay] = []
        # Foods used at the tail of the previous week. Repairs must avoid these
        # too, otherwise rebuilding an early day of this week can reintroduce a
        # food eaten two days ago in the previous week.
        carried_in = _window_exclusions(history)

        for day_index in range(7):
            absolute_day = (week - 1) * 7 + day_index
            plan_date = start_date + timedelta(days=absolute_day) if start_date else None

            day = _build_day(
                recommender, week, day_index, plan_date,
                calorie_target, protein_target, goal,
                _window_exclusions(history), rng,
            )
            week_days.append(day)
            history.append(day.food_ids)

        # ---------------- weekly macro check + greedy repair -------------
        for _ in range(max_repair_passes):
            mean_calories = float(np.mean([d.calories for d in week_days]))
            mean_protein = float(np.mean([d.protein_g for d in week_days]))
            calorie_drift = (mean_calories - calorie_target) / calorie_target
            protein_drift = (mean_protein - protein_target) / protein_target

            if abs(calorie_drift) <= tolerance and abs(protein_drift) <= tolerance:
                break

            # Rebuild the single worst-fitting day with a corrected budget.
            # Correcting the *request* (rather than swapping meals directly)
            # keeps the repair inside the same recommendation logic, so a
            # repaired day is generated exactly like any other day.
            deviations = [
                abs(d.calories - calorie_target) / calorie_target
                + abs(d.protein_g - protein_target) / protein_target
                for d in week_days
            ]
            worst = int(np.argmax(deviations))

            corrected_calories = float(np.clip(
                calorie_target * (1.0 - calorie_drift),
                calorie_target * 0.7, calorie_target * 1.3,
            ))
            corrected_protein = float(np.clip(
                protein_target * (1.0 - protein_drift),
                protein_target * 0.7, protein_target * 1.3,
            ))

            # Exclude only the five-day neighbourhood of the day being
            # rebuilt -- excluding the whole week would starve the candidate
            # pool and force the recommender to ignore exclusions entirely.
            low = max(0, worst - VARIETY_WINDOW_DAYS)
            neighbourhood: set[str] = set()
            for position in range(low, min(len(week_days), worst + VARIETY_WINDOW_DAYS + 1)):
                if position != worst:
                    neighbourhood |= week_days[position].food_ids
            # Days near the start of the week are still inside the previous
            # week's five-day window.
            if worst < VARIETY_WINDOW_DAYS:
                neighbourhood |= carried_in

            old = week_days[worst]
            week_days[worst] = _build_day(
                recommender, week, old.day_index, old.plan_date,
                corrected_calories, corrected_protein, goal, neighbourhood, rng,
            )
            repairs += 1

        days.extend(week_days)

        # Reseed the rolling window from the (possibly repaired) week so the
        # variety constraint carries correctly across the week boundary.
        history = deque(
            [d.food_ids for d in week_days[-VARIETY_WINDOW_DAYS:]],
            maxlen=VARIETY_WINDOW_DAYS,
        )

    plan = MealPlan(
        weeks=weeks,
        goal=goal,
        calorie_target=calorie_target,
        protein_target=protein_target,
        days=days,
        start_date=start_date,
        repairs_applied=repairs,
        seed=seed,
    )
    variety = plan.variety_report()
    logger.info(
        "Generated %d-week plan: %d days, %d repairs, variety %.2f, tolerance met: %s",
        weeks, len(days), repairs, variety["variety_ratio"], plan.within_tolerance(tolerance),
    )
    return plan


def needs_regeneration(
    old_calories: float,
    new_calories: float,
    old_protein: float,
    new_protein: float,
    threshold: float = 0.07,
) -> tuple[bool, dict[str, float]]:
    """Should the user be prompted to regenerate their plan? (Plan section 6.)

    Returns ``(needs_regeneration, drift_detail)``.  The caller *prompts* --
    it never silently overwrites an active plan, because a plan the user is
    mid-way through following is their data, not ours.
    """
    calorie_drift = abs(new_calories - old_calories) / max(old_calories, 1.0)
    protein_drift = abs(new_protein - old_protein) / max(old_protein, 1.0)
    return (
        calorie_drift > threshold or protein_drift > threshold,
        {
            "calorie_drift_pct": round(calorie_drift * 100, 2),
            "protein_drift_pct": round(protein_drift * 100, 2),
            "threshold_pct": round(threshold * 100, 2),
        },
    )


__all__ = [
    "DAYS_OF_WEEK",
    "DEFAULT_WEEKS",
    "VARIETY_WINDOW_DAYS",
    "MACRO_TOLERANCE",
    "PlannedDay",
    "MealPlan",
    "generate_meal_plan",
    "needs_regeneration",
]
