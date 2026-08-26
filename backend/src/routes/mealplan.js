/**
 * FR7 — Two-Month Meal Plan Generation.
 *
 * Generating a plan supersedes the previous one rather than deleting it: a
 * partial unique index in the schema allows only one `active` plan per user,
 * while archived plans stay queryable so the user keeps their history.
 */
import { Router } from 'express';

import { requireAuth } from '../middleware/auth.js';
import { mealPlanSchema, validateBody } from '../middleware/validate.js';
import { mlClient } from '../services/mlClient.js';
import { asyncHandler, AppError, NotFoundError } from '../utils/errors.js';

export const mealPlanRouter = Router();

mealPlanRouter.use(requireAuth);

/** Flatten the ML service's nested plan into `meal_plan_items` rows. */
function toItemRows(planId, plan) {
  const rows = [];
  for (const day of plan.days) {
    for (const [slot, slotPlan] of Object.entries(day.slots)) {
      slotPlan.items.forEach((item, position) => {
        rows.push({
          meal_plan_id: planId,
          week_number: day.week,
          day_index: day.day_index,
          day_of_week: day.day_of_week,
          plan_date: day.date,
          meal_slot: slot,
          position,
          food_id: item.food_id,
          food_name: item.name,
          servings: item.servings,
          calories: item.calories,
          protein_g: item.protein_g,
          carbs_g: item.carbs_g,
          fat_g: item.fat_g,
          fiber_g: item.fiber_g,
        });
      });
    }
  }
  return rows;
}

/** POST /api/mealplan/generate */
mealPlanRouter.post(
  '/generate',
  validateBody(mealPlanSchema),
  asyncHandler(async (req, res) => {
    const { data: prediction, error: predictionError } = await req.db
      .from('predictions')
      .select('calorie_target, protein_target')
      .eq('user_id', req.user.id)
      .order('created_at', { ascending: false })
      .limit(1)
      .maybeSingle();

    if (predictionError) {
      throw new AppError(`Failed to read targets: ${predictionError.message}`, 500);
    }
    if (!prediction) {
      throw new NotFoundError('No prediction yet. Call POST /api/predict first.');
    }

    const { data: profile } = await req.db
      .from('profiles')
      .select('fitness_goal')
      .eq('user_id', req.user.id)
      .maybeSingle();

    const goal = profile?.fitness_goal ?? 'maintenance';
    const startDate = req.body.start_date ?? new Date().toISOString().slice(0, 10);

    const plan = await mlClient.mealPlan({
      calorie_target: Number(prediction.calorie_target),
      protein_target: Number(prediction.protein_target),
      fitness_goal: goal,
      weeks: req.body.weeks,
      start_date: startDate,
      seed: req.body.seed ?? Math.floor(Math.random() * 100000),
    });

    // Retire the previous active plan before inserting the new one, so the
    // one-active-plan-per-user index cannot be violated.
    const { error: supersedeError } = await req.db
      .from('meal_plans')
      .update({ status: 'superseded' })
      .eq('user_id', req.user.id)
      .eq('status', 'active');

    if (supersedeError) {
      throw new AppError(`Failed to supersede old plan: ${supersedeError.message}`, 500);
    }

    const { data: planRow, error: planError } = await req.db
      .from('meal_plans')
      .insert({
        user_id: req.user.id,
        week_count: plan.weeks,
        start_date: startDate,
        status: 'active',
        calorie_target: plan.calorie_target,
        protein_target: plan.protein_target,
        fitness_goal: plan.goal,
        seed: plan.seed,
      })
      .select()
      .single();

    if (planError) throw new AppError(`Failed to create plan: ${planError.message}`, 500);

    const rows = toItemRows(planRow.id, plan);
    // Chunked insert: 224+ rows in one statement risks hitting request limits.
    const CHUNK = 200;
    for (let index = 0; index < rows.length; index += CHUNK) {
      const { error: itemsError } = await req.db
        .from('meal_plan_items')
        .insert(rows.slice(index, index + CHUNK));
      if (itemsError) {
        throw new AppError(`Failed to save plan items: ${itemsError.message}`, 500);
      }
    }

    res.status(201).json({
      meal_plan: planRow,
      item_count: rows.length,
      weekly_summary: plan.weekly_summary,
      variety: plan.variety,
      within_tolerance: plan.within_tolerance,
    });
  }),
);

/** GET /api/mealplan/active — the plan the dashboard displays. */
mealPlanRouter.get(
  '/active',
  asyncHandler(async (req, res) => {
    const { data: plan, error } = await req.db
      .from('meal_plans')
      .select('*')
      .eq('user_id', req.user.id)
      .eq('status', 'active')
      .maybeSingle();

    if (error) throw new AppError(`Failed to read plan: ${error.message}`, 500);
    if (!plan) throw new NotFoundError('No active meal plan.');

    const { data: items, error: itemsError } = await req.db
      .from('meal_plan_items')
      .select('*')
      .eq('meal_plan_id', plan.id)
      .order('week_number')
      .order('day_index')
      .order('meal_slot')
      .order('position');

    if (itemsError) throw new AppError(`Failed to read plan items: ${itemsError.message}`, 500);
    res.json({ meal_plan: plan, items: items ?? [] });
  }),
);

/** GET /api/mealplan/:id */
mealPlanRouter.get(
  '/:id',
  asyncHandler(async (req, res) => {
    const { data: plan, error } = await req.db
      .from('meal_plans')
      .select('*')
      .eq('id', req.params.id)
      .maybeSingle();

    // RLS already blocks other users' plans; a missing row is simply a 404.
    if (error) throw new AppError(`Failed to read plan: ${error.message}`, 500);
    if (!plan) throw new NotFoundError('Meal plan not found.');

    const { data: items } = await req.db
      .from('meal_plan_items')
      .select('*')
      .eq('meal_plan_id', plan.id)
      .order('week_number')
      .order('day_index')
      .order('meal_slot')
      .order('position');

    res.json({ meal_plan: plan, items: items ?? [] });
  }),
);
