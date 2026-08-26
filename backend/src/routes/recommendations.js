/**
 * FR6 — Personalised Meal Recommendation.
 *
 * Targets come from the user's most recent stored prediction rather than being
 * recomputed per request: recommendations must match the numbers the dashboard
 * is showing, and a fresh prediction on every page load would make them drift.
 */
import { Router } from 'express';

import { requireAuth } from '../middleware/auth.js';
import { recommendationQuerySchema, validateQuery } from '../middleware/validate.js';
import { mlClient } from '../services/mlClient.js';
import { asyncHandler, AppError, NotFoundError } from '../utils/errors.js';

export const recommendationsRouter = Router();

recommendationsRouter.use(requireAuth);

async function latestTargets(db, userId) {
  const { data: prediction, error } = await db
    .from('predictions')
    .select('calorie_target, protein_target')
    .eq('user_id', userId)
    .order('created_at', { ascending: false })
    .limit(1)
    .maybeSingle();

  if (error) throw new AppError(`Failed to read targets: ${error.message}`, 500);
  if (!prediction) {
    throw new NotFoundError('No prediction yet. Call POST /api/predict first.');
  }

  const { data: profile } = await db
    .from('profiles')
    .select('fitness_goal')
    .eq('user_id', userId)
    .maybeSingle();

  return {
    calorie_target: Number(prediction.calorie_target),
    protein_target: Number(prediction.protein_target),
    fitness_goal: profile?.fitness_goal ?? 'maintenance',
  };
}

/**
 * GET /api/recommendations
 *
 * With `?meal_slot=lunch` returns a ranked menu for that slot; without it,
 * returns a composed plan for the whole day.
 */
recommendationsRouter.get(
  '/',
  validateQuery(recommendationQuerySchema),
  asyncHandler(async (req, res) => {
    const { meal_slot: mealSlot, top_n: topN } = req.validatedQuery;
    const targets = await latestTargets(req.db, req.user.id);

    const result = await mlClient.recommend({
      calorie_target: targets.calorie_target,
      protein_target: targets.protein_target,
      fitness_goal: targets.fitness_goal,
      ...(mealSlot ? { meal_slot: mealSlot, top_n: topN } : {}),
    });

    res.json({ targets, ...result });
  }),
);
