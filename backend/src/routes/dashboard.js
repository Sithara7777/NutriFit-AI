/**
 * FR10 — Dashboard and Reporting.
 *
 * One aggregated call so the dashboard renders in a single round-trip instead
 * of five. Returns everything the proposal names: current BMI and category,
 * calorie/protein targets plus their history, this week's meal plan at a
 * glance, and the weight trend series.
 */
import { Router } from 'express';

import { requireAuth } from '../middleware/auth.js';
import { asyncHandler, AppError } from '../utils/errors.js';
import { round1 } from '../utils/nutrition.js';

export const dashboardRouter = Router();

dashboardRouter.use(requireAuth);

/** Which plan week covers today, given the plan's start date? */
function currentWeekNumber(startDate, weekCount) {
  if (!startDate) return 1;
  const start = new Date(`${startDate}T00:00:00Z`);
  const elapsedDays = Math.floor((Date.now() - start.getTime()) / 86_400_000);
  if (elapsedDays < 0) return 1;
  return Math.min(Math.floor(elapsedDays / 7) + 1, weekCount);
}

/** GET /api/dashboard */
dashboardRouter.get(
  '/',
  asyncHandler(async (req, res) => {
    const userId = req.user.id;

    // Independent reads — issue them concurrently rather than serially.
    const [profileResult, weightResult, predictionResult, planResult] = await Promise.all([
      req.db.from('profiles').select('*').eq('user_id', userId).maybeSingle(),
      req.db
        .from('weight_logs')
        .select('weight_kg, bmi, bmi_category, logged_at')
        .eq('user_id', userId)
        .order('logged_at', { ascending: true })
        .limit(200),
      req.db
        .from('predictions')
        .select('calorie_target, protein_target, bmr, tdee, source, created_at')
        .eq('user_id', userId)
        .order('created_at', { ascending: true })
        .limit(200),
      req.db
        .from('meal_plans')
        .select('*')
        .eq('user_id', userId)
        .eq('status', 'active')
        .maybeSingle(),
    ]);

    if (profileResult.error) {
      throw new AppError(`Failed to read profile: ${profileResult.error.message}`, 500);
    }

    const profile = profileResult.data ?? null;
    const weightLogs = weightResult.data ?? [];
    const predictions = predictionResult.data ?? [];
    const activePlan = planResult.data ?? null;

    const latestWeight = weightLogs.at(-1) ?? null;
    const latestPrediction = predictions.at(-1) ?? null;

    // --- this week's plan ------------------------------------------------
    let currentWeek = null;
    if (activePlan) {
      const weekNumber = currentWeekNumber(activePlan.start_date, activePlan.week_count);
      const { data: items } = await req.db
        .from('meal_plan_items')
        .select('week_number, day_index, day_of_week, plan_date, meal_slot, position, food_name, servings, calories, protein_g, carbs_g, fat_g')
        .eq('meal_plan_id', activePlan.id)
        .eq('week_number', weekNumber)
        .order('day_index')
        .order('meal_slot')
        .order('position');

      currentWeek = { week_number: weekNumber, items: items ?? [] };
    }

    res.json({
      profile,
      bmi: latestWeight
        ? {
            value: Number(latestWeight.bmi),
            category: latestWeight.bmi_category,
            weight_kg: Number(latestWeight.weight_kg),
            logged_at: latestWeight.logged_at,
          }
        : null,
      targets: latestPrediction
        ? {
            calorie_target: Number(latestPrediction.calorie_target),
            protein_target: Number(latestPrediction.protein_target),
            bmr: latestPrediction.bmr == null ? null : Number(latestPrediction.bmr),
            tdee: latestPrediction.tdee == null ? null : Number(latestPrediction.tdee),
            source: latestPrediction.source,
            created_at: latestPrediction.created_at,
          }
        : null,
      target_history: predictions.map((row) => ({
        created_at: row.created_at,
        calorie_target: Number(row.calorie_target),
        protein_target: Number(row.protein_target),
      })),
      weight_trend: weightLogs.map((row) => ({
        logged_at: row.logged_at,
        weight_kg: Number(row.weight_kg),
        bmi: Number(row.bmi),
      })),
      progress_summary:
        weightLogs.length > 1
          ? {
              entries: weightLogs.length,
              change_kg: round1(
                Number(weightLogs.at(-1).weight_kg) - Number(weightLogs.at(0).weight_kg),
              ),
            }
          : null,
      active_meal_plan: activePlan,
      current_week: currentWeek,
      onboarding_complete: Boolean(profile && latestWeight && latestPrediction),
    });
  }),
);
