/**
 * FR8 — Progress Monitoring, and FR9 — Recommendation Adjustment.
 *
 * Logging a weight triggers the full adjustment loop:
 *   new weight -> recompute BMI -> re-predict targets -> compare against the
 *   active plan -> *prompt* if the drift exceeds the threshold.
 *
 * The prompt is the important design decision. The system never silently
 * replaces a plan the user is part-way through following — that plan is their
 * data, and overwriting it without consent would be both a usability failure
 * and an ethical one. The response returns `needs_regeneration` and the drift
 * figures; the user decides.
 */
import { Router } from 'express';

import { config } from '../config.js';
import { requireAuth } from '../middleware/auth.js';
import { progressSchema, validateBody } from '../middleware/validate.js';
import { mlClient } from '../services/mlClient.js';
import { asyncHandler, AppError, NotFoundError } from '../utils/errors.js';
import { bmiCategory, calculateBmi, round1 } from '../utils/nutrition.js';
import { buildPredictionPayload, storePrediction } from './predict.js';

export const progressRouter = Router();

progressRouter.use(requireAuth);

/** POST /api/progress — log a new weight and re-evaluate the plan. */
progressRouter.post(
  '/',
  validateBody(progressSchema),
  asyncHandler(async (req, res) => {
    const { data: profile, error: profileError } = await req.db
      .from('profiles')
      .select('height_cm')
      .eq('user_id', req.user.id)
      .maybeSingle();

    if (profileError) throw new AppError(`Failed to read profile: ${profileError.message}`, 500);
    if (!profile) throw new NotFoundError('Create a profile first (POST /api/profile).');

    // --- 1. log the weight and recompute BMI -----------------------------
    const bmi = round1(calculateBmi(req.body.weight_kg, Number(profile.height_cm)));
    const { data: weightLog, error: weightError } = await req.db
      .from('weight_logs')
      .insert({
        user_id: req.user.id,
        weight_kg: req.body.weight_kg,
        bmi,
        bmi_category: bmiCategory(bmi),
        ...(req.body.logged_at ? { logged_at: req.body.logged_at } : {}),
      })
      .select()
      .single();

    if (weightError) throw new AppError(`Failed to log weight: ${weightError.message}`, 500);

    // --- 2. what was the active plan built against? ----------------------
    const { data: activePlan } = await req.db
      .from('meal_plans')
      .select('id, calorie_target, protein_target')
      .eq('user_id', req.user.id)
      .eq('status', 'active')
      .maybeSingle();

    // --- 3. re-predict with the new weight -------------------------------
    const payload = await buildPredictionPayload(req.db, req.user.id);
    const prediction = await mlClient.predict(payload);
    await storePrediction(req.db, req.user.id, prediction);

    // --- 4. has the requirement drifted far enough to matter? ------------
    let drift = null;
    if (activePlan) {
      drift = await mlClient.driftCheck({
        old_calorie_target: Number(activePlan.calorie_target),
        new_calorie_target: prediction.calorie_target,
        old_protein_target: Number(activePlan.protein_target),
        new_protein_target: prediction.protein_target,
        threshold: config.driftThreshold,
      });
    }

    res.status(201).json({
      weight_log: weightLog,
      prediction,
      drift,
      // The frontend shows a "Your targets have changed — regenerate?" banner
      // when this is true. Nothing is regenerated automatically.
      needs_regeneration: drift?.needs_regeneration ?? false,
      message: drift?.needs_regeneration
        ? 'Your calculated targets have moved significantly. Consider regenerating your meal plan.'
        : 'Weight logged. Your current meal plan is still on target.',
    });
  }),
);

/** GET /api/progress — weight history for the trend chart. */
progressRouter.get(
  '/',
  asyncHandler(async (req, res) => {
    const { data, error } = await req.db
      .from('weight_logs')
      .select('id, weight_kg, bmi, bmi_category, logged_at')
      .eq('user_id', req.user.id)
      .order('logged_at', { ascending: true })
      .limit(500);

    if (error) throw new AppError(`Failed to read weight history: ${error.message}`, 500);

    const logs = data ?? [];
    const first = logs.at(0);
    const latest = logs.at(-1);

    res.json({
      weight_logs: logs,
      summary: logs.length
        ? {
            entries: logs.length,
            starting_weight_kg: Number(first.weight_kg),
            latest_weight_kg: Number(latest.weight_kg),
            change_kg: round1(Number(latest.weight_kg) - Number(first.weight_kg)),
            latest_bmi: Number(latest.bmi),
            latest_bmi_category: latest.bmi_category,
          }
        : null,
    });
  }),
);
