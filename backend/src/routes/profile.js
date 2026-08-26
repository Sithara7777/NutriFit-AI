/**
 * FR2 — User Profile Management, and FR3 — BMI calculation.
 *
 * Height and the training variables live on the profile; **weight does not**.
 * Weight changes over time and is therefore stored as a series in
 * `weight_logs`, with the profile joining to the most recent entry. That
 * separation is what makes the progress dashboard (FR8) and the
 * recommendation-adjustment loop (FR9) possible at all.
 */
import { Router } from 'express';

import { requireAuth } from '../middleware/auth.js';
import { profileSchema, validateBody } from '../middleware/validate.js';
import { asyncHandler, AppError, NotFoundError } from '../utils/errors.js';
import { bmiCategory, calculateBmi, deriveActivityLevel, round1 } from '../utils/nutrition.js';

export const profileRouter = Router();

profileRouter.use(requireAuth);

/** GET /api/profile — current profile plus latest weight/BMI. */
profileRouter.get(
  '/',
  asyncHandler(async (req, res) => {
    const { data: profile, error } = await req.db
      .from('profiles')
      .select('*')
      .eq('user_id', req.user.id)
      .maybeSingle();

    if (error) throw new AppError(`Failed to read profile: ${error.message}`, 500);
    if (!profile) throw new NotFoundError('No profile yet. POST /api/profile to create one.');

    const { data: weight } = await req.db
      .from('weight_logs')
      .select('weight_kg, bmi, bmi_category, logged_at')
      .eq('user_id', req.user.id)
      .order('logged_at', { ascending: false })
      .limit(1)
      .maybeSingle();

    res.json({ profile, latest_weight: weight ?? null });
  }),
);

/**
 * POST /api/profile — create or update the profile.
 *
 * Upsert rather than separate create/update: onboarding and "edit my details"
 * are the same operation from the user's point of view, and a profile is
 * one-per-user by primary key.
 *
 * Submitting a weight also writes a `weight_logs` entry, so the very first
 * onboarding produces a usable BMI and a first point on the progress chart.
 */
profileRouter.post(
  '/',
  validateBody(profileSchema),
  asyncHandler(async (req, res) => {
    const body = req.body;
    const activityLevel = deriveActivityLevel(body.workout_frequency, body.session_duration_h);

    const profileRow = {
      user_id: req.user.id,
      age: body.age,
      gender: body.gender,
      height_cm: body.height_cm,
      activity_level: activityLevel,
      workout_frequency: body.workout_frequency,
      session_duration_h: body.session_duration_h,
      experience_level: body.experience_level,
      fitness_goal: body.fitness_goal,
      body_fat_pct: body.body_fat_pct ?? null,
      body_fat_source: body.body_fat_pct == null ? 'estimated_deurenberg' : 'measured',
    };

    const { data: profile, error } = await req.db
      .from('profiles')
      .upsert(profileRow, { onConflict: 'user_id' })
      .select()
      .single();

    if (error) throw new AppError(`Failed to save profile: ${error.message}`, 500);

    const bmi = round1(calculateBmi(body.weight_kg, body.height_cm));
    const { data: weightLog, error: weightError } = await req.db
      .from('weight_logs')
      .insert({
        user_id: req.user.id,
        weight_kg: body.weight_kg,
        bmi,
        bmi_category: bmiCategory(bmi),
      })
      .select()
      .single();

    if (weightError) throw new AppError(`Failed to log weight: ${weightError.message}`, 500);

    res.status(201).json({ profile, latest_weight: weightLog });
  }),
);
