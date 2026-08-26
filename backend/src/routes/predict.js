/**
 * FR4 / FR5 — Calorie and Protein Requirement Prediction.
 *
 * The backend does not implement any nutrition science itself: it assembles the
 * feature payload from the stored profile plus the latest weight log, calls the
 * Python ML service, and persists the result. Keeping the science in one place
 * (the `nutrifit` package) is what stops the web tier and the model drifting
 * apart.
 */
import { Router } from 'express';

import { requireAuth } from '../middleware/auth.js';
import { mlClient } from '../services/mlClient.js';
import { asyncHandler, AppError, NotFoundError } from '../utils/errors.js';

export const predictRouter = Router();

predictRouter.use(requireAuth);

/**
 * Build the ML request payload for a user. Shared by /api/predict and the
 * recalculation path in /api/progress.
 */
export async function buildPredictionPayload(db, userId) {
  const { data: profile, error } = await db
    .from('profiles')
    .select('*')
    .eq('user_id', userId)
    .maybeSingle();

  if (error) throw new AppError(`Failed to read profile: ${error.message}`, 500);
  if (!profile) throw new NotFoundError('Create a profile first (POST /api/profile).');

  const { data: weight } = await db
    .from('weight_logs')
    .select('weight_kg')
    .eq('user_id', userId)
    .order('logged_at', { ascending: false })
    .limit(1)
    .maybeSingle();

  if (!weight) throw new NotFoundError('No weight recorded. POST /api/progress first.');

  return {
    age: profile.age,
    gender: profile.gender,
    height_cm: Number(profile.height_cm),
    weight_kg: Number(weight.weight_kg),
    fitness_goal: profile.fitness_goal,
    workout_frequency: profile.workout_frequency,
    session_duration_h: Number(profile.session_duration_h),
    experience_level: profile.experience_level,
    body_fat_pct: profile.body_fat_pct == null ? null : Number(profile.body_fat_pct),
  };
}

/** Persist a prediction so the dashboard can chart target history. */
export async function storePrediction(db, userId, prediction) {
  const { data, error } = await db
    .from('predictions')
    .insert({
      user_id: userId,
      calorie_target: prediction.calorie_target,
      protein_target: prediction.protein_target,
      bmr: prediction.bmr,
      tdee: prediction.tdee,
      bmi: prediction.bmi,
      model_version: prediction.model_version,
      source: prediction.source,
    })
    .select()
    .single();

  if (error) throw new AppError(`Failed to store prediction: ${error.message}`, 500);
  return data;
}

/** POST /api/predict — recompute and store the user's daily targets. */
predictRouter.post(
  '/',
  asyncHandler(async (req, res) => {
    const payload = await buildPredictionPayload(req.db, req.user.id);
    const prediction = await mlClient.predict(payload);
    const stored = await storePrediction(req.db, req.user.id, prediction);

    res.json({ prediction, stored_id: stored.id });
  }),
);

/** GET /api/predict/history — target history for the dashboard chart. */
predictRouter.get(
  '/history',
  asyncHandler(async (req, res) => {
    const { data, error } = await req.db
      .from('predictions')
      .select('id, calorie_target, protein_target, bmi, source, created_at')
      .eq('user_id', req.user.id)
      .order('created_at', { ascending: true })
      .limit(200);

    if (error) throw new AppError(`Failed to read prediction history: ${error.message}`, 500);
    res.json({ predictions: data ?? [] });
  }),
);
