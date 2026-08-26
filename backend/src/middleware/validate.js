/**
 * Zod-based request validation.
 *
 * Validating at the edge means a bad payload produces a 400 with field-level
 * detail, and handlers can assume their inputs are already well-formed. The
 * bounds mirror the Pydantic schema in the ML service and the CHECK
 * constraints in the database, so all three layers agree.
 */
import { z } from 'zod';

import { ValidationError } from '../utils/errors.js';

export function validateBody(schema) {
  return (req, _res, next) => {
    const result = schema.safeParse(req.body);
    if (!result.success) {
      const details = result.error.issues.map((issue) => ({
        field: issue.path.join('.') || '(root)',
        message: issue.message,
      }));
      return next(new ValidationError('Request body failed validation.', details));
    }
    req.body = result.data;
    return next();
  };
}

export function validateQuery(schema) {
  return (req, _res, next) => {
    const result = schema.safeParse(req.query);
    if (!result.success) {
      const details = result.error.issues.map((issue) => ({
        field: issue.path.join('.') || '(root)',
        message: issue.message,
      }));
      return next(new ValidationError('Query parameters failed validation.', details));
    }
    req.validatedQuery = result.data;
    return next();
  };
}

// --------------------------------------------------------------------------
// Shared schemas
// --------------------------------------------------------------------------
export const GOALS = ['fat_loss', 'maintenance', 'muscle_gain'];
export const MEAL_SLOTS = ['breakfast', 'lunch', 'dinner', 'snack'];

export const profileSchema = z.object({
  age: z.number().int().min(16).max(80),
  gender: z.enum(['Male', 'Female']),
  height_cm: z.number().min(120).max(230),
  weight_kg: z.number().min(30).max(250),
  fitness_goal: z.enum(GOALS),
  workout_frequency: z.number().int().min(0).max(7),
  session_duration_h: z.number().min(0.1).max(5).default(1.25),
  experience_level: z.number().int().min(1).max(3).default(2),
  body_fat_pct: z.number().min(3).max(60).nullable().optional(),
});

export const progressSchema = z.object({
  weight_kg: z.number().min(30).max(250),
  logged_at: z.string().datetime().optional(),
});

export const mealPlanSchema = z.object({
  weeks: z.number().int().min(1).max(12).default(8),
  start_date: z
    .string()
    .regex(/^\d{4}-\d{2}-\d{2}$/, 'start_date must be YYYY-MM-DD')
    .optional(),
  seed: z.number().int().min(0).max(2 ** 31 - 1).optional(),
});

export const recommendationQuerySchema = z.object({
  meal_slot: z.enum(MEAL_SLOTS).optional(),
  top_n: z.coerce.number().int().min(1).max(25).default(5),
});
