/**
 * BMI helpers — the Node-side mirror of `nutrifit.nutrition`.
 *
 * Only BMI and its WHO classification are duplicated here, deliberately: the
 * backend must classify a weight log without a network round-trip to the ML
 * service, and BMI is a two-line formula with no model behind it. Everything
 * else (BMR, TDEE, targets) stays in Python so there is exactly one
 * implementation of the parts that matter.
 *
 * `backend/tests/nutrition.test.js` asserts these values against the same
 * hand-computed figures used in `ml/tests/test_nutrition.py`, so the two
 * implementations cannot silently diverge.
 */

/** Body Mass Index in kg/m². */
export function calculateBmi(weightKg, heightCm) {
  const heightM = heightCm / 100;
  return weightKg / (heightM * heightM);
}

/** WHO BMI classification. */
export function bmiCategory(bmi) {
  if (bmi < 18.5) return 'underweight';
  if (bmi < 25) return 'normal';
  if (bmi < 30) return 'overweight';
  return 'obese';
}

/** Round to one decimal place, matching the numeric(4,1) DB columns. */
export function round1(value) {
  return Math.round(value * 10) / 10;
}

/**
 * FAO/WHO activity band derived from weekly training volume.
 * Mirrors `nutrifit.nutrition.derive_activity_level` so the profile stored in
 * Supabase agrees with what the ML service would compute.
 */
export function deriveActivityLevel(workoutFrequency, sessionDurationH) {
  const weeklyHours = workoutFrequency * sessionDurationH;
  if (weeklyHours < 1.5) return 'sedentary';
  if (weeklyHours < 3.0) return 'light';
  if (weeklyHours < 5.0) return 'moderate';
  if (weeklyHours < 7.0) return 'active';
  return 'very_active';
}
