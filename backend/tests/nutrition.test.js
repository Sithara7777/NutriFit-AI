/**
 * Unit tests for the Node-side BMI/activity helpers.
 *
 * These assert the *same* hand-computed values as `ml/tests/test_nutrition.py`,
 * which is what guarantees the JavaScript and Python implementations cannot
 * silently diverge.
 */
import { describe, expect, test } from '@jest/globals';

import {
  bmiCategory,
  calculateBmi,
  deriveActivityLevel,
  round1,
} from '../src/utils/nutrition.js';

describe('calculateBmi', () => {
  test('matches the Python implementation for 80 kg at 180 cm', () => {
    expect(calculateBmi(80, 180)).toBeCloseTo(24.691, 3);
  });

  test('matches the Python implementation for 60 kg at 165 cm', () => {
    expect(calculateBmi(60, 165)).toBeCloseTo(22.039, 3);
  });

  test('is monotonic in weight', () => {
    expect(calculateBmi(90, 180)).toBeGreaterThan(calculateBmi(80, 180));
  });
});

describe('bmiCategory (WHO boundaries)', () => {
  test.each([
    [17.0, 'underweight'],
    [18.4, 'underweight'],
    [18.5, 'normal'],
    [22.0, 'normal'],
    [24.9, 'normal'],
    [25.0, 'overweight'],
    [27.5, 'overweight'],
    [29.9, 'overweight'],
    [30.0, 'obese'],
    [40.0, 'obese'],
  ])('BMI %s -> %s', (bmi, expected) => {
    expect(bmiCategory(bmi)).toBe(expected);
  });
});

describe('deriveActivityLevel', () => {
  test.each([
    [1, 1.0, 'sedentary'],
    [2, 1.0, 'light'],
    [4, 1.0, 'moderate'],
    [5, 1.2, 'active'],
    [6, 1.5, 'very_active'],
  ])('%s days x %s h -> %s', (frequency, duration, expected) => {
    expect(deriveActivityLevel(frequency, duration)).toBe(expected);
  });

  test('4 days x 1.25 h lands in the active band (boundary case)', () => {
    // 5.0 weekly hours is NOT < 5.0, so it falls into "active".
    // The Python test asserts the same boundary.
    expect(deriveActivityLevel(4, 1.25)).toBe('active');
  });
});

describe('round1', () => {
  test('rounds to one decimal place', () => {
    expect(round1(24.6913)).toBe(24.7);
    expect(round1(24.64)).toBe(24.6);
  });
});
