/**
 * Typed wrapper around the Node backend.
 *
 * Every call attaches the current Supabase JWT. The browser never talks to the
 * Python ML service directly — it is not reachable from the internet, and all
 * ML calls are brokered by the authenticated Node API.
 */
import { supabase } from './supabase.js';

const BASE = import.meta.env.VITE_BACKEND_URL ?? '';

export class ApiError extends Error {
  constructor(message, status, code, details) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

async function request(path, { method = 'GET', body, signal } = {}) {
  const {
    data: { session },
  } = await supabase.auth.getSession();

  if (!session?.access_token) {
    throw new ApiError('You are not signed in.', 401, 'unauthorized');
  }

  const response = await fetch(`${BASE}/api${path}`, {
    method,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${session.access_token}`,
    },
    ...(body ? { body: JSON.stringify(body) } : {}),
    signal,
  });

  const text = await response.text();
  let payload = null;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    payload = { message: text };
  }

  if (!response.ok) {
    throw new ApiError(
      payload?.message ?? `Request failed with status ${response.status}`,
      response.status,
      payload?.error,
      payload?.details,
    );
  }
  return payload;
}

export const api = {
  getDashboard: () => request('/dashboard'),

  getProfile: () => request('/profile'),
  saveProfile: (profile) => request('/profile', { method: 'POST', body: profile }),

  predict: () => request('/predict', { method: 'POST' }),
  getPredictionHistory: () => request('/predict/history'),

  getRecommendations: (mealSlot, topN = 5) =>
    request(
      mealSlot
        ? `/recommendations?meal_slot=${encodeURIComponent(mealSlot)}&top_n=${topN}`
        : '/recommendations',
    ),

  generateMealPlan: (options = {}) =>
    request('/mealplan/generate', { method: 'POST', body: { weeks: 8, ...options } }),
  getActiveMealPlan: () => request('/mealplan/active'),

  logProgress: (weightKg) =>
    request('/progress', { method: 'POST', body: { weight_kg: weightKg } }),
  getProgress: () => request('/progress'),
};
