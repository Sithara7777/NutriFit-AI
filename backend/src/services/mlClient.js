/**
 * HTTP client for the Python ML microservice.
 *
 * The ML service is an internal dependency: it is never reachable from the
 * browser, and every call passes through here so that timeouts, error shaping
 * and logging are handled in exactly one place.
 */
import axios from 'axios';

import { config } from '../config.js';
import { ServiceUnavailableError } from '../utils/errors.js';

const client = axios.create({
  baseURL: config.mlService.baseUrl,
  timeout: config.mlService.timeoutMs,
  headers: { 'Content-Type': 'application/json' },
});

/**
 * Translate an axios failure into a domain error the route layer can return
 * without leaking internal hostnames or stack traces to the client.
 */
function toServiceError(error, operation) {
  if (error.response) {
    const detail = error.response.data?.detail ?? error.response.data?.error ?? 'unknown error';
    return new ServiceUnavailableError(
      `ML service rejected ${operation}: ${JSON.stringify(detail)}`,
      error.response.status >= 500 ? 503 : 502,
    );
  }
  if (error.code === 'ECONNABORTED') {
    return new ServiceUnavailableError(`ML service timed out during ${operation}.`);
  }
  return new ServiceUnavailableError(
    `ML service unreachable during ${operation}. Is it running on ${config.mlService.baseUrl}?`,
  );
}

export const mlClient = {
  async health() {
    try {
      const { data } = await client.get('/health', { timeout: 5000 });
      return data;
    } catch (error) {
      throw toServiceError(error, 'health check');
    }
  },

  /** Predict calorie + protein targets for one profile. */
  async predict(profile) {
    try {
      const { data } = await client.post('/predict', profile);
      return data;
    } catch (error) {
      throw toServiceError(error, 'prediction');
    }
  },

  /** Meal suggestions for one slot, or a composed day when slot is omitted. */
  async recommend(payload) {
    try {
      const { data } = await client.post('/recommend', payload);
      return data;
    } catch (error) {
      throw toServiceError(error, 'recommendation');
    }
  },

  /** Generate a multi-week meal plan. Uses the longer timeout budget. */
  async mealPlan(payload) {
    try {
      const { data } = await client.post('/mealplan', payload, {
        timeout: config.mlService.planTimeoutMs,
      });
      return data;
    } catch (error) {
      throw toServiceError(error, 'meal plan generation');
    }
  },

  /** Decide whether a changed target warrants prompting for a new plan. */
  async driftCheck(payload) {
    try {
      const { data } = await client.post('/drift-check', payload);
      return data;
    } catch (error) {
      throw toServiceError(error, 'drift check');
    }
  },
};
