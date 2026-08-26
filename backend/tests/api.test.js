/**
 * API-level tests for routing, authentication and validation.
 *
 * Scope: these exercise the guard rails that must hold regardless of whether a
 * live Supabase project is attached — unauthenticated access is refused, unknown
 * routes 404, malformed JSON 400. Full round-trip tests against real Supabase
 * data belong in the integration suite documented in `docs/TESTING.md`.
 */
import { describe, expect, test } from '@jest/globals';
import request from 'supertest';

import { createApp } from '../src/app.js';

const app = createApp();

const PROTECTED_ROUTES = [
  ['get', '/api/profile'],
  ['post', '/api/profile'],
  ['post', '/api/predict'],
  ['get', '/api/predict/history'],
  ['get', '/api/recommendations'],
  ['post', '/api/mealplan/generate'],
  ['get', '/api/mealplan/active'],
  ['post', '/api/progress'],
  ['get', '/api/progress'],
  ['get', '/api/dashboard'],
];

describe('health', () => {
  test('GET /api/health returns ok without auth', async () => {
    const response = await request(app).get('/api/health');
    expect(response.status).toBe(200);
    expect(response.body.status).toBe('ok');
    expect(response.body.service).toBe('nutrifit-backend');
  });
});

describe('authentication guard', () => {
  test.each(PROTECTED_ROUTES)('%s %s requires a token', async (method, path) => {
    const response = await request(app)[method](path);
    expect(response.status).toBe(401);
    expect(response.body.error).toBe('unauthorized');
  });

  test('rejects a malformed Authorization header', async () => {
    const response = await request(app)
      .get('/api/dashboard')
      .set('Authorization', 'Basic abc123');
    expect(response.status).toBe(401);
  });

  test('rejects an empty bearer token', async () => {
    const response = await request(app).get('/api/dashboard').set('Authorization', 'Bearer ');
    expect(response.status).toBe(401);
  });
});

describe('error handling', () => {
  test('unknown route returns a consistent 404 shape', async () => {
    const response = await request(app).get('/api/does-not-exist');
    expect(response.status).toBe(404);
    expect(response.body.error).toBe('not_found');
    expect(response.body.message).toContain('/api/does-not-exist');
  });

  test('malformed JSON returns 400, not 500', async () => {
    const response = await request(app)
      .post('/api/profile')
      .set('Content-Type', 'application/json')
      .send('{"age": ');
    expect(response.status).toBe(400);
    expect(response.body.error).toBe('validation_error');
  });

  test('responses never leak a stack trace', async () => {
    const response = await request(app).get('/api/dashboard');
    expect(JSON.stringify(response.body)).not.toMatch(/at .*\.js:\d+/);
  });
});

describe('security headers', () => {
  test('helmet sets protective headers', async () => {
    const response = await request(app).get('/api/health');
    expect(response.headers['x-content-type-options']).toBe('nosniff');
    expect(response.headers['x-powered-by']).toBeUndefined();
  });
});
