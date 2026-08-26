/**
 * Environment configuration with fail-fast validation.
 *
 * Missing configuration is detected at start-up rather than on the first
 * request that needs it, so a misconfigured deployment refuses to boot instead
 * of returning confusing 500s under load.
 */
import dotenv from 'dotenv';

dotenv.config();

const isTest = process.env.NODE_ENV === 'test';

export const config = {
  env: process.env.NODE_ENV ?? 'development',
  port: Number(process.env.PORT ?? 4000),

  supabase: {
    url: process.env.SUPABASE_URL ?? '',
    anonKey: process.env.SUPABASE_ANON_KEY ?? '',
    serviceRoleKey: process.env.SUPABASE_SERVICE_ROLE_KEY ?? '',
  },

  mlService: {
    baseUrl: process.env.ML_SERVICE_URL ?? 'http://127.0.0.1:8000',
    timeoutMs: Number(process.env.ML_SERVICE_TIMEOUT_MS ?? 20000),
    // The eight-week planner does real work; it gets a longer budget.
    planTimeoutMs: Number(process.env.ML_SERVICE_PLAN_TIMEOUT_MS ?? 120000),
  },

  cors: {
    origins: (process.env.CORS_ORIGINS ?? 'http://localhost:5173,http://localhost:3000')
      .split(',')
      .map((origin) => origin.trim())
      .filter(Boolean),
  },

  /** Recompute-triggered plan regeneration prompt threshold (fraction). */
  driftThreshold: Number(process.env.DRIFT_THRESHOLD ?? 0.07),
};

/**
 * Validate required settings. Called from server.js, not at import time, so
 * the test suite can import the app without a live Supabase project.
 */
export function assertConfig() {
  const missing = [];
  if (!config.supabase.url) missing.push('SUPABASE_URL');
  if (!config.supabase.anonKey) missing.push('SUPABASE_ANON_KEY');

  if (missing.length > 0 && !isTest) {
    throw new Error(
      `Missing required environment variable(s): ${missing.join(', ')}.\n` +
        'Copy backend/.env.example to backend/.env and fill in your Supabase credentials.',
    );
  }
}
