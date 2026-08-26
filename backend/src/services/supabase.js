/**
 * Supabase client factory.
 *
 * The important detail: every user-scoped query runs through a client built
 * with **that user's JWT**, not the service-role key. That means Postgres Row
 * Level Security is the thing actually enforcing data isolation. If a handler
 * ever forgot a `.eq('user_id', ...)` filter, RLS would still return nothing —
 * defence in depth rather than trusting application code alone.
 *
 * The service-role client exists only for operations that legitimately span
 * users (seeding the shared food catalogue) and is never used to serve a
 * request on a user's behalf.
 */
import { createClient } from '@supabase/supabase-js';

import { config } from '../config.js';

/**
 * A Supabase client scoped to one authenticated user. RLS applies.
 * @param {string} accessToken - the caller's Supabase JWT
 */
export function userClient(accessToken) {
  return createClient(config.supabase.url, config.supabase.anonKey, {
    global: { headers: { Authorization: `Bearer ${accessToken}` } },
    auth: { persistSession: false, autoRefreshToken: false },
  });
}

/** Anonymous client, used only to verify a token. */
export function anonClient() {
  return createClient(config.supabase.url, config.supabase.anonKey, {
    auth: { persistSession: false, autoRefreshToken: false },
  });
}

/**
 * Service-role client — bypasses RLS. Administrative use only
 * (catalogue seeding). Never use this to serve a user request.
 */
export function serviceClient() {
  if (!config.supabase.serviceRoleKey) {
    throw new Error('SUPABASE_SERVICE_ROLE_KEY is not configured.');
  }
  return createClient(config.supabase.url, config.supabase.serviceRoleKey, {
    auth: { persistSession: false, autoRefreshToken: false },
  });
}
