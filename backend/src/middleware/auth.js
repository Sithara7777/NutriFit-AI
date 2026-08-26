/**
 * Supabase JWT authentication guard.
 *
 * Every `/api/*` route except `/api/health` requires a valid bearer token.
 * The token is verified against Supabase (not merely decoded), and the
 * resulting per-user client is attached to the request so that all downstream
 * queries run under Row Level Security.
 */
import { anonClient, userClient } from '../services/supabase.js';
import { UnauthorizedError } from '../utils/errors.js';

export async function requireAuth(req, res, next) {
  try {
    const header = req.headers.authorization ?? '';
    if (!header.startsWith('Bearer ')) {
      throw new UnauthorizedError('Missing Authorization: Bearer <token> header.');
    }

    const token = header.slice('Bearer '.length).trim();
    if (!token) {
      throw new UnauthorizedError('Empty bearer token.');
    }

    // Verify with Supabase. Decoding the JWT locally would accept a forged or
    // revoked token; this asks the auth server whether it is genuinely valid.
    const { data, error } = await anonClient().auth.getUser(token);
    if (error || !data?.user) {
      throw new UnauthorizedError('Invalid or expired token.');
    }

    req.user = { id: data.user.id, email: data.user.email };
    req.accessToken = token;
    req.db = userClient(token); // RLS-scoped client
    next();
  } catch (error) {
    next(error);
  }
}
