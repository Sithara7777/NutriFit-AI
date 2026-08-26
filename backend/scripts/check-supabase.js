/**
 * Supabase setup verifier.
 *
 *   node backend/scripts/check-supabase.js
 *
 * Run this after each Supabase step. It checks, in order:
 *   1. .env is filled in (no leftover placeholders)
 *   2. the project is reachable
 *   3. db/schema.sql ran   -> all six tables exist
 *   4. foods_seed.sql ran  -> the catalogue is populated
 *   5. Row Level Security is actually switched on
 *
 * Every failure prints the exact fix, so you never have to guess which step
 * went wrong.
 */
import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.join(here, '..', '.env') });

const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const YELLOW = '\x1b[33m';
const DIM = '\x1b[2m';
const RESET = '\x1b[0m';

const ok = (msg) => console.log(`${GREEN}[ OK ]${RESET} ${msg}`);
const bad = (msg, fix) => {
  console.log(`${RED}[FAIL]${RESET} ${msg}`);
  if (fix) console.log(`${DIM}       -> ${fix}${RESET}`);
};
const warn = (msg, note) => {
  console.log(`${YELLOW}[WARN]${RESET} ${msg}`);
  if (note) console.log(`${DIM}       -> ${note}${RESET}`);
};

const EXPECTED_TABLES = [
  'profiles',
  'weight_logs',
  'predictions',
  'foods',
  'meal_plans',
  'meal_plan_items',
];

const EXPECTED_FOOD_COUNT = 593;

let failures = 0;

// --------------------------------------------------------------- 1. config
console.log('\n=== 1. Environment ===');

const url = process.env.SUPABASE_URL ?? '';
const anonKey = process.env.SUPABASE_ANON_KEY ?? '';
const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY ?? '';

if (!url || url.includes('YOUR-PROJECT-REF')) {
  bad('SUPABASE_URL not set', 'Set it in backend/.env');
  failures++;
} else {
  ok(`SUPABASE_URL   ${url}`);
}

for (const [label, value] of [
  ['SUPABASE_ANON_KEY', anonKey],
  ['SUPABASE_SERVICE_ROLE_KEY', serviceKey],
]) {
  if (!value || value.startsWith('PASTE_') || value.startsWith('your-')) {
    bad(`${label} still a placeholder`, 'Supabase Dashboard -> Settings -> API Keys');
    failures++;
  } else {
    ok(`${label.padEnd(26)} ${value.slice(0, 12)}…${value.slice(-4)} (${value.length} chars)`);
  }
}

if (failures > 0) {
  console.log(`\n${RED}Fill in backend/.env before continuing.${RESET}\n`);
  process.exit(1);
}

const admin = createClient(url, serviceKey, { auth: { persistSession: false } });
const anon = createClient(url, anonKey, { auth: { persistSession: false } });

// ------------------------------------------------------------ 2. reachable
console.log('\n=== 2. Connectivity ===');

/**
 * Decode a Supabase JWT payload without verifying the signature.
 *
 * This is a *configuration* check, not an authentication one — we only want to
 * read the `ref` and `role` claims to confirm the key belongs to this project
 * and carries the role we expect. Verification is Supabase's job.
 */
function decodeJwtPayload(token) {
  try {
    const [, payload] = token.split('.');
    return JSON.parse(Buffer.from(payload, 'base64url').toString('utf8'));
  } catch {
    return null;
  }
}

const projectRef = new URL(url).hostname.split('.')[0];

for (const [label, token, expectedRole] of [
  ['anon key', anonKey, 'anon'],
  ['service_role key', serviceKey, 'service_role'],
]) {
  const claims = decodeJwtPayload(token);
  if (!claims) {
    warn(`${label} is not a decodable JWT`, 'If you copied a new-style sb_publishable_/sb_secret_ key, use the legacy anon/service_role keys instead');
    continue;
  }
  if (claims.ref !== projectRef) {
    bad(
      `${label} belongs to project "${claims.ref}", not "${projectRef}"`,
      'You copied a key from a different Supabase project',
    );
    failures++;
  } else if (claims.role !== expectedRole) {
    bad(`${label} has role "${claims.role}", expected "${expectedRole}"`, 'Keys are swapped in .env');
    failures++;
  } else {
    const expires = claims.exp ? new Date(claims.exp * 1000).toISOString().slice(0, 10) : 'n/a';
    ok(`${label.padEnd(17)} ref=${claims.ref} role=${claims.role} expires=${expires}`);
  }
}

// Live probe against a real table, which is what the application actually
// does. The bare `/rest/v1/` root is NOT a valid health check — Supabase does
// not serve the OpenAPI spec to anon clients, so it returns 401 even when the
// key is perfectly good.
try {
  const response = await fetch(`${url}/rest/v1/foods?select=food_id&limit=1`, {
    headers: { apikey: anonKey, Authorization: `Bearer ${anonKey}` },
  });
  if (response.ok) {
    ok(`REST API reachable (HTTP ${response.status})`);
  } else if (response.status === 401) {
    bad('401 querying the foods table', 'The anon key was rejected — re-copy it from the dashboard');
    failures++;
  } else if (response.status === 404) {
    bad('foods table not found via REST', 'Run db/schema.sql');
    failures++;
  } else {
    warn(`REST API returned HTTP ${response.status}`, await response.text());
  }
} catch (error) {
  bad(`Cannot reach ${url}`, error.message);
  process.exit(1);
}

// Auth endpoint — this is what actually matters for sign-up / sign-in.
try {
  const response = await fetch(`${url}/auth/v1/settings`, {
    headers: { apikey: anonKey, Authorization: `Bearer ${anonKey}` },
  });
  if (response.ok) {
    const settings = await response.json();
    ok('Auth endpoint reachable');
    if (settings.mailer_autoconfirm === false && settings.external?.email !== false) {
      warn(
        'Email confirmation is ENABLED',
        'Sign-in will fail until you confirm each address. Turn it off for development: Authentication -> Sign In / Providers -> Email -> uncheck "Confirm email"',
      );
    } else if (settings.mailer_autoconfirm === true) {
      ok('Email auto-confirm is ON — accounts work immediately (correct for development)');
    }
  } else {
    warn(`Auth settings returned HTTP ${response.status}`);
  }
} catch (error) {
  warn('Could not read auth settings', error.message);
}

// ---------------------------------------------------------------- 3. schema
console.log('\n=== 3. Schema (db/schema.sql) ===');
const missingTables = [];

for (const table of EXPECTED_TABLES) {
  const { error } = await admin.from(table).select('*', { count: 'exact', head: true });
  if (error) {
    missingTables.push(table);
    bad(`table "${table}" not found`, error.message);
  } else {
    ok(`table "${table}"`);
  }
}

if (missingTables.length > 0) {
  console.log(
    `\n${RED}Schema incomplete.${RESET} Run db/schema.sql in the Supabase SQL Editor.\n`,
  );
  process.exit(1);
}

// Views
const { error: viewError } = await admin
  .from('user_dashboard')
  .select('*', { count: 'exact', head: true });
if (viewError) {
  warn('view "user_dashboard" missing', 'Re-run the tail end of db/schema.sql');
} else {
  ok('view  "user_dashboard"');
}

// ------------------------------------------------------------------ 4. seed
console.log('\n=== 4. Food catalogue (foods_seed.sql) ===');
const { count: foodCount, error: foodError } = await admin
  .from('foods')
  .select('*', { count: 'exact', head: true });

if (foodError) {
  bad('Could not count foods', foodError.message);
  failures++;
} else if (!foodCount) {
  bad('foods table is empty', 'Run data/processed/foods_seed.sql in the SQL Editor');
  failures++;
} else {
  ok(`${foodCount} food items seeded`);
  if (foodCount !== EXPECTED_FOOD_COUNT) {
    warn(
      `expected ${EXPECTED_FOOD_COUNT}, found ${foodCount}`,
      'Not necessarily wrong — depends on your catalogue build options',
    );
  }

  const { data: slots } = await admin.from('foods').select('meal_type');
  const bySlot = {};
  for (const row of slots ?? []) bySlot[row.meal_type] = (bySlot[row.meal_type] ?? 0) + 1;
  console.log(`${DIM}       per slot: ${JSON.stringify(bySlot)}${RESET}`);

  const thin = Object.entries(bySlot).filter(([, n]) => n < 20);
  if (thin.length > 0) {
    warn(`slots with <20 items: ${thin.map(([s]) => s).join(', ')}`, 'May breach the 5-day variety rule');
  }
}

// ------------------------------------------------------------------- 5. RLS
console.log('\n=== 5. Row Level Security ===');

// `foods` is readable only by authenticated users. An anonymous client must
// therefore come back empty — if it returns rows, RLS is not enforcing.
const { data: anonFoods, error: anonError } = await anon.from('foods').select('food_id').limit(5);

if (anonError) {
  ok(`anonymous read of "foods" blocked (${anonError.code ?? 'error'})`);
} else if ((anonFoods ?? []).length === 0) {
  ok('anonymous read of "foods" returns 0 rows — RLS active');
} else {
  bad(
    `anonymous client read ${anonFoods.length} rows from "foods" — RLS is NOT enforcing`,
    'Re-run the "Row Level Security" section of db/schema.sql',
  );
  failures++;
}

// profiles must be unreadable anonymously
const { data: anonProfiles, error: profileError } = await anon.from('profiles').select('user_id').limit(1);
if (profileError || (anonProfiles ?? []).length === 0) {
  ok('anonymous read of "profiles" blocked / empty — RLS active');
} else {
  bad('anonymous client can read "profiles" — RLS is NOT enforcing', 'Re-run db/schema.sql');
  failures++;
}

// -------------------------------------------------------------- 6. ML check
console.log('\n=== 6. ML service (optional at this stage) ===');
const mlUrl = process.env.ML_SERVICE_URL ?? 'http://127.0.0.1:8000';
try {
  const response = await fetch(`${mlUrl}/health`, { signal: AbortSignal.timeout(3000) });
  const health = await response.json();
  if (health.status === 'ok') {
    ok(`ML service healthy at ${mlUrl}`);
  } else {
    warn(`ML service degraded: ${health.detail}`, 'Run: python ml/scripts/export_models.py');
  }
} catch {
  warn(`ML service not running at ${mlUrl}`, 'Start it later: uvicorn app.main:app --port 8000');
}

// ----------------------------------------------------------------- summary
console.log();
if (failures === 0) {
  console.log(`${GREEN}Supabase is configured correctly. Next: start the three services.${RESET}\n`);
  process.exit(0);
}
console.log(`${RED}${failures} problem(s) found — see the fixes above.${RESET}\n`);
process.exit(1);
