-- ===========================================================================
-- NutriFit-AI - Supabase / PostgreSQL schema
-- ===========================================================================
-- Run this in the Supabase SQL Editor (Dashboard -> SQL Editor -> New query).
-- Safe to re-run: every statement is idempotent.
--
-- Security model
-- --------------
-- Row Level Security is enabled on every user-scoped table, with policies
-- keyed on `auth.uid()`. This is enforced by Postgres itself, so even if the
-- API layer had a bug, one user's rows remain unreachable from another user's
-- session. `foods` is the single exception: it is a shared reference catalogue
-- containing no personal data, so it is world-readable but writable only by
-- the service role.
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- 1. profiles   (FR2 - User Profile Management)
-- ---------------------------------------------------------------------------
create table if not exists public.profiles (
    user_id            uuid primary key references auth.users (id) on delete cascade,
    age                integer      not null check (age between 16 and 80),
    gender             text         not null check (gender in ('Male', 'Female')),
    height_cm          numeric(5,1) not null check (height_cm between 120 and 230),
    activity_level     text         not null default 'moderate'
                                    check (activity_level in
                                          ('sedentary','light','moderate','active','very_active')),
    workout_frequency  integer      not null check (workout_frequency between 0 and 7),
    session_duration_h numeric(4,2) not null default 1.25
                                    check (session_duration_h between 0.1 and 5.0),
    experience_level   integer      not null default 2 check (experience_level between 1 and 3),
    fitness_goal       text         not null
                                    check (fitness_goal in ('fat_loss','maintenance','muscle_gain')),
    -- Optional: users who have had a body-composition scan get a materially
    -- better BMR estimate (Katch-McArdle instead of Mifflin-St Jeor).
    body_fat_pct       numeric(4,1) check (body_fat_pct between 3 and 60),
    body_fat_source    text         default 'estimated_deurenberg'
                                    check (body_fat_source in ('measured','estimated_deurenberg')),
    created_at         timestamptz  not null default now(),
    updated_at         timestamptz  not null default now()
);

comment on table public.profiles is
    'One row per user. Height lives here; weight lives in weight_logs because it changes.';

-- ---------------------------------------------------------------------------
-- 2. weight_logs   (FR3 BMI, FR8 Progress Monitoring)
-- ---------------------------------------------------------------------------
create table if not exists public.weight_logs (
    id            uuid         primary key default gen_random_uuid(),
    user_id       uuid         not null references auth.users (id) on delete cascade,
    weight_kg     numeric(5,1) not null check (weight_kg between 30 and 250),
    bmi           numeric(4,1) not null,
    bmi_category  text         not null
                               check (bmi_category in ('underweight','normal','overweight','obese')),
    logged_at     timestamptz  not null default now()
);

create index if not exists weight_logs_user_time_idx
    on public.weight_logs (user_id, logged_at desc);

-- ---------------------------------------------------------------------------
-- 3. predictions   (FR4, FR5 - stores every prediction for auditability)
-- ---------------------------------------------------------------------------
create table if not exists public.predictions (
    id              uuid         primary key default gen_random_uuid(),
    user_id         uuid         not null references auth.users (id) on delete cascade,
    calorie_target  numeric(7,1) not null check (calorie_target between 1000 and 7000),
    protein_target  numeric(6,1) not null check (protein_target between 30 and 400),
    bmr             numeric(7,1),
    tdee            numeric(7,1),
    bmi             numeric(4,1),
    model_version   text         not null default '1.0.0',
    -- 'model' when a trained pipeline answered; 'formula' when the ML service
    -- degraded to the deterministic equations. Recording this makes the
    -- Reliability NFR auditable after the fact.
    source          text         not null default 'model' check (source in ('model','formula')),
    created_at      timestamptz  not null default now()
);

create index if not exists predictions_user_time_idx
    on public.predictions (user_id, created_at desc);

-- ---------------------------------------------------------------------------
-- 4. foods   (shared reference catalogue - seeded from foods_seed.sql)
-- ---------------------------------------------------------------------------
create table if not exists public.foods (
    food_id         text primary key,
    name            text         not null,
    category        text,
    meal_type       text         not null
                                 check (meal_type in ('breakfast','lunch','dinner','snack')),
    calories        numeric(7,2) not null check (calories > 0),
    protein_g       numeric(6,2) not null check (protein_g >= 0),
    carbs_g         numeric(6,2) not null check (carbs_g >= 0),
    fat_g           numeric(6,2) not null check (fat_g >= 0),
    fiber_g         numeric(6,2) default 0,
    sugar_g         numeric(6,2) default 0,
    sodium_mg       numeric(8,2) default 0,
    cholesterol_mg  numeric(8,2) default 0,
    protein_density numeric(6,2),
    source          text,
    created_at      timestamptz  not null default now()
);

create index if not exists foods_meal_type_idx on public.foods (meal_type);
create index if not exists foods_name_idx      on public.foods (lower(name));

-- ---------------------------------------------------------------------------
-- 5. meal_plans   (FR7 - Two-Month Meal Plan Generation)
-- ---------------------------------------------------------------------------
create table if not exists public.meal_plans (
    id             uuid         primary key default gen_random_uuid(),
    user_id        uuid         not null references auth.users (id) on delete cascade,
    week_count     integer      not null default 8 check (week_count between 1 and 12),
    start_date     date         not null default current_date,
    status         text         not null default 'active'
                                check (status in ('active','superseded','archived')),
    calorie_target numeric(7,1) not null,
    protein_target numeric(6,1) not null,
    fitness_goal   text         not null
                                check (fitness_goal in ('fat_loss','maintenance','muscle_gain')),
    seed           integer      not null default 42,
    created_at     timestamptz  not null default now()
);

create index if not exists meal_plans_user_status_idx
    on public.meal_plans (user_id, status, created_at desc);

-- At most one active plan per user. Superseding rather than deleting keeps the
-- user's history intact, which the progress dashboard relies on.
create unique index if not exists meal_plans_one_active_per_user
    on public.meal_plans (user_id) where (status = 'active');

-- ---------------------------------------------------------------------------
-- 6. meal_plan_items
-- ---------------------------------------------------------------------------
create table if not exists public.meal_plan_items (
    id           uuid         primary key default gen_random_uuid(),
    meal_plan_id uuid         not null references public.meal_plans (id) on delete cascade,
    week_number  integer      not null check (week_number between 1 and 12),
    day_index    integer      not null check (day_index between 0 and 6),
    day_of_week  text         not null,
    plan_date    date,
    meal_slot    text         not null
                              check (meal_slot in ('breakfast','lunch','dinner','snack')),
    -- A slot holds 1-3 foods; `position` preserves their order.
    position     integer      not null default 0 check (position between 0 and 5),
    food_id      text         references public.foods (food_id) on delete set null,
    food_name    text         not null,
    servings     numeric(4,2) not null default 1.0 check (servings > 0 and servings <= 5),
    calories     numeric(7,2) not null,
    protein_g    numeric(6,2) not null,
    carbs_g      numeric(6,2) not null,
    fat_g        numeric(6,2) not null,
    fiber_g      numeric(6,2) default 0,
    unique (meal_plan_id, week_number, day_index, meal_slot, position)
);

create index if not exists meal_plan_items_plan_idx
    on public.meal_plan_items (meal_plan_id, week_number, day_index);

-- ===========================================================================
-- Row Level Security
-- ===========================================================================
alter table public.profiles        enable row level security;
alter table public.weight_logs     enable row level security;
alter table public.predictions     enable row level security;
alter table public.meal_plans      enable row level security;
alter table public.meal_plan_items enable row level security;
alter table public.foods           enable row level security;

-- --- profiles --------------------------------------------------------------
drop policy if exists "own profile: select" on public.profiles;
create policy "own profile: select" on public.profiles
    for select using (auth.uid() = user_id);

drop policy if exists "own profile: insert" on public.profiles;
create policy "own profile: insert" on public.profiles
    for insert with check (auth.uid() = user_id);

drop policy if exists "own profile: update" on public.profiles;
create policy "own profile: update" on public.profiles
    for update using (auth.uid() = user_id) with check (auth.uid() = user_id);

drop policy if exists "own profile: delete" on public.profiles;
create policy "own profile: delete" on public.profiles
    for delete using (auth.uid() = user_id);

-- --- weight_logs -----------------------------------------------------------
drop policy if exists "own weight logs: select" on public.weight_logs;
create policy "own weight logs: select" on public.weight_logs
    for select using (auth.uid() = user_id);

drop policy if exists "own weight logs: insert" on public.weight_logs;
create policy "own weight logs: insert" on public.weight_logs
    for insert with check (auth.uid() = user_id);

drop policy if exists "own weight logs: delete" on public.weight_logs;
create policy "own weight logs: delete" on public.weight_logs
    for delete using (auth.uid() = user_id);

-- --- predictions -----------------------------------------------------------
drop policy if exists "own predictions: select" on public.predictions;
create policy "own predictions: select" on public.predictions
    for select using (auth.uid() = user_id);

drop policy if exists "own predictions: insert" on public.predictions;
create policy "own predictions: insert" on public.predictions
    for insert with check (auth.uid() = user_id);

-- --- meal_plans ------------------------------------------------------------
drop policy if exists "own meal plans: select" on public.meal_plans;
create policy "own meal plans: select" on public.meal_plans
    for select using (auth.uid() = user_id);

drop policy if exists "own meal plans: insert" on public.meal_plans;
create policy "own meal plans: insert" on public.meal_plans
    for insert with check (auth.uid() = user_id);

drop policy if exists "own meal plans: update" on public.meal_plans;
create policy "own meal plans: update" on public.meal_plans
    for update using (auth.uid() = user_id) with check (auth.uid() = user_id);

drop policy if exists "own meal plans: delete" on public.meal_plans;
create policy "own meal plans: delete" on public.meal_plans
    for delete using (auth.uid() = user_id);

-- --- meal_plan_items -------------------------------------------------------
-- Items have no user_id of their own; ownership is derived from the parent
-- plan. The EXISTS sub-query is what makes that enforceable in Postgres.
drop policy if exists "own plan items: select" on public.meal_plan_items;
create policy "own plan items: select" on public.meal_plan_items
    for select using (
        exists (select 1 from public.meal_plans p
                 where p.id = meal_plan_items.meal_plan_id and p.user_id = auth.uid())
    );

drop policy if exists "own plan items: insert" on public.meal_plan_items;
create policy "own plan items: insert" on public.meal_plan_items
    for insert with check (
        exists (select 1 from public.meal_plans p
                 where p.id = meal_plan_items.meal_plan_id and p.user_id = auth.uid())
    );

drop policy if exists "own plan items: delete" on public.meal_plan_items;
create policy "own plan items: delete" on public.meal_plan_items
    for delete using (
        exists (select 1 from public.meal_plans p
                 where p.id = meal_plan_items.meal_plan_id and p.user_id = auth.uid())
    );

-- --- foods -----------------------------------------------------------------
-- Shared reference data: readable by any signed-in user, never writable by one.
drop policy if exists "foods: read for authenticated" on public.foods;
create policy "foods: read for authenticated" on public.foods
    for select to authenticated using (true);

-- ===========================================================================
-- Triggers
-- ===========================================================================
create or replace function public.touch_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists profiles_touch_updated_at on public.profiles;
create trigger profiles_touch_updated_at
    before update on public.profiles
    for each row execute function public.touch_updated_at();

-- ===========================================================================
-- Convenience view: latest weight + latest prediction per user
-- ===========================================================================
-- security_invoker makes the view run with the *querying* user's privileges,
-- so the underlying RLS policies still apply. Without it a view silently
-- becomes an RLS bypass.
create or replace view public.user_dashboard
with (security_invoker = true)
as
select
    p.user_id,
    p.age,
    p.gender,
    p.height_cm,
    p.fitness_goal,
    p.activity_level,
    p.workout_frequency,
    w.weight_kg      as latest_weight_kg,
    w.bmi            as latest_bmi,
    w.bmi_category   as latest_bmi_category,
    w.logged_at      as weight_logged_at,
    pr.calorie_target,
    pr.protein_target,
    pr.created_at    as prediction_created_at
from public.profiles p
left join lateral (
    select * from public.weight_logs wl
     where wl.user_id = p.user_id
     order by wl.logged_at desc limit 1
) w on true
left join lateral (
    select * from public.predictions pd
     where pd.user_id = p.user_id
     order by pd.created_at desc limit 1
) pr on true;

-- ===========================================================================
-- Next step: seed the food catalogue
--   Run data/processed/foods_seed.sql (generated by ml/scripts/prepare_data.py)
-- ===========================================================================
