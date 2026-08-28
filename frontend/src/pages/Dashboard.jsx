/**
 * FR10 â€” Dashboard and Reporting.
 *
 * Shows everything the proposal specifies: current BMI and category, calorie
 * and protein targets with history, this week's meal plan at a glance, and the
 * weight trend line.
 */
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import {
  AccessibleChart,
  Alert,
  BMI_TONE,
  Button,
  Card,
  EmptyState,
  GOAL_LABELS,
  SLOT_LABELS,
  Spinner,
  StatCard,
} from '../components/ui.jsx';
import { api } from '../lib/api.js';
import { usePageTitle } from '../hooks/usePageTitle.js';

const DAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
const SLOTS = ['breakfast', 'lunch', 'dinner', 'snack'];

function formatDate(value) {
  return new Date(value).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

export default function Dashboard() {
  usePageTitle('Dashboard');
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    api
      .getDashboard()
      .then((result) => active && setData(result))
      .catch((err) => active && setError(err))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, []);

  if (loading) return <Spinner label="Loading your dashboardâ€¦" />;
  if (error) return <Alert tone="error">{error.message}</Alert>;

  if (!data?.profile) {
    return (
      <EmptyState title="Welcome to NutriFit-AI" to="/profile" cta="Set up your profile">
        Tell us a little about yourself and we'll calculate your personalised daily calorie and
        protein targets, then build you an eight-week meal plan.
      </EmptyState>
    );
  }

  const { profile, bmi, targets, weight_trend: weightTrend, current_week: currentWeek } = data;

  // Group this week's items into a day x slot grid.
  const grid = {};
  for (const item of currentWeek?.items ?? []) {
    grid[item.day_of_week] ??= {};
    (grid[item.day_of_week][item.meal_slot] ??= []).push(item);
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Dashboard</h1>
          <p className="mt-1 text-sm text-slate-600">
            Goal: <strong>{GOAL_LABELS[profile.fitness_goal]}</strong> Â· {profile.workout_frequency}{' '}
            sessions/week Â· {profile.activity_level.replace('_', ' ')}
          </p>
        </div>
        <Button as={Link} to="/progress" variant="secondary">
          Log today's weight
        </Button>
      </div>

      {!targets && (
        <Alert
          tone="warn"
          title="No targets yet"
          action={
            <Button as={Link} to="/profile">
              Calculate targets
            </Button>
          }
        >
          Save your profile to generate your calorie and protein targets.
        </Alert>
      )}

      {/* ---- headline stats ------------------------------------------- */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Body Mass Index"
          value={bmi ? bmi.value.toFixed(1) : 'â€”'}
          hint={bmi ? `${bmi.category} Â· ${bmi.weight_kg} kg` : 'Log a weight to see your BMI'}
          tone={bmi ? BMI_TONE[bmi.category] : 'default'}
        />
        <StatCard
          label="Daily calories"
          value={targets ? Math.round(targets.calorie_target) : 'â€”'}
          unit="kcal"
          hint={targets ? `TDEE â‰ˆ ${Math.round(targets.tdee ?? 0)} kcal` : undefined}
          tone="brand"
        />
        <StatCard
          label="Daily protein"
          value={targets ? Math.round(targets.protein_target) : 'â€”'}
          unit="g"
          hint={
            targets && bmi
              ? `${(targets.protein_target / bmi.weight_kg).toFixed(1)} g per kg bodyweight`
              : undefined
          }
          tone="brand"
        />
        <StatCard
          label="Weight change"
          value={
            data.progress_summary
              ? `${data.progress_summary.change_kg > 0 ? '+' : ''}${data.progress_summary.change_kg}`
              : 'â€”'
          }
          unit={data.progress_summary ? 'kg' : undefined}
          hint={
            data.progress_summary
              ? `across ${data.progress_summary.entries} entries`
              : 'Log twice to see a trend'
          }
        />
      </div>

      {targets?.source === 'formula' && (
        <Alert tone="warn" title="Estimated without the trained model">
          The prediction service is running in fallback mode and used the physiological formulas
          directly. Your targets are still valid, but the ML models are not currently loaded.
        </Alert>
      )}

      {/* ---- charts ---------------------------------------------------- */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Weight trend" subtitle="Every weight you have logged">
          {weightTrend.length > 1 ? (
            <AccessibleChart
              title="Weight trend over time"
              summary={
                `Line chart of ${weightTrend.length} weight entries, from ` +
                `${weightTrend[0].weight_kg} kg on ${formatDate(weightTrend[0].logged_at)} to ` +
                `${weightTrend.at(-1).weight_kg} kg on ${formatDate(weightTrend.at(-1).logged_at)}.`
              }
              columns={['Date', 'Weight (kg)', 'BMI']}
              rows={weightTrend.map((d) => [
                formatDate(d.logged_at),
                d.weight_kg,
                d.bmi.toFixed(1),
              ])}
            >
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={weightTrend.map((d) => ({ ...d, label: formatDate(d.logged_at) }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="label" tick={{ fontSize: 12 }} />
                  <YAxis domain={['dataMin - 2', 'dataMax + 2']} tick={{ fontSize: 12 }} unit="kg" />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="weight_kg"
                    name="Weight (kg)"
                    stroke="#24704a"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </AccessibleChart>
          ) : (
            <p className="py-10 text-center text-sm text-slate-600">
              Log at least two weights to see your trend.
            </p>
          )}
        </Card>

        <Card title="Target history" subtitle="How your targets changed as you progressed">
          {data.target_history.length > 1 ? (
            <AccessibleChart
              title="Calorie and protein target history"
              summary={
                `Line chart of ${data.target_history.length} recalculations. Calories moved from ` +
                `${Math.round(data.target_history[0].calorie_target)} to ` +
                `${Math.round(data.target_history.at(-1).calorie_target)} kcal, protein from ` +
                `${Math.round(data.target_history[0].protein_target)} to ` +
                `${Math.round(data.target_history.at(-1).protein_target)} g.`
              }
              columns={['Date', 'Calories (kcal)', 'Protein (g)']}
              rows={data.target_history.map((d) => [
                formatDate(d.created_at),
                Math.round(d.calorie_target),
                Math.round(d.protein_target),
              ])}
            >
              <ResponsiveContainer width="100%" height={260}>
                <LineChart
                  data={data.target_history.map((d) => ({ ...d, label: formatDate(d.created_at) }))}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="label" tick={{ fontSize: 12 }} />
                  <YAxis yAxisId="kcal" tick={{ fontSize: 12 }} />
                  <YAxis yAxisId="g" orientation="right" tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Line
                    yAxisId="kcal"
                    type="monotone"
                    dataKey="calorie_target"
                    name="Calories (kcal)"
                    stroke="#24704a"
                    strokeWidth={2}
                  />
                  <Line
                    yAxisId="g"
                    type="monotone"
                    dataKey="protein_target"
                    name="Protein (g)"
                    stroke="#b45309"
                    strokeWidth={2}
                    strokeDasharray="5 3"
                  />
                </LineChart>
              </ResponsiveContainer>
            </AccessibleChart>
          ) : (
            <p className="py-10 text-center text-sm text-slate-600">
              Your targets will be charted here as they are recalculated.
            </p>
          )}
        </Card>
      </div>

      {/* ---- this week's plan ------------------------------------------ */}
      <Card
        title="This week's meal plan"
        subtitle={
          currentWeek
            ? `Week ${currentWeek.week_number} of ${data.active_meal_plan.week_count}`
            : undefined
        }
        action={
          <Button as={Link} to="/meal-plan" variant="secondary">
            View full plan
          </Button>
        }
      >
        {currentWeek?.items?.length ? (
          // tabIndex makes a horizontally scrolling region keyboard-scrollable
          // (WCAG 2.1.1); the accessible name tells the user what it contains.
          <div
            className="overflow-x-auto"
            tabIndex={0}
            role="region"
            aria-label={`Meal plan for week ${currentWeek.week_number}, scrollable`}
          >
            <table className="w-full min-w-[720px] border-collapse text-sm">
              <caption className="sr-only">
                Week {currentWeek.week_number} meal plan, by day and meal slot
              </caption>
              <thead>
                <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-600">
                  <th scope="col" className="py-2 pr-3">
                    Day
                  </th>
                  {SLOTS.map((slot) => (
                    <th scope="col" key={slot} className="py-2 pr-3">
                      {SLOT_LABELS[slot]}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {DAY_ORDER.filter((day) => grid[day]).map((day) => (
                  <tr key={day} className="border-b border-slate-100 align-top">
                    <th scope="row" className="py-3 pr-3 text-left font-medium text-slate-700">
                      <span aria-hidden="true">{day.slice(0, 3)}</span>
                      <span className="sr-only">{day}</span>
                    </th>
                    {SLOTS.map((slot) => (
                      <td key={slot} className="py-3 pr-3 text-slate-700">
                        {(grid[day][slot] ?? []).map((item) => (
                          <div key={`${item.food_name}-${item.position}`} className="mb-1">
                            <span className="block">{item.food_name}</span>
                            <span className="text-xs text-slate-600">
                              {Number(item.servings) !== 1 && `${item.servings}Ã— Â· `}
                              {Math.round(item.calories)} kcal Â· {Math.round(item.protein_g)} g
                              <span className="sr-only"> protein</span>
                              <span aria-hidden="true"> P</span>
                            </span>
                          </div>
                        ))}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="py-8 text-center">
            <p className="text-sm text-slate-600">You don't have a meal plan yet.</p>
            <Button as={Link} to="/meal-plan" className="mt-4">
              Generate my two-month plan
            </Button>
          </div>
        )}
      </Card>
    </div>
  );
}
