/** FR7 â€” Two-Month Meal Plan Generation. */
import { useEffect, useState } from 'react';

import {
  Alert,
  Button,
  Card,
  EmptyState,
  SLOT_LABELS,
  Spinner,
  StatCard,
} from '../components/ui.jsx';
import { api } from '../lib/api.js';
import { usePageTitle } from '../hooks/usePageTitle.js';

const DAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
const SLOTS = ['breakfast', 'lunch', 'dinner', 'snack'];

export default function MealPlan() {
  usePageTitle('Two-Month Meal Plan');
  const [plan, setPlan] = useState(null);
  const [items, setItems] = useState([]);
  const [week, setWeek] = useState(1);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const result = await api.getActiveMealPlan();
      setPlan(result.meal_plan);
      setItems(result.items);
      setWeek(1);
    } catch (err) {
      if (err.status === 404) {
        setPlan(null);
        setItems([]);
      } else {
        setError(err);
      }
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleGenerate() {
    setGenerating(true);
    setError(null);
    setNotice(null);
    try {
      const result = await api.generateMealPlan({ weeks: 8 });
      setNotice(
        `Generated ${result.item_count} meals across ${result.meal_plan.week_count} weeks. ` +
          (result.within_tolerance
            ? 'Every week is within Â±5% of your targets.'
            : 'Some weeks fall outside the Â±5% tolerance â€” your food catalogue may be limited.'),
      );
      await load();
    } catch (err) {
      setError(err);
    } finally {
      setGenerating(false);
    }
  }

  if (loading) return <Spinner label="Loading your meal planâ€¦" />;

  if (!plan) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-bold text-slate-900">Two-month meal plan</h1>
        {error && <Alert tone="error">{error.message}</Alert>}
        <EmptyState title="No meal plan yet">
          Generate an eight-week plan built around your calorie and protein targets. Meals do not
          repeat within any five-day window, and each week is checked against your targets.
          <div className="mt-5">
            <Button onClick={handleGenerate} disabled={generating}>
              {generating ? 'Generating (this takes a moment)â€¦' : 'Generate my two-month plan'}
            </Button>
          </div>
        </EmptyState>
      </div>
    );
  }

  const weekItems = items.filter((item) => item.week_number === week);
  const grid = {};
  for (const item of weekItems) {
    grid[item.day_of_week] ??= {};
    (grid[item.day_of_week][item.meal_slot] ??= []).push(item);
  }

  const dayTotals = DAY_ORDER.map((day) => {
    const dayItems = weekItems.filter((item) => item.day_of_week === day);
    return {
      day,
      calories: dayItems.reduce((sum, item) => sum + Number(item.calories), 0),
      protein: dayItems.reduce((sum, item) => sum + Number(item.protein_g), 0),
    };
  }).filter((row) => row.calories > 0);

  const meanCalories = dayTotals.length
    ? dayTotals.reduce((sum, row) => sum + row.calories, 0) / dayTotals.length
    : 0;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Two-month meal plan</h1>
          <p className="mt-1 text-sm text-slate-600">
            Starting {new Date(plan.start_date).toLocaleDateString()} Â· {plan.week_count} weeks
          </p>
        </div>
        <Button variant="secondary" onClick={handleGenerate} disabled={generating}>
          {generating ? 'Regeneratingâ€¦' : 'Regenerate plan'}
        </Button>
      </div>

      {notice && <Alert tone="success">{notice}</Alert>}
      {error && <Alert tone="error">{error.message}</Alert>}

      <div className="grid gap-4 sm:grid-cols-3">
        <StatCard
          label="Daily calorie target"
          value={Math.round(plan.calorie_target)}
          unit="kcal"
          tone="brand"
        />
        <StatCard
          label="Week average delivered"
          value={Math.round(meanCalories)}
          unit="kcal"
          hint={
            plan.calorie_target
              ? `${(((meanCalories - plan.calorie_target) / plan.calorie_target) * 100).toFixed(1)}% vs target`
              : undefined
          }
        />
        <StatCard
          label="Daily protein target"
          value={Math.round(plan.protein_target)}
          unit="g"
          tone="brand"
        />
      </div>

      <nav aria-label="Meal plan weeks">
        <ul className="flex list-none flex-wrap gap-2 p-0">
          {Array.from({ length: plan.week_count }, (_, index) => index + 1).map((value) => (
            <li key={value}>
              <button
                type="button"
                onClick={() => setWeek(value)}
                // Selection is signalled by colour; aria-current carries the
                // same information to assistive technology (WCAG 1.4.1).
                aria-current={week === value ? 'true' : undefined}
                className={`rounded-lg px-3 py-1.5 text-sm font-medium transition focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-600 focus-visible:ring-offset-2 ${
                  week === value
                    ? 'bg-brand-600 text-white'
                    : 'border border-slate-300 bg-white text-slate-700 hover:bg-slate-50'
                }`}
              >
                Week {value}
              </button>
            </li>
          ))}
        </ul>
      </nav>

      <Card title={`Week ${week}`}>
        <div
          className="overflow-x-auto"
          tabIndex={0}
          role="region"
          aria-label={`Week ${week} meal plan, scrollable`}
        >
          <table className="w-full min-w-[820px] border-collapse text-sm">
            <caption className="sr-only">
              Week {week} of {plan.week_count}: meals by day and slot, with daily totals
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
                <th scope="col" className="py-2 text-right">
                  Total
                </th>
              </tr>
            </thead>
            <tbody>
              {DAY_ORDER.filter((day) => grid[day]).map((day) => {
                const totals = dayTotals.find((row) => row.day === day);
                return (
                  <tr key={day} className="border-b border-slate-100 align-top">
                    <th scope="row" className="py-3 pr-3 text-left font-medium text-slate-700">
                      {day}
                    </th>
                    {SLOTS.map((slot) => (
                      <td key={slot} className="py-3 pr-3 text-slate-700">
                        {(grid[day][slot] ?? []).map((item) => (
                          <div key={`${item.food_name}-${item.position}`} className="mb-1.5">
                            <span className="block leading-tight">{item.food_name}</span>
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
                    <td className="py-3 text-right text-xs text-slate-600">
                      <span className="block font-semibold text-slate-700">
                        {Math.round(totals?.calories ?? 0)} kcal
                      </span>
                      {Math.round(totals?.protein ?? 0)} g protein
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
