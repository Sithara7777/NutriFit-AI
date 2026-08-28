/**
 * FR6 â€” Personalised Meal Recommendation.
 *
 * Every suggestion shows its full macro breakdown, not just a name â€” the
 * proposal's Expected Outputs list "nutritional information and calorie
 * analysis for recommended meals" as a deliverable in its own right.
 */
import { useEffect, useState } from 'react';

import {
  Alert,
  Button,
  Card,
  SLOT_LABELS,
  Spinner,
  StatCard,
} from '../components/ui.jsx';
import { api } from '../lib/api.js';
import { usePageTitle } from '../hooks/usePageTitle.js';

const SLOTS = ['breakfast', 'lunch', 'dinner', 'snack'];

function MacroBar({ label, value, unit = 'g' }) {
  // The number and its label are visually adjacent but separate elements, so
  // they are joined into one accessible string to avoid "23" being announced
  // with no indication of what it measures.
  return (
    <div className="text-center">
      <p className="text-sm font-semibold text-slate-900">
        <span className="sr-only">{`${label}: ${Math.round(value)} ${unit}`}</span>
        <span aria-hidden="true">
          {Math.round(value)}
          <span className="ml-0.5 text-xs font-normal text-slate-600">{unit}</span>
        </span>
      </p>
      <p aria-hidden="true" className="text-[11px] uppercase tracking-wide text-slate-600">
        {label}
      </p>
    </div>
  );
}

function MealCard({ item }) {
  return (
    <li className="list-none rounded-lg border border-slate-200 p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="font-medium text-slate-900">{item.name}</p>
          <p className="text-xs text-slate-600">
            {item.category}
            {Number(item.servings) !== 1 && ` Â· ${item.servings} servings`}
          </p>
        </div>
        <span className="shrink-0 rounded-full bg-brand-50 px-2.5 py-1 text-xs font-medium text-brand-700">
          {Math.round(item.calories)} kcal
        </span>
      </div>
      <div className="mt-3 grid grid-cols-4 gap-2 border-t border-slate-100 pt-3">
        <MacroBar label="Protein" value={item.protein_g} />
        <MacroBar label="Carbs" value={item.carbs_g} />
        <MacroBar label="Fat" value={item.fat_g} />
        <MacroBar label="Fibre" value={item.fiber_g} />
      </div>
    </li>
  );
}

export default function Recommendations() {
  usePageTitle('Meal Recommendations');
  const [slot, setSlot] = useState(null);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    api
      .getRecommendations(slot, 6)
      .then((result) => active && setData(result))
      .catch((err) => active && setError(err))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [slot]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Meal recommendations</h1>
        <p className="mt-1 text-sm text-slate-600">
          Content-based suggestions matched to your remaining macro budget.
        </p>
      </div>

      <div className="flex flex-wrap gap-2" role="group" aria-label="Filter by meal slot">
        <Button
          variant={slot === null ? 'primary' : 'secondary'}
          aria-pressed={slot === null}
          onClick={() => setSlot(null)}
        >
          Full day
        </Button>
        {SLOTS.map((value) => (
          <Button
            key={value}
            variant={slot === value ? 'primary' : 'secondary'}
            aria-pressed={slot === value}
            onClick={() => setSlot(value)}
          >
            {SLOT_LABELS[value]}
          </Button>
        ))}
      </div>

      {error && (
        <Alert tone="error" title="Could not load recommendations">
          {error.message}
        </Alert>
      )}

      {loading && <Spinner label="Finding meals for youâ€¦" />}

      {!loading && !error && data && (
        <>
          {data.targets && (
            <div className="grid gap-4 sm:grid-cols-2">
              <StatCard
                label="Daily calorie target"
                value={Math.round(data.targets.calorie_target)}
                unit="kcal"
                tone="brand"
              />
              <StatCard
                label="Daily protein target"
                value={Math.round(data.targets.protein_target)}
                unit="g"
                tone="brand"
              />
            </div>
          )}

          {/* Single-slot menu */}
          {data.suggestions && (
            <Card
              title={`${SLOT_LABELS[data.meal_slot]} suggestions`}
              subtitle="Ranked by how well they fit your remaining budget for this slot."
            >
              <ul className="grid list-none gap-3 p-0 md:grid-cols-2">
                {data.suggestions.map((item) => (
                  <MealCard key={item.food_id} item={item} />
                ))}
              </ul>
            </Card>
          )}

          {/* Composed full day */}
          {data.slots && (
            <>
              <Card title="Today's totals">
                <div className="grid grid-cols-4 gap-3">
                  <MacroBar label="Calories" value={data.totals.calories} unit="kcal" />
                  <MacroBar label="Protein" value={data.totals.protein_g} />
                  <MacroBar label="Carbs" value={data.totals.carbs_g} />
                  <MacroBar label="Fat" value={data.totals.fat_g} />
                </div>
                <p className="mt-3 text-center text-xs text-slate-500">
                  Target: {Math.round(data.targets.calories ?? data.targets.calorie_target)} kcal Â·{' '}
                  {Math.round(data.targets.protein_g ?? data.targets.protein_target)} g protein
                </p>
              </Card>

              <div className="grid gap-4 md:grid-cols-2">
                {SLOTS.filter((value) => data.slots[value]).map((value) => {
                  const slotPlan = data.slots[value];
                  return (
                    <Card
                      key={value}
                      title={SLOT_LABELS[value]}
                      subtitle={`${Math.round(slotPlan.totals.calories)} kcal Â· ${Math.round(
                        slotPlan.totals.protein_g,
                      )} g protein`}
                    >
                      <ul className="list-none space-y-3 p-0">
                        {slotPlan.items.map((item, index) => (
                          <MealCard key={`${item.food_id}-${index}`} item={item} />
                        ))}
                      </ul>
                    </Card>
                  );
                })}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
