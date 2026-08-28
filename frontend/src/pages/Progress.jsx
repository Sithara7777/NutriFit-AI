/**
 * FR8 â€” Progress Monitoring, and FR9 â€” Recommendation Adjustment.
 *
 * Logging a weight triggers a recalculation server-side. If the new targets
 * differ from the active plan by more than the drift threshold, the user is
 * *prompted* to regenerate â€” the plan is never silently overwritten.
 */
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
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
  Field,
  Spinner,
  StatCard,
  inputClass,
} from '../components/ui.jsx';
import { api } from '../lib/api.js';
import { usePageTitle } from '../hooks/usePageTitle.js';

export default function Progress() {
  usePageTitle('Progress');
  const navigate = useNavigate();
  const [history, setHistory] = useState(null);
  const [weight, setWeight] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  async function load() {
    try {
      setHistory(await api.getProgress());
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function handleSubmit(event) {
    event.preventDefault();
    setSaving(true);
    setError(null);
    setResult(null);
    try {
      const response = await api.logProgress(Number(weight));
      setResult(response);
      setWeight('');
      await load();
    } catch (err) {
      setError(err);
    } finally {
      setSaving(false);
    }
  }

  if (loading) return <Spinner label="Loading your progressâ€¦" />;

  const logs = history?.weight_logs ?? [];
  const summary = history?.summary;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Progress</h1>
        <p className="mt-1 text-sm text-slate-600">
          Log your weight regularly â€” your targets are recalculated each time.
        </p>
      </div>

      {error && <Alert tone="error">{error.message}</Alert>}

      {result && (
        <Alert
          tone={result.needs_regeneration ? 'warn' : 'success'}
          title={result.needs_regeneration ? 'Your targets have changed' : 'Weight logged'}
          action={
            result.needs_regeneration ? (
              <Button onClick={() => navigate('/meal-plan')}>Regenerate my meal plan</Button>
            ) : null
          }
        >
          {result.message}
          {result.drift && (
            <p className="mt-2 text-xs">
              Calorie target moved {result.drift.calorie_drift_pct}% Â· protein{' '}
              {result.drift.protein_drift_pct}% (threshold {result.drift.threshold_pct}%). New
              targets: {Math.round(result.prediction.calorie_target)} kcal Â·{' '}
              {Math.round(result.prediction.protein_target)} g protein.
            </p>
          )}
        </Alert>
      )}

      {summary && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard label="Starting weight" value={summary.starting_weight_kg} unit="kg" />
          <StatCard label="Current weight" value={summary.latest_weight_kg} unit="kg" tone="brand" />
          <StatCard
            label="Change"
            value={`${summary.change_kg > 0 ? '+' : ''}${summary.change_kg}`}
            unit="kg"
            hint={`over ${summary.entries} entries`}
          />
          <StatCard
            label="Current BMI"
            value={summary.latest_bmi.toFixed(1)}
            hint={summary.latest_bmi_category}
            tone={BMI_TONE[summary.latest_bmi_category]}
          />
        </div>
      )}

      <Card title="Log a new weight">
        <form onSubmit={handleSubmit} className="flex flex-wrap items-end gap-4">
          <div className="w-48">
            <Field label="Weight (kg)" required hint="Between 30 and 250 kg">
              <input
                type="number"
                required
                min={30}
                max={250}
                step="0.1"
                inputMode="decimal"
                autoComplete="off"
                className={inputClass}
                value={weight}
                onChange={(event) => setWeight(event.target.value)}
              />
            </Field>
          </div>
          <Button type="submit" disabled={saving}>
            {saving ? 'Savingâ€¦' : 'Log weight'}
          </Button>
        </form>
      </Card>

      <Card title="Weight history">
        {logs.length > 1 ? (
          <AccessibleChart
            title="Weight and BMI history"
            summary={
              `Line chart of ${logs.length} entries. Weight moved from ` +
              `${logs[0].weight_kg} kg to ${logs.at(-1).weight_kg} kg; BMI from ` +
              `${Number(logs[0].bmi).toFixed(1)} to ${Number(logs.at(-1).bmi).toFixed(1)}.`
            }
            columns={['Date', 'Weight (kg)', 'BMI', 'Category']}
            rows={logs.map((log) => [
              new Date(log.logged_at).toLocaleDateString(),
              log.weight_kg,
              Number(log.bmi).toFixed(1),
              log.bmi_category,
            ])}
          >
            <ResponsiveContainer width="100%" height={300}>
              <LineChart
                data={logs.map((log) => ({
                  ...log,
                  label: new Date(log.logged_at).toLocaleDateString(undefined, {
                    month: 'short',
                    day: 'numeric',
                  }),
                  weight_kg: Number(log.weight_kg),
                  bmi: Number(log.bmi),
                }))}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="label" tick={{ fontSize: 12 }} />
                <YAxis
                  yAxisId="kg"
                  domain={['dataMin - 2', 'dataMax + 2']}
                  tick={{ fontSize: 12 }}
                  unit="kg"
                />
                <YAxis
                  yAxisId="bmi"
                  orientation="right"
                  domain={['dataMin - 1', 'dataMax + 1']}
                  tick={{ fontSize: 12 }}
                />
                <Tooltip />
                <Line
                  yAxisId="kg"
                  type="monotone"
                  dataKey="weight_kg"
                  name="Weight (kg)"
                  stroke="#24704a"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
                <Line
                  yAxisId="bmi"
                  type="monotone"
                  dataKey="bmi"
                  name="BMI"
                  stroke="#64748b"
                  strokeWidth={2}
                  strokeDasharray="4 4"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </AccessibleChart>
        ) : (
          <p className="py-10 text-center text-sm text-slate-600">
            Log at least two weights to see your trend.
          </p>
        )}

        {logs.length > 0 && (
          <div
            className="mt-6 overflow-x-auto"
            tabIndex={0}
            role="region"
            aria-label="Recent weight entries, scrollable"
          >
            <table className="w-full min-w-[420px] border-collapse text-sm">
              <caption className="sr-only">Your 12 most recent weight entries</caption>
              <thead>
                <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-600">
                  <th scope="col" className="py-2">
                    Date
                  </th>
                  <th scope="col" className="py-2">
                    Weight
                  </th>
                  <th scope="col" className="py-2">
                    BMI
                  </th>
                  <th scope="col" className="py-2">
                    Category
                  </th>
                </tr>
              </thead>
              <tbody>
                {[...logs].reverse().slice(0, 12).map((log) => (
                  <tr key={log.id} className="border-b border-slate-100">
                    <th scope="row" className="py-2 text-left font-normal text-slate-700">
                      {new Date(log.logged_at).toLocaleDateString()}
                    </th>
                    <td className="py-2 font-medium text-slate-800">{log.weight_kg} kg</td>
                    <td className="py-2 text-slate-700">{Number(log.bmi).toFixed(1)}</td>
                    <td className="py-2 capitalize text-slate-700">{log.bmi_category}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
