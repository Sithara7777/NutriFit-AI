/** FR2 â€” User Profile Management (also the onboarding flow). */
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import {
  Alert,
  Button,
  Card,
  Disclaimer,
  Field,
  GOAL_LABELS,
  Spinner,
  inputClass,
} from '../components/ui.jsx';
import { api } from '../lib/api.js';
import { usePageTitle } from '../hooks/usePageTitle.js';

const EMPTY = {
  age: '',
  gender: 'Male',
  height_cm: '',
  weight_kg: '',
  fitness_goal: 'maintenance',
  workout_frequency: '3',
  session_duration_h: '1.25',
  experience_level: '2',
  body_fat_pct: '',
};

export default function Profile() {
  usePageTitle('Profile');
  const navigate = useNavigate();
  const [form, setForm] = useState(EMPTY);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState(null);
  const [isNew, setIsNew] = useState(true);

  useEffect(() => {
    let active = true;
    api
      .getProfile()
      .then(({ profile, latest_weight: latestWeight }) => {
        if (!active) return;
        setIsNew(false);
        setForm({
          age: String(profile.age),
          gender: profile.gender,
          height_cm: String(profile.height_cm),
          weight_kg: latestWeight ? String(latestWeight.weight_kg) : '',
          fitness_goal: profile.fitness_goal,
          workout_frequency: String(profile.workout_frequency),
          session_duration_h: String(profile.session_duration_h),
          experience_level: String(profile.experience_level),
          body_fat_pct: profile.body_fat_pct == null ? '' : String(profile.body_fat_pct),
        });
      })
      .catch((error) => {
        // 404 simply means this user has not onboarded yet.
        if (active && error.status !== 404) {
          setStatus({ tone: 'error', message: error.message });
        }
      })
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, []);

  function update(key, value) {
    setForm((previous) => ({ ...previous, [key]: value }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setSaving(true);
    setStatus(null);

    try {
      await api.saveProfile({
        age: Number(form.age),
        gender: form.gender,
        height_cm: Number(form.height_cm),
        weight_kg: Number(form.weight_kg),
        fitness_goal: form.fitness_goal,
        workout_frequency: Number(form.workout_frequency),
        session_duration_h: Number(form.session_duration_h),
        experience_level: Number(form.experience_level),
        body_fat_pct: form.body_fat_pct === '' ? null : Number(form.body_fat_pct),
      });

      // Saving a profile immediately produces targets, so the dashboard is
      // never shown in a half-configured state.
      await api.predict();

      setStatus({ tone: 'success', message: 'Profile saved and targets recalculated.' });
      setTimeout(() => navigate('/'), 700);
    } catch (error) {
      setStatus({
        tone: 'error',
        message: error.details
          ? `${error.message} (${error.details.map((d) => `${d.field}: ${d.message}`).join('; ')})`
          : error.message,
      });
    } finally {
      setSaving(false);
    }
  }

  if (loading) return <Spinner label="Loading your profileâ€¦" />;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">
          {isNew ? 'Set up your profile' : 'Your profile'}
        </h1>
        <p className="mt-1 text-sm text-slate-600">
          These details drive your calorie and protein targets. Keep them up to date.
        </p>
      </div>

      {isNew && <Disclaimer />}

      <form onSubmit={handleSubmit} className="space-y-6">
        <Card title="About you">
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Age" required hint="16â€“80 years">
              <input
                type="number"
                required
                min={16}
                max={80}
                className={inputClass}
                value={form.age}
                onChange={(event) => update('age', event.target.value)}
              />
            </Field>

            <Field label="Gender" required hint="Used by the BMR equation.">
              <select
                className={inputClass}
                value={form.gender}
                onChange={(event) => update('gender', event.target.value)}
              >
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
            </Field>

            <Field label="Height (cm)" required hint="120â€“230 cm">
              <input
                type="number"
                required
                min={120}
                max={230}
                step="0.1"
                className={inputClass}
                value={form.height_cm}
                onChange={(event) => update('height_cm', event.target.value)}
              />
            </Field>

            <Field label="Current weight (kg)" required hint="30â€“250 kg">
              <input
                type="number"
                required
                min={30}
                max={250}
                step="0.1"
                className={inputClass}
                value={form.weight_kg}
                onChange={(event) => update('weight_kg', event.target.value)}
              />
            </Field>

            <Field
              label="Body fat % (optional)"
              hint="If you've had a body-composition scan, entering it gives a more accurate estimate (Katchâ€“McArdle instead of Mifflinâ€“St Jeor). Leave blank and we'll estimate it."
            >
              <input
                type="number"
                min={3}
                max={60}
                step="0.1"
                className={inputClass}
                value={form.body_fat_pct}
                onChange={(event) => update('body_fat_pct', event.target.value)}
              />
            </Field>
          </div>
        </Card>

        <Card title="Training">
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Workout days per week" required>
              <input
                type="number"
                required
                min={0}
                max={7}
                className={inputClass}
                value={form.workout_frequency}
                onChange={(event) => update('workout_frequency', event.target.value)}
              />
            </Field>

            <Field label="Typical session length (hours)" required hint="e.g. 1.25 for 75 minutes">
              <input
                type="number"
                required
                min={0.1}
                max={5}
                step="0.05"
                className={inputClass}
                value={form.session_duration_h}
                onChange={(event) => update('session_duration_h', event.target.value)}
              />
            </Field>

            <Field label="Experience level" required>
              <select
                className={inputClass}
                value={form.experience_level}
                onChange={(event) => update('experience_level', event.target.value)}
              >
                <option value="1">Beginner</option>
                <option value="2">Intermediate</option>
                <option value="3">Advanced</option>
              </select>
            </Field>

            <Field label="Fitness goal" required>
              <select
                className={inputClass}
                value={form.fitness_goal}
                onChange={(event) => update('fitness_goal', event.target.value)}
              >
                {Object.entries(GOAL_LABELS).map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </Field>
          </div>
        </Card>

        {status && <Alert tone={status.tone}>{status.message}</Alert>}

        <div className="flex gap-3">
          <Button type="submit" disabled={saving}>
            {saving ? 'Savingâ€¦' : isNew ? 'Create profile & calculate targets' : 'Save changes'}
          </Button>
          {!isNew && (
            <Button type="button" variant="secondary" onClick={() => navigate('/')}>
              Cancel
            </Button>
          )}
        </div>
      </form>
    </div>
  );
}
