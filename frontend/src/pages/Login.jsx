/** FR1 â€” User Registration and Authentication. */
import { useState } from 'react';

import { Alert, Button, Disclaimer, Field, inputClass } from '../components/ui.jsx';
import { useAuth } from '../hooks/useAuth.jsx';
import { usePageTitle } from '../hooks/usePageTitle.js';

export default function Login() {
  usePageTitle('Sign in');
  const { signIn, signUp } = useAuth();
  const [mode, setMode] = useState('signin');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [status, setStatus] = useState(null);
  const [busy, setBusy] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    setStatus(null);

    if (password.length < 8) {
      setStatus({ tone: 'error', message: 'Password must be at least 8 characters.' });
      return;
    }

    setBusy(true);
    try {
      const { error } =
        mode === 'signin' ? await signIn(email, password) : await signUp(email, password);

      if (error) {
        setStatus({ tone: 'error', message: error.message });
      } else if (mode === 'signup') {
        setStatus({
          tone: 'success',
          message:
            'Account created. If your Supabase project has email confirmation enabled, check your inbox before signing in.',
        });
      }
      // On successful sign-in the auth listener redirects automatically.
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex min-h-full items-center justify-center px-4 py-12">
      <div className="w-full max-w-md space-y-6">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-brand-600">NutriFit&#8209;AI</h1>
          <p className="mt-1 text-sm text-slate-600">
            Personalised nutrition and meal planning for gym users
          </p>
        </div>

        <form
          onSubmit={handleSubmit}
          className="space-y-4 rounded-xl border border-slate-200 bg-white p-6 shadow-sm"
        >
          <div className="flex rounded-lg bg-slate-100 p-1" role="group" aria-label="Choose action">
            {[
              ['signin', 'Sign in'],
              ['signup', 'Create account'],
            ].map(([value, label]) => (
              <button
                key={value}
                type="button"
                aria-pressed={mode === value}
                onClick={() => {
                  setMode(value);
                  setStatus(null);
                }}
                className={`flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-600 ${
                  mode === value ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-700'
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          <Field label="Email" required>
            <input
              type="email"
              required
              autoComplete="email"
              className={inputClass}
              value={email}
              onChange={(event) => setEmail(event.target.value)}
            />
          </Field>

          <Field label="Password" required hint="At least 8 characters.">
            <input
              type="password"
              required
              minLength={8}
              autoComplete={mode === 'signin' ? 'current-password' : 'new-password'}
              className={inputClass}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
          </Field>

          {status && <Alert tone={status.tone}>{status.message}</Alert>}

          <Button type="submit" disabled={busy} className="w-full">
            {busy ? 'Please waitâ€¦' : mode === 'signin' ? 'Sign in' : 'Create account'}
          </Button>
          {/* Announces busy state to screen readers without a visual change. */}
          <span aria-live="polite" className="sr-only">
            {busy ? 'Submitting, please wait' : ''}
          </span>
        </form>

        <Disclaimer />
      </div>
    </div>
  );
}
