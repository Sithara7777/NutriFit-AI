/**
 * Small shared presentational components.
 *
 * Accessibility notes (WCAG 2.1 AA, Cardiff Met EDGE — ETHICAL):
 * - every interactive element carries a visible `focus-visible` ring, so the
 *   interface is operable by keyboard alone (2.4.7 Focus Visible);
 * - form controls use explicit `id`/`htmlFor` pairing rather than relying on
 *   implicit label nesting, which several screen readers announce unreliably
 *   (1.3.1 Info and Relationships, 3.3.2 Labels or Instructions);
 * - validation errors are wired to their input via `aria-describedby` and
 *   `aria-invalid` (3.3.1 Error Identification);
 * - status messages announce themselves via live regions (4.1.3 Status Messages);
 * - colour is never the sole carrier of meaning — tone is always paired with
 *   text (1.4.1 Use of Colour).
 */
import { cloneElement, isValidElement, useId } from 'react';
import { Link } from 'react-router-dom';

export function Card({ title, subtitle, children, className = '', action }) {
  const headingId = useId();
  return (
    <section
      className={`rounded-xl border border-slate-200 bg-white p-5 shadow-sm ${className}`}
      aria-labelledby={title ? headingId : undefined}
    >
      {(title || action) && (
        <header className="mb-4 flex items-start justify-between gap-4">
          <div>
            {title && (
              <h2 id={headingId} className="text-base font-semibold text-slate-900">
                {title}
              </h2>
            )}
            {subtitle && <p className="mt-0.5 text-sm text-slate-500">{subtitle}</p>}
          </div>
          {action}
        </header>
      )}
      {children}
    </section>
  );
}

export function StatCard({ label, value, unit, hint, tone = 'default' }) {
  const tones = {
    default: 'text-slate-900',
    brand: 'text-brand-700',
    warn: 'text-amber-700',
    danger: 'text-red-700',
  };
  // The value is often a bare number; the label alone gives it meaning, so the
  // two are announced together as one accessible name.
  const accessibleValue = [value, unit].filter(Boolean).join(' ');
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-600">{label}</p>
      <p
        className={`mt-2 text-3xl font-semibold ${tones[tone] ?? tones.default}`}
        aria-label={`${label}: ${accessibleValue}${hint ? `. ${hint}` : ''}`}
      >
        <span aria-hidden="true">
          {value}
          {unit && <span className="ml-1 text-lg font-normal text-slate-600">{unit}</span>}
        </span>
      </p>
      {hint && (
        <p className="mt-1 text-xs text-slate-600" aria-hidden="true">
          {hint}
        </p>
      )}
    </div>
  );
}

const FOCUS_RING =
  'focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-600 focus-visible:ring-offset-2';

export function Button({
  children,
  variant = 'primary',
  className = '',
  as: Component = 'button',
  type,
  ...props
}) {
  const variants = {
    primary: 'bg-brand-600 text-white hover:bg-brand-700 disabled:bg-brand-600/50',
    secondary: 'bg-white text-slate-700 border border-slate-300 hover:bg-slate-50',
    ghost: 'text-slate-700 hover:bg-slate-100',
    danger: 'bg-red-700 text-white hover:bg-red-800',
  };
  // An explicit type prevents a <button> inside a form defaulting to submit.
  const resolvedType = Component === 'button' ? (type ?? 'button') : type;
  return (
    <Component
      type={resolvedType}
      className={`inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition disabled:cursor-not-allowed ${FOCUS_RING} ${variants[variant]} ${className}`}
      {...props}
    >
      {children}
    </Component>
  );
}

/**
 * A labelled form control.
 *
 * The accessibility wiring (`id`, `aria-describedby`, `aria-invalid`,
 * `aria-required`) is injected into the child element automatically via
 * `cloneElement`, so every existing call site gains a correctly associated
 * label without being rewritten. A render function is also accepted for the
 * rare case of a control that needs the props applied somewhere non-obvious.
 */
export function Field({ label, hint, error, children, required }) {
  const id = useId();
  const hintId = `${id}-hint`;
  const errorId = `${id}-error`;
  const describedBy = [hint ? hintId : null, error ? errorId : null]
    .filter(Boolean)
    .join(' ');

  const controlProps = {
    id,
    'aria-describedby': describedBy || undefined,
    'aria-invalid': error ? true : undefined,
    'aria-required': required || undefined,
  };

  return (
    <div className="block">
      <label htmlFor={id} className="mb-1 block text-sm font-medium text-slate-700">
        {label}
        {required && (
          <>
            <span aria-hidden="true" className="ml-0.5 text-red-600">
              *
            </span>
            <span className="sr-only"> (required)</span>
          </>
        )}
      </label>

      {typeof children === 'function'
        ? children(controlProps)
        : isValidElement(children)
          ? cloneElement(children, controlProps)
          : children}

      {hint && (
        <span id={hintId} className="mt-1 block text-xs text-slate-600">
          {hint}
        </span>
      )}
      {error && (
        <span id={errorId} className="mt-1 block text-xs text-red-700">
          {error}
        </span>
      )}
    </div>
  );
}

// `aria-invalid` is not one of Tailwind's built-in aria variants, so it is
// written as an arbitrary variant rather than `aria-invalid:`.
export const inputClass =
  'w-full rounded-lg border border-slate-300 px-3 py-2 text-sm shadow-sm outline-none focus:border-brand-600 focus-visible:ring-2 focus-visible:ring-brand-600 focus-visible:ring-offset-1 aria-[invalid=true]:border-red-600';

/**
 * Status / error message.
 *
 * `role="alert"` for errors so assistive technology interrupts and announces
 * immediately; `role="status"` (polite) otherwise so success messages do not
 * cut across whatever the user is doing.
 */
export function Alert({ tone = 'info', title, children, action }) {
  const tones = {
    info: 'border-sky-300 bg-sky-50 text-sky-900',
    success: 'border-emerald-300 bg-emerald-50 text-emerald-900',
    warn: 'border-amber-300 bg-amber-50 text-amber-900',
    error: 'border-red-300 bg-red-50 text-red-900',
  };
  // Text prefix so the meaning does not depend on colour alone.
  const prefixes = {
    info: 'Information',
    success: 'Success',
    warn: 'Warning',
    error: 'Error',
  };
  const isError = tone === 'error';
  return (
    <div
      role={isError ? 'alert' : 'status'}
      aria-live={isError ? 'assertive' : 'polite'}
      className={`rounded-lg border p-4 text-sm ${tones[tone]}`}
    >
      <span className="sr-only">{prefixes[tone]}: </span>
      {title && <p className="mb-1 font-semibold">{title}</p>}
      <div>{children}</div>
      {action && <div className="mt-3">{action}</div>}
    </div>
  );
}

export function Spinner({ label = 'Loading…' }) {
  return (
    <div
      role="status"
      aria-live="polite"
      className="flex items-center justify-center gap-3 py-12 text-sm text-slate-600"
    >
      <span
        aria-hidden="true"
        className="h-5 w-5 animate-spin rounded-full border-2 border-slate-300 border-t-brand-600 motion-reduce:animate-none"
      />
      {label}
    </div>
  );
}

export function EmptyState({ title, children, to, cta }) {
  return (
    <div className="rounded-xl border border-dashed border-slate-300 bg-white p-10 text-center">
      <h3 className="text-base font-semibold text-slate-900">{title}</h3>
      <div className="mx-auto mt-2 max-w-md text-sm text-slate-600">{children}</div>
      {to && cta && (
        <Button as={Link} to={to} className="mt-5">
          {cta}
        </Button>
      )}
    </div>
  );
}

/**
 * Accessible wrapper for a Recharts figure.
 *
 * Recharts emits an SVG of unlabelled paths, which is silence to a screen
 * reader. This wraps the chart in a `<figure>` with a text summary, hides the
 * decorative SVG from the accessibility tree, and exposes the underlying
 * numbers as a visually-hidden data table — so the *information* is available
 * to everyone even though the *picture* is not (WCAG 1.1.1 Non-text Content).
 */
export function AccessibleChart({ title, summary, columns, rows, children }) {
  const captionId = useId();
  return (
    <figure className="m-0">
      <figcaption id={captionId} className="sr-only">
        {title}. {summary}
      </figcaption>

      <div aria-hidden="true">{children}</div>

      {rows?.length > 0 && (
        <table className="sr-only">
          <caption>{title} — underlying data</caption>
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column} scope="col">
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr key={index}>
                {row.map((cell, cellIndex) =>
                  cellIndex === 0 ? (
                    <th key={cellIndex} scope="row">
                      {cell}
                    </th>
                  ) : (
                    <td key={cellIndex}>{cell}</td>
                  ),
                )}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </figure>
  );
}

/**
 * Medical disclaimer.
 *
 * Implements the ethical commitment made in the proposal (§11.7 / Legal &
 * Ethical Feasibility): the system offers nutritional guidance, not medical
 * advice. Shown in the footer on every page and again during onboarding.
 */
export function Disclaimer({ compact = false }) {
  if (compact) {
    return (
      <p className="text-xs text-slate-600">
        NutriFit-AI provides general nutritional guidance and is not a substitute for professional
        medical or dietetic advice.
      </p>
    );
  }
  return (
    <Alert tone="warn" title="Important">
      NutriFit-AI provides general nutritional guidance based on established sports-nutrition
      formulas and machine-learning estimates. It is <strong>not</strong> a medical device and does
      not replace advice from a qualified doctor, dietitian or nutritionist. Consult a healthcare
      professional before making significant dietary changes, particularly if you have a medical
      condition, food allergies, or are pregnant.
    </Alert>
  );
}

export const GOAL_LABELS = {
  fat_loss: 'Fat loss',
  maintenance: 'Maintenance',
  muscle_gain: 'Muscle gain',
};

export const SLOT_LABELS = {
  breakfast: 'Breakfast',
  lunch: 'Lunch',
  dinner: 'Dinner',
  snack: 'Snack',
};

export const BMI_TONE = {
  underweight: 'warn',
  normal: 'brand',
  overweight: 'warn',
  obese: 'danger',
};
