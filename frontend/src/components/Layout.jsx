import { NavLink, useNavigate } from 'react-router-dom';

import { useAuth } from '../hooks/useAuth.jsx';
import { Button, Disclaimer } from './ui.jsx';

const NAV = [
  { to: '/', label: 'Dashboard', end: true },
  { to: '/meal-plan', label: 'Meal Plan' },
  { to: '/recommendations', label: 'Recommendations' },
  { to: '/progress', label: 'Progress' },
  { to: '/profile', label: 'Profile' },
];

export default function Layout({ children }) {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();

  async function handleSignOut() {
    await signOut();
    navigate('/login');
  }

  return (
    <div className="flex min-h-full flex-col">
      {/*
        Skip link (WCAG 2.4.1 Bypass Blocks). Visually hidden until focused,
        so keyboard and screen-reader users can jump past the navigation
        instead of tabbing through it on every page.
      */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-brand-700 focus:px-4 focus:py-2 focus:text-sm focus:font-medium focus:text-white"
      >
        Skip to main content
      </a>

      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-4 px-4 py-3">
          <span className="text-lg font-bold text-brand-700">NutriFit&#8209;AI</span>

          <nav aria-label="Main navigation">
            <ul className="flex list-none flex-wrap gap-1 p-0">
              {NAV.map((item) => (
                <li key={item.to}>
                  <NavLink
                    to={item.to}
                    end={item.end}
                    // `aria-current="page"` tells assistive technology which
                    // page is active — the colour change alone does not.
                    className={({ isActive }) =>
                      `block rounded-lg px-3 py-1.5 text-sm font-medium transition focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-600 focus-visible:ring-offset-2 ${
                        isActive
                          ? 'bg-brand-50 text-brand-700'
                          : 'text-slate-700 hover:bg-slate-100 hover:text-slate-900'
                      }`
                    }
                  >
                    {item.label}
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>

          <div className="ml-auto flex items-center gap-3">
            <span className="hidden text-sm text-slate-600 sm:inline">
              <span className="sr-only">Signed in as </span>
              {user?.email}
            </span>
            <Button variant="secondary" onClick={handleSignOut}>
              Sign out
            </Button>
          </div>
        </div>
      </header>

      <main
        id="main-content"
        tabIndex={-1}
        className="mx-auto w-full max-w-6xl flex-1 px-4 py-8 focus:outline-none"
      >
        {children}
      </main>

      <footer className="border-t border-slate-200 bg-white">
        <div className="mx-auto max-w-6xl px-4 py-5">
          <Disclaimer compact />
        </div>
      </footer>
    </div>
  );
}
