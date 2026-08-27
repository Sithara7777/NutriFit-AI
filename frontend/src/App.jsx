import { Navigate, Route, Routes } from 'react-router-dom';

import Layout from './components/Layout.jsx';
import { Spinner } from './components/ui.jsx';
import { useAuth } from './hooks/useAuth.jsx';
import Dashboard from './pages/Dashboard.jsx';
import Login from './pages/Login.jsx';
import MealPlan from './pages/MealPlan.jsx';
import Profile from './pages/Profile.jsx';
import Progress from './pages/Progress.jsx';
import Recommendations from './pages/Recommendations.jsx';

function RequireAuth({ children }) {
  const { session, loading } = useAuth();
  if (loading) return <Spinner label="Checking your session…" />;
  if (!session) return <Navigate to="/login" replace />;
  return <Layout>{children}</Layout>;
}

export default function App() {
  const { session, loading } = useAuth();

  return (
    <Routes>
      <Route
        path="/login"
        element={loading ? <Spinner /> : session ? <Navigate to="/" replace /> : <Login />}
      />
      <Route
        path="/"
        element={
          <RequireAuth>
            <Dashboard />
          </RequireAuth>
        }
      />
      <Route
        path="/profile"
        element={
          <RequireAuth>
            <Profile />
          </RequireAuth>
        }
      />
      <Route
        path="/recommendations"
        element={
          <RequireAuth>
            <Recommendations />
          </RequireAuth>
        }
      />
      <Route
        path="/meal-plan"
        element={
          <RequireAuth>
            <MealPlan />
          </RequireAuth>
        }
      />
      <Route
        path="/progress"
        element={
          <RequireAuth>
            <Progress />
          </RequireAuth>
        }
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
