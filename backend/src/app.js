/**
 * Express application factory.
 *
 * Exported separately from `server.js` so the test suite can mount the app
 * with supertest without binding a real port.
 */
import cors from 'cors';
import express from 'express';
import rateLimit from 'express-rate-limit';
import helmet from 'helmet';
import morgan from 'morgan';

import { config } from './config.js';
import { errorHandler, notFoundHandler } from './middleware/errorHandler.js';
import { dashboardRouter } from './routes/dashboard.js';
import { mealPlanRouter } from './routes/mealplan.js';
import { predictRouter } from './routes/predict.js';
import { profileRouter } from './routes/profile.js';
import { progressRouter } from './routes/progress.js';
import { recommendationsRouter } from './routes/recommendations.js';
import { mlClient } from './services/mlClient.js';
import { asyncHandler } from './utils/errors.js';

export function createApp() {
  const app = express();

  app.disable('x-powered-by');
  app.use(helmet());
  app.use(
    cors({
      origin: config.cors.origins,
      credentials: true,
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    }),
  );
  app.use(express.json({ limit: '1mb' }));

  if (config.env !== 'test') {
    app.use(morgan('dev'));
  }

  // Meal-plan generation is the one genuinely expensive operation, so it gets
  // its own tighter budget rather than sharing the general allowance.
  const generalLimiter = rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 300,
    standardHeaders: true,
    legacyHeaders: false,
    skip: () => config.env === 'test',
  });
  const expensiveLimiter = rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 20,
    standardHeaders: true,
    legacyHeaders: false,
    skip: () => config.env === 'test',
    message: { error: 'rate_limited', message: 'Too many meal-plan generations. Try again later.' },
  });

  app.use('/api', generalLimiter);
  app.use('/api/mealplan/generate', expensiveLimiter);

  // ---- public ------------------------------------------------------------
  app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', service: 'nutrifit-backend', version: '1.0.0', env: config.env });
  });

  app.get(
    '/api/health/ml',
    asyncHandler(async (req, res) => {
      res.json(await mlClient.health());
    }),
  );

  // ---- authenticated -----------------------------------------------------
  app.use('/api/profile', profileRouter);
  app.use('/api/predict', predictRouter);
  app.use('/api/recommendations', recommendationsRouter);
  app.use('/api/mealplan', mealPlanRouter);
  app.use('/api/progress', progressRouter);
  app.use('/api/dashboard', dashboardRouter);

  app.use(notFoundHandler);
  app.use(errorHandler);

  return app;
}
