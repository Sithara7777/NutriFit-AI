/**
 * Centralised error handling — one consistent error shape for every route.
 *
 * Clients always receive `{ error, message, details? }`. Internal details
 * (stack traces, database messages) are logged server-side but never returned,
 * because error text is a classic information-disclosure vector.
 */
import { AppError } from '../utils/errors.js';

export function notFoundHandler(req, res) {
  res.status(404).json({
    error: 'not_found',
    message: `No route matches ${req.method} ${req.originalUrl}`,
  });
}

// Express identifies error middleware by arity, so `next` must stay declared.
// eslint-disable-next-line no-unused-vars
export function errorHandler(error, req, res, next) {
  if (error instanceof AppError) {
    if (error.statusCode >= 500) {
      console.error(`[${error.code}] ${error.message}`);
    }
    return res.status(error.statusCode).json({
      error: error.code,
      message: error.message,
      ...(error.details ? { details: error.details } : {}),
    });
  }

  if (error?.type === 'entity.parse.failed') {
    return res.status(400).json({
      error: 'validation_error',
      message: 'Request body is not valid JSON.',
    });
  }

  console.error('Unhandled error:', error);
  return res.status(500).json({
    error: 'internal_error',
    message: 'An unexpected error occurred.',
  });
}
