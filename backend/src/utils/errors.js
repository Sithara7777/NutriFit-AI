/** Domain error types, so routes never construct raw HTTP responses. */

export class AppError extends Error {
  constructor(message, statusCode = 500, code = 'internal_error', details = undefined) {
    super(message);
    this.name = this.constructor.name;
    this.statusCode = statusCode;
    this.code = code;
    this.details = details;
    Error.captureStackTrace?.(this, this.constructor);
  }
}

export class ValidationError extends AppError {
  constructor(message, details) {
    super(message, 400, 'validation_error', details);
  }
}

export class UnauthorizedError extends AppError {
  constructor(message = 'Authentication required.') {
    super(message, 401, 'unauthorized');
  }
}

export class NotFoundError extends AppError {
  constructor(message = 'Resource not found.') {
    super(message, 404, 'not_found');
  }
}

export class ConflictError extends AppError {
  constructor(message) {
    super(message, 409, 'conflict');
  }
}

export class ServiceUnavailableError extends AppError {
  constructor(message, statusCode = 503) {
    super(message, statusCode, 'service_unavailable');
  }
}

/** Wrap an async route handler so rejections reach the error middleware. */
export function asyncHandler(handler) {
  return (req, res, next) => Promise.resolve(handler(req, res, next)).catch(next);
}
