/** HTTP entry point. */
import { createApp } from './app.js';
import { assertConfig, config } from './config.js';
import { mlClient } from './services/mlClient.js';

assertConfig();

const app = createApp();

const server = app.listen(config.port, () => {
  console.log(`NutriFit-AI backend listening on http://localhost:${config.port}`);
  console.log(`  environment : ${config.env}`);
  console.log(`  ML service  : ${config.mlService.baseUrl}`);
  console.log(`  CORS origins: ${config.cors.origins.join(', ')}`);

  // Probe the ML service at start-up so a misconfiguration is visible
  // immediately rather than on the first user prediction.
  mlClient
    .health()
    .then((health) => {
      console.log(`  ML status   : ${health.status} (models: ${JSON.stringify(health.models_loaded)})`);
      if (health.status === 'degraded') {
        console.warn(`  WARNING     : ML service degraded - ${health.detail}`);
      }
    })
    .catch((error) => {
      console.warn(`  WARNING     : ML service unreachable - ${error.message}`);
      console.warn('                Start it: cd services/ml-service && uvicorn app.main:app --port 8000');
    });
});

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    console.log(`\n${signal} received, shutting down.`);
    server.close(() => process.exit(0));
  });
}
