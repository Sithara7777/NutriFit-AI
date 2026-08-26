import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    rollupOptions: {
      output: {
        // Split the two heavy third-party dependencies into their own chunks
        // so a code change to the app does not invalidate the browser's cached
        // copy of the charting and auth libraries.
        manualChunks: {
          charts: ['recharts'],
          supabase: ['@supabase/supabase-js'],
        },
      },
    },
  },
  server: {
    port: 5173,
    // The browser never talks to the Node backend cross-origin in dev; Vite
    // proxies /api so cookies and relative URLs behave the same as in prod.
    proxy: {
      '/api': {
        target: process.env.VITE_BACKEND_URL ?? 'http://localhost:4000',
        changeOrigin: true,
      },
    },
  },
});
