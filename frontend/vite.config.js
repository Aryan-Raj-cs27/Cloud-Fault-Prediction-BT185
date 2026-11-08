import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The browser always talks to the same origin under /api. In development Vite proxies it to
// FastAPI; in production FastAPI serves this build and the API from one origin (app.api:server),
// so the session cookie is first-party and SameSite=Strict.
const apiProxy = {
  '/api': {
    target: process.env.FAULTAWARE_API ?? 'http://127.0.0.1:8000',
    changeOrigin: false,
    rewrite: (path) => path.replace(/^\/api/, ''),
  },
};

export default defineConfig({
  plugins: [react()],
  server: { host: '127.0.0.1', port: 5173, strictPort: true, proxy: apiProxy },
  preview: { host: '127.0.0.1', port: 4173, strictPort: true, proxy: apiProxy },
  build: {
    sourcemap: false,
    chunkSizeWarningLimit: 700,
    rollupOptions: { output: { manualChunks: { charts: ['recharts'] } } },
  },
});
