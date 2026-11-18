import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const apiTarget = loadEnv(mode, '.', 'VITE_').VITE_API_TARGET || 'http://127.0.0.1:8000';
  const websocketTarget = apiTarget.replace(/^http/, 'ws');

  return {
    plugins: [react()],
    server: {
      allowedHosts: [
        'olympics-undergo-retake.ngrok-free.dev',
        '.ngrok-free.dev',
      ],
      proxy: {
        '/api': apiTarget,
        '/ws': { target: websocketTarget, ws: true },
      },
    },
    build: { outDir: '../dashboard', emptyOutDir: true },
  };
});
