import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '')

  return {
    plugins: [react()],
    // Allow .mha and .nii.gz to be treated as static assets when imported
    assetsInclude: ['**/*.mha', '**/*.nii.gz', '**/*.nii'],
    server: {
      fs: {
        // Allow serving files from the project root (including imgs/ directory)
        // so that /@fs/<absolute-path> URLs for .mha files work correctly
        strict: false,
        allow: [resolve(__dirname, '.'), resolve(__dirname, 'imgs')],
      },
      proxy: {
        // Classifier API (port 8001) — must be listed BEFORE the catch-all /api rule
        '/api/classifier': {
          target: env.VITE_CLASSIFIER_API_TARGET || 'http://127.0.0.1:8001',
          changeOrigin: true,
        },
        // MedGemma API (port 8000)
        '/api': {
          target: env.VITE_MEDGEMMA_API_TARGET || 'http://127.0.0.1:8000',
          changeOrigin: true,
        },
      },
    },
  }
})
