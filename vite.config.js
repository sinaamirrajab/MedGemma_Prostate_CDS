import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import { dirname, resolve } from 'path'
import { fileURLToPath } from 'url'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const rootDir = dirname(fileURLToPath(import.meta.url))
  const env = loadEnv(mode, '.', '')
  const configuredRoots = (env.VITE_ALLOWED_FILE_ROOTS || '')
    .split(':')
    .map((v) => v.trim())
    .filter(Boolean)
  const allowedRoots = [
    resolve(rootDir, '.'),
    resolve(rootDir, 'imgs'),
    ...configuredRoots,
  ]

  return {
    plugins: [react()],
    // Allow .mha and .nii.gz to be treated as static assets when imported
    assetsInclude: ['**/*.mha', '**/*.nii.gz', '**/*.nii'],
    server: {
      fs: {
        // Keep strict filesystem serving and allow only explicit trusted roots.
        strict: true,
        allow: allowedRoots,
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
