import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { viteSingleFile } from 'vite-plugin-singlefile'

// `npm run build` produces one self-contained dist/index.html. The composer works without a backend;
// the classification page calls the FastAPI server (server/app.py), which also serves dist/.
// In `npm run dev`, /api is proxied to that server on :8000.
export default defineConfig({
  plugins: [react(), viteSingleFile()],
  server: { proxy: { '/api': 'http://localhost:8000' } },
})
