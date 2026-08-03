import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dentro do docker compose o backend responde em http://backend:8000.
// Rodando na maquina, em http://localhost:8000.
const alvo = process.env.VITE_API_TARGET || 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      '/api': { target: alvo, changeOrigin: true },
    },
  },
})
