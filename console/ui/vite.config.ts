import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// The console server (FastAPI) listens on 127.0.0.1:7780 and serves
// `console/ui/dist/` at `/` in production. In dev, Vite proxies the API and
// the WebSocket stream to it so the UI can run on its own port.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:7780", changeOrigin: true },
      "/ws": { target: "ws://127.0.0.1:7780", ws: true },
    },
  },
  build: { outDir: "dist", sourcemap: false },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/__tests__/setup.ts"],
    css: false,
  },
});
