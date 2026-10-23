import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies the read-only backend on 127.0.0.1:8000.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
});
