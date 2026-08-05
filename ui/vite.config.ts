import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The UI never talks to CockroachDB — it talks to the FastAPI bridge, which
// talks to the kernel. In dev that bridge is a separate process, so /api is
// proxied rather than hard-coded into the client (which would bake a hostname
// into the bundle and break the moment it is served from anywhere else).
const API_TARGET = process.env.RECALL_API_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: Number(process.env.RECALL_UI_PORT ?? 5173),
    strictPort: false,
    proxy: {
      "/api": {
        target: API_TARGET,
        changeOrigin: true,
        // Server-Sent Events must not be buffered by the proxy or the live
        // feed arrives in one lump when the connection closes.
        configure: (proxy) => {
          proxy.on("proxyRes", (proxyRes) => {
            if (proxyRes.headers["content-type"]?.includes("text/event-stream")) {
              proxyRes.headers["cache-control"] = "no-cache";
            }
          });
        },
      },
    },
  },
});
