import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// 开发期把 API 请求代理到 zylo serve（默认端口 8000），
// 避免浏览器跨域；生产部署由 FastAPI 直接挂载构建产物（M5）
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/healthz": "http://127.0.0.1:8000",
    },
  },
});
