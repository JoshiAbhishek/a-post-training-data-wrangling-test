import { configDefaults, defineConfig } from "vitest/config";

// Tests sit next to the code they test, anywhere in the repo but hidden folders.
export default defineConfig({ test: { include: ["**/*.test.ts"], exclude: [...configDefaults.exclude, "**/.*/**"] } });
