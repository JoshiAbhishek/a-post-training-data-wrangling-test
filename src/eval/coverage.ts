// Print how many tasks cover each category. Run: npm run tasks:coverage
import { fileURLToPath } from "node:url";
import { CATEGORIES, coverage, loadTasks } from "./task.ts";

const tasks = loadTasks(fileURLToPath(new URL("../../tasks/", import.meta.url))).map((t) => t.task);
const counts = coverage(tasks);
const width = Math.max(...CATEGORIES.map((c) => c.length));

console.log(`${tasks.length} tasks\n`);
for (const c of CATEGORIES)
  console.log(`${c.padEnd(width)}  ${String(counts[c]).padStart(3)}  ${"█".repeat(counts[c])}`);
console.log(
  `\nheld-out examples: ${tasks.reduce((n, t) => n + t.heldOut.length, 0)}, shown: ${tasks.reduce((n, t) => n + t.examples.length, 0)}`,
);
