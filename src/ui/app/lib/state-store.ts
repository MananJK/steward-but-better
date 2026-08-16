/**
 * File-backed state store for the dashboard.
 *
 * State lives in a data directory OUTSIDE `public/` (which is a static-asset
 * directory snapshotted at build time and therefore unusable for runtime
 * state). All read-modify-write cycles are serialized through a per-file
 * promise chain so concurrent requests cannot interleave; writes themselves
 * are atomic (temp file + rename) with EPERM/EBUSY retries for Windows.
 */
import { promises as fs } from "node:fs";
import path from "node:path";

const DATA_DIR = process.env.DATA_DIR
  ? path.resolve(process.env.DATA_DIR)
  : path.join(process.cwd(), "data");

export const STATE_FILES = {
  live: path.join(DATA_DIR, "live_incident.json"),
  investigations: path.join(DATA_DIR, "active_investigations.json"),
  inquiry: path.join(DATA_DIR, "current_inquiry.json"),
} as const;

const locks = new Map<string, Promise<unknown>>();

/** Serialize mutations per file: read-modify-write cycles never interleave. */
function withFileLock<T>(file: string, fn: () => Promise<T>): Promise<T> {
  const previous = locks.get(file) ?? Promise.resolve();
  const next = previous.then(fn, fn);
  locks.set(
    file,
    next.catch(() => undefined)
  );
  return next;
}

export async function readJson<T>(file: string, fallback: T): Promise<T> {
  try {
    const raw = await fs.readFile(file, "utf-8");
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

export async function writeJsonAtomic(file: string, payload: unknown): Promise<void> {
  await fs.mkdir(path.dirname(file), { recursive: true });
  const tempPath = `${file}.tmp`;
  const jsonContent = `${JSON.stringify(payload, null, 2)}\n`;
  const maxRetries = 3;

  for (let attempt = 1; attempt <= maxRetries; attempt += 1) {
    try {
      await fs.writeFile(tempPath, jsonContent, "utf-8");
      const handle = await fs.open(tempPath, "r+");
      await handle.sync();
      await handle.close();
      await fs.rename(tempPath, file);
      return;
    } catch (error) {
      const err = error as NodeJS.ErrnoException;
      if ((err.code === "EPERM" || err.code === "EBUSY") && attempt < maxRetries) {
        await new Promise((resolve) => setTimeout(resolve, 100));
        continue;
      }
      throw error;
    }
  }
}

/** Locked read-modify-write; the mutator receives the current value. */
export function mutateJson<T>(
  file: string,
  fallback: T,
  mutator: (current: T) => T | Promise<T>
): Promise<T> {
  return withFileLock(file, async () => {
    const current = await readJson<T>(file, fallback);
    const next = await mutator(current);
    await writeJsonAtomic(file, next);
    return next;
  });
}

export async function writeJson(file: string, payload: unknown): Promise<void> {
  return withFileLock(file, () => writeJsonAtomic(file, payload));
}

export function isFileLockError(error: unknown): boolean {
  const err = error as NodeJS.ErrnoException | undefined;
  const code = err?.code ?? "";
  return code === "EPERM" || code === "EACCES" || code === "EBUSY";
}
