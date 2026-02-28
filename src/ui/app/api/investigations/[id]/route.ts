import { promises as fs } from "node:fs";
import path from "node:path";

import { NextResponse } from "next/server";

import type { ActiveInvestigation } from "@/app/types/incident";

export const runtime = "nodejs";

async function readInvestigations(filePath: string): Promise<ActiveInvestigation[]> {
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    const parsed = JSON.parse(raw) as unknown;
    return Array.isArray(parsed) ? (parsed as ActiveInvestigation[]) : [];
  } catch {
    return [];
  }
}

async function writeJsonAtomic(filePath: string, payload: unknown): Promise<void> {
  const jsonContent = `${JSON.stringify(payload, null, 2)}\n`;
  const tempPath = `${filePath}.tmp`;
  await fs.writeFile(tempPath, jsonContent, "utf-8");
  await fs.rename(tempPath, filePath);
}

export async function DELETE(_: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  const investigationsPath = path.join(process.cwd(), "public", "active_investigations.json");

  try {
    const investigations = await readInvestigations(investigationsPath);
    const filtered = investigations.filter((item) => item.id !== id);

    await fs.mkdir(path.dirname(investigationsPath), { recursive: true });
    await writeJsonAtomic(investigationsPath, filtered);

    return NextResponse.json(
      {
        ok: true,
        removed: investigations.length - filtered.length,
      },
      { status: 200 }
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to dismiss investigation";
    return NextResponse.json({ ok: false, error: message }, { status: 500 });
  }
}
