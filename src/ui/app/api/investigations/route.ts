import { promises as fs } from "node:fs";
import path from "node:path";

import { NextResponse } from "next/server";

import type { ActiveInvestigation } from "@/app/types/incident";

export const runtime = "nodejs";

async function readInvestigations(): Promise<ActiveInvestigation[]> {
  const investigationsPath = path.join(process.cwd(), "public", "active_investigations.json");

  try {
    const raw = await fs.readFile(investigationsPath, "utf-8");
    const parsed = JSON.parse(raw) as unknown;
    return Array.isArray(parsed) ? (parsed as ActiveInvestigation[]) : [];
  } catch {
    return [];
  }
}

export async function GET() {
  try {
    const investigations = await readInvestigations();
    return NextResponse.json({ investigations }, { status: 200 });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to load investigations";
    return NextResponse.json({ investigations: [], error: message }, { status: 500 });
  }
}
