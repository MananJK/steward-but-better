import { NextResponse } from "next/server";

import { STATE_FILES, readJson } from "@/app/lib/state-store";
import type { ActiveInvestigation } from "@/app/types/incident";

export const runtime = "nodejs";

export async function GET() {
  try {
    const investigations = await readJson<ActiveInvestigation[]>(
      STATE_FILES.investigations,
      []
    );
    return NextResponse.json(
      { investigations: Array.isArray(investigations) ? investigations : [] },
      { status: 200 }
    );
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Failed to load investigations";
    return NextResponse.json(
      { investigations: [], error: message },
      { status: 500 }
    );
  }
}
