import { NextResponse } from "next/server";

import { STATE_FILES, mutateJson } from "@/app/lib/state-store";
import type { ActiveInvestigation } from "@/app/types/incident";

export const runtime = "nodejs";

export async function DELETE(
  _: Request,
  context: { params: Promise<{ id: string }> }
) {
  const { id } = await context.params;

  try {
    let removed = 0;
    const filtered = await mutateJson<ActiveInvestigation[]>(
      STATE_FILES.investigations,
      [],
      (investigations) => {
        const list = Array.isArray(investigations) ? investigations : [];
        removed = list.filter((item) => item.id === id).length;
        return list.filter((item) => item.id !== id);
      }
    );

    return NextResponse.json(
      { ok: true, removed, remaining: filtered.length },
      { status: 200 }
    );
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Failed to dismiss investigation";
    return NextResponse.json({ ok: false, error: message }, { status: 500 });
  }
}
