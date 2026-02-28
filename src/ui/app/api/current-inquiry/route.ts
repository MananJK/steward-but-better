import { promises as fs } from "node:fs";
import path from "node:path";

import { NextResponse } from "next/server";

type InquiryPayload = Record<string, unknown>;

type CurrentInquiryRecord = {
  manual_clear_required: boolean;
  dismissed: boolean;
  updated_at: string;
  inquiry: InquiryPayload | null;
};

export const runtime = "nodejs";

function getCurrentInquiryPath(): string {
  return path.join(process.cwd(), "public", "current_inquiry.json");
}

async function writeJsonAtomic(filePath: string, payload: unknown): Promise<void> {
  const jsonContent = `${JSON.stringify(payload, null, 2)}\n`;
  const tempPath = `${filePath}.tmp`;

  try {
    await fs.writeFile(tempPath, jsonContent, "utf-8");
    await fs.rename(tempPath, filePath);
  } catch (error) {
    const err = error as NodeJS.ErrnoException;
    if (err.code === "EPERM") {
      await new Promise((resolve) => setTimeout(resolve, 100));
      await fs.writeFile(tempPath, jsonContent, "utf-8");
      await fs.rename(tempPath, filePath);
      return;
    }
    throw error;
  }
}

async function readCurrentInquiry(): Promise<CurrentInquiryRecord> {
  const filePath = getCurrentInquiryPath();

  try {
    const raw = await fs.readFile(filePath, "utf-8");
    const parsed = JSON.parse(raw) as Partial<CurrentInquiryRecord>;
    return {
      manual_clear_required: parsed.manual_clear_required === true,
      dismissed: parsed.dismissed === true,
      updated_at: typeof parsed.updated_at === "string" ? parsed.updated_at : "",
      inquiry: parsed.inquiry ?? null,
    };
  } catch {
    return {
      manual_clear_required: false,
      dismissed: true,
      updated_at: "",
      inquiry: null,
    };
  }
}

export async function GET() {
  try {
    const record = await readCurrentInquiry();
    const shouldShowInquiry = record.manual_clear_required && !record.dismissed && record.inquiry !== null;

    return NextResponse.json(
      {
        manual_clear_required: shouldShowInquiry,
        updated_at: record.updated_at,
        inquiry: shouldShowInquiry ? record.inquiry : null,
      },
      { status: 200 }
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to load inquiry";
    return NextResponse.json(
      {
        manual_clear_required: false,
        updated_at: "",
        inquiry: null,
        error: message,
      },
      { status: 500 }
    );
  }
}

export async function DELETE() {
  const filePath = getCurrentInquiryPath();

  try {
    const existing = await readCurrentInquiry();
    const cleared: CurrentInquiryRecord = {
      ...existing,
      manual_clear_required: false,
      dismissed: true,
      updated_at: new Date().toISOString(),
      inquiry: null,
    };

    await fs.mkdir(path.dirname(filePath), { recursive: true });
    await writeJsonAtomic(filePath, cleared);

    return NextResponse.json({ ok: true }, { status: 200 });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to dismiss inquiry";
    return NextResponse.json({ ok: false, error: message }, { status: 500 });
  }
}
