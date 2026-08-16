import { NextResponse } from "next/server";

import { STATE_FILES, mutateJson, readJson } from "@/app/lib/state-store";

export const runtime = "nodejs";

type InquiryPayload = Record<string, unknown>;

type CurrentInquiryRecord = {
  manual_clear_required: boolean;
  dismissed: boolean;
  updated_at: string;
  inquiry: InquiryPayload | null;
};

const EMPTY_RECORD: CurrentInquiryRecord = {
  manual_clear_required: false,
  dismissed: true,
  updated_at: "",
  inquiry: null,
};

export async function GET() {
  try {
    const record = await readJson<Partial<CurrentInquiryRecord>>(
      STATE_FILES.inquiry,
      EMPTY_RECORD
    );
    const shouldShow =
      record.manual_clear_required === true &&
      record.dismissed !== true &&
      record.inquiry != null;

    return NextResponse.json(
      {
        manual_clear_required: shouldShow,
        updated_at: record.updated_at ?? "",
        inquiry: shouldShow ? record.inquiry : null,
      },
      { status: 200 }
    );
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Failed to load inquiry";
    return NextResponse.json(
      { manual_clear_required: false, updated_at: "", inquiry: null, error: message },
      { status: 500 }
    );
  }
}

export async function DELETE() {
  try {
    await mutateJson<Partial<CurrentInquiryRecord>>(STATE_FILES.inquiry, EMPTY_RECORD, (existing) => ({
      ...EMPTY_RECORD,
      ...existing,
      manual_clear_required: false,
      dismissed: true,
      updated_at: new Date().toISOString(),
      inquiry: null,
    }));

    return NextResponse.json({ ok: true }, { status: 200 });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Failed to dismiss inquiry";
    return NextResponse.json({ ok: false, error: message }, { status: 500 });
  }
}
