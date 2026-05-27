import { NextResponse } from 'next/server';
import { readInstallStatus, isAutoInstallSupported, resolveSam3Command } from '@/lib/magic-layer/runner';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  const resolved = await resolveSam3Command();
  if (resolved.source === 'env' && resolved.argv?.length) {
    return NextResponse.json(
      {
        installed: true,
        installing: false,
        failed: false,
        autoInstallSupported: isAutoInstallSupported(),
        canFallback: false,
        message: 'External SAM3 command configured for Magic Layer segmentation.',
      },
      { headers: { 'Cache-Control': 'no-store' } },
    );
  }

  if (!isAutoInstallSupported()) {
    return NextResponse.json(
      { installed: false, installing: false, failed: false, autoInstallSupported: false, canFallback: true, message: 'Auto-install not supported on this platform.' },
      { headers: { 'Cache-Control': 'no-store' } },
    );
  }
  const status = await readInstallStatus();
  return NextResponse.json(
    { ...status, autoInstallSupported: true },
    { headers: { 'Cache-Control': 'no-store' } },
  );
}
