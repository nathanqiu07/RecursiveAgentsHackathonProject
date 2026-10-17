import type { BenchmarkStore, VerificationClaim } from './benchmark-store';
import { readPath, renderTemplate } from './verifier-adapter';
import { checkpointReference, PublicCheckpointError, verifyPublicCheckpoint } from './github-checkpoint';

class VerificationError extends Error {}
type RuntimeEnv = Record<string, unknown>;

async function readResponse(response: Response): Promise<unknown> {
  if (!response.ok) { await response.body?.cancel(); throw new VerificationError(`Verification endpoint returned HTTP ${response.status}`); }
  if (!response.body) throw new VerificationError('Verification endpoint returned an empty response');
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let length = 0;
  try {
    while (true) {
      const next = await reader.read();
      if (next.done) break;
      length += next.value.byteLength;
      if (length > 1024 * 1024) throw new VerificationError('Verification response exceeds 1 MiB');
      chunks.push(next.value);
    }
  } finally { await reader.cancel().catch(() => {}); }
  const bytes = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  try { return JSON.parse(new TextDecoder().decode(bytes)); }
  catch { throw new VerificationError('Verification endpoint must return valid JSON'); }
}

function message(value: unknown): string | null {
  return value === undefined || value === null ? null : (typeof value === 'string' ? value : JSON.stringify(value)).slice(0, 4000);
}

export async function evaluateClaim(store: BenchmarkStore, claim: VerificationClaim, env: RuntimeEnv,
  request: typeof fetch = fetch, checkpointRequest: typeof fetch = fetch): Promise<void> {
  const { benchmark, submission } = claim;
  const verifier = benchmark.verifier;
  try {
    checkpointReference(submission.base_checkpoint_url);
    if (claim.phase === 'submit') {
      await verifyPublicCheckpoint(submission.base_checkpoint_url, checkpointRequest,
        AbortSignal.timeout(Math.max(1, claim.deadline_at - store.now())));
    }
    const headers: Record<string, string> = { Accept: 'application/json' };
    if (verifier.auth_env) {
      const allowedOrigins = typeof env.BENCHMARK_VERIFIER_ALLOWED_ORIGINS === 'string'
        ? env.BENCHMARK_VERIFIER_ALLOWED_ORIGINS.split(',').map((origin) => origin.trim()) : [];
      if (!allowedOrigins.includes(new URL(verifier.url).origin)) {
        throw new VerificationError('This verifier origin is not configured for hosted server credentials');
      }
      const credential = env[verifier.auth_env];
      if (typeof credential !== 'string' || !credential) throw new VerificationError(`Verification credential ${verifier.auth_env} is not configured`);
      headers.Authorization = `Bearer ${credential}`;
    }
    let url = verifier.url;
    const options: RequestInit = {
      method: claim.phase === 'submit' ? verifier.method : 'GET', headers, redirect: 'error',
      signal: AbortSignal.timeout(Math.max(1, Math.min(8000, claim.deadline_at - store.now()))),
    };
    if (claim.phase === 'submit') {
      headers['Content-Type'] = 'application/json'; headers['Idempotency-Key'] = submission.id;
      options.body = JSON.stringify(renderTemplate(verifier.request_template, benchmark, submission));
    } else {
      if (!submission.external_job_id) throw new VerificationError('Verification job reference is missing');
      url = verifier.poll_url.replaceAll('{{job_id}}', encodeURIComponent(submission.external_job_id));
    }
    if (verifier.auth_env && new URL(url).origin !== new URL(verifier.url).origin) throw new VerificationError('Authenticated polling endpoint origin changed');
    const response = await request(url, options);
    const result = await readResponse(response);
    const rawStatus = readPath(result, verifier.status_path);
    const status = ['string', 'number', 'boolean'].includes(typeof rawStatus) ? String(rawStatus) : '';
    const feedback = message(readPath(result, verifier.feedback_path));
    if (verifier.failure_statuses.includes(status)) {
      await store.finish(claim, status === 'rejected' ? 'rejected' : 'failed', null, feedback, feedback || `Verifier reported ${status}`);
      return;
    }
    if (verifier.mode === 'synchronous' || verifier.success_statuses.includes(status)) {
      if (verifier.mode === 'synchronous' && (response.status === 202 || (status && !verifier.success_statuses.includes(status)))) {
        throw new VerificationError('Verifier returned an unfinished evaluation. Configure queued verification or completed statuses.');
      }
      const score = readPath(result, verifier.score_path);
      if (typeof score !== 'number' || !Number.isFinite(score)) throw new VerificationError('Verifier did not return a finite numeric score at the configured score path');
      if ((benchmark.score_min !== null && score < benchmark.score_min) || (benchmark.score_max !== null && score > benchmark.score_max)) {
        throw new VerificationError('Verified score is outside the benchmark score range');
      }
      await store.finish(claim, 'verified', score, feedback);
      return;
    }
    const external = claim.phase === 'poll' ? submission.external_job_id : readPath(result, verifier.job_id_path);
    if (!['string', 'number'].includes(typeof external) || external === null || !String(external).length || String(external).length > 500
      || (typeof external === 'number' && !Number.isFinite(external))) throw new VerificationError('Queued verifier did not return a valid job reference');
    if (claim.phase === 'poll' && !status) throw new VerificationError('Queued verifier did not return a job status');
    await store.pending(claim, String(external), ['queued', 'pending'].includes(status) ? 'queued' : 'running');
  } catch (error) {
    if (error instanceof PublicCheckpointError) {
      await store.finish(claim, error.rejected ? 'rejected' : 'failed', null, null, error.message);
      return;
    }
    const detail = error instanceof VerificationError ? error.message : 'Could not reach or read the verification endpoint';
    if (claim.phase === 'poll' && !(error instanceof VerificationError)) {
      await store.pending(claim, submission.external_job_id!, 'running', detail);
    } else {
      await store.finish(claim, 'failed', null, null, claim.phase === 'submit' && !(error instanceof VerificationError)
        ? `${detail}. Dispatch outcome is unknown; check the verifier before submitting again.` : detail);
    }
  }
}

/** Requests advance durable work briefly; further reads resume long-running remote jobs. */
export async function advanceVerification(store: BenchmarkStore, env: RuntimeEnv): Promise<void> {
  const until = Date.now() + 24000;
  // Leave room for a ten-second public-checkpoint read plus an eight-second verifier request.
  while (Date.now() < until - 18000) {
    const claims: VerificationClaim[] = [];
    for (let i = 0; i < 4; i++) {
      const claim = await store.claimDue();
      if (!claim) break;
      claims.push(claim);
    }
    await Promise.all(claims.map((claim) => evaluateClaim(store, claim, env)));
    const next = await store.nextDue();
    if (next === null || next > until - 18000) return;
    const delay = Math.max(100, next - Date.now());
    await new Promise((resolve) => setTimeout(resolve, delay));
  }
}
