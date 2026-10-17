import { env, waitUntil } from 'cloudflare:workers';
import { z } from 'zod';
import { BenchmarkStore } from '../../../lib/benchmark-store';
import { ApiError } from '../../../lib/api-error';
import { advanceVerification } from '../../../lib/verification-worker';
import { resourceApi } from '../../../lib/resource-api';
import { benchmarkDefinitionSchema, solutionSubmissionSchema } from '../../../shared/api-schemas';
import { agentHarnessApi } from '../../../lib/agent-harness';
import { agentAccess } from '../../../lib/agent-access';
import { libraryFeedApi } from '../../../lib/library-feed';

export const dynamic = 'force-dynamic';
const headers = { 'Cache-Control': 'no-store' };

async function body(request: Request): Promise<unknown> {
  if (!request.headers.get('content-type')?.toLowerCase().includes('application/json')) throw new ApiError(415, 'Use application/json for API requests');
  const reader = request.body?.getReader();
  if (!reader) throw new ApiError(400, 'A JSON request body is required');
  const chunks: Uint8Array[] = [];
  let bytes = 0;
  try {
    while (true) {
      const item = await reader.read();
      if (item.done) break;
      bytes += item.value.byteLength;
      if (bytes > 256 * 1024) throw new ApiError(413, 'Request body must be at most 256 KiB');
      chunks.push(item.value);
    }
  } finally { await reader.cancel().catch(() => {}); }
  const buffer = new Uint8Array(bytes);
  let offset = 0;
  for (const chunk of chunks) { buffer.set(chunk, offset); offset += chunk.byteLength; }
  try { return JSON.parse(new TextDecoder().decode(buffer)); }
  catch { throw new ApiError(400, 'Request body must contain valid JSON'); }
}

function pagination(url: URL) {
  const limit = Number(url.searchParams.get('limit') ?? 100);
  const offset = Number(url.searchParams.get('offset') ?? 0);
  if (!Number.isSafeInteger(limit) || limit < 1 || limit > 500 || !Number.isSafeInteger(offset) || offset < 0 || offset > 2 ** 31) {
    throw new ApiError(400, 'Use a limit between 1 and 500 and a nonnegative offset');
  }
  return { limit, offset };
}

async function route(request: Request): Promise<Response> {
  const url = new URL(request.url);
  const path = url.pathname.replace(/^\/api\/?/, '').split('/').filter(Boolean).map(decodeURIComponent);
  try {
    if (request.method === 'POST') {
      const origin = request.headers.get('origin');
      if (origin && origin !== url.origin) throw new ApiError(403, 'Origin is not allowed');
    }
    if (path[0] === 'health' && path.length === 1 && request.method === 'GET') {
      return Response.json({ status: 'ok', database: 'd1' }, { headers });
    }
    if (path[0] === 'schemas' && request.method === 'GET' && path.length === 2) {
      if (path[1] === 'benchmark') return Response.json(benchmarkDefinitionSchema, { headers });
      if (path[1] === 'submission') return Response.json(solutionSubmissionSchema, { headers });
    }
    if (path[0] === 'agents') {
      const configuration = env as unknown as Record<string, unknown>;
      const denied = agentAccess(request, path, configuration);
      if (denied) return denied;
      const response = await agentHarnessApi(request, path, configuration);
      if (response) return response;
    }
    if (path[0] === 'library') {
      const configuration = env as unknown as Record<string, unknown>;
      if (typeof configuration.AGENT_HARNESS_URL !== 'string' || !configuration.AGENT_HARNESS_URL) {
        return Response.json({ error: 'Connect the agent harness to view the Library research feed.' }, { status: 503, headers });
      }
      const denied = agentAccess(request, ['agents', 'runs'], configuration);
      if (denied) return denied;
      const response = await libraryFeedApi(request, path, configuration);
      if (response) return response;
    }
    if (!env.DB) throw new ApiError(503, 'Persistent database is not connected');
    const store = new BenchmarkStore(env.DB);
    if (path[0] === 'benchmarks' || path[0] === 'submissions') {
      waitUntil(advanceVerification(store, env as unknown as Record<string, unknown>).catch(() => {}));
    }
    if (path[0] === 'benchmarks') {
      if (path.length === 1 && request.method === 'GET') return Response.json({ benchmarks: await store.listBenchmarks() }, { headers });
      if (path.length === 1 && request.method === 'POST') return Response.json(await store.createBenchmark(await body(request)), { status: 201, headers });
      if (path.length === 2 && request.method === 'GET') return Response.json(await store.getBenchmark(path[1]), { headers });
      if (path.length === 3 && path[2] === 'leaderboard' && request.method === 'GET') return Response.json(await store.leaderboard(path[1]), { headers });
      if (path.length === 3 && path[2] === 'submissions') {
        if (request.method === 'GET') {
          const { limit, offset } = pagination(url);
          return Response.json(await store.listSubmissions(path[1], limit, offset), { headers });
        }
        if (request.method === 'POST') {
          const submission = await store.createSubmission(path[1], await body(request));
          waitUntil(advanceVerification(store, env as unknown as Record<string, unknown>).catch(() => {}));
          return Response.json(submission, { status: 202, headers: { ...headers, Location: `/api/submissions/${submission.id}` } });
        }
      }
    }
    if (path[0] === 'submissions' && path.length === 2 && request.method === 'GET') return Response.json(await store.getSubmission(path[1]), { headers });
    const resourceResponse = await resourceApi(request, path, { ...env, DB: env.DB });
    if (resourceResponse) return resourceResponse;
    return Response.json({ error: 'API route not found' }, { status: 404, headers });
  } catch (error) {
    if (error instanceof z.ZodError) return Response.json({ error: 'Invalid request', details: error.issues.map(({ path, message }) => ({ path, message })) }, { status: 400, headers });
    if (error instanceof ApiError || (error instanceof Error && 'status' in error && typeof error.status === 'number')) {
      return Response.json({ error: error.message }, { status: (error as ApiError).status, headers });
    }
    console.error('Dashboard API request failed', error instanceof Error ? error.message : 'Unexpected error');
    return Response.json({ error: 'Unable to complete request' }, { status: 500, headers });
  }
}

export const GET = route;
export const POST = route;
