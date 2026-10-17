import { z } from 'zod';

const UUID = /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i;
const MAX_RESPONSE_BYTES = 16 * 1024 * 1024;
const MAX_REQUEST_BYTES = 256 * 1024;
const headers = { 'Cache-Control': 'no-store' };
const identifier = z.string().min(1).max(100).regex(/^[A-Za-z0-9][A-Za-z0-9._-]*$/);
const runRequest = z.object({
  role: z.enum(['researcher', 'attempt']),
  objective: z.string().trim().min(1).max(20000),
  benchmark_id: z.string().regex(UUID).nullable().default(null),
  allocation: z.object({
    resource_id: identifier, node_id: identifier,
    cpus: z.number().finite().positive().max(256),
    memory_mb: z.number().int().min(64).max(1048576),
    scratch_mb: z.number().int().min(16).max(1048576),
    timeout_seconds: z.number().int().min(30).max(86400),
  }).strict(),
  max_steps: z.number().int().min(1).max(100).default(40),
  harness: z.enum(['builtin', 'codex', 'claude-code', 'nemoclaw']).optional(),
  model_profile_id: z.string().regex(UUID).nullable().optional(),
  model: z.string().max(256).regex(/^[A-Za-z0-9._/:-]*$/).nullable().optional(),
}).strict().superRefine((value, context) => {
  if (value.role === 'attempt' && !value.benchmark_id) {
    context.addIssue({ code: 'custom', path: ['benchmark_id'], message: 'Choose a benchmark for an attempt agent.' });
  }
});

type Route = { method: 'GET' | 'POST'; destination: string; kind: string; runId?: string; provider?: string };
type Problem = { error: string; uncertain?: boolean; details?: { path: string[]; message: string }[] };

function json(value: unknown, status = 200, extra?: Record<string, string>) {
  return Response.json(value, { status, headers: { ...headers, ...extra } });
}

function connection(env: Record<string, unknown>): { url: string; key?: string } | null {
  const raw = env.AGENT_HARNESS_URL;
  if (typeof raw !== 'string' || !raw) return null;
  try {
    const url = new URL(raw);
    const key = env.AGENT_HARNESS_API_KEY;
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password
      || url.pathname !== '/' || url.search || url.hash || /[\s\\\u0000-\u001f\u007f]/.test(raw)
      || (key !== undefined && key !== '' && (typeof key !== 'string' || /[^\x20-\x7e]/.test(key)))) return null;
    return { url: url.origin, key: typeof key === 'string' && key ? key : undefined };
  } catch { return null; }
}

function query(url: URL, allowed: Record<string, { initial: number; max: number }>) {
  const result = new URLSearchParams();
  const names = new Set<string>();
  for (const [key, value] of url.searchParams) {
    if (!Object.hasOwn(allowed, key) || names.has(key) || !/^\d+$/.test(value)) return null;
    names.add(key);
  }
  for (const [key, setting] of Object.entries(allowed)) {
    const raw = url.searchParams.get(key);
    const value = raw === null ? setting.initial : Number(raw);
    if (!Number.isSafeInteger(value) || value < (key === 'limit' ? 1 : 0) || value > setting.max) return null;
    result.set(key, String(value));
  }
  return result;
}

function route(path: string[]): Route | null {
  if (path[1] === 'settings') {
    if (path.length === 3 && ['providers', 'readiness', 'harnesses'].includes(path[2])) return {
      method: 'GET', destination: `/settings/${path[2]}`, kind: path[2],
    };
    if (path[2] === 'harnesses' && path[3] === 'profiles') {
      if (path.length === 4) return { method: 'POST', destination: '/settings/harnesses/profiles', kind: 'profile_save' };
      if (path.length === 6 && UUID.test(path[4]) && path[5] === 'remove') return {
        method: 'POST', destination: `/settings/harnesses/profiles/${path[4]}/remove`, kind: 'profile_remove',
      };
    }
    if (path[2] !== 'providers' || !['crusoe', 'you'].includes(path[3])) return null;
    const provider = path[3];
    if (path.length === 4) return { method: 'POST', destination: `/settings/providers/${provider}`, kind: 'provider_save', provider };
    if (path.length === 5 && (['test', 'remove'].includes(path[4]) || (provider === 'crusoe' && path[4] === 'models'))) return {
      method: path[4] === 'models' ? 'GET' : 'POST', destination: `/settings/providers/${provider}/${path[4]}`,
      kind: path[4] === 'models' ? 'models' : `provider_${path[4]}`, provider,
    };
    return null;
  }
  if (path.length === 2 && ['resources', 'status'].includes(path[1])) {
    return { method: 'GET', destination: `/${path[1]}`, kind: path[1] };
  }
  if (path[1] !== 'runs') return null;
  if (path.length === 2) return { method: 'GET', destination: '/runs', kind: 'runs' };
  if (!UUID.test(path[2])) return null;
  const runId = path[2].toLowerCase();
  if (path.length === 3) return { method: 'GET', destination: `/runs/${runId}`, kind: 'run', runId };
  if (path.length !== 4 || !['events', 'cancel', 'publication'].includes(path[3])) return null;
  return { method: path[3] === 'cancel' ? 'POST' : 'GET',
    destination: `/runs/${runId}/${path[3]}`, kind: path[3], runId };
}

async function boundedBody(response: Response | Request, maximum: number) {
  if (!response.body) return new Uint8Array();
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let bytes = 0;
  try {
    while (true) {
      const part = await reader.read();
      if (part.done) break;
      bytes += part.value.byteLength;
      if (bytes > maximum) throw new Error('Body limit exceeded');
      chunks.push(part.value);
    }
  } finally { await reader.cancel().catch(() => {}); }
  const content = new Uint8Array(bytes);
  let offset = 0;
  for (const part of chunks) { content.set(part, offset); offset += part.byteLength; }
  return content;
}

function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function validResponse(value: unknown, selected: Route, launch: boolean) {
  if (!record(value)) return false;
  if (selected.kind === 'resources') return Array.isArray(value.resources);
  if (selected.kind === 'status') return typeof value.workers_online === 'number'
    && Number.isInteger(value.workers_online) && value.workers_online >= 0
    && (value.last_worker_seen_at === null || (typeof value.last_worker_seen_at === 'number' && Number.isFinite(value.last_worker_seen_at)))
    && typeof value.queued_runs === 'number' && Number.isInteger(value.queued_runs) && value.queued_runs >= 0
    && typeof value.running_runs === 'number' && Number.isInteger(value.running_runs) && value.running_runs >= 0;
  if (selected.kind === 'events') return Array.isArray(value.events)
    && typeof value.next_cursor === 'number' && Number.isSafeInteger(value.next_cursor) && value.next_cursor >= 0;
  if (selected.kind === 'runs' && !launch) return Array.isArray(value.runs)
    && typeof value.total === 'number' && Number.isSafeInteger(value.total) && value.total >= 0;
  if (typeof value.id !== 'string' || !UUID.test(value.id)) return false;
  if (selected.kind !== 'publication' && selected.runId && value.id.toLowerCase() !== selected.runId) return false;
  return true;
}

const keyInput = z.string().max(4096).regex(/^[\x20-\x7e]*$/);
const modelInput = z.string().max(256).regex(/^[A-Za-z0-9._/:-]*$/);
const providerState = z.object({
  configured: z.boolean(), api_key_set: z.boolean(), enabled: z.boolean(),
  source: z.enum(['saved', 'environment', 'none']), model: z.string().max(256).nullable().optional(),
  check: z.object({ ok: z.boolean(), message: z.string().max(500), checked_at: z.number().finite() }).nullable(),
});
const settingsResponse = z.object({ providers: z.object({ crusoe: providerState, you: providerState }), revision: z.number().int().nonnegative() });
const profileInput = z.object({ id: z.string().regex(UUID).optional(), name: z.string().trim().min(1).max(100),
  harness: z.enum(['builtin', 'codex', 'claude-code', 'nemoclaw']), model: modelInput.min(1),
  base_url: z.string().max(2048).nullable().optional(), api_key: keyInput.optional(),
  auth_mode: z.enum(['codex-oauth', 'api-key', 'environment']),
}).strict();
const harnessResponse = z.object({
  default_harness: z.enum(['builtin', 'codex', 'claude-code', 'nemoclaw']),
  harnesses: z.array(z.object({ id: z.enum(['builtin', 'codex', 'claude-code', 'nemoclaw']), label: z.string(),
    available: z.boolean(), message: z.string(), models: z.array(z.object({ id: z.string(), label: z.string().optional() })) })),
  profiles: z.array(z.object({ id: z.string().regex(UUID), name: z.string(), harness: z.enum(['builtin', 'codex', 'claude-code', 'nemoclaw']),
    model: z.string(), base_url: z.string().nullable(), api_key_set: z.boolean(), enabled: z.boolean(),
    auth_mode: z.enum(['codex-oauth', 'api-key', 'environment']) })),
});
const readinessResponse = z.object({ ready: z.boolean(), checks: z.array(z.object({ id: z.string(), label: z.string(),
  ok: z.boolean(), required: z.boolean(), message: z.string() })) });

function settingsOutput(value: unknown, kind: string) {
  const schema = kind === 'models' ? z.object({ models: z.array(z.object({ id: z.string(), supported_parameters: z.array(z.string()).optional() })) })
    : kind === 'readiness' ? readinessResponse : kind === 'harnesses' || kind.startsWith('profile_') ? harnessResponse : settingsResponse;
  return schema.safeParse(value);
}

const fieldNames = new Set(['role', 'objective', 'benchmark_id', 'allocation', 'resource_id', 'node_id',
  'cpus', 'memory_mb', 'scratch_mb', 'timeout_seconds', 'max_steps']);

function upstreamProblem(status: number, body: unknown, launch: boolean, mutation: boolean): Problem {
  if (launch && (status >= 500 || (status >= 300 && status < 400))) return { error: 'The launch could not be confirmed. Refresh the run list before retrying the same launch request.', uncertain: true };
  if (status === 400 || status === 422) {
    const issues = record(body) && Array.isArray(body.detail) ? body.detail : [];
    const details = issues.slice(0, 30).flatMap((issue: unknown) => {
      if (!record(issue) || !Array.isArray(issue.loc)) return [];
      const path = issue.loc.filter((item): item is string => typeof item === 'string' && fieldNames.has(item));
      return path.length ? [{ path, message: 'Check this value.' }] : [];
    });
    return { error: 'The agent harness rejected these settings. Check the values and try again.', ...(details.length ? { details } : {}) };
  }
  if (status === 401 || status === 403) return { error: 'The dashboard cannot authenticate with the agent harness. Check the server connection settings.' };
  if (status === 404) return { error: 'The requested agent run or result was not found.' };
  if (status === 409) return { error: 'This action is not available in the current run state. Refresh to see its latest status.' };
  if (status === 429) return { error: 'The agent harness is busy. Try again shortly.' };
  if (mutation && (status >= 500 || (status >= 300 && status < 400))) return { error: 'Cancellation could not be confirmed. Refresh to check the run.', uncertain: true };
  return { error: 'The agent harness could not complete this request. Check the harness service.' };
}

/** Fixed-route, credential-owning bridge shared by the local and hosted dashboard. */
export async function agentHarnessApi(request: Request, path: string[], env: Record<string, unknown>,
  requester: typeof fetch = fetch): Promise<Response | null> {
  if (path[0] !== 'agents') return null;
  const url = new URL(request.url);
  const settings = connection(env);
  if (path.length === 2 && path[1] === 'config') {
    if (request.method !== 'GET') return json({ error: 'Use GET for connection status.' }, 405, { Allow: 'GET' });
    if ([...url.searchParams].length) return json({ error: 'Unexpected query parameters.' }, 400);
    return json({ configured: settings !== null,
      ...(settings ? {} : { message: 'Connect the agent harness in the dashboard server settings to launch and monitor agents.' }) });
  }
  const selected = route(path);
  if (!selected) return json({ error: 'Agent route not found.' }, 404);
  const launch = selected.kind === 'runs' && request.method === 'POST';
  const providerRoute = path[1] === 'settings';
  const expectedMethod = launch ? 'POST' : selected.method;
  if (request.method !== expectedMethod) {
    return json({ error: 'This method is not supported for the agent route.' }, 405,
      { Allow: selected.kind === 'runs' ? 'GET, POST' : selected.method });
  }
  const pagination = selected.kind === 'runs' && !launch
    ? query(url, { limit: { initial: 50, max: 100 }, offset: { initial: 0, max: Number.MAX_SAFE_INTEGER } })
    : selected.kind === 'events'
      ? query(url, { after: { initial: 0, max: Number.MAX_SAFE_INTEGER }, limit: { initial: 100, max: 200 } })
      : query(url, {});
  if (!pagination) return json({ error: 'Invalid pagination or unexpected query parameters.' }, 400);
  if (!settings) return json({ error: 'The dashboard is not connected to an agent harness. Configure the server connection first.' }, 503);

  let payload: unknown;
  const outgoingHeaders: Record<string, string> = { Accept: 'application/json', 'Accept-Encoding': 'identity' };
  if (settings.key) outgoingHeaders.Authorization = `Bearer ${settings.key}`;
  if (providerRoute && request.method === 'POST' && ['provider_save', 'profile_save'].includes(selected.kind)) {
    if (request.headers.get('Content-Type')?.split(';')[0].trim().toLowerCase() !== 'application/json') return json({ error: 'Use application/json for connection settings.' }, 415);
    try {
      const input: unknown = JSON.parse(new TextDecoder().decode(await boundedBody(request, 16384)));
      const schema = selected.kind === 'profile_save' ? profileInput
        : selected.provider === 'crusoe' ? z.object({ api_key: keyInput.optional(), model: modelInput.optional() }).strict()
          : z.object({ api_key: keyInput.optional() }).strict();
      const valid = schema.safeParse(input);
      if (!valid.success) return json({ error: 'Check the connection settings. Only supported fields and printable credentials are accepted.' }, 422);
      payload = valid.data;
      outgoingHeaders['Content-Type'] = 'application/json';
    } catch { return json({ error: 'Connection settings must be valid JSON of at most 16 KiB.' }, 400); }
  } else if (launch) {
    const idempotencyKey = request.headers.get('Idempotency-Key');
    if (idempotencyKey !== null) {
      if (!/^[\x21-\x7e]{1,200}$/.test(idempotencyKey)) return json({ error: 'Invalid launch request key.' }, 400);
      outgoingHeaders['Idempotency-Key'] = idempotencyKey;
    }
    if (request.headers.get('Content-Type')?.split(';')[0].trim().toLowerCase() !== 'application/json') return json({ error: 'Use application/json for agent launch requests.' }, 415);
    try {
      const content = await boundedBody(request, MAX_REQUEST_BYTES);
      const input: unknown = JSON.parse(new TextDecoder().decode(content));
      const validated = runRequest.safeParse(input);
      if (!validated.success) {
        return json({ error: 'Check the agent launch settings.', details: validated.error.issues.map((issue) => ({
          path: issue.path.map(String), message: issue.message,
        })) }, 422);
      }
      payload = validated.data;
      outgoingHeaders['Content-Type'] = 'application/json';
    } catch {
      return json({ error: 'The launch body must be valid JSON of at most 256 KiB.' }, 400);
    }
  } else if (request.method === 'POST') {
    try {
      const input = new TextDecoder().decode(await boundedBody(request, 1024)).trim();
      if (input && input !== '{}') return json({ error: 'This action does not accept extra settings.' }, 400);
    } catch { return json({ error: 'Invalid action request.' }, 400); }
  }
  const destination = settings.url + selected.destination + (pagination.size ? `?${pagination}` : '');
  let response: Response;
  try {
    response = await requester(destination, { method: expectedMethod, headers: outgoingHeaders,
      redirect: 'error', signal: AbortSignal.any([request.signal, AbortSignal.timeout(providerRoute ? 45000 : 15000)]),
      ...(payload === undefined ? {} : { body: JSON.stringify(payload) }) });
  } catch {
    if (providerRoute) return json({ error: 'The connection request could not be confirmed. Refresh its status before trying again.' }, 502);
    return json(launch ? { error: 'The launch could not be confirmed. Refresh the run list before retrying the same launch request.', uncertain: true }
      : request.method === 'POST' ? { error: 'Cancellation could not be confirmed. Refresh to check the run.', uncertain: true }
        : { error: 'The agent harness is unavailable. Check that the service is running.' }, 502);
  }
  let upstream: unknown;
  try {
    if (response.headers.get('Content-Encoding') && response.headers.get('Content-Encoding') !== 'identity') throw new Error('Unsupported encoding');
    upstream = JSON.parse(new TextDecoder().decode(await boundedBody(response, MAX_RESPONSE_BYTES)));
  } catch {
    if (providerRoute) return json({ error: 'The connection service returned an unreadable response. Refresh to check its status.' }, 502);
    return json(launch ? { error: 'The launch acknowledgement was unreadable. Refresh the run list before retrying the same launch request.', uncertain: true }
      : request.method === 'POST' ? { error: 'Cancellation could not be confirmed. Refresh to check the run.', uncertain: true }
        : { error: 'The agent harness returned an unreadable response.' }, 502);
  }
  if (!response.ok && providerRoute) return json({ error: response.status === 401 || response.status === 403
    ? 'The dashboard cannot authenticate with the harness settings service.'
    : response.status === 409 ? 'These connection settings are not ready for this action. Refresh and check the saved values.'
      : response.status === 400 || response.status === 422 ? 'Check the connection fields and try again.'
        : 'The settings service could not complete this request. Refresh its status before trying again.' },
  response.status >= 400 && response.status < 500 ? response.status : 502);
  if (!response.ok) return json(upstreamProblem(response.status, upstream, launch, request.method === 'POST'),
    response.status >= 400 && response.status < 500 ? response.status : 502);
  if (providerRoute) {
    const sanitized = settingsOutput(upstream, selected.kind);
    return response.status === 200 && sanitized.success ? json(sanitized.data)
      : json({ error: 'The connection service returned an invalid response.' }, 502);
  }
  if ((launch && response.status !== 202) || (!launch && response.status !== 200)
    || !validResponse(upstream, selected, launch)) {
    return json({ error: launch ? 'The launch acknowledgement was invalid. Refresh the run list before retrying the same launch request.'
      : 'The agent harness returned an invalid response.', ...(launch ? { uncertain: true } : {}) }, 502);
  }
  return json(upstream, response.status);
}
