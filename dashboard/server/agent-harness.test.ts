import assert from 'node:assert/strict';
import test from 'node:test';
import { agentHarnessApi } from './agent-harness.ts';

const RUN = '11111111-1111-4111-8111-111111111111';
const BENCHMARK = '22222222-2222-4222-8222-222222222222';
const SECRET = 'harness-server-only-secret';
const env = { AGENT_HARNESS_URL: 'http://127.0.0.1:8003', AGENT_HARNESS_API_KEY: SECRET };
const body = {
  role: 'researcher', objective: 'Investigate a new recurrence.', max_steps: 40,
  allocation: { resource_id: 'cpu', node_id: 'node', cpus: 1, memory_mb: 512,
    scratch_mb: 64, timeout_seconds: 600 },
};
const result = { id: RUN, status: 'queued', request: body, state: {}, cleanup_required: false };
const reply = (value: unknown, status = 200, headers?: Record<string, string>) => Response.json(value, { status, headers });
function request(path: string, method = 'GET', value?: unknown, headers: Record<string, string> = {}) {
  return new Request('http://localhost:3001/api/agents/' + path, { method,
    headers: { ...(value === undefined ? {} : { 'Content-Type': 'application/json' }), ...headers },
    ...(value === undefined ? {} : { body: JSON.stringify(value) }) });
}
const fetcher = (handler: (url: string, init: RequestInit) => Response | Promise<Response>): typeof fetch =>
  ((url: string | URL | Request, init: RequestInit = {}) => handler(String(url), init)) as typeof fetch;
async function call(path: string, method = 'GET', value?: unknown, handler = fetcher(() => reply(result)),
  headers: Record<string, string> = {}, settings: Record<string, unknown> = env) {
  const source = request(path, method, value, headers);
  const segments = ['agents', ...new URL(source.url).pathname.slice('/api/agents/'.length).split('/')];
  const response = await agentHarnessApi(source, segments, settings, handler);
  assert.ok(response);
  return { response, body: await response.json() as Record<string, unknown> };
}

test('connection status is explicit, does not probe, and never exposes connection credentials', async () => {
  const never = fetcher(() => { throw new Error('Should not probe'); });
  const configured = await call('config', 'GET', undefined, never);
  assert.equal(configured.response.status, 200);
  assert.deepEqual(configured.body, { configured: true });
  assert.ok(!JSON.stringify(configured.body).includes(SECRET));
  assert.ok(!JSON.stringify(configured.body).includes('8003'));
  const missing = await call('config', 'GET', undefined, never, {}, {});
  assert.equal(missing.body.configured, false);
  assert.ok(String(missing.body.message).includes('Connect'));
  assert.equal((await call('runs', 'GET', undefined, never, {}, {})).response.status, 503);
});

test('malformed connection origins are treated as unavailable without exposing configuration', async () => {
  for (const url of ['https://user:secret@harness.example', 'file:///tmp/service',
    'https://harness.example/private', 'https://harness.example?token=secret',
    'https://harness.example/#secret', 'https://harness.example\\bad']) {
    const actual = await call('config', 'GET', undefined, undefined, {}, { AGENT_HARNESS_URL: url });
    assert.equal(actual.body.configured, false);
    assert.ok(!JSON.stringify(actual.body).includes(url));
  }
});

test('run listing uses only configured origin and its own bearer token, with bounded pagination', async () => {
  let calls = 0;
  const upstream = fetcher((url, init) => {
    calls += 1;
    assert.equal(url, 'http://127.0.0.1:8003/runs?limit=25&offset=50');
    assert.equal(init.method, 'GET');
    assert.equal(init.redirect, 'error');
    assert.ok(init.signal instanceof AbortSignal);
    const sent = new Headers(init.headers);
    assert.equal(sent.get('Authorization'), `Bearer ${SECRET}`);
    assert.equal(sent.get('Cookie'), null);
    assert.equal(sent.get('X-Leak'), null);
    return reply({ runs: [result], total: 90, limit: 25, offset: 50 }, 200,
      { 'Set-Cookie': 'secret=response', Location: 'https://internal.example/' });
  });
  const actual = await call('runs?offset=50&limit=25', 'GET', undefined, upstream,
    { Authorization: 'Bearer browser-secret', Cookie: 'private-cookie', 'X-Leak': 'no' });
  assert.equal(actual.response.status, 200);
  assert.equal(actual.response.headers.get('Cache-Control'), 'no-store');
  assert.equal(actual.response.headers.get('Set-Cookie'), null);
  assert.equal(actual.response.headers.get('Location'), null);
  assert.equal(calls, 1);
});

test('discovery and fixed run read routes forward only their intended capability', async () => {
  const cases = [
    ['resources', '/resources', { resources: [] }],
    ['status', '/status', { workers_online: 1, last_worker_seen_at: 1000, queued_runs: 2, running_runs: 1 }],
    [`runs/${RUN}`, `/runs/${RUN}`, result],
    [`runs/${RUN}/events?after=10&limit=200`, `/runs/${RUN}/events?after=10&limit=200`, { events: [], next_cursor: 10 }],
    [`runs/${RUN}/publication`, `/runs/${RUN}/publication`, { id: BENCHMARK, status: 'pending_verification' }],
  ] as const;
  for (const [path, destination, value] of cases) {
    const actual = await call(path, 'GET', undefined, fetcher((url) => {
      assert.equal(url, env.AGENT_HARNESS_URL + destination);
      return reply(value);
    }));
    assert.equal(actual.response.status, 200);
    assert.deepEqual(actual.body, value);
  }
});

test('shared experiment inventory keeps busy pools visible without accepting extra control routes', async () => {
  const value = { source: 'resource_manager', checked_at: 1000, resources: [{
    id: 'a5000-one', node_id: 'gpu-one', description: 'A5000 GPU VM', status: 'busy', runtime: 'remote_vm',
    available_slots: 0, total_slots: 1, held_jobs: 0, available_cpus: 8,
    available_memory_mb: 65536, memory_mb: 65536, max_timeout_seconds: 21600,
  }] };
  const actual = await call('resource-status', 'GET', undefined, fetcher((url, init) => {
    assert.equal(url, env.AGENT_HARNESS_URL + '/resource-status');
    assert.equal(init.method, 'GET');
    assert.equal(new Headers(init.headers).get('Authorization'), `Bearer ${SECRET}`);
    return reply(value);
  }));
  assert.equal(actual.response.status, 200);
  assert.deepEqual(actual.body, value);
  assert.equal((await call('resource-status', 'POST')).response.status, 405);
  assert.equal((await call('resource-status?node_id=arbitrary')).response.status, 400);
});

test('launcher validates and forwards a normalized role request and stable idempotency key', async () => {
  const actual = await call('runs', 'POST', { ...body, role: 'attempt', benchmark_id: BENCHMARK },
    fetcher((url, init) => {
      assert.equal(url, env.AGENT_HARNESS_URL + '/runs');
      assert.equal(init.method, 'POST');
      assert.equal(new Headers(init.headers).get('Idempotency-Key'), 'launch-unique-123');
      assert.deepEqual(JSON.parse(String(init.body)), { ...body, role: 'attempt', benchmark_id: BENCHMARK });
      return reply(result, 202);
    }), { 'Idempotency-Key': 'launch-unique-123' });
  assert.equal(actual.response.status, 202);
  assert.equal(actual.body.id, RUN);
});

test('one-click research forwards only benchmark context and preserves its idempotency key', async () => {
  const actual = await call('research-runs', 'POST', { benchmark_id: BENCHMARK }, fetcher((url, init) => {
    assert.equal(url, env.AGENT_HARNESS_URL + '/research-runs');
    assert.equal(init.method, 'POST');
    assert.equal(new Headers(init.headers).get('Idempotency-Key'), 'one-click-research');
    assert.deepEqual(JSON.parse(String(init.body)), { benchmark_id: BENCHMARK });
    return reply({ ...result, request: { ...body, allocation: null, benchmark_id: BENCHMARK } }, 202);
  }), { 'Idempotency-Key': 'one-click-research' });
  assert.equal(actual.response.status, 202);
  for (const extra of [{ objective: 'user option' }, { allocation: null }, { max_steps: 1 }, { count: 3 }]) {
    const invalid = await call('research-runs', 'POST', { benchmark_id: BENCHMARK, ...extra },
      fetcher(() => { throw new Error('Do not forward extra controls'); }));
    assert.equal(invalid.response.status, 422);
  }
  assert.equal((await call('research-runs', 'GET')).response.status, 405);
});

test('launch defaults match the harness request contract without accepting caller-selected images', async () => {
  const input = { ...body } as Record<string, unknown>;
  delete input.max_steps;
  const actual = await call('runs', 'POST', input, fetcher((_url, init) => {
    assert.deepEqual(JSON.parse(String(init.body)), { ...body, benchmark_id: null });
    return reply(result, 202);
  }));
  assert.equal(actual.response.status, 202);
  for (const change of [
    { role: 'worker' }, { role: 'attempt' }, { benchmark_id: 'not-a-uuid' },
    { max_steps: 0 }, { max_steps: 101 }, { max_steps: '40' }, { objective: ' ' },
    { objective: 'x'.repeat(20001) }, { image: 'untrusted:latest' },
    { allocation: { ...body.allocation, cpus: 0 } },
    { allocation: { ...body.allocation, cpus: 257 } },
    { allocation: { ...body.allocation, memory_mb: 63 } },
    { allocation: { ...body.allocation, scratch_mb: 15 } },
    { allocation: { ...body.allocation, timeout_seconds: 29 } },
    { allocation: { ...body.allocation, timeout_seconds: 86401 } },
    { allocation: { ...body.allocation, command: 'host-shell' } },
    { allocation: { ...body.allocation, node_id: '../escape' } },
  ]) {
    const invalid = await call('runs', 'POST', { ...body, ...change }, fetcher(() => {
      assert.fail('Invalid settings must not dispatch');
    }));
    assert.equal(invalid.response.status, 422);
  }
});

test('invalid route, method, path identifiers and unexpected query never reach the harness', async () => {
  const never = fetcher(() => { assert.fail('Must not dispatch'); });
  for (const [path, method, status] of [
    ['resources?destination=https://attacker.example', 'GET', 400],
    ['runs?limit=101', 'GET', 400], ['runs?offset=-1', 'GET', 400],
    ['runs?offset=9007199254740992', 'GET', 400], ['runs?limit=1&limit=2', 'GET', 400],
    ['runs?limit=', 'GET', 400], [`runs/${RUN}/events?limit=201`, 'GET', 400],
    [`runs/${RUN}/events?after=-1`, 'GET', 400], [`runs/${RUN}?next=/jobs`, 'GET', 400],
    ['config?probe=true', 'GET', 400], ['config', 'POST', 405],
    ['resources', 'POST', 405], [`runs/${RUN}/cancel`, 'GET', 405],
    [`runs/${RUN}/events`, 'POST', 405], ['runs', 'DELETE', 405],
    ['runs/not-a-uuid', 'GET', 404], [`runs/${RUN}/stdin`, 'POST', 404],
    ['jobs', 'GET', 404], ['health', 'GET', 404],
  ] as const) {
    assert.equal((await call(path, method, undefined, never)).response.status, status, path);
  }
  assert.equal(await agentHarnessApi(request('config'), ['benchmarks'], env, never), null);
});

test('cancellation is a fixed empty mutation and does not forward user-supplied body or headers', async () => {
  let calls = 0;
  const actual = await call(`runs/${RUN}/cancel`, 'POST', {}, fetcher((url, init) => {
    calls += 1;
    assert.equal(url, env.AGENT_HARNESS_URL + `/runs/${RUN}/cancel`);
    assert.equal(init.method, 'POST');
    assert.equal(init.body, undefined);
    assert.equal(new Headers(init.headers).get('Idempotency-Key'), null);
    return reply({ ...result, cancel_requested: true });
  }), { 'Idempotency-Key': 'irrelevant' });
  assert.equal(actual.response.status, 200);
  assert.equal(calls, 1);
  assert.equal((await call(`runs/${RUN}/cancel`, 'POST', { url: 'https://attacker.example' },
    fetcher(() => { assert.fail('Extra cancellation settings must not dispatch'); }))).response.status, 400);
});

test('upstream errors are safe field messages and never disclose keys, URLs, or reflected values', async () => {
  for (const status of [400, 401, 403, 404, 409, 422, 429, 500, 503]) {
    const actual = await call('runs', 'POST', body, fetcher(() => reply({
      error: SECRET, detail: [{ loc: ['body', 'allocation', 'memory_mb'],
        msg: `Cannot use ${SECRET}`, input: SECRET }, { loc: ['body', SECRET], msg: SECRET }],
    }, status)));
    assert.equal(actual.response.status, status < 500 ? status : 502);
    assert.ok(!JSON.stringify(actual.body).includes(SECRET));
    assert.ok(!JSON.stringify(actual.body).includes('8003'));
    if (status === 422) assert.deepEqual(actual.body.details,
      [{ path: ['allocation', 'memory_mb'], message: 'Check this value.' }]);
    assert.equal(actual.body.uncertain, status >= 500 ? true : undefined);
  }
});

test('lost launch acknowledgements stay uncertain and are never dispatched twice automatically', async () => {
  for (const mode of ['network', 'timeout', 'malformed', 'wrong-status', 'wrong-id', 'redirect']) {
    let calls = 0;
    const actual = await call('runs', 'POST', body, fetcher(() => {
      calls += 1;
      if (mode === 'network') throw new Error(`Network failure with ${SECRET}`);
      if (mode === 'timeout') throw new DOMException(`Timeout containing ${SECRET}`, 'TimeoutError');
      if (mode === 'malformed') return new Response(`invalid ${SECRET}`, { status: 202 });
      if (mode === 'wrong-status') return reply(result, 200);
      if (mode === 'wrong-id') return reply({ ...result, id: 'not-a-uuid' }, 202);
      return reply({ error: SECRET }, 302, { Location: 'https://attacker.example' });
    }), { 'Idempotency-Key': 'preserve-this-launch' });
    assert.equal(actual.response.status, 502, mode);
    assert.equal(actual.body.uncertain, true, mode);
    assert.ok(!JSON.stringify(actual.body).includes(SECRET));
    assert.equal(calls, 1, mode);
  }
});

test('unreadable, compressed, wrong-run and oversized read responses are rejected safely', async () => {
  for (const value of [
    new Response(SECRET),
    reply(result, 200, { 'Content-Encoding': 'gzip' }),
    reply({ ...result, id: BENCHMARK }),
    new Response('x'.repeat(16 * 1024 * 1024 + 1)),
  ]) {
    const actual = await call(`runs/${RUN}`, 'GET', undefined, fetcher(() => value));
    assert.equal(actual.response.status, 502);
    assert.ok(!JSON.stringify(actual.body).includes(SECRET));
  }
});

test('body and launch key validation reject malformed requests before dispatch', async () => {
  const never = fetcher(() => { assert.fail('Malformed request must not dispatch'); });
  assert.equal((await call('runs', 'POST', body, never, { 'Idempotency-Key': 'has spaces' })).response.status, 400);
  const invalidJSON = new Request('http://localhost/api/agents/runs', { method: 'POST',
    headers: { 'Content-Type': 'application/json' }, body: 'invalid' });
  assert.equal((await agentHarnessApi(invalidJSON, ['agents', 'runs'], env, never))?.status, 400);
  const huge = new Request('http://localhost/api/agents/runs', { method: 'POST',
    headers: { 'Content-Type': 'application/json' }, body: 'x'.repeat(256 * 1024 + 1) });
  assert.equal((await agentHarnessApi(huge, ['agents', 'runs'], env, never))?.status, 400);
  const wrongType = new Request('http://localhost/api/agents/runs', { method: 'POST', body: JSON.stringify(body) });
  assert.equal((await agentHarnessApi(wrongType, ['agents', 'runs'], env, never))?.status, 415);
});
