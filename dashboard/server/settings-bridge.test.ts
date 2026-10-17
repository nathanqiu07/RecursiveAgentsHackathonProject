import assert from 'node:assert/strict';
import test from 'node:test';
import { agentHarnessApi } from './agent-harness.ts';

const SECRET = 'dummy-provider-secret-never-live';
const state = { configured: true, api_key_set: true, enabled: true, source: 'saved', check: null };
const settings = { revision: 1, providers: { crusoe: { ...state, model: 'model-a' }, you: state } };
const catalog = { default_harness: 'codex', harnesses: [{ id: 'codex', label: 'Codex', available: true,
  message: 'Signed in', models: [{ id: 'model-a' }] }], profiles: [] };

async function bridge(path: string, method = 'GET', body?: unknown, reply: unknown = settings, status = 200) {
  let sent = 0;
  let outgoing: RequestInit | undefined;
  let destination = '';
  const requester = (async (url: string, init: RequestInit) => {
    sent++; outgoing = init; destination = url;
    return Response.json(reply, { status });
  }) as typeof fetch;
  const request = new Request('http://localhost/api/agents/' + path, { method,
    headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
  const response = await agentHarnessApi(request, ['agents', ...path.split('?')[0].split('/')], {
    AGENT_HARNESS_URL: 'http://127.0.0.1:8003', AGENT_HARNESS_API_KEY: 'backend-auth',
  }, requester);
  assert.ok(response);
  return { response, body: await response.json(), sent, outgoing, destination };
}

test('provider saves forward secrets only in a bounded POST body and strip extra response keys', async () => {
  const reflected = { ...settings, api_key: SECRET, providers: { ...settings.providers,
    you: { ...state, api_key: SECRET } } };
  const result = await bridge('settings/providers/you', 'POST', { api_key: SECRET }, reflected);
  assert.equal(result.response.status, 200);
  assert.equal(result.destination, 'http://127.0.0.1:8003/settings/providers/you');
  assert.equal(new Headers(result.outgoing?.headers).get('Authorization'), 'Bearer backend-auth');
  assert.deepEqual(JSON.parse(String(result.outgoing?.body)), { api_key: SECRET });
  assert.ok(!JSON.stringify(result.body).includes(SECRET));
  assert.equal(result.response.headers.get('Cache-Control'), 'no-store');
});

test('settings reject unexpected routes, methods, fields, and credential query strings', async () => {
  for (const [path, method, body] of [
    ['settings/providers/other', 'POST', { api_key: SECRET }],
    ['settings/providers/you', 'GET', undefined],
    ['settings/providers/you', 'POST', { endpoint: 'https://other.test', api_key: SECRET }],
    ['settings/providers/you/test', 'POST', { api_key: SECRET }],
    ['settings/providers?api_key=' + SECRET, 'GET', undefined],
    ['settings/providers/you', 'POST', { api_key: 'bad\nheader' }],
  ] as const) {
    const result = await bridge(path, method, body);
    assert.equal(result.sent, 0);
    assert.ok(result.response.status >= 400);
    assert.ok(!JSON.stringify(result.body).includes(SECRET));
  }
});

test('model profiles and readiness use their own response allowlists', async () => {
  const profile = { name: 'My model', harness: 'codex', model: 'model-a', auth_mode: 'codex-oauth' };
  const saved = await bridge('settings/harnesses/profiles', 'POST', profile, { ...catalog, secret: SECRET });
  assert.equal(saved.response.status, 200);
  assert.deepEqual(saved.body, catalog);
  const status = { ready: false, checks: [{ id: 'worker', label: 'Worker', ok: false, required: true,
    message: 'Start a worker', api_key: SECRET }] };
  const readiness = await bridge('settings/readiness', 'GET', undefined, status);
  assert.equal(readiness.response.status, 200);
  assert.ok(!JSON.stringify(readiness.body).includes(SECRET));
  const models = await bridge('settings/providers/crusoe/models', 'GET', undefined,
    { models: [{ id: 'model-a', supported_parameters: ['tools'], api_key: SECRET }] });
  assert.deepEqual(models.body, { models: [{ id: 'model-a', supported_parameters: ['tools'] }] });
});

test('settings errors never forward upstream echoes of a submitted key', async () => {
  for (const status of [400, 401, 409, 422, 500]) {
    const result = await bridge('settings/providers/you', 'POST', { api_key: SECRET },
      { detail: 'Rejected ' + SECRET }, status);
    assert.ok(result.response.status >= 400);
    assert.ok(!JSON.stringify(result.body).includes(SECRET));
    assert.ok(!JSON.stringify(result.body).includes('Cancellation'));
  }
});
