import assert from 'node:assert/strict';
import { once } from 'node:events';
import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import test from 'node:test';
import { createApp } from './app.ts';
import { ResourceStore } from './store.ts';

test('dashboard HTTP routes launch and inspect agents through their server-only harness connection', async () => {
  const runId = '11111111-1111-4111-8111-111111111111';
  const calls: { path: string; method: string; key?: string; authorization?: string; body?: unknown }[] = [];
  const harness = createServer(async (request, response) => {
    let raw = '';
    for await (const part of request) raw += part;
    calls.push({ path: request.url!, method: request.method!, key: request.headers['idempotency-key'] as string | undefined,
      authorization: request.headers.authorization, body: raw ? JSON.parse(raw) : undefined });
    const body = request.url === '/status'
      ? { workers_online: 1, last_worker_seen_at: 1000, queued_runs: 1, running_runs: 0 }
      : { id: runId, status: request.url?.endsWith('/cancel') ? 'cancelled' : 'queued' };
    response.writeHead(request.method === 'POST' && request.url === '/runs' ? 202 : 200,
      { 'Content-Type': 'application/json' });
    response.end(JSON.stringify(body));
  }).listen(0, '127.0.0.1');
  await once(harness, 'listening');
  const previousUrl = process.env.AGENT_HARNESS_URL;
  const previousKey = process.env.AGENT_HARNESS_API_KEY;
  process.env.AGENT_HARNESS_URL = `http://127.0.0.1:${(harness.address() as AddressInfo).port}`;
  process.env.AGENT_HARNESS_API_KEY = 'server-only-test-key';
  const store = new ResourceStore(':memory:');
  const server = createApp(store).listen(0, '127.0.0.1');
  await once(server, 'listening');
  const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}/api/agents`;
  try {
    assert.deepEqual(await (await fetch(base + '/config')).json(), { configured: true });
    assert.equal(calls.length, 0);
    const launch = await fetch(base + '/runs', { method: 'POST', headers: {
      'Content-Type': 'application/json', 'Idempotency-Key': 'browser-launch-1',
    }, body: JSON.stringify({ role: 'researcher', objective: 'Investigate an idea', allocation: {
      resource_id: 'cpu', node_id: 'node', cpus: 1, memory_mb: 128, scratch_mb: 16, timeout_seconds: 300,
    } }) });
    assert.equal(launch.status, 202);
    assert.equal((await launch.json()).id, runId);
    assert.equal(calls[0].key, 'browser-launch-1');
    assert.equal(calls[0].authorization, 'Bearer server-only-test-key');
    assert.equal(calls[0].method, 'POST');
    assert.equal(calls[0].path, '/runs');
    assert.equal((calls[0].body as { max_steps: number }).max_steps, 40);
    const status = await fetch(base + '/status');
    assert.equal((await status.json()).workers_online, 1);
    const cancel = await fetch(base + `/runs/${runId}/cancel`, { method: 'POST' });
    assert.equal((await cancel.json()).status, 'cancelled');
    assert.equal(calls[2].path, `/runs/${runId}/cancel`);
    const blocked = await fetch(base + '/runs', { method: 'POST', headers: {
      'Content-Type': 'application/json', Origin: 'https://unrelated.example',
    }, body: '{}' });
    assert.equal(blocked.status, 403);
    assert.equal(calls.length, 3);
  } finally {
    if (previousUrl === undefined) delete process.env.AGENT_HARNESS_URL;
    else process.env.AGENT_HARNESS_URL = previousUrl;
    if (previousKey === undefined) delete process.env.AGENT_HARNESS_API_KEY;
    else process.env.AGENT_HARNESS_API_KEY = previousKey;
    await Promise.all([new Promise<void>((resolve) => server.close(() => resolve())),
      new Promise<void>((resolve) => harness.close(() => resolve()))]);
    store.close();
  }
});
