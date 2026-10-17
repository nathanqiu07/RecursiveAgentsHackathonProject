import assert from 'node:assert/strict';
import test from 'node:test';
import { nemotronStatus, startNemotronBaseline } from './nemotron-verifier.ts';
import { nemotronBenchmark, nemotronModel } from '../shared/nemotron-benchmark.ts';
import { BenchmarkStore } from './benchmark-store.ts';
import { ResourceStore } from './store.ts';

const env = { BENCHMARK_VERIFIER_NEMOTRON_KEY: 'private-verifier-key' };
const metrics = { mmlu: 100 * 45 / 57, mmlu_correct: 45, mmlu_count: 57,
  gsm8k: 100 * 12 / 16, gsm8k_correct: 12, gsm8k_count: 16, ppl: 5.5, identity: 'frozen-mini-suite' };
const mock = (handler: (url: string, init: RequestInit) => Response) =>
  ((url: string | URL | Request, init: RequestInit = {}) => Promise.resolve(handler(String(url), init))) as typeof fetch;

test('mini status stays unconfigured without a key and never requests a nonlocal origin', async () => {
  const request = mock(() => { throw new Error('No request should be made'); });
  assert.equal((await nemotronStatus({}, request)).configured, false);
  assert.equal((await nemotronStatus({ ...env, NEMOTRON_VERIFIER_URL: 'https://external.example' }, request)).configured, false);
  assert.equal((await startNemotronBaseline({}, request)).status, 503);
});

test('mini status exposes only measured mini metrics and retains credentials server-side', async () => {
  const state = await nemotronStatus(env, mock((url, options) => {
    assert.equal((options.headers as Record<string, string>).Authorization, 'Bearer private-verifier-key');
    assert.equal(options.redirect, 'error');
    return Response.json(url.endsWith('/health') ? { ready: true } : { metrics: { ...metrics, api_key: 'must-not-return' } });
  }));
  assert.equal(state.ready, true);
  assert.equal(state.baseline_ready, true);
  assert.deepEqual(state.baseline, metrics);
  assert.doesNotMatch(JSON.stringify(state), /private-verifier-key|must-not-return/);
});

test('wrong suite counts or inconsistent percentages never become a ready mini baseline', async () => {
  for (const baseline of [{ ...metrics, mmlu_count: 14042 }, { ...metrics, gsm8k: 99 }, { ...metrics, ppl: '5.5' }]) {
    const state = await nemotronStatus(env, mock((url) => Response.json(url.endsWith('/health') ? { ready: true } : { metrics: baseline })));
    assert.equal(state.baseline_ready, false);
    assert.equal(state.baseline, null);
  }
});

test('a missing GPU or suite stays pending without requesting baseline data', async () => {
  let requests = 0;
  const state = await nemotronStatus(env, mock(() => { requests++; return Response.json({ ready: false }); }));
  assert.equal(requests, 1);
  assert.equal(state.reachable, true);
  assert.equal(state.baseline, null);
});

test('configured offline pools expose safe per-node pending baselines without claiming readiness', async () => {
  const requests: string[] = [];
  const state = await nemotronStatus(env, mock((url) => {
    requests.push(url);
    return Response.json(url.endsWith('/health')
      ? { ready: false, baseline_ready: false, configured_nodes: 3, online_nodes: 0, baseline_status: 'unmeasured' }
      : { metrics: null, node_baselines: ['a5000-01', 'a5000-02', 'a5000-03'].map((node_id) => ({
        node_id, status: 'unmeasured', metrics: null, key: 'must-not-return',
      })) });
  }));
  assert.equal(requests.length, 2);
  assert.equal(state.reachable, true);
  assert.equal(state.ready, false);
  assert.equal(state.baseline_ready, false);
  assert.equal(state.node_baselines?.length, 3);
  assert.ok(state.node_baselines?.every((node) => node.metrics === null && node.status === 'unmeasured'));
  assert.doesNotMatch(JSON.stringify(state), /must-not-return|private-verifier-key/);
});

test('per-node measured results preserve mini counts and strip extra engine fields', async () => {
  const state = await nemotronStatus(env, mock((url) => Response.json(url.endsWith('/health')
    ? { ready: true, baseline_ready: true, configured_nodes: 1, online_nodes: 1, baseline_status: 'completed' }
    : { metrics, node_baselines: [{ node_id: 'a5000-01', status: 'completed', metrics: { ...metrics, key: 'must-not-return' } }] })));
  assert.equal(state.baseline_ready, true);
  assert.deepEqual(state.node_baselines, [{ node_id: 'a5000-01', status: 'completed', metrics }]);
  assert.doesNotMatch(JSON.stringify(state), /must-not-return/);
});

test('baseline start uses a fixed empty request and sanitizes engine errors', async () => {
  const result = await startNemotronBaseline(env, mock((url, options) => {
    assert.equal(url, 'http://127.0.0.1:9000/baseline');
    assert.equal(options.method, 'POST');
    assert.equal(options.body, undefined);
    return Response.json({ id: 'd7417600-6067-4c23-8d46-c8c20ebed5cf', status: 'queued', secret: 'must-not-return' }, { status: 202 });
  }));
  assert.equal(result.status, 202);
  assert.doesNotMatch(JSON.stringify(result), /must-not-return/);
  const failed = await startNemotronBaseline(env, mock(() => new Response('private-verifier-key', { status: 401 })));
  assert.equal(failed.status, 503);
  assert.doesNotMatch(JSON.stringify(failed), /private-verifier-key/);
});

test('the A5000 contract accepts a registered deployment and rejects alternate targets and claimed scores', () => {
  const resources = new ResourceStore(':memory:');
  try {
    const store = new BenchmarkStore(resources);
    const benchmark = store.createBenchmark({ ...nemotronBenchmark, verifier: { url: 'http://127.0.0.1:9000/evaluate' } });
    const payload = { repository: 'https://github.com/ShardulAgg/vllm', commit: 'a'.repeat(40), deployment: 'candidate',
      model: nemotronModel, gpu: 'NVIDIA RTX A5000', precision: 'NVFP4' };
    const input = { title: 'Candidate', base_checkpoint_url: `https://github.com/ShardulAgg/vllm/commit/${'b'.repeat(40)}`, payload };
    assert.equal(store.createSubmission(benchmark.id, input).status, 'queued');
    for (const invalid of [{ ...payload, gpu: 'NVIDIA RTX 5090' }, { ...payload, precision: 'BF16' }, { ...payload, score: 999 }]) {
      assert.throws(() => store.createSubmission(benchmark.id, { ...input, payload: invalid }));
    }
    assert.equal(store.leaderboard(benchmark.id).total, 0);
  } finally { resources.close(); }
});
