import { z } from 'zod';
import type { NemotronVerifierStatus } from '../shared/nemotron-status.ts';

const baselineSchema = z.object({
  mmlu: z.number().finite().min(0).max(100),
  gsm8k: z.number().finite().min(0).max(100),
  ppl: z.number().finite().min(1), identity: z.string().min(1).max(256),
  mmlu_correct: z.number().int().min(0).max(57), mmlu_count: z.literal(57),
  gsm8k_correct: z.number().int().min(0).max(16), gsm8k_count: z.literal(16),
}).refine((value) => Math.abs(value.mmlu - 100 * value.mmlu_correct / 57) < 0.000001
  && Math.abs(value.gsm8k - 100 * value.gsm8k_correct / 16) < 0.000001, 'Mini percentages must match correct counts');
const healthSchema = z.object({ ready: z.boolean(), baseline_ready: z.boolean().optional(),
  configured_nodes: z.number().int().min(1).max(3).optional(), online_nodes: z.number().int().min(0).max(3).optional(),
  baseline_status: z.enum(['queued', 'running', 'completed', 'failed', 'rejected', 'pending', 'missing', 'unmeasured', 'unavailable', 'cancelled']).optional() });
const jobSchema = z.object({ id: z.string().uuid(), status: z.enum(['queued', 'running', 'completed', 'failed', 'rejected']) });
const nodeBaselineSchema = z.object({ node_id: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9_.-]{0,59}$/),
  status: z.enum(['queued', 'running', 'completed', 'failed', 'rejected', 'cancelled', 'unmeasured']), metrics: baselineSchema.nullable() });

function connection(env: NodeJS.ProcessEnv) {
  const key = env.BENCHMARK_VERIFIER_NEMOTRON_KEY || env.NEMOTRON_VERIFIER_API_KEY;
  try {
    const url = new URL(env.NEMOTRON_VERIFIER_URL || 'http://127.0.0.1:9000');
    if (!key || url.protocol !== 'http:' || !['127.0.0.1', 'localhost'].includes(url.hostname)
      || url.username || url.password || url.search || url.hash || !['', '/'].includes(url.pathname)) return null;
    return { url: url.origin, key };
  } catch { return null; }
}

async function readJSON(response: Response) {
  if (!response.body) throw new Error('Empty verifier response');
  let size = 0;
  const chunks: Uint8Array[] = [];
  for await (const chunk of response.body) {
    size += chunk.byteLength;
    if (size > 65536) throw new Error('Verifier response exceeded limit');
    chunks.push(chunk);
  }
  return JSON.parse(Buffer.concat(chunks).toString('utf8')) as unknown;
}

export async function nemotronStatus(env = process.env, request: typeof fetch = fetch): Promise<NemotronVerifierStatus> {
  const settings = connection(env);
  const state: NemotronVerifierStatus = {
    configured: settings !== null, reachable: false, ready: false, baseline_ready: false, baseline: null,
    message: 'Start the local stack to connect the mini verifier.',
  };
  if (!settings) return state;
  try {
    const options = { headers: { Authorization: `Bearer ${settings.key}` }, redirect: 'error' as const,
      signal: AbortSignal.timeout(8000) };
    const health = await request(`${settings.url}/health`, options);
    if (!health.ok) throw new Error('Verifier unavailable');
    const status = healthSchema.parse(await readJSON(health));
    state.reachable = true;
    state.ready = status.ready;
    state.baseline_status = status.baseline_status;
    state.configured_nodes = status.configured_nodes;
    state.online_nodes = status.online_nodes;
    state.message = 'Connect an RTX A5000 resource node and prepare the mini suite.';
    // A configured pool can expose safe pending per-node records before any
    // GPU is online. Legacy single-endpoint setup still requires readiness.
    if (!status.ready && status.configured_nodes === undefined) return state;
    const baseline = await request(`${settings.url}/baseline`, options);
    if (!baseline.ok) throw new Error('Baseline unavailable');
    const result = z.object({ metrics: baselineSchema.nullable(), node_baselines: z.array(nodeBaselineSchema).max(3).optional() }).parse(await readJSON(baseline));
    state.baseline = result.metrics;
    state.baseline_ready = status.ready && result.metrics !== null;
    state.node_baselines = result.node_baselines;
    state.message = !status.ready ? 'Connect the configured RTX A5000 nodes to measure their reference baselines.'
      : result.metrics ? 'Mini baseline measured on your RTX A5000.'
      : status.baseline_status === 'queued' || status.baseline_status === 'running'
        ? 'Measuring the reference mini baseline…' : 'Measure the unmodified model to establish the mini baseline.';
    return state;
  } catch {
    // Neither upstream error messages nor server connection details enter the browser.
    return { ...state, ready: false, baseline_ready: false, baseline: null,
      message: 'The local mini verifier is unavailable. Check local service status.' };
  }
}

export async function startNemotronBaseline(env = process.env, request: typeof fetch = fetch) {
  const settings = connection(env);
  if (!settings) return { status: 503, body: { error: 'Start the local stack to connect the mini verifier.' } };
  try {
    const response = await request(`${settings.url}/baseline`, { method: 'POST',
      headers: { Authorization: `Bearer ${settings.key}` }, redirect: 'error', signal: AbortSignal.timeout(15000) });
    if (!response.ok) return { status: response.status === 401 ? 503 : response.status === 409 ? 409 : 503,
      body: { error: 'The mini verifier is not ready. Connect a resource node and prepare the evaluation suite first.' } };
    return { status: 202, body: jobSchema.parse(await readJSON(response)) };
  } catch { return { status: 503, body: { error: 'Unable to start the mini baseline. Check local service status.' } }; }
}
