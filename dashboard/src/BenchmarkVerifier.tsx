import type { Benchmark } from '../shared/benchmarks';
import { isNemotronBenchmark, nemotronBaselines } from '../shared/nemotron-benchmark';
import type { MiniQualityBaseline, NemotronVerifierStatus } from '../shared/nemotron-status';
import './benchmark-verifier.css';

export type BenchmarkVerifierProps = {
  benchmark: Benchmark;
  verifier?: NemotronVerifierStatus;
  onMeasure?: () => void;
  measuring?: boolean;
};
type NodeBaseline = { node_id: string; status: string; metrics: MiniQualityBaseline | null };
type PoolVerifierStatus = NemotronVerifierStatus & {
  configured_nodes?: number; online_nodes?: number; node_baselines?: NodeBaseline[];
};

const number = (value: number) => new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 }).format(value);
const percent = (value: number, correct?: number, count?: number) => `${number(value)}%${correct !== undefined && count !== undefined ? ` · ${correct}/${count}` : ''}`;
const statusLabels: Record<string, string> = { completed: 'Measured', unmeasured: 'Pending', missing: 'Pending',
  queued: 'Queued', running: 'Measuring', failed: 'Failed', unavailable: 'Unavailable',
  cancelled: 'Cancelled', rejected: 'Rejected' };
const statusLabel = (status: string) => statusLabels[status] || status.replaceAll('_', ' ');
const recipes = {
  mmlu: '0-shot · greedy answer letter · up to 8 generated tokens · thinking disabled',
  gsm8k: '0-shot chain of thought · greedy · up to 512 generated tokens · final #### numeric answer',
  ppl: 'First 4,096 test tokens · 2,048-token windows · stride 1,024 · overlapping targets masked',
};

function BaselineRows({ verifier }: { verifier?: NemotronVerifierStatus }) {
  const rows = (verifier as PoolVerifierStatus | undefined)?.node_baselines || [];
  const fallback = verifier?.baseline;
  const baselines: NodeBaseline[] = rows.length ? rows : fallback
    ? [{ node_id: 'Measured GPU', status: 'completed', metrics: fallback }] : [];
  return <div className="verifier-baseline-table-wrap">
    {baselines.length ? <table className="verifier-baseline-table" aria-label="Reference quality by GPU node"><thead><tr>
      <th scope="col">GPU node</th><th scope="col">Reference</th><th scope="col">MMLU</th><th scope="col">GSM8K</th><th scope="col">PPL</th>
    </tr></thead><tbody>{baselines.map((node) => <tr key={node.node_id}>
      <th scope="row">{node.node_id}{node.metrics?.identity && <details className="verifier-manifest"><summary>Manifest</summary><code>{node.metrics.identity}</code></details>}</th>
      <td><span className={`verifier-state verifier-state-${node.status === 'completed' ? 'ready' : node.status === 'running' || node.status === 'queued' ? 'busy' : 'pending'}`}>{statusLabel(node.status)}</span></td>
      <td>{node.metrics ? percent(node.metrics.mmlu, node.metrics.mmlu_correct, node.metrics.mmlu_count) : '—'}</td>
      <td>{node.metrics ? percent(node.metrics.gsm8k, node.metrics.gsm8k_correct, node.metrics.gsm8k_count) : '—'}</td>
      <td>{node.metrics ? number(node.metrics.ppl) : '—'}</td>
    </tr>)}</tbody></table> : <p className="verifier-empty">Reference scores are pending. Connect a GPU node and measure its unmodified reference model to establish the baseline.</p>}
  </div>;
}

function NemotronInstructions({ verifier, onMeasure, measuring }: Omit<BenchmarkVerifierProps, 'benchmark'>) {
  const busy = measuring || ['queued', 'running'].includes(verifier?.baseline_status || '');
  const resourceStatus = verifier as PoolVerifierStatus | undefined;
  return <>
    <section className="verifier-section" aria-labelledby="verifier-setup-heading">
      <div className="verifier-section-heading"><h3 id="verifier-setup-heading">Set up the independent verifier</h3><span className={`verifier-state verifier-state-${verifier?.baseline_ready ? 'ready' : verifier?.reachable ? 'busy' : 'pending'}`}>{verifier?.baseline_ready ? 'Baselines ready' : verifier?.ready ? 'Baseline required' : 'Setup required'}</span></div>
      <ol className="verifier-setup-steps">
        <li><strong>Run the local services.</strong> Keep the verifier, its job coordinator and all agent harnesses on this machine.</li>
        <li><strong>Connect the A5000 resource nodes.</strong> Register approved reference and candidate deployments with their exact source repositories and commits. The GPU VMs run inference and jobs.</li>
        <li><strong>Prepare the frozen mini suite.</strong> Measure an immutable reference on each physical GPU before evaluating candidates on that GPU.</li>
      </ol>
      <details className="verifier-details"><summary>Local startup command</summary><pre>python3 dev/local_stack.py up</pre><p>Resource nodes and their inference deployments must be configured before measurement can start.</p></details>
    </section>

    <section className="verifier-section" aria-labelledby="verifier-baseline-heading">
      <div className="verifier-section-heading"><h3 id="verifier-baseline-heading">Reference baselines by GPU</h3>{resourceStatus?.configured_nodes !== undefined && <span className="verifier-note">{resourceStatus.online_nodes ?? 0} of {resourceStatus.configured_nodes} nodes connected</span>}</div>
      <p>Candidates use the reference from the same physical GPU and evaluation manifest. These resources are shared with Experiments, with one active VM per GPU node.</p>
      <BaselineRows verifier={verifier} />
      <div className="verifier-measure-controls"><p role="status">{verifier?.message || 'Connect the local verifier to check setup and measure reference scores.'}</p>
        {onMeasure && <button className="button" type="button" disabled={!verifier?.ready || verifier.baseline_ready || busy} onClick={onMeasure}>{verifier?.baseline_ready ? 'Baselines measured' : busy ? 'Measuring baselines…' : 'Measure mini baselines'}</button>}
      </div>
    </section>

    <section className="verifier-section" aria-labelledby="verifier-quality-heading">
      <h3 id="verifier-quality-heading">1. Pass the fast quality checks</h3>
      <div className="verifier-quality-grid">{nemotronBaselines.map((metric) => <article key={metric.key}><h4>{metric.name}</h4><p>{metric.setting}</p><p>{recipes[metric.key]}</p><strong>{metric.gate}</strong></article>)}</div>
      <p className="verifier-note">Frozen samples use seed 42. MMLU uses greedy answer-letter extraction; GSM8K uses final numeric exact match. PPL uses teacher-forced prompt-token log probabilities. Invalid or truncated answers count as incorrect.</p>
      <p className="verifier-note">A failed gate rejects the candidate before throughput measurement and receives no leaderboard score. Mini results screen for regressions; their accuracy percentages describe these small samples.</p>
    </section>

    <section className="verifier-section" aria-labelledby="verifier-throughput-heading">
      <h3 id="verifier-throughput-heading">2. Measure C20 throughput</h3>
      <div className="verifier-workload"><div><strong>20</strong><span>concurrent requests</span></div><div><strong>2,048</strong><span>input tokens each</span></div><div><strong>1,024</strong><span>output tokens each</span></div></div>
      <p>One warmup request, then 20 timed requests. Output tokens divided by end-to-end wall time determines the score, including prefill, queueing and API overhead. EOS stopping and prefix/output caching are disabled.</p>
      <p>The engine can queue requests and reduce active sequences to fit 24 GB VRAM while preserving the offered workload. RTX A5000 uses W4A16 compute with the released NVFP4 weights.</p>
      <p className="verifier-target"><strong>10–20 minute execution target on a prepared VM.</strong> Actual duration is reported. Queue wait, VM boot and model loading are reported separately; longer valid runs remain eligible.</p>
    </section>
  </>;
}

export default function BenchmarkVerifier({ benchmark, verifier, onMeasure, measuring }: BenchmarkVerifierProps) {
  const nemotron = isNemotronBenchmark(benchmark);
  const contract = benchmark.verifier;
  return <div className="benchmark-verifier-view">
    <header className="verifier-view-header"><h2>Verifier instructions</h2><p>{nemotron ? 'The local verifier measures quality and throughput independently on the shared A5000 resource pool.' : 'The configured verifier evaluates each validated submission and returns the official result.'}</p></header>
    {nemotron && <NemotronInstructions verifier={verifier} onMeasure={onMeasure} measuring={measuring} />}
    <section className="verifier-section" aria-labelledby="verifier-delivery-heading">
      <h3 id="verifier-delivery-heading">Evaluation and scoring contract</h3>
      <dl className="verifier-contract-fields">
        <div><dt>Evaluation endpoint</dt><dd><code>{contract.method} {contract.url}</code></dd></div>
        {contract.mode === 'asynchronous' && <div><dt>Job status endpoint</dt><dd><code>GET {contract.poll_url}</code></dd></div>}
        <div><dt>Result delivery</dt><dd>{contract.mode === 'asynchronous' ? 'Queued evaluation; the dashboard polls the returned job reference.' : 'The evaluation response contains the final result.'}</dd></div>
        <div><dt>Leaderboard metric</dt><dd>{benchmark.score_name} · {benchmark.score_direction === 'higher' ? 'Higher' : 'Lower'} is better{benchmark.score_min !== null || benchmark.score_max !== null ? ` · accepted range ${benchmark.score_min === null ? 'unbounded' : number(benchmark.score_min)} to ${benchmark.score_max === null ? 'unbounded' : number(benchmark.score_max)}` : ''}</dd></div>
        <div><dt>Verified result</dt><dd>A finite score is recorded only after the verifier reports success. Pending, failed and rejected evaluations receive no ranked score.</dd></div>
      </dl>
      <div className="verifier-schema-links"><a href={`/api/benchmarks/${encodeURIComponent(benchmark.id)}`} target="_blank" rel="noreferrer">Benchmark definition ↗</a><a href="/api/schemas/submission" target="_blank" rel="noreferrer">Submission envelope schema ↗</a><a href="/api/schemas/benchmark" target="_blank" rel="noreferrer">Benchmark schema ↗</a></div>
      <details className="verifier-details"><summary>Full verifier configuration and response mapping</summary><pre>{JSON.stringify(contract, null, 2)}</pre></details>
      <details className="verifier-details"><summary>Accepted submission payload schema</summary><pre>{JSON.stringify(benchmark.submission_schema, null, 2)}</pre></details>
      {nemotron && <details className="verifier-details"><summary>Full immutable evaluation recipe</summary><p className="verifier-full-recipe">{benchmark.prompt}</p></details>}
    </section>
  </div>;
}
