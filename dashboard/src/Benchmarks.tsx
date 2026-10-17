import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react';
import type { Benchmark, BenchmarkInput, LeaderboardEntry, Submission } from '../shared/benchmarks';
import { isNemotronBenchmark, nemotronBaselines, nemotronBenchmark, nemotronMemoryReferenceUrl, nemotronReferenceUrl } from '../shared/nemotron-benchmark';
import type { NemotronVerifierStatus } from '../shared/nemotron-status';
import BenchmarkVerifier from './BenchmarkVerifier';
import BenchmarkAgents from './BenchmarkAgents';
import './benchmarks.css';

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, options);
  const body = await response.json() as { error?: string; details?: { path?: (string | number)[]; message: string }[] };
  if (!response.ok) {
    const detail = Array.isArray(body.details) ? body.details.map((item) =>
      `${item.path?.join('.') ? `${item.path.join('.')}: ` : ''}${item.message}`).join('; ') : '';
    throw new Error(detail || body.error || 'Unable to complete this request');
  }
  return body as T;
}

const requestTemplate = {
  submission: '{{submission.payload}}', submission_id: '{{submission.id}}', benchmark_id: '{{benchmark.id}}',
  base_checkpoint_url: '{{submission.base_checkpoint_url}}',
};
const repositorySchema = {
  type: 'object',
  properties: {
    repository: { type: 'string', format: 'uri' },
    commit: { type: 'string', pattern: '^(?:[a-fA-F0-9]{40}|[a-fA-F0-9]{64})$' },
  },
  required: ['repository', 'commit'],
  additionalProperties: false,
};
const text = (form: FormData, key: string) => String(form.get(key) ?? '').trim();
const optionalNumber = (form: FormData, key: string) => text(form, key) === '' ? null : Number(form.get(key));
const statuses = (value: string) => value.split(',').map((item) => item.trim()).filter(Boolean);
const score = (value: number | null) => value === null ? '—' : new Intl.NumberFormat(undefined, { maximumSignificantDigits: 8 }).format(value);
const date = (value: number) => new Date(value).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
const statusLabels = { queued: 'Queued', running: 'Running', verified: 'Verified', failed: 'Failed', rejected: 'Rejected' };

function parseJSON(value: string, label: string) {
  try { return JSON.parse(value); }
  catch { throw new Error(`${label} must be valid JSON.`); }
}

type BaselineControls = { verifier?: NemotronVerifierStatus; measuring?: boolean; onMeasure?: () => void };

function NemotronBaselines({ verifier, measuring, onMeasure }: BaselineControls) {
  const busy = measuring || ['queued', 'running'].includes(verifier?.baseline_status ?? '');
  return <section className="nemotron-baselines" aria-label="Nemotron baseline scores">
    <div className="nemotron-baseline-heading"><h3>Mini quality baseline</h3><span>{verifier?.baseline ? 'Measured on your RTX A5000' : 'RTX A5000 measurements pending'}</span></div>
    <p className="benchmark-form-help">{verifier?.configured_nodes ? `${verifier.online_nodes ?? 0} of ${verifier.configured_nodes} GPU nodes connected · Shared with Experiments. ` : ''}10–20 minute execution target. Slow valid runs remain eligible.</p>
    <div className="nemotron-metrics">{nemotronBaselines.map((metric) => <div className="nemotron-metric" key={metric.name}>
      <h4>{metric.name}</h4><strong>{verifier?.baseline ? `${score(verifier.baseline[metric.key])}${metric.key === 'ppl' ? '' : '%'}` : 'Pending'}</strong><p>{metric.setting}</p>
      <span>{metric.gate}</span>
    </div>)}</div>
    {onMeasure && <div className="nemotron-baseline-controls"><p role="status">{verifier?.message ?? 'Connect the local mini verifier to measure the reference model.'}</p>
      <button type="button" className="button" disabled={!verifier?.ready || verifier.baseline_ready || busy} onClick={onMeasure}>{verifier?.baseline_ready ? 'Baseline measured' : busy ? 'Measuring baseline…' : 'Measure mini baseline'}</button></div>}
    <p className="nemotron-reference"><a href={nemotronReferenceUrl} target="_blank" rel="noreferrer">NVIDIA’s BF16 base-model references ↗</a>: MMLU 78.63% · GSM8K 91.74%; PPL not reported. These use a different checkpoint and recipe. The verifier measures the fixed NVFP4 mini suite on your RTX A5000 before ranking solutions. RTX A5000 uses W4A16 compute for NVFP4 weights.</p>
  </section>;
}

function NemotronWorkload() {
  return <><div className="nemotron-tags"><span>1 × RTX A5000 · 24 GB</span><span>NVFP4 weights · W4A16 compute</span><span>C20 · 2K input · 1K output</span></div>
    <p className="nemotron-memory-note">Keep 20 clients in flight; limit active engine sequences if needed. Fit on 24 GB still needs measurement: <a href={nemotronMemoryReferenceUrl} target="_blank" rel="noreferrer">NVIDIA’s stock W4A16 profile starts at 32 GB ↗</a>.</p>
  </>;
}

function NemotronChallenge({ onSetup, ...controls }: { onSetup: () => void } & BaselineControls) {
  return <article className="nemotron-challenge" aria-labelledby="nemotron-challenge-title">
    <div className="nemotron-challenge-heading"><span className="nemotron-eyebrow">Benchmark 01</span><span className="nemotron-setup-status">{controls.verifier?.baseline_ready ? 'Quality baseline ready' : 'Local setup required'}</span></div>
    <h2 id="nemotron-challenge-title">Optimize Nemotron Lightning</h2>
    <p className="nemotron-challenge-description">Make Nemotron 3.5 Lightning 30B-A3B faster on a single RTX A5000 24 GB while preserving model quality.</p>
    <NemotronWorkload />
    <NemotronBaselines {...controls} />
    <div className="nemotron-challenge-footer"><p>The included local verifier checks mini MMLU, PPL, and GSM8K before the C20 throughput run. Connect a job VM on your RTX A5000 server to measure the reference and evaluate solutions.</p>
      <button className="button button-primary" onClick={onSetup}>Set up benchmark</button></div>
  </article>;
}

function CreateBenchmark({ preset, onClose, onCreated, ...controls }: { preset?: Omit<BenchmarkInput, 'verifier'>; onClose: () => void; onCreated: (benchmark: Benchmark) => void } & BaselineControls) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [mode, setMode] = useState<'synchronous' | 'asynchronous'>(preset ? 'asynchronous' : 'synchronous');
  const [pollUrl, setPollUrl] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => { if (!dialog.current?.open) dialog.current?.showModal(); }, []);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError('');
    const form = new FormData(event.currentTarget);
    setSaving(true);
    try {
      const schema = parseJSON(text(form, 'submission_schema'), 'Submission schema');
      if (!schema || typeof schema !== 'object' || Array.isArray(schema)) throw new Error('Submission schema must be a JSON Schema object.');
      const definition: BenchmarkInput = {
        title: text(form, 'title'), prompt: text(form, 'prompt'),
        submission_instructions: text(form, 'submission_instructions'), submission_schema: schema,
        score_name: text(form, 'score_name'), score_direction: text(form, 'score_direction') as 'higher' | 'lower',
        score_min: optionalNumber(form, 'score_min'), score_max: optionalNumber(form, 'score_max'),
        verifier: {
          url: text(form, 'url'), method: text(form, 'method') as 'POST' | 'PUT', mode,
          request_template: parseJSON(text(form, 'request_template'), 'Request body template'),
          score_path: text(form, 'score_path'), feedback_path: text(form, 'feedback_path'),
          job_id_path: text(form, 'job_id_path') || 'id', status_path: text(form, 'status_path') || 'status',
          success_statuses: statuses(text(form, 'success_statuses') || 'completed,succeeded,verified'),
          failure_statuses: statuses(text(form, 'failure_statuses') || 'failed,cancelled,rejected,error'),
          poll_url: mode === 'asynchronous' ? text(form, 'poll_url') : '',
          poll_interval_seconds: Number(form.get('poll_interval_seconds') || 5),
          timeout_seconds: Number(form.get('timeout_seconds') || 3600), auth_env: text(form, 'auth_env'),
        },
      };
      onCreated(await api<Benchmark>('/benchmarks', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(definition),
      }));
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Unable to create benchmark');
    } finally { setSaving(false); }
  }

  return <dialog ref={dialog} className="node-dialog benchmark-dialog" onCancel={(event) => {
    if (saving) event.preventDefault(); else onClose();
  }} aria-labelledby="create-benchmark-title">
    <form onSubmit={submit}>
      <div className="section-heading"><h2 id="create-benchmark-title">Create benchmark</h2>
        <button className="icon-button" type="button" onClick={onClose} disabled={saving} aria-label="Close">×</button></div>
      <p className="benchmark-dialog-intro">Define the challenge, what agents submit, and how their solutions are scored.</p>
      {preset && <NemotronBaselines {...controls} />}
      <div className="benchmark-form">
        <label>Benchmark name<input name="title" required maxLength={200} defaultValue={preset?.title} placeholder="e.g. Inference throughput challenge" autoFocus /></label>
        <label>Open-ended objective<textarea name="prompt" required maxLength={30000} rows={4} defaultValue={preset?.prompt} placeholder="Describe the outcome, available inputs, and constraints. Leave the approach to the agent." /></label>
        <label>Submission instructions<textarea name="submission_instructions" required maxLength={10000} rows={3} defaultValue={preset?.submission_instructions} placeholder="Explain each required field, where artifacts should live, and anything the verifier needs to evaluate a solution." /></label>
        <label>Submission format · JSON Schema
          <textarea name="submission_schema" className="code-input" required rows={10} defaultValue={JSON.stringify(preset?.submission_schema ?? repositorySchema, null, 2)} spellCheck={false} />
          <span className="benchmark-field-note">Define every required field and its constraints. Submissions are validated against this schema before verification.</span>
        </label>
        <div className="benchmark-form-divider" />
        <h3>Verification</h3>
        <label>Verification endpoint<input name="url" type="url" required maxLength={2000} placeholder="https://your-verifier.com/evaluate" onChange={(event) => {
          if (preset) setPollUrl(event.target.value.trim().replace(/\/evaluate\/?$/, '') + '/evaluations/{{job_id}}');
        }} /></label>
        {preset && <p className="benchmark-form-help">Use your RTX A5000 evaluator’s reachable endpoint. It must measure the unmodified NVFP4 reference, rerun the mini MMLU, PPL, and GSM8K suite for each candidate, and reject failed quality checks before returning a score.</p>}
        <div className="benchmark-form-row">
          <label>Evaluation mode<select value={mode} onChange={(event) => setMode(event.target.value as typeof mode)}>
            <option value="synchronous">Returns a score immediately</option><option value="asynchronous">Queues an evaluation job</option>
          </select></label>
          <label>Score direction<select name="score_direction" defaultValue={preset?.score_direction ?? 'higher'}><option value="higher">Higher is better</option><option value="lower">Lower is better</option></select></label>
        </div>
        <label>Score label<input name="score_name" required maxLength={100} defaultValue={preset?.score_name ?? 'Score'} placeholder="e.g. Throughput (tokens/s)" /></label>
        {mode === 'asynchronous' && <label>Job status endpoint<input name="poll_url" required maxLength={2000} value={pollUrl} onChange={(event) => setPollUrl(event.target.value)} placeholder="https://your-verifier.com/jobs/{{job_id}}" />
          <span className="benchmark-field-note">Use <code>{'{{job_id}}'}</code> for the job reference returned by your verification API.</span></label>}
        <details className="benchmark-advanced">
          <summary>API format and scoring settings</summary>
          <div className="benchmark-form">
            <div className="benchmark-form-row">
              <label>Request method<select name="method" defaultValue="POST"><option>POST</option><option>PUT</option></select></label>
              <label>Evaluation timeout (seconds)<input name="timeout_seconds" type="number" required min={1} max={86400} defaultValue={preset ? 21600 : 3600} /></label>
            </div>
            <label>Request body template<textarea name="request_template" className="code-input" required rows={6} defaultValue={JSON.stringify(requestTemplate, null, 2)} spellCheck={false} />
              <span className="benchmark-field-note">Match your API's JSON format. <code>{'{{submission.payload}}'}</code> inserts the complete submitted solution. Nested fields such as <code>{'{{submission.payload.repository}}'}</code> are supported.</span>
            </label>
            <div className="benchmark-form-row">
              <label>Score response field<input name="score_path" maxLength={200} defaultValue="score" placeholder="e.g. result.metrics.score" /></label>
              <label>Feedback response field<input name="feedback_path" maxLength={200} defaultValue="feedback" /></label>
            </div>
            <p className="benchmark-form-help">Use dotted paths for nested response fields, or <code>$</code> for the full response. The score must be a finite number.</p>
            <div className="benchmark-form-row">
              <label>Minimum score · optional<input name="score_min" type="number" step="any" defaultValue={preset?.score_min ?? ''} placeholder="No minimum" /></label>
              <label>Maximum score · optional<input name="score_max" type="number" step="any" defaultValue={preset?.score_max ?? ''} placeholder="No maximum" /></label>
            </div>
            {mode === 'asynchronous' && <>
              <div className="benchmark-form-row">
                <label>Job ID response field<input name="job_id_path" defaultValue="id" maxLength={200} /></label>
                <label>Status response field<input name="status_path" defaultValue="status" maxLength={200} /></label>
              </div>
              <label>Completed statuses<input name="success_statuses" defaultValue="completed, succeeded, verified" required /><span className="benchmark-field-note">Separate your API's status values with commas.</span></label>
              <label>Failure statuses<input name="failure_statuses" defaultValue="failed, cancelled, rejected, error" required /></label>
              <label>Check status every (seconds)<input name="poll_interval_seconds" type="number" min={1} max={3600} defaultValue={5} required /></label>
            </>}
            <label>Authentication environment variable · optional<input name="auth_env" pattern="BENCHMARK_VERIFIER_[A-Z0-9_]+" defaultValue={preset ? 'BENCHMARK_VERIFIER_NEMOTRON_KEY' : ''} placeholder="BENCHMARK_VERIFIER_SERVICE_KEY" />
              <span className="benchmark-field-note">Name of a server environment variable containing the verifier's bearer token. Credentials stay on the server.</span></label>
          </div>
        </details>
      </div>
      {error && <p className="error-text" role="alert">{error}</p>}
      <div className="dialog-actions"><button className="button" type="button" onClick={onClose} disabled={saving}>Cancel</button>
        <button className="button button-primary" disabled={saving}>{saving ? 'Creating…' : 'Create benchmark'}</button></div>
    </form>
  </dialog>;
}

function SubmitSolution({ benchmark, onClose, onSubmitted }: {
  benchmark: Benchmark; onClose: () => void; onSubmitted: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => { if (!dialog.current?.open) dialog.current?.showModal(); }, []);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    setSaving(true);
    setError('');
    try {
      const payload = parseJSON(text(form, 'payload'), 'Solution');
      await api<Submission>(`/benchmarks/${benchmark.id}/submissions`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: text(form, 'title'), submitted_by: text(form, 'submitted_by'),
          base_checkpoint_url: text(form, 'base_checkpoint_url'), payload }),
      });
      onSubmitted();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Unable to submit solution');
    } finally { setSaving(false); }
  }

  return <dialog ref={dialog} className="node-dialog benchmark-dialog" onCancel={(event) => {
    if (saving) event.preventDefault(); else onClose();
  }} aria-labelledby="submit-solution-title">
    <form onSubmit={submit}>
      <div className="section-heading"><h2 id="submit-solution-title">Submit solution</h2>
        <button className="icon-button" type="button" onClick={onClose} disabled={saving} aria-label="Close">×</button></div>
      <p className="benchmark-dialog-intro">{benchmark.title}</p>
      <p className="benchmark-submit-instructions">{benchmark.submission_instructions}</p>
      <div className="benchmark-form">
        <label>Solution name<input name="title" required maxLength={200} placeholder="Name this submission" autoFocus /></label>
        <label>Submitted by<input name="submitted_by" required maxLength={200} placeholder="Agent, team, or your name" /></label>
        <label>Base checkpoint · public GitHub<input name="base_checkpoint_url" type="url" required maxLength={2000} placeholder="https://github.com/owner/repo/commit/FULL_COMMIT_SHA" />
          <span className="benchmark-field-note">Link to a public GitHub commit or tree pinned to a full 40- or 64-character commit hash. Every submission requires this base checkpoint in addition to the benchmark's payload.</span></label>
        <details className="benchmark-advanced"><summary>Required submission format</summary><pre className="benchmark-schema">{JSON.stringify(benchmark.submission_schema, null, 2)}</pre></details>
        <label>Solution · JSON<textarea name="payload" className="code-input" required rows={10} spellCheck={false} placeholder={'{\n  "repository": "https://github.com/your-team/solution",\n  "commit": "full commit hash"\n}'} />
          <span className="benchmark-field-note">Provide the fields required by this benchmark's submission format. The verifier evaluates this exact payload.</span></label>
      </div>
      {error && <p className="error-text" role="alert">{error}</p>}
      <div className="dialog-actions"><button className="button" type="button" disabled={saving} onClick={onClose}>Cancel</button>
        <button className="button button-primary" disabled={saving}>{saving ? 'Submitting…' : 'Submit for verification'}</button></div>
    </form>
  </dialog>;
}

type Detail = { benchmark: Benchmark; submissions: Submission[]; total: number; entries: LeaderboardEntry[] };
type BenchmarkView = 'overview' | 'verifier' | 'agents' | 'submissions' | 'leaderboard';
const benchmarkViews: BenchmarkView[] = ['overview', 'verifier', 'agents', 'submissions', 'leaderboard'];

export default function Benchmarks({ active }: { active: boolean }) {
  const [benchmarks, setBenchmarks] = useState<Benchmark[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<Detail | null>(null);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [creating, setCreating] = useState<'custom' | 'nemotron' | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [verifier, setVerifier] = useState<NemotronVerifierStatus>();
  const [measuring, setMeasuring] = useState(false);
  const [view, setView] = useState<BenchmarkView>('overview');

  const load = useCallback(async (signal?: AbortSignal) => {
    try {
      const [list, details, verifierState] = await Promise.all([
        api<{ benchmarks: Benchmark[] }>('/benchmarks', { signal }),
        selectedId ? Promise.all([
          api<Benchmark>(`/benchmarks/${selectedId}`, { signal }),
          api<{ submissions: Submission[]; total: number }>(`/benchmarks/${selectedId}/submissions?limit=100&offset=${offset}`, { signal }),
          api<{ entries: LeaderboardEntry[] }>(`/benchmarks/${selectedId}/leaderboard`, { signal }),
        ]) : null,
        api<NemotronVerifierStatus>('/verifiers/nemotron/status', { signal }).catch(() => undefined),
      ]);
      if (signal?.aborted) return;
      setBenchmarks(list.benchmarks);
      setVerifier(verifierState);
      if (details) setDetail({ benchmark: details[0], submissions: details[1].submissions, total: details[1].total, entries: details[2].entries });
      setError('');
    } catch (failure) {
      if (!signal?.aborted) setError(failure instanceof Error ? failure.message : 'Unable to load benchmarks');
    } finally {
      if (!signal?.aborted) { setLoading(false); setRefreshing(false); }
    }
  }, [selectedId, offset]);

  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    void load(controller.signal);
    const interval = window.setInterval(() => { void load(controller.signal); }, 5000);
    return () => { controller.abort(); window.clearInterval(interval); };
  }, [active, load]);

  function openBenchmark(id: string) {
    setDetail(null); setSelectedId(id); setOffset(0); setView('overview'); setMessage(''); setError('');
  }

  async function measureBaseline() {
    setMeasuring(true); setError('');
    try {
      const job = await api<{ id: string; status: string }>('/verifiers/nemotron/baseline', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}',
      });
      setMessage(job.status === 'completed' ? 'The mini baseline is already measured.' : 'Reference mini quality check queued. Scores will update automatically.');
      await load();
    } catch (failure) { setError(failure instanceof Error ? failure.message : 'Unable to start the mini baseline'); }
    finally { setMeasuring(false); }
  }

  const benchmark = detail?.benchmark;
  return <div className="benchmarks">
    {selectedId && <nav className="benchmark-breadcrumb" aria-label="Benchmark navigation"><button onClick={() => {
      setSelectedId(null); setDetail(null); setOffset(0); setMessage(''); setError('');
    }}>Benchmarks</button><span aria-hidden="true">/</span><span>{benchmark?.title ?? 'Loading…'}</span></nav>}
    <div className="benchmarks-header">
      <div><h1>{selectedId ? benchmark?.title ?? 'Benchmark' : 'Benchmarks'}{benchmark && <span className="benchmark-version">Version {benchmark.version}</span>}</h1>
        <p>{selectedId ? 'Open-ended solutions. Independently verified scores.' : 'Create open-ended challenges and rank solutions by verified results.'}</p></div>
      <div className="benchmark-actions">
        <button className="button" onClick={() => { setRefreshing(true); void load(); }} disabled={loading || refreshing}>{refreshing ? 'Refreshing…' : 'Refresh'}</button>
        {selectedId ? <><button className="button" disabled={!benchmark} onClick={() => setView('agents')}>Launch agents</button><button className="button button-primary" disabled={!benchmark || (isNemotronBenchmark(benchmark) && !verifier?.baseline_ready)} onClick={() => setSubmitting(true)}>Submit solution</button></>
          : <button className="button button-primary" onClick={() => setCreating('custom')}>Create benchmark</button>}
      </div>
    </div>
    {error && <p className="error-banner" role="alert">{error}</p>}
    {message && <p className="benchmark-success" role="status">{message}</p>}
    {loading ? <p className="benchmark-loading" role="status">Loading benchmarks…</p>
      : selectedId ? !detail ? <p className="benchmark-loading" role="status">{error ? 'Benchmark could not be loaded.' : 'Loading benchmark…'}</p> : <>
        <nav className="benchmark-views" aria-label="Benchmark sections" role="tablist">{([
          ['overview', 'Overview'], ['verifier', 'Verifier instructions'], ['agents', 'Agents'],
          ['submissions', `Submissions (${detail.total})`], ['leaderboard', `Leaderboard (${detail.entries.length})`],
        ] as [BenchmarkView, string][]).map(([id, label]) => <button key={id} id={`benchmark-tab-${id}`} role="tab" aria-selected={view === id}
          tabIndex={view === id ? 0 : -1} aria-controls={`benchmark-panel-${id}`} onClick={() => setView(id)} onKeyDown={(event) => {
            const index = benchmarkViews.indexOf(id);
            const next = event.key === 'ArrowRight' ? (index + 1) % benchmarkViews.length : event.key === 'ArrowLeft' ? (index + benchmarkViews.length - 1) % benchmarkViews.length : event.key === 'Home' ? 0 : event.key === 'End' ? benchmarkViews.length - 1 : -1;
            if (next < 0) return;
            event.preventDefault(); setView(benchmarkViews[next]);
            document.getElementById(`benchmark-tab-${benchmarkViews[next]}`)?.focus();
          }}>{label}</button>)}</nav>
        <section role="tabpanel" id="benchmark-panel-overview" aria-labelledby="benchmark-tab-overview" hidden={view !== 'overview'}>
        {isNemotronBenchmark(detail.benchmark) && <NemotronWorkload />}
        <div className="benchmark-description">
          <div><h2>Objective</h2><p className="benchmark-prompt">{isNemotronBenchmark(detail.benchmark) ? detail.benchmark.prompt.split('\n')[0] : detail.benchmark.prompt}</p>
            <p className="benchmark-overview-flow">{isNemotronBenchmark(detail.benchmark) ? 'Prepare the reference in Verifier instructions, launch agents to explore improvements, then submit a pinned implementation.' : 'Read the verifier and submission instructions, launch agents to explore improvements, then submit a pinned implementation.'} Verified results appear on the leaderboard.</p>
            <div className="benchmark-actions"><button className="button" onClick={() => setView('verifier')}>Set up verification</button><button className="button button-primary" onClick={() => setView('agents')}>Launch benchmark agents</button></div>
          </div>
          <aside className="benchmark-verifier" aria-label="Verification details"><dl>
            <div><dt>Scoring</dt><dd>{detail.benchmark.score_name} · {detail.benchmark.score_direction === 'higher' ? 'Higher' : 'Lower'} is better</dd></div>
            <div><dt>Evaluation</dt><dd>{detail.benchmark.verifier.mode === 'asynchronous' ? 'Queued evaluation' : 'Immediate evaluation'}</dd></div>
            <div><dt>Submissions</dt><dd>{detail.benchmark.submission_count} total · {detail.benchmark.verified_count} verified</dd></div>
            {isNemotronBenchmark(detail.benchmark) && <><div><dt>Shared resources</dt><dd>{verifier?.configured_nodes ? `${verifier.online_nodes ?? 0} of ${verifier.configured_nodes} GPU nodes connected` : 'GPU connection status unavailable'} · Experiments and verification</dd></div><div><dt>Fast check target</dt><dd>10–20 minutes · slow valid runs remain eligible</dd></div></>}
          </dl></aside>
        </div>
        {isNemotronBenchmark(detail.benchmark) && <div className="benchmark-readiness-callout"><strong>{verifier?.baseline_ready ? 'Reference baseline ready' : 'Reference baseline pending'}</strong><p>{verifier?.message ?? 'Connect your local verifier and resource nodes.'}</p><button className="button" onClick={() => setView('verifier')}>View verifier instructions</button></div>}
        </section>
        <section role="tabpanel" id="benchmark-panel-verifier" aria-labelledby="benchmark-tab-verifier" hidden={view !== 'verifier'}>
          <BenchmarkVerifier benchmark={detail.benchmark} verifier={verifier} measuring={measuring} onMeasure={() => { void measureBaseline(); }} />
        </section>
        <section role="tabpanel" id="benchmark-panel-agents" aria-labelledby="benchmark-tab-agents" hidden={view !== 'agents'}>
          <BenchmarkAgents benchmark={detail.benchmark} active={active && view === 'agents'} />
        </section>
        <section className="benchmark-section" role="tabpanel" id="benchmark-panel-leaderboard" aria-labelledby="benchmark-tab-leaderboard" hidden={view !== 'leaderboard'}>
          <div className="section-heading"><div><h2 id="leaderboard-title">Leaderboard</h2><p className="muted">Verified results, ranked by {detail.benchmark.score_name.toLowerCase()}.</p></div></div>
          <div className="nodes-table-container"><table className="nodes-table benchmark-data-table" aria-label="Benchmark leaderboard"><thead><tr>
            <th scope="col">Rank</th><th scope="col">Solution</th><th scope="col">Submitted by</th><th scope="col">{detail.benchmark.score_name}</th><th scope="col">Verified</th>
          </tr></thead><tbody>{detail.entries.length === 0 ? <tr><td className="nodes-empty" colSpan={5}>No verified scores yet. Completed submissions that pass verification will appear here.</td></tr>
            : detail.entries.map((entry) => <tr key={entry.id}><td className="benchmark-rank">{entry.rank}</td><th scope="row">{entry.title}
              {entry.base_checkpoint_url && <a className="cell-secondary" href={entry.base_checkpoint_url} target="_blank" rel="noreferrer">Base checkpoint ↗</a>}
            </th><td>{entry.submitted_by}</td><td className="benchmark-score">{score(entry.score)}</td><td className="muted">{entry.completed_at ? date(entry.completed_at) : '—'}</td></tr>)}</tbody></table></div>
        </section>
        <section className="benchmark-section" role="tabpanel" id="benchmark-panel-submissions" aria-labelledby="benchmark-tab-submissions" hidden={view !== 'submissions'}>
          <div className="benchmark-instructions"><h2>Submission instructions</h2><p>{detail.benchmark.submission_instructions}</p>
            <details className="benchmark-submission-payload"><summary>View full submission format</summary><pre className="benchmark-schema">{JSON.stringify(detail.benchmark.submission_schema, null, 2)}</pre></details>
          </div>
          {isNemotronBenchmark(detail.benchmark) && !verifier?.baseline_ready && <p className="benchmark-form-help">Measure the reference baseline in Verifier instructions before submitting a candidate.</p>}
          <div className="section-heading"><div><h2 id="submissions-title">Submissions <span className="node-count">{detail.total}</span></h2><p className="muted">Verification progress and feedback for each solution.</p></div></div>
          <div className="nodes-table-container"><table className="nodes-table benchmark-data-table" aria-label="Benchmark submissions"><thead><tr>
            <th scope="col">Solution</th><th scope="col">Submitted by</th><th scope="col">Status</th><th scope="col">{detail.benchmark.score_name}</th><th scope="col">Submitted</th>
          </tr></thead><tbody>{detail.submissions.length === 0 ? <tr><td className="nodes-empty" colSpan={5}>No submissions yet.</td></tr> : detail.submissions.map((submission) => <tr key={submission.id}>
            <th scope="row">{submission.title}
              {submission.base_checkpoint_url && <a className="cell-secondary" href={submission.base_checkpoint_url} target="_blank" rel="noreferrer">Base checkpoint ↗</a>}
              {submission.feedback && <span className="benchmark-submission-note">{submission.feedback}</span>}
              {submission.error && <span className="benchmark-submission-note error-text">{submission.error}</span>}
              <details className="benchmark-submission-payload"><summary>View submitted solution</summary><pre>{JSON.stringify(submission.payload, null, 2)}</pre></details>
            </th>
            <td>{submission.submitted_by}</td><td><span className={`benchmark-status benchmark-status-${submission.status}`}>{statusLabels[submission.status]}</span>
              {submission.external_job_id && <div className="benchmark-run-reference">Job {submission.external_job_id}</div>}</td>
            <td className="benchmark-score">{score(submission.score)}</td><td className="muted">{date(submission.created_at)}</td>
          </tr>)}</tbody></table></div>
          {detail.total > 100 && <div className="benchmark-pagination"><p className="benchmark-form-help">Showing {offset + 1}–{offset + detail.submissions.length} of {detail.total} submissions.</p>
            <div className="benchmark-actions"><button className="button" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 100))}>Previous</button>
              <button className="button" disabled={offset + detail.submissions.length >= detail.total} onClick={() => setOffset(offset + 100)}>Next</button></div></div>}
        </section>
      </> : benchmarks.length === 0 ? !error && <NemotronChallenge verifier={verifier} measuring={measuring} onMeasure={() => { void measureBaseline(); }} onSetup={() => setCreating('nemotron')} /> : <div className="nodes-table-container"><table className="benchmark-list" aria-label="Benchmarks"><thead><tr>
        <th scope="col">Benchmark</th><th scope="col">Submissions</th><th scope="col">Best verified score</th><th scope="col">Created</th>
      </tr></thead><tbody>{benchmarks.map((item) => <tr key={item.id}>
        <td><button className="benchmark-name" onClick={() => openBenchmark(item.id)}>{item.title}</button><p className="benchmark-list-prompt">{item.prompt}</p></td>
        <td>{item.submission_count}<span className="cell-secondary">{item.verified_count} verified</span></td>
        <td className="benchmark-score">{score(item.best_score)}<span className="cell-secondary">{item.score_name} · {item.score_direction === 'higher' ? 'Higher' : 'Lower'} is better</span></td><td className="muted">{date(item.created_at)}</td>
      </tr>)}</tbody></table></div>}
    {creating && <CreateBenchmark verifier={verifier} measuring={measuring} onMeasure={() => { void measureBaseline(); }} preset={creating === 'nemotron' ? nemotronBenchmark : undefined} onClose={() => setCreating(null)} onCreated={(created) => {
      setCreating(null); setBenchmarks((items) => [created, ...items]); setSelectedId(created.id); setOffset(0); setView('overview'); setDetail(null);
      setMessage('Benchmark created. Submit a solution to start verification.');
    }} />}
    {submitting && benchmark && <SubmitSolution benchmark={benchmark} onClose={() => setSubmitting(false)} onSubmitted={() => {
      setSubmitting(false); setView('submissions'); setMessage('Solution submitted. Its verification status will update automatically.'); void load();
    }} />}
  </div>;
}
