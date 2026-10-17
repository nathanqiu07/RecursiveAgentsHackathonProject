import { useEffect, useRef, useState, type FormEvent } from 'react';
import type { Benchmark } from '../shared/benchmarks';
import { harnessLabel, type HarnessCatalog, type HarnessId, type ModelProfile } from './harnesses';
import './benchmark-agents.css';

type Role = 'attempt' | 'researcher';
type RunStatus = 'queued' | 'running' | 'submitted' | 'completed' | 'failed' | 'cancelled' | 'needs_attention';
type Pool = { id: string; node_id: string | null; cpus: number; memory_mb: number; scratch_mb: number; max_timeout_seconds: number; enabled?: boolean };
type RunRequest = {
  role: Role; objective: string; benchmark_id: string; harness: HarnessId; model_profile_id: string | null;
  model: string | null; max_steps: number;
  allocation: { resource_id: string; node_id: string; cpus: number; memory_mb: number; scratch_mb: number; timeout_seconds: number };
};
type Run = {
  id: string; request: RunRequest; status: RunStatus; created_at: number; updated_at: number;
  cancel_requested: boolean; cleanup_required: boolean; error: string | null;
  state: { steps?: number; experiment_count?: number; publication?: { kind: string; status: string; id?: string } };
  result: { summary?: string } | null;
};
type Configuration = { configured: boolean; message?: string; requires_sign_in?: boolean; sign_in_url?: string };
type Readiness = { ready: boolean; checks: { id: string; label: string; ok: boolean; required: boolean; message: string }[] };
type Workers = { workers_online: number; queued_runs: number; running_runs: number };
type LaunchItem = { body: string; key: string; run?: Run; error?: string };
type PendingBatch = { signature: string; items: LaunchItem[] };
const labels: Record<RunStatus, string> = { queued: 'Queued', running: 'Running', submitted: 'Submitted for verification', completed: 'Finished', failed: 'Failed', cancelled: 'Cancelled', needs_attention: 'Needs attention' };
const date = (seconds: number) => new Date(seconds * 1000).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
const errorText = (value: unknown) => value instanceof Error ? value.message : 'The agent service could not complete this request.';
const eligible = (pools: Pool[]) => pools.filter((pool) => pool.enabled !== false && pool.node_id && pool.cpus > 0 && pool.memory_mb >= 64 && pool.scratch_mb >= 16 && pool.max_timeout_seconds >= 30);

class RequestError extends Error {
  constructor(message: string, readonly status: number) { super(message); }
}

async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api/agents${path}`, { ...options, cache: 'no-store' });
  let body: Record<string, unknown>;
  try { body = await response.json() as Record<string, unknown>; }
  catch { throw new RequestError('The agent service returned an unreadable response.', response.status); }
  if (!response.ok) {
    const details = Array.isArray(body.details) ? body.details : Array.isArray(body.detail) ? body.detail : [];
    const messages = details.map((item: { message?: string; msg?: string }) => item.message || item.msg || '').filter(Boolean).join('; ');
    throw new RequestError(messages || (typeof body.error === 'string' ? body.error : typeof body.detail === 'string' ? body.detail : 'The agent request failed.'), response.status);
  }
  return body as T;
}

function allowedProfiles(catalog: HarnessCatalog | null, harness: HarnessId): ModelProfile[] {
  const requiresKey = harness === 'builtin' && !catalog?.harnesses.find((item) => item.id === harness)?.models.length;
  return (catalog?.profiles || []).filter((profile) => profile.harness === harness && profile.enabled
    && (!requiresKey || (profile.auth_mode === 'api-key' && profile.api_key_set && Boolean(profile.base_url))));
}

function defaultObjective(benchmark: Benchmark, role: Role) {
  return role === 'researcher'
    ? `Investigate a distinct way to improve ${benchmark.title}. Read the complete benchmark contract, review prior work, test the idea with real experiments, and submit a supported paper to the Library.`
    : `Improve ${benchmark.title}. Read the complete benchmark contract, implement and test a concrete approach, and submit a candidate with an immutable public source checkpoint for independent verification.`;
}

export default function BenchmarkAgents({ benchmark, active = true }: { benchmark: Benchmark; active?: boolean }) {
  const [configuration, setConfiguration] = useState<Configuration | null>(null);
  const [catalog, setCatalog] = useState<HarnessCatalog | null>(null);
  const [readiness, setReadiness] = useState<Readiness | null>(null);
  const [workers, setWorkers] = useState<Workers | null>(null);
  const [pools, setPools] = useState<Pool[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [historyTotal, setHistoryTotal] = useState(0);
  const [checked, setChecked] = useState(false);
  const [loading, setLoading] = useState(true);
  const [discoveryError, setDiscoveryError] = useState('');
  const [historyError, setHistoryError] = useState('');
  const [launchError, setLaunchError] = useState('');
  const [notice, setNotice] = useState('');
  const [refreshToken, setRefreshToken] = useState(0);
  const [role, setRole] = useState<Role>('attempt');
  const [objective, setObjective] = useState(() => defaultObjective(benchmark, 'attempt'));
  const [harness, setHarness] = useState<HarnessId>('codex');
  const [profileId, setProfileId] = useState('');
  const [model, setModel] = useState('');
  const [poolId, setPoolId] = useState('');
  const [count, setCount] = useState(1);
  const [minutes, setMinutes] = useState(120);
  const [steps, setSteps] = useState(40);
  const [launching, setLaunching] = useState(false);
  const [cancelling, setCancelling] = useState<string[]>([]);
  const pending = useRef<PendingBatch | null>(null);
  const guard = useRef(false);
  const initialized = useRef(false);
  const mounted = useRef(true);
  const benchmarkId = benchmark.id;
  const currentBenchmark = useRef(benchmarkId);
  currentBenchmark.current = benchmarkId;

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  useEffect(() => {
    setRole('attempt'); setObjective(defaultObjective(benchmark, 'attempt'));
    setRuns([]); setNotice(''); setLaunchError(''); pending.current = null;
  }, [benchmarkId]); // A different benchmark gets its own launch request and history.

  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    const signal = controller.signal;
    let timer: number | undefined;
    setLoading(true);
    async function poll() {
      try {
        const config = await api<Configuration>('/config', { signal });
        if (signal.aborted) return;
        setConfiguration(config);
        if (!config.configured) { setChecked(false); setDiscoveryError(''); return; }
        const [history, capacity, harnesses, dependencies, status] = await Promise.allSettled([
          api<{ runs: Run[]; total: number }>('/runs?limit=100&offset=0', { signal }),
          api<{ resources: Pool[] }>('/resources', { signal }),
          api<HarnessCatalog>('/settings/harnesses', { signal }),
          api<Readiness>('/settings/readiness', { signal }),
          api<Workers>('/status', { signal }),
        ]);
        if (signal.aborted) return;
        if (history.status === 'fulfilled') {
          const matching = history.value.runs.filter((run) => run.request.benchmark_id === benchmarkId);
          setRuns((current) => {
            const ids = new Set(matching.map((run) => run.id));
            return [...matching, ...current.filter((run) => !ids.has(run.id))].sort((a, b) => b.created_at - a.created_at).slice(0, 100);
          });
          setHistoryTotal(history.value.total); setHistoryError('');
        } else setHistoryError(errorText(history.reason));
        const failures = [capacity, harnesses, dependencies, status].filter((result) => result.status === 'rejected');
        setChecked(failures.length === 0);
        setDiscoveryError(failures.length ? 'Current model, worker, readiness, or resource availability could not be checked. Refresh before starting another agent.' : '');
        if (capacity.status === 'fulfilled') setPools(eligible(capacity.value.resources));
        if (dependencies.status === 'fulfilled') setReadiness(dependencies.value);
        if (status.status === 'fulfilled') setWorkers(status.value);
        if (harnesses.status === 'fulfilled') {
          setCatalog(harnesses.value);
          if (!initialized.current) {
            const preferred = harnesses.value.harnesses.find((item) => item.id === 'codex' && item.available)?.id
              || harnesses.value.default_harness;
            setHarness(preferred);
            const profiles = allowedProfiles(harnesses.value, preferred);
            setProfileId(profiles.find((profile) => profile.auth_mode === 'codex-oauth')?.id || profiles[0]?.id || '');
            initialized.current = true;
          }
        }
      } catch (failure) {
        if (!signal.aborted) { setChecked(false); setDiscoveryError(errorText(failure)); }
      } finally {
        if (!signal.aborted) { setLoading(false); timer = window.setTimeout(() => { void poll(); }, 5000); }
      }
    }
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [active, benchmarkId, refreshToken]);

  const selectedHarness = catalog?.harnesses.find((item) => item.id === harness);
  const profiles = allowedProfiles(catalog, harness);
  const selectedProfile = profiles.find((profile) => profile.id === profileId);
  const needsProfile = harness === 'claude-code' || harness === 'nemoclaw' || (harness === 'builtin' && !selectedHarness?.models.length);
  const harnessReady = selectedHarness?.available === true && (needsProfile ? Boolean(selectedProfile) : !profileId || Boolean(selectedProfile));
  const chosenPools = poolId ? pools.filter((pool) => pool.id === poolId) : pools;
  const maxCount = Math.min(3, chosenPools.length);
  const actualCount = Math.min(count, maxCount);
  const signature = JSON.stringify({ benchmarkId, role, objective: objective.trim(), harness, profileId, model, poolId, count, minutes, steps });
  const retrying = pending.current?.signature === signature && pending.current.items.some((item) => !item.run);
  const ready = configuration?.configured === true && checked && harnessReady && readiness?.ready === true && (workers?.workers_online || 0) > 0 && actualCount > 0;
  const selectedMaxSeconds = chosenPools.length ? Math.min(...chosenPools.slice(0, Math.max(1, actualCount)).map((pool) => pool.max_timeout_seconds)) : null;
  useEffect(() => {
    if (selectedMaxSeconds !== null && !pending.current) setMinutes((value) => Math.min(value, selectedMaxSeconds / 60));
  }, [selectedMaxSeconds]);
  const blocked = readiness?.checks.filter((check) => check.required && !check.ok) || [];
  const running = runs.filter((run) => ['queued', 'running'].includes(run.status)).length;
  const signInUrl = configuration?.sign_in_url?.startsWith('/signin-with-chatgpt?') ? configuration.sign_in_url : null;

  function chooseHarness(value: HarnessId) {
    setHarness(value); setModel('');
    const available = allowedProfiles(catalog, value);
    setProfileId(available.find((profile) => profile.auth_mode === 'codex-oauth')?.id || available[0]?.id || '');
  }

  function chooseRole(value: Role) {
    if (objective === defaultObjective(benchmark, role)) setObjective(defaultObjective(benchmark, value));
    setRole(value);
  }

  async function launch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (guard.current) return;
    setLaunchError(''); setNotice('');
    if (!objective.trim()) { setLaunchError('Give these agents an objective.'); return; }
    if (!retrying && !ready) { setLaunchError('Complete the readiness checks and connect an available resource before launching.'); return; }
    if (!Number.isInteger(steps) || steps < 1 || steps > 100 || !Number.isFinite(minutes) || minutes < .5 || minutes > 1440) {
      setLaunchError('Choose 1–100 steps and a run budget between 30 seconds and 24 hours.'); return;
    }
    if (!retrying) {
      const selected = chosenPools.slice(0, actualCount);
      const seconds = Math.floor(minutes * 60);
      if (selected.some((pool) => seconds > pool.max_timeout_seconds)) {
        setLaunchError('This run budget exceeds a selected resource’s limit. Reduce it or choose a different resource.'); return;
      }
      pending.current = { signature, items: selected.map((pool, index) => {
        const request: RunRequest = {
          role, objective: objective.trim() + (selected.length > 1 ? `\nIndependent agent ${index + 1} of ${selected.length}: investigate a distinct approach and keep your own experiment evidence.` : ''),
          benchmark_id: benchmarkId, harness, model_profile_id: profileId || null, model: profileId ? null : model || null, max_steps: steps,
          allocation: { resource_id: pool.id, node_id: pool.node_id!, cpus: Math.min(8, pool.cpus), memory_mb: pool.memory_mb,
            scratch_mb: pool.scratch_mb, timeout_seconds: seconds },
        };
        return { body: JSON.stringify(request), key: crypto.randomUUID() };
      }) };
    }
    const batch = pending.current!;
    guard.current = true; setLaunching(true);
    await Promise.allSettled(batch.items.map(async (item) => {
      if (item.run) return;
      try {
        item.run = await api<Run>('/runs', { method: 'POST', headers: { 'Content-Type': 'application/json', 'Idempotency-Key': item.key }, body: item.body });
        item.error = undefined;
      } catch (failure) { item.error = errorText(failure); }
    }));
    if (mounted.current && currentBenchmark.current === benchmarkId) {
      const created = batch.items.flatMap((item) => item.run ? [item.run] : []);
      const failed = batch.items.filter((item) => !item.run);
      setRuns((current) => [...created, ...current.filter((run) => !created.some((item) => item.id === run.id))].sort((a, b) => b.created_at - a.created_at));
      if (created.length) setNotice(`${created.length} ${created.length === 1 ? 'agent is' : 'agents are'} assigned to ${benchmark.title}. Each run has its own job VM and experiment history.`);
      if (failed.length) setLaunchError(`${failed.length} launch ${failed.length === 1 ? 'request needs' : 'requests need'} attention. ${failed[0].error || ''} Retry the unchanged request to safely recover accepted runs.`);
      else if (pending.current === batch) pending.current = null;
      setRefreshToken((value) => value + 1);
    }
    if (mounted.current) setLaunching(false);
    guard.current = false;
  }

  async function cancel(run: Run) {
    if (cancelling.includes(run.id)) return;
    setCancelling((ids) => [...ids, run.id]); setLaunchError('');
    try {
      const updated = await api<Run>(`/runs/${run.id}/cancel`, { method: 'POST' });
      if (mounted.current && currentBenchmark.current === benchmarkId) setRuns((current) => current.map((item) => item.id === updated.id ? updated : item));
    } catch (failure) { if (mounted.current && currentBenchmark.current === benchmarkId) setLaunchError(errorText(failure)); }
    finally { if (mounted.current) setCancelling((ids) => ids.filter((id) => id !== run.id)); }
  }

  return <section className="benchmark-agents" aria-label="Agents for this benchmark">
    <header className="bma-header"><div><h3>Spin out agents</h3><p>Launch independent research or optimization runs for this benchmark. Agents use the same resources as Experiments and verification.</p></div>
      <button className="button" type="button" onClick={() => setRefreshToken((value) => value + 1)} disabled={loading}>{loading ? 'Checking…' : 'Refresh'}</button></header>
    {configuration && !configuration.configured && <div className="bma-warning" role="status"><p>{configuration.message || 'Connect the local agent harness in Connections before launching.'}</p>{signInUrl && <a className="button" href={signInUrl} target="_top">Sign in to manage agents</a>}</div>}
    {discoveryError && <p className="bma-warning" role="status">{discoveryError}</p>}
    <div className="bma-health" aria-live="polite"><span>{workers ? `${workers.workers_online} local ${workers.workers_online === 1 ? 'worker' : 'workers'} online` : 'Checking workers'}</span><span>{checked ? `${pools.length} ${pools.length === 1 ? 'resource' : 'resources'} available` : 'Resource availability unconfirmed'}</span><span>{running} active for this benchmark</span></div>
    <form className="bma-launch" onSubmit={launch}>
      <div className="bma-form-grid">
        <label>Agent type<select value={role} disabled={launching} onChange={(event) => chooseRole(event.target.value as Role)}><option value="attempt">Optimization agent</option><option value="researcher">Research agent</option></select><span>{role === 'attempt' ? 'Test an approach and submit a candidate for scoring.' : 'Investigate an idea and submit a supported paper to the Library.'}</span></label>
        <label>Harness<select value={harness} disabled={launching} onChange={(event) => chooseHarness(event.target.value as HarnessId)}>{catalog?.harnesses.length ? catalog.harnesses.map((item) => <option key={item.id} value={item.id}>{item.label}{item.available ? '' : ' · needs setup'}</option>) : <option value="codex">Codex · checking availability</option>}</select><span>{selectedHarness?.message || 'Model turns and orchestration run locally.'}</span></label>
        <label>Model profile<select value={profileId} disabled={launching} onChange={(event) => { setProfileId(event.target.value); setModel(''); }}><option value="">{needsProfile ? 'Choose a saved profile' : harness === 'codex' ? 'Existing Codex sign-in' : 'Runtime default'}</option>{profiles.map((profile) => <option key={profile.id} value={profile.id}>{profile.name} · {profile.model}</option>)}</select><span>{selectedProfile ? `Uses ${selectedProfile.model} through ${harnessLabel(harness)}.` : 'Configure saved model profiles in Connections.'}</span></label>
        {!profileId && !needsProfile && <label>Model<select value={model} disabled={launching} onChange={(event) => setModel(event.target.value)}><option value="">Runtime default</option>{selectedHarness?.models.map((item) => <option key={item.id} value={item.id}>{item.label || item.id}</option>)}</select></label>}
        <label>Resource<select value={poolId} disabled={launching} onChange={(event) => { setPoolId(event.target.value); setCount(1); }}><option value="">Spread across available resources</option>{pools.map((pool) => <option key={pool.id} value={pool.id}>{pool.id} · {(pool.memory_mb / 1024).toFixed(0)} GiB RAM</option>)}</select><span>Each parallel agent receives a different available pool.</span></label>
        <label>Parallel agents<select value={retrying ? count : Math.min(count, Math.max(1, maxCount))} disabled={launching || maxCount < 1} onChange={(event) => setCount(Number(event.target.value))}>{Array.from({ length: retrying ? Math.max(count, maxCount) : Math.max(1, maxCount) }, (_, index) => <option key={index + 1} value={index + 1}>{index + 1}</option>)}</select><span>Runs wait when all local workers are busy.</span></label>
        <label>Run budget · minutes<input type="number" min="0.5" max={retrying ? 1440 : selectedMaxSeconds !== null ? selectedMaxSeconds / 60 : 1440} step="0.5" value={minutes} disabled={launching} onChange={(event) => setMinutes(Number(event.target.value))} /><span>{selectedMaxSeconds !== null ? `Up to ${Math.floor(selectedMaxSeconds / 60)} minutes on these resources.` : 'Limited by the selected resource.'}</span></label>
        <label>Maximum steps<input type="number" min="1" max="100" value={steps} disabled={launching} onChange={(event) => setSteps(Number(event.target.value))} /></label>
      </div>
      <label className="bma-objective">Objective<textarea rows={3} maxLength={20000} required value={objective} disabled={launching} onChange={(event) => setObjective(event.target.value)} /></label>
      {blocked.length > 0 && <details className="bma-readiness"><summary>{blocked.length} readiness {blocked.length === 1 ? 'check needs' : 'checks need'} attention</summary><ul>{blocked.map((check) => <li key={check.id}><strong>{check.label}:</strong> {check.message}</li>)}</ul></details>}
      {checked && !pools.length && <p className="bma-warning">No resource is available right now. Bring the A5000 nodes online or wait for a running VM to finish cleanup.</p>}
      {checked && !harnessReady && <p className="bma-warning">Choose an available harness and model profile. Check local sign-in and saved profiles in Connections.</p>}
      <footer className="bma-launch-footer"><p>Assigned to <strong>{benchmark.title}</strong>. Submissions receive independent verification.</p><button className="button button-primary" type="submit" disabled={launching || (!ready && !retrying)}>{launching ? 'Starting agents…' : retrying ? 'Retry pending launch' : `Launch ${actualCount > 1 ? `${actualCount} agents` : 'agent'}`}</button></footer>
    </form>
    {launchError && <p className="bma-warning" role="alert">{launchError}</p>}
    {notice && <p className="bma-notice" role="status">{notice}</p>}
    <div className="bma-runs-heading"><h3>Agent runs</h3><span>Assigned to this benchmark</span></div>
    {historyError && <p className="bma-warning" role="status">{historyError} Showing the last known history.</p>}
    {!runs.length ? <p className="bma-empty">{loading ? 'Loading benchmark agents…' : 'No recent agents are assigned to this benchmark. Launch one to start an independent run.'}</p> : <ul className="bma-run-list">{runs.map((run) => <li key={run.id}>
      <div className="bma-run-top"><strong>{run.request.role === 'researcher' ? 'Research agent' : 'Optimization agent'}</strong><span className={`bma-status bma-status-${run.status}`}>{labels[run.status] || run.status.replaceAll('_', ' ')}</span></div>
      <p>{run.request.objective}</p><div className="bma-run-meta"><span>{harnessLabel(run.request.harness)} · {run.request.allocation.resource_id}</span><span>{run.state.steps || 0}/{run.request.max_steps} steps · {run.state.experiment_count || 0} experiments</span><time dateTime={new Date(run.created_at * 1000).toISOString()}>{date(run.created_at)}</time></div>
      <progress max={run.request.max_steps} value={Math.min(run.state.steps || 0, run.request.max_steps)} aria-label="Completed agent steps" />
      {run.result?.summary && <p className="bma-outcome">{run.result.summary}</p>}{run.error && <p className="bma-run-error">{run.error.replaceAll('_', ' ')}</p>}
      {(run.cleanup_required || (run.cancel_requested && ['queued', 'running'].includes(run.status))) && <p className="bma-run-note">{run.cleanup_required ? 'VM cleanup still needs confirmation.' : 'Cancellation requested; waiting for the worker.'}</p>}
      <div className="bma-run-actions"><span>Run {run.id.slice(0, 8)}</span>{['queued', 'running'].includes(run.status) && <button type="button" className="button" disabled={cancelling.includes(run.id) || run.cancel_requested} onClick={() => { void cancel(run); }}>{run.cancel_requested ? 'Stopping…' : cancelling.includes(run.id) ? 'Requesting…' : 'Stop agent'}</button>}</div>
    </li>)}</ul>}
    {historyTotal > 100 && <p className="bma-history-note">This view matches the latest 100 agent runs to this benchmark. Older history is available in Experiments.</p>}
  </section>;
}
