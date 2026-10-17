import { useEffect, useRef, useState } from 'react';
import type { Benchmark } from '../shared/benchmarks';
import './benchmark-agents.css';

type RunStatus = 'queued' | 'running' | 'submitted' | 'completed' | 'failed' | 'cancelled' | 'needs_attention';
type Run = {
  id: string; request: { benchmark_id: string | null; role: string }; status: RunStatus;
  created_at: number; cancel_requested: boolean; cleanup_required: boolean; error: string | null;
  state: { experiment_count?: number; research_summary?: { hypothesis?: string; source_urls?: string[]; papers_read?: number; access_connected?: boolean }; research_proposal?: { hypothesis?: string }; research_idea?: { hypothesis?: string } };
  result: { summary?: string } | null;
};
type Configuration = { configured: boolean; message?: string };
const labels: Record<RunStatus, string> = { queued: 'Queued', running: 'Researching', submitted: 'Submitted to Library', completed: 'Finished', failed: 'Failed', cancelled: 'Stopped', needs_attention: 'Needs attention' };
const errorText = (value: unknown) => value instanceof Error ? value.message : 'The agent service could not complete this request.';

async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api/agents${path}`, { ...options, cache: 'no-store' });
  const body = await response.json() as { error?: string; detail?: unknown };
  if (!response.ok) throw new Error(typeof body.error === 'string' ? body.error : typeof body.detail === 'string' ? body.detail : 'The agent request could not complete.');
  return body as T;
}

export default function BenchmarkAgents({ benchmark, active = true, launchToken = 0 }: { benchmark: Benchmark; active?: boolean; launchToken?: number }) {
  const [configuration, setConfiguration] = useState<Configuration | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [refresh, setRefresh] = useState(0);
  const [stopping, setStopping] = useState<string[]>([]);
  const pending = useRef<{ benchmark: string; key: string } | null>(null);
  const guard = useRef(false);
  const mounted = useRef(true);
  const currentBenchmark = useRef(benchmark.id);
  currentBenchmark.current = benchmark.id;
  const lastLaunchToken = useRef(0);

  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  useEffect(() => { pending.current = null; setRuns([]); setNotice(''); setError(''); }, [benchmark.id]);
  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    let timer: number | undefined;
    async function poll() {
      try {
        const config = await api<Configuration>('/config', { signal: controller.signal });
        if (controller.signal.aborted) return;
        setConfiguration(config);
        if (!config.configured) return;
        const history = await api<{ runs: Run[] }>('/runs?limit=100&offset=0', { signal: controller.signal });
        if (controller.signal.aborted) return;
        const matching = history.runs.filter((run) => run.request.benchmark_id === benchmark.id);
        setRuns((previous) => {
          const ids = new Set(matching.map((run) => run.id));
          return [...matching, ...previous.filter((run) => !ids.has(run.id))].sort((a, b) => b.created_at - a.created_at).slice(0, 100);
        });
      } catch (failure) { if (!controller.signal.aborted) setError(errorText(failure)); }
      finally { if (!controller.signal.aborted) { setLoading(false); timer = window.setTimeout(() => { void poll(); }, 5000); } }
    }
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [active, benchmark.id, refresh]);

  async function launch() {
    if (guard.current) return;
    guard.current = true; setLaunching(true); setError(''); setNotice('');
    const id = benchmark.id;
    if (!pending.current || pending.current.benchmark !== id) pending.current = { benchmark: id, key: crypto.randomUUID() };
    const request = pending.current;
    try {
      const run = await api<Run>('/research-runs', { method: 'POST', headers: { 'Content-Type': 'application/json', 'Idempotency-Key': request.key }, body: JSON.stringify({ benchmark_id: id }) });
      if (mounted.current && currentBenchmark.current === id) {
        setRuns((previous) => [run, ...previous.filter((item) => item.id !== run.id)]);
        setNotice('Research agent started. It will search papers, consult the Library, and request experiment resources when needed.');
        if (pending.current === request) pending.current = null;
        setRefresh((value) => value + 1);
      }
    } catch (failure) {
      if (mounted.current && currentBenchmark.current === id) setError(`${errorText(failure)} Retry to recover the same launch safely.`);
    } finally { guard.current = false; if (mounted.current) setLaunching(false); }
  }

  useEffect(() => {
    if (launchToken <= lastLaunchToken.current) return;
    lastLaunchToken.current = launchToken;
    void launch();
  }, [launchToken]);

  async function stop(run: Run) {
    setStopping((ids) => [...ids, run.id]);
    try {
      const updated = await api<Run>(`/runs/${run.id}/cancel`, { method: 'POST' });
      if (mounted.current && currentBenchmark.current === benchmark.id) setRuns((previous) => previous.map((item) => item.id === updated.id ? updated : item));
    } catch (failure) { if (mounted.current) setError(errorText(failure)); }
    finally { if (mounted.current) setStopping((ids) => ids.filter((id) => id !== run.id)); }
  }

  return <section className="benchmark-agents" aria-label="Agents for this benchmark">
    <header className="bma-header"><div><h2>Research agents</h2><p>Give an agent this benchmark. It develops its own method, runs experiments, and writes an original research paper.</p></div>
      <button type="button" className="button button-primary" disabled={launching || (!pending.current && (!configuration?.configured || loading))} onClick={() => { void launch(); }}>{launching ? 'Spinning out agent…' : pending.current ? 'Retry agent launch' : 'Spin out research agent'}</button></header>
    <div className="bma-capabilities"><span>You.com MCP · References</span><span>Library API · Publish original paper</span><span>Resource API · Experiments</span></div>
    <p className="bma-note">Each click starts an independent agent. The benchmark supplies its context automatically. Research starts locally; GPU resources are assigned when an experiment needs them.</p>
    {configuration && !configuration.configured && <p className="bma-warning" role="status">{configuration.message || 'The local agent service needs to be connected.'}</p>}
    {notice && <p className="bma-notice" role="status">{notice}</p>}
    {error && <p className="bma-warning" role="alert">{error}</p>}
    <div className="bma-header"><h3>Agent progress</h3><button type="button" className="button" onClick={() => setRefresh((value) => value + 1)}>Refresh</button></div>
    {loading ? <p className="bma-note">Loading agents…</p> : !runs.length ? <p className="bma-note">No research agents yet. Spin one out to start exploring.</p> : <div className="bma-runs">{runs.map((run) => <article className="bma-run" key={run.id}>
      <div className="bma-run-heading"><strong>{run.request.role === 'researcher' ? 'Research agent' : 'Optimization agent'} · {run.id.slice(0, 8)}</strong><span className={`bma-status bma-status-${run.status}`}>{labels[run.status]}</span></div>
      {(run.state.research_summary?.hypothesis || run.state.research_proposal?.hypothesis || run.state.research_idea?.hypothesis) && <p>{run.state.research_summary?.hypothesis || run.state.research_proposal?.hypothesis || run.state.research_idea?.hypothesis}</p>}
      <p className="bma-note">{new Date(run.created_at * 1000).toLocaleString()} · {run.state.research_summary?.papers_read || 0} references read · {run.state.experiment_count || 0} experiments{run.state.research_summary?.access_connected ? ' · You.com connected' : ''}</p>
      {!!run.state.research_summary?.source_urls?.length && <div className="bma-capabilities">{run.state.research_summary.source_urls.filter((url) => url.startsWith('https://')).map((url, index) => <a key={url} href={url} target="_blank" rel="noreferrer">Reference {index + 1} ↗</a>)}</div>}
      {run.result?.summary && <p>{run.result.summary}</p>}{run.error && <p className="bma-warning">{run.error === 'research_gpu_capacity_unavailable' ? 'Research is saved. Connect a GPU resource node to conduct the planned experiments.' : run.error === 'you_mcp_unavailable' ? 'You.com could not be reached. Retry later or add a You.com key in Connections.' : run.error}</p>}
      {run.cancel_requested && <p className="bma-note">Stopping; waiting for experiment cleanup.</p>}
      {['queued', 'running'].includes(run.status) && <button type="button" className="button" disabled={run.cancel_requested || stopping.includes(run.id)} onClick={() => { void stop(run); }}>{run.cancel_requested || stopping.includes(run.id) ? 'Stopping…' : 'Stop agent'}</button>}
    </article>)}</div>}
  </section>;
}
