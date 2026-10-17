import { useEffect, useRef, useState, type FormEvent, type MutableRefObject } from 'react';
import './agents.css';
import { harnessLabel, type HarnessCatalog, type HarnessId } from './harnesses';
import LibraryNovelty, { type LibraryNoveltyReview } from './LibraryNovelty';

type Role = 'researcher' | 'attempt';
type RunStatus = 'queued' | 'running' | 'submitted' | 'completed' | 'failed' | 'cancelled' | 'needs_attention';
type Allocation = { resource_id: string; node_id: string; cpus: number; memory_mb: number; scratch_mb: number; timeout_seconds: number };
type RunRequest = { role: Role; objective: string; benchmark_id: string | null; allocation: Allocation; max_steps: number; harness?: HarnessId; model_profile_id?: string | null; model?: string | null };
type Experiment = { id: string; command: string; purpose: string; timeout_seconds: number; exit_code: number; output: string; truncated: boolean; step?: number; command_sha256?: string };
type Novelty = { approved: boolean; rationale: string; compared_paper_ids: string[]; related_paper_ids?: string[]; scope?: string; snapshot_sha256?: string; idea?: string };
type PublicationRef = { kind: 'paper' | 'attempt'; status: string; id?: string };
type Run = {
  id: string; request: RunRequest; status: RunStatus; created_at: number; started_at: number | null; updated_at: number;
  cancel_requested: boolean; cleanup_required: boolean; compute_job_id: string | null; sandbox_id: string | null;
  error: string | null; result: { summary?: string; verification_status?: string; published?: boolean } | null;
  state: { steps?: number; experiments?: Experiment[]; novelty?: Novelty; publication?: PublicationRef; runtime?: { harness?: HarnessId; model?: string | null }; runtime_cleanup_required?: boolean };
};
type AgentEvent = { sequence: number; kind: string; data: Record<string, unknown>; created_at: number };
type Pool = { id: string; node_id: string | null; description?: string; enabled?: boolean; cpus: number; memory_mb: number; scratch_mb: number; max_timeout_seconds: number; max_concurrent_jobs?: number };
type BenchmarkChoice = { id: string; title: string };
type Configuration = { configured: boolean; message?: string; requires_sign_in?: boolean; sign_in_url?: string };
type WorkerStatus = { workers_online: number; last_worker_seen_at: number | null; queued_runs: number; running_runs: number };
type Publication = {
  id: string; status: string; title?: string; score?: number | null; feedback?: string | null; error?: string | null;
  blocked_reason?: string | null; payload?: unknown; base_checkpoint_url?: string;
  submission?: { title?: string; paper_text?: string | null; paper_url?: string | null; authors?: string[] };
  report?: { verdict: string; summary: string; evidence?: string[] } | null;
  experimental_report?: { verdict: string; summary: string; evidence?: string[] } | null;
  novelty?: LibraryNoveltyReview | null;
};
type PendingLaunch = { body: string; key: string };

const runLabels: Record<RunStatus, string> = {
  queued: 'Queued', running: 'Running', submitted: 'Submitted', completed: 'Finished',
  failed: 'Failed', cancelled: 'Cancelled', needs_attention: 'Needs attention',
};
const toolLabels: Record<string, string> = {
  search_library: 'Searching the Library', read_paper: 'Reading a paper', read_benchmark: 'Reading the benchmark',
  web_search: 'Researching online', web_read: 'Reading an online source', execute: 'Running a sandbox command',
  review_novelty: 'Reviewing the proposed idea', submit_paper: 'Preparing a paper for Library',
  submit_attempt: 'Preparing a benchmark submission', finish: 'Finishing the run',
};
const reasonLabels: Record<string, string> = {
  step_budget_exceeded: 'The agent reached its step limit.', model_context_budget_exceeded: 'The agent reached its research context limit.',
  run_time_budget_exceeded: 'The run reached its time limit.', cancelled_by_user: 'Cancellation was requested.',
  worker_stopped: 'The worker stopped. Inspect this run before starting a replacement.',
  interrupted_run_not_replayed: 'This run was interrupted and was not restarted automatically.',
  publication_outcome_unknown_do_not_resubmit: 'The submission may have been accepted. Check the destination before launching a replacement.',
  publication_already_started_do_not_replay: 'A submission was already started. Check the destination before launching a replacement.',
  agent_runtime_error: 'The agent could not continue. Review its latest activity and experiments.',
};
const PAGE_SIZE = 20;
const pendingHarnessCatalog: HarnessCatalog = {
  default_harness: 'codex', profiles: [],
  harnesses: (['codex', 'builtin', 'claude-code', 'nemoclaw'] as HarnessId[]).map((id) => ({
    id, label: harnessLabel(id), available: false, models: [],
    message: 'Harness availability has not been confirmed. Refresh the Agents tab before launching.',
  })),
};
const date = (seconds: number) => new Date(seconds * 1000).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
const readable = (value: string) => value.replaceAll('_', ' ');
const roleLabel = (role: Role) => role === 'researcher' ? 'Researcher' : 'Attempt';
const failureText = (value: string) => reasonLabels[value] || readable(value);
const errorText = (error: unknown, fallback: string) => error instanceof Error ? error.message : fallback;
const validPools = (pools: Pool[]) => pools.filter((pool) => pool.enabled !== false && pool.node_id && pool.cpus > 0 && pool.memory_mb >= 64 && pool.scratch_mb >= 16 && pool.max_timeout_seconds >= 30);

class RequestError extends Error {
  status: number;
  constructor(message: string, status: number) { super(message); this.status = status; }
}

async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api${path}`, { ...options, cache: 'no-store' });
  let body: Record<string, unknown>;
  try { body = await response.json() as Record<string, unknown>; }
  catch { throw new RequestError(response.ok ? 'The service returned an unreadable response.' : 'Unable to reach the agent service.', response.status); }
  if (!response.ok) {
    const details = Array.isArray(body.details) ? body.details : Array.isArray(body.detail) ? body.detail : [];
    const detail = details.map((item: { message?: string; msg?: string }) => item.message || item.msg || '').filter(Boolean).join('; ');
    throw new RequestError(detail || (typeof body.error === 'string' ? body.error : typeof body.detail === 'string' ? body.detail : 'Unable to complete this request.'), response.status);
  }
  return body as T;
}

function Status({ status }: { status: string }) {
  return <span className={`agent-status agent-status-${status}`}>{runLabels[status as RunStatus] || readable(status)}</span>;
}

function eventLabel(event: AgentEvent): string {
  if (event.kind === 'tool_call') return toolLabels[String(event.data.name)] || 'Agent activity';
  if (event.kind === 'sandbox_allocated') return 'Sandbox allocated';
  if (event.kind === 'experiment_started') return `${event.data.purpose === 'test' ? 'Test' : event.data.purpose === 'setup' ? 'Setup' : 'Experiment'} started`;
  if (event.kind === 'experiment_result') return `${event.data.purpose === 'test' ? 'Test' : event.data.purpose === 'setup' ? 'Setup' : 'Experiment'} ${event.data.exit_code === 0 ? 'completed' : 'failed'}`;
  if (event.kind === 'novelty_review') return event.data.approved ? 'Idea passed researcher novelty review' : 'Idea needs a new approach';
  if (event.kind === 'final_novelty_review') return event.data.approved ? 'Paper passed researcher novelty review' : 'Paper needs a novelty review';
  if (event.kind === 'evidence_review') return event.data.supported ? 'Evidence review passed' : 'Evidence needs more support';
  if (event.kind === 'publication_started') return event.data.kind === 'paper' ? 'Submitting paper to Library' : 'Submitting benchmark candidate';
  if (event.kind === 'publication_accepted') return event.data.kind === 'paper' ? 'Library accepted the paper for verification' : 'Benchmark accepted the candidate for verification';
  if (event.kind === 'sandbox_cleanup') return event.data.confirmed ? 'Sandbox cleanup confirmed' : 'Waiting for sandbox cleanup';
  return readable(event.kind);
}

function currentPhase(run: Run, events: AgentEvent[]) {
  if (run.cancel_requested && run.status === 'running') return 'Stopping the agent and cleaning up its sandbox';
  if (run.status === 'queued') return 'Waiting for a worker and sandbox capacity';
  if (run.status === 'submitted') return run.state.publication?.kind === 'paper' ? 'Paper submitted to Library for independent verification' : 'Candidate submitted for independent benchmark scoring';
  if (run.status !== 'running') return run.error ? failureText(run.error) : runLabels[run.status];
  const latest = [...events].reverse().find((event) => event.kind !== 'sandbox_cleanup');
  return latest ? eventLabel(latest) : 'Starting the run';
}

function LaunchAgent({ role: initialRole, initialRequest, pools, benchmarks, benchmarkError, harnesses, availabilityChecked, pending, onClose, onCreated }: {
  role: Role; pools: Pool[]; benchmarks: BenchmarkChoice[]; benchmarkError: string; harnesses: HarnessCatalog; availabilityChecked: boolean;
  initialRequest: RunRequest | null;
  pending: MutableRefObject<PendingLaunch | null>; onClose: () => void; onCreated: (run: Run) => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [role, setRole] = useState<Role>(initialRole);
  const [objective, setObjective] = useState(initialRequest?.objective || '');
  const [benchmarkId, setBenchmarkId] = useState(initialRequest?.benchmark_id || '');
  const [poolId, setPoolId] = useState(pools[0]?.id || '');
  const [cpus, setCpus] = useState(1);
  const [memory, setMemory] = useState(1024);
  const [scratch, setScratch] = useState(128);
  const [timeout, setTimeoutBudget] = useState(600);
  const [steps, setSteps] = useState(40);
  const [harness, setHarness] = useState<HarnessId>(initialRequest ? initialRequest.harness || 'builtin' : harnesses.default_harness || 'codex');
  const [profileId, setProfileId] = useState(initialRequest?.model_profile_id || '');
  const [model, setModel] = useState(initialRequest?.model || '');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const guard = useRef(false);
  const requestController = useRef<AbortController | null>(null);
  const pool = pools.find((item) => item.id === poolId);
  const selectedHarness = harnesses.harnesses.find((item) => item.id === harness);
  const requiresKeyedProfile = harness === 'builtin' && selectedHarness?.models.length === 0;
  const profiles = harnesses.profiles.filter((item) => item.harness === harness && item.enabled
    && (!requiresKeyedProfile || (item.auth_mode === 'api-key' && item.api_key_set && Boolean(item.base_url))));
  const selectedProfile = profiles.find((item) => item.id === profileId);
  const requiresProfile = harness === 'claude-code' || harness === 'nemoclaw' || requiresKeyedProfile;
  const harnessReady = availabilityChecked && selectedHarness?.available === true && (requiresProfile ? Boolean(selectedProfile) : !profileId || Boolean(selectedProfile));

  function chooseHarness(value: HarnessId) {
    const keyed = value === 'builtin' && harnesses.harnesses.find((item) => item.id === value)?.models.length === 0;
    const required = value === 'claude-code' || value === 'nemoclaw' || keyed;
    const profile = harnesses.profiles.find((item) => item.harness === value && item.enabled
      && (!keyed || (item.auth_mode === 'api-key' && item.api_key_set && Boolean(item.base_url))));
    setHarness(value); setProfileId(required ? profile?.id || '' : ''); setModel('');
  }

  useEffect(() => {
    if (!dialog.current?.open) dialog.current?.showModal();
    return () => requestController.current?.abort();
  }, []);

  useEffect(() => {
    if (!pool) return;
    setCpus(Math.min(1, pool.cpus)); setMemory(Math.min(1024, pool.memory_mb));
    setScratch(Math.min(128, pool.scratch_mb)); setTimeoutBudget(Math.min(1200, pool.max_timeout_seconds));
  }, [pool?.id]); // Capacity changes are checked again when launching.

  async function launch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (guard.current) return;
    setError('');
    if (!harnessReady) { setError('Choose an available harness and model profile. Manage connections to complete setup.'); return; }
    if (!pool?.node_id) { setError('Choose an available sandbox pool.'); return; }
    if (role === 'attempt' && !benchmarkId) { setError('Choose the benchmark this agent should work on.'); return; }
    if (!objective.trim()) { setError('Describe what this agent should investigate or improve.'); return; }
    if (cpus <= 0 || cpus > pool.cpus || memory > pool.memory_mb || scratch > pool.scratch_mb || timeout > pool.max_timeout_seconds) {
      setError('The available pool capacity changed. Adjust the resource limits before launching.'); return;
    }
    const request: RunRequest = {
      role, objective: objective.trim(), benchmark_id: benchmarkId || null, max_steps: steps,
      harness, model_profile_id: profileId || null, model: profileId ? null : model.trim() || null,
      allocation: { resource_id: pool.id, node_id: pool.node_id, cpus, memory_mb: memory, scratch_mb: scratch, timeout_seconds: timeout },
    };
    const body = JSON.stringify(request);
    if (pending.current?.body !== body) pending.current = { body, key: crypto.randomUUID() };
    const key = pending.current.key;
    const controller = new AbortController();
    requestController.current = controller;
    guard.current = true; setSaving(true);
    try {
      const run = await api<Run>('/agents/runs', { method: 'POST', signal: controller.signal,
        headers: { 'Content-Type': 'application/json', 'Idempotency-Key': key }, body });
      if (controller.signal.aborted) return;
      pending.current = null;
      onCreated(run);
    } catch (failure) {
      if (!controller.signal.aborted) {
        const retry = !(failure instanceof RequestError) || failure.status >= 500;
        setError(`${errorText(failure, 'Unable to launch the agent.')}${retry ? ' Retry this unchanged request to safely check whether it was accepted.' : ''}`);
      }
    } finally { if (!controller.signal.aborted) { guard.current = false; setSaving(false); } }
  }

  return <dialog ref={dialog} className="node-dialog agent-launch-dialog" aria-labelledby="launch-agent-title" onCancel={(event) => {
    if (saving) event.preventDefault(); else onClose();
  }}>
    <form onSubmit={launch}>
      <div className="section-heading"><h2 id="launch-agent-title">{initialRequest ? 'Spin out a new agent' : 'New agent'}</h2><button className="icon-button" type="button" onClick={onClose} disabled={saving} aria-label="Close launch controls">×</button></div>
      {initialRequest && <p className="agent-new-run-note">Review these copied settings before launching an independent run. It gets a fresh sandbox and experiment history; saved profiles use their current settings.</p>}
      <fieldset className="agent-role-options" disabled={saving}><legend>Agent type</legend>
        <label className={role === 'researcher' ? 'agent-role-option is-selected' : 'agent-role-option'}>
          <input type="radio" name="agent-role" value="researcher" checked={role === 'researcher'} onChange={() => setRole('researcher')} />
          <span><strong>Researcher</strong><span>Explore a new idea, run experiments, and submit a paper to Library.</span></span>
        </label>
        <label className={role === 'attempt' ? 'agent-role-option is-selected' : 'agent-role-option'}>
          <input type="radio" name="agent-role" value="attempt" checked={role === 'attempt'} onChange={() => setRole('attempt')} />
          <span><strong>Attempt</strong><span>Apply existing research, test an approach, and submit a benchmark candidate.</span></span>
        </label>
      </fieldset>
      <div className="agent-launch-fields">
        <div className="agent-harness-fields">
          <label>Agent harness<select value={harness} disabled={saving} onChange={(event) => chooseHarness(event.target.value as HarnessId)}>{harnesses.harnesses.map((item) => <option key={item.id} value={item.id}>{item.label}{item.id === harnesses.default_harness ? ' · default' : ''}{!item.available ? ' · needs setup' : ''}</option>)}</select><span className="agent-field-note">{selectedHarness?.message || 'Choose an agent harness.'}</span></label>
          <label>Model profile<select value={profileId} disabled={saving} onChange={(event) => { setProfileId(event.target.value); setModel(''); }}><option value="">{requiresKeyedProfile ? 'Choose a saved API key profile' : requiresProfile ? 'Choose a saved model profile' : harness === 'codex' ? 'Existing Codex sign-in · choose model below' : 'Harness environment · choose model below'}</option>{profiles.map((profile) => <option key={profile.id} value={profile.id}>{profile.name} · {profile.model}</option>)}</select><span className="agent-field-note">{selectedProfile ? selectedProfile.auth_mode === 'codex-oauth' ? 'Uses the existing Codex sign-in on the worker.' : selectedProfile.auth_mode === 'api-key' ? 'Uses this profile’s encrypted API credentials.' : 'Uses the worker environment.' : 'Save multiple models per harness in Connections.'}</span></label>
          {!profileId && !requiresProfile && <label>Model<select value={model} disabled={saving} onChange={(event) => setModel(event.target.value)}><option value="">Runtime default</option>{model && !selectedHarness?.models.some((item) => item.id === model) && <option value={model}>{model}</option>}{(selectedHarness?.models || []).map((item) => <option key={item.id} value={item.id}>{item.label || item.id}</option>)}</select></label>}
          {!harnessReady && <p className="agent-warning">{!availabilityChecked ? 'Current harness or sandbox availability could not be checked. Refresh the Agents tab before launching.' : requiresKeyedProfile ? 'Choose a saved API key profile with a base URL. Add one in Connections if none are available.' : selectedHarness?.available && requiresProfile ? 'Choose a saved model profile. Add one in Connections if none are available.' : selectedHarness?.message || 'This harness is unavailable. Complete its setup in Connections before launching.'}</p>}
        </div>
        <label>Objective<textarea required maxLength={20000} rows={4} value={objective} disabled={saving} onChange={(event) => setObjective(event.target.value)} placeholder={role === 'researcher' ? 'What new idea or research question should the agent investigate?' : 'What should the agent improve? Include the public checkpoint and any available solution artifacts.'} autoFocus /></label>
        <label>Benchmark{role === 'researcher' && ' · optional'}
          <select value={benchmarkId} required={role === 'attempt'} disabled={saving || !!benchmarkError} onChange={(event) => setBenchmarkId(event.target.value)}>
            <option value="">{role === 'attempt' ? 'Choose a benchmark' : 'No assigned benchmark'}</option>
            {benchmarks.map((benchmark) => <option key={benchmark.id} value={benchmark.id}>{benchmark.title}</option>)}
          </select>
          {benchmarkError ? <span className="error-text">{benchmarkError}</span> : role === 'attempt' && benchmarks.length === 0 && <span className="agent-field-note">Create a benchmark before launching an attempt.</span>}
        </label>
        <label>Sandbox pool<select value={poolId} onChange={(event) => setPoolId(event.target.value)} required disabled={saving || pools.length === 0}>
          <option value="" disabled>Choose an available pool</option>
          {pools.map((item) => <option key={item.id} value={item.id}>{item.id} · {item.node_id}</option>)}
        </select><span className="agent-field-note">{pool ? pool.description || `Up to ${pool.cpus} CPU cores and ${pool.memory_mb.toLocaleString()} MiB memory available.` : 'No eligible sandbox pools are currently available. Register capacity with ResourceManager to launch a run.'}</span></label>
        <fieldset className="agent-budget-fields" disabled={saving || !pool}><legend>Run limits</legend>
          <label>CPU cores<input type="number" required min={0.001} max={pool?.cpus ?? 256} step={0.001} value={cpus} onChange={(event) => setCpus(Number(event.target.value))} /></label>
          <label>Memory (MiB)<input type="number" required min={64} max={pool?.memory_mb ?? 1048576} step={1} value={memory} onChange={(event) => setMemory(Number(event.target.value))} /></label>
          <label>Scratch space (MiB)<input type="number" required min={16} max={pool?.scratch_mb ?? 1048576} step={1} value={scratch} onChange={(event) => setScratch(Number(event.target.value))} /></label>
          <label>Time limit (seconds)<input type="number" required min={30} max={pool?.max_timeout_seconds ?? 86400} step={1} value={timeout} onChange={(event) => setTimeoutBudget(Number(event.target.value))} /></label>
          <label>Maximum steps<input type="number" required min={1} max={100} step={1} value={steps} onChange={(event) => setSteps(Number(event.target.value))} /></label>
        </fieldset>
        <p className="agent-field-note">Every run gets a separate sandbox. Runs wait for an available worker and capacity. Submitted papers and solutions are independently verified.</p>
      </div>
      {error && <p className="error-banner agent-form-error" role="alert">{error}</p>}
      <div className="dialog-actions"><button className="button" type="button" onClick={onClose} disabled={saving}>Cancel</button>
        <button className="button button-primary" disabled={saving || !pool || !harnessReady || (role === 'attempt' && (!benchmarks.length || !!benchmarkError))}>{saving ? 'Launching…' : `Launch ${role}`}</button></div>
    </form>
  </dialog>;
}

function PublicationView({ reference, publication, error }: { reference: PublicationRef; publication: Publication | null; error: string }) {
  const paper = reference.kind === 'paper';
  const experimentalReport = publication?.experimental_report || publication?.report;
  return <section className="agent-detail-section" aria-labelledby="agent-submission-heading">
    <div className="section-heading"><h3 id="agent-submission-heading">{paper ? 'Paper submission' : 'Benchmark submission'}</h3>{publication && <Status status={paper ? publication.status : publication.report?.verdict || publication.status} />}</div>
    {reference.status !== 'acknowledged' ? <p className="agent-warning">The submission was started, but its acceptance is unconfirmed. Check the destination before launching a replacement.</p> : <>
      <p className="agent-detail-copy">{paper ? 'Submitted to Library for independent verification.' : 'Submitted to the benchmark for independent scoring.'}</p>
      {reference.id && <p className="agent-reference">Submission <code>{reference.id}</code></p>}
      {error && <p className="error-text" role="alert">{error} {publication ? 'Showing the last available verification result.' : 'Verification details are unavailable.'}</p>}
      {!publication && !error && <p className="muted agent-detail-copy">Loading verification status…</p>}
      {publication && <>
        {(publication.submission?.title || publication.title) && <p className="agent-submission-title">{publication.submission?.title || publication.title}</p>}
        {typeof publication.score === 'number' && Number.isFinite(publication.score) && <p className="agent-verified-score">Verified score <strong>{new Intl.NumberFormat(undefined, { maximumSignificantDigits: 8 }).format(publication.score)}</strong></p>}
        {paper && <LibraryNovelty review={publication.novelty} />}
        {paper ? <div className="agent-library-review" aria-labelledby="independent-library-experiments-title"><div className="agent-library-review-heading"><h4 id="independent-library-experiments-title">Independent experiment verification</h4>{experimentalReport ? <Status status={experimentalReport.verdict} /> : <span className="agent-status">No report</span>}</div>
          {experimentalReport ? <><p className="agent-detail-copy">{experimentalReport.summary}</p>{!!experimentalReport.evidence?.length && <details className="agent-disclosure"><summary>Experiment verification evidence</summary><ul>{experimentalReport.evidence.map((item, index) => <li key={index}>{item}</li>)}</ul></details>}</> : <p className="agent-field-note">No experiment verification report is available for this submission.</p>}
        </div> : publication.report && <p className="agent-detail-copy">{publication.report.summary}</p>}
        {publication.feedback && <p className="agent-detail-copy">{publication.feedback}</p>}
        {(publication.error || publication.blocked_reason) && <p className="error-text">{publication.error || publication.blocked_reason}</p>}
        {!paper && !!publication.report?.evidence?.length && <details className="agent-disclosure"><summary>Verification evidence</summary><ul>{publication.report.evidence.map((item, index) => <li key={index}>{item}</li>)}</ul></details>}
        {publication.submission?.paper_text && <details className="agent-disclosure"><summary>Submitted paper</summary><pre>{publication.submission.paper_text}</pre></details>}
        {Object.hasOwn(publication, 'payload') && <details className="agent-disclosure"><summary>Submitted solution</summary><pre>{JSON.stringify(publication.payload, null, 2)}</pre></details>}
        {publication.base_checkpoint_url?.startsWith('https://github.com/') && <a className="agent-text-link" href={publication.base_checkpoint_url} target="_blank" rel="noreferrer">View base checkpoint ↗</a>}
      </>}
    </>}
  </section>;
}

function RunDetail({ id, active, refreshToken, harnesses, onClose, onUpdated, onSpinOut }: { id: string; active: boolean; refreshToken: number; harnesses: HarnessCatalog | null; onClose: () => void; onUpdated: () => void; onSpinOut: (request: RunRequest) => void }) {
  const [run, setRun] = useState<Run | null>(null);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [publication, setPublication] = useState<Publication | null>(null);
  const [error, setError] = useState('');
  const [eventsError, setEventsError] = useState('');
  const [publicationError, setPublicationError] = useState('');
  const [cancelling, setCancelling] = useState(false);
  const [cancelError, setCancelError] = useState('');
  const [localRefresh, setLocalRefresh] = useState(0);
  const [visibleEvents, setVisibleEvents] = useState(20);
  const eventCursor = useRef(0);
  const eventCache = useRef<AgentEvent[]>([]);
  const cancelController = useRef<AbortController | null>(null);
  const cancelGuard = useRef(false);

  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    const signal = controller.signal;
    let timer: number | undefined;
    async function poll() {
      try {
        const latest = await api<Run>(`/agents/runs/${id}`, { signal });
        if (signal.aborted) return;
        setRun(latest); setError('');
        const loadEvents = async () => {
          try {
            let more = true;
            while (more && !signal.aborted) {
              const page = await api<{ events: AgentEvent[]; next_cursor: number }>(`/agents/runs/${id}/events?after=${eventCursor.current}&limit=200`, { signal });
              if (signal.aborted) return;
              if (page.next_cursor < eventCursor.current || (page.events.length && page.next_cursor <= eventCursor.current)) throw new Error('The activity stream returned an invalid cursor.');
              const seen = new Set(eventCache.current.map((event) => event.sequence));
              eventCache.current = [...eventCache.current, ...page.events.filter((event) => !seen.has(event.sequence))];
              eventCursor.current = page.next_cursor;
              setEvents(eventCache.current);
              more = page.events.length === 200;
            }
            if (!signal.aborted) setEventsError('');
          } catch (failure) { if (!signal.aborted) setEventsError(errorText(failure, 'Unable to refresh activity.')); }
        };
        const loadPublication = async () => {
          if (latest.state.publication?.status !== 'acknowledged') return;
          try {
            const result = await api<Publication>(`/agents/runs/${id}/publication`, { signal });
            if (!signal.aborted) { setPublication(result); setPublicationError(''); }
          } catch (failure) { if (!signal.aborted) setPublicationError(errorText(failure, 'Unable to refresh verification.')); }
        };
        await Promise.all([loadEvents(), loadPublication()]);
      } catch (failure) { if (!signal.aborted) setError(errorText(failure, 'Unable to refresh this run.')); }
      finally { if (!signal.aborted) timer = window.setTimeout(() => { void poll(); }, 4000); }
    }
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); cancelController.current?.abort(); cancelGuard.current = false; setCancelling(false); };
  }, [active, id, refreshToken, localRefresh]);

  async function cancel() {
    if (cancelGuard.current) return;
    cancelGuard.current = true; setCancelling(true); setCancelError('');
    const controller = new AbortController();
    cancelController.current = controller;
    try {
      const cancelled = await api<Run>(`/agents/runs/${id}/cancel`, { method: 'POST', signal: controller.signal });
      if (controller.signal.aborted) return;
      setRun(cancelled); setLocalRefresh((value) => value + 1); onUpdated();
    } catch (failure) { if (!controller.signal.aborted) setCancelError(errorText(failure, 'Unable to cancel this run.')); }
    finally { if (!controller.signal.aborted) { cancelGuard.current = false; setCancelling(false); } }
  }

  const eventExperiments = events.filter((event) => event.kind === 'experiment_result'
    && typeof event.data.id === 'string' && typeof event.data.command === 'string'
    && typeof event.data.output === 'string' && typeof event.data.exit_code === 'number')
    .map((event) => event.data as unknown as Experiment);
  const experiments = [...new Map([...(run?.state.experiments || []), ...eventExperiments]
    .map((experiment) => [experiment.id, experiment])).values()];
  const completedIds = new Set(experiments.map((experiment) => experiment.id));
  const inProgress = events.filter((event) => event.kind === 'experiment_started' && !completedIds.has(String(event.data.id)));
  const novelty = run?.state.novelty;
  const modelProfile = harnesses?.profiles.find((profile) => profile.id === run?.request.model_profile_id);
  const stepCount = run?.state.steps || 0;
  return <section className="agent-run-detail" aria-labelledby="agent-run-heading">
    <div className="section-heading"><h2 id="agent-run-heading">Run details</h2><div className="agent-detail-actions">{run && <button className="button" type="button" onClick={() => onSpinOut({ ...run.request, harness: run.state.runtime?.harness || run.request.harness || 'builtin', model: run.state.runtime?.model || run.request.model || null })}>Spin out new agent</button>}<button className="button" onClick={onClose}>Close details</button></div></div>
    {error && <p className="error-banner" role="alert">{error} {run && 'Showing the last available run state.'}</p>}
    {!run && !error && <p className="agent-loading">Loading run…</p>}
    {run && <>
      <div className="agent-run-title"><span className="agent-role-label">{roleLabel(run.request.role)}</span><Status status={run.status} /></div>
      <p className="agent-run-objective">{run.request.objective}</p>
      <div className="agent-current-phase" aria-live="polite"><strong>{currentPhase(run, events)}</strong></div>
      <div className="agent-step-budget"><span>{stepCount} of {run.request.max_steps} steps used</span><span>{run.request.allocation.timeout_seconds.toLocaleString()} second time limit</span></div>
      <progress className="agent-step-progress" aria-label="Agent step budget used" value={Math.min(stepCount, run.request.max_steps)} max={run.request.max_steps} />
      <p className="agent-field-note">Steps measure the run budget, not the likelihood of finding a solution.</p>
      <dl className="agent-run-meta"><div><dt>Harness</dt><dd>{harnessLabel(run.state.runtime?.harness || run.request.harness)}</dd></div><div><dt>Model</dt><dd>{run.state.runtime?.model || run.request.model || modelProfile?.model || (run.request.model_profile_id ? 'Saved model profile' : 'Runtime default')}</dd></div>{modelProfile && <div><dt>Model profile</dt><dd>{modelProfile.name}</dd></div>}<div><dt>Started</dt><dd>{run.started_at ? date(run.started_at) : 'Waiting to start'}</dd></div><div><dt>Sandbox pool</dt><dd>{run.request.allocation.resource_id}</dd></div><div><dt>Allocation</dt><dd>{run.request.allocation.cpus} CPU · {run.request.allocation.memory_mb.toLocaleString()} MiB</dd></div></dl>
      {(run.status === 'queued' || run.status === 'running') && <div className="agent-cancel-action"><button className="button" disabled={cancelling || run.cancel_requested} onClick={() => { void cancel(); }}>{cancelling || run.cancel_requested ? 'Stopping…' : 'Cancel run'}</button><span className="agent-field-note">{run.cancel_requested ? 'Waiting for the worker to confirm that the sandbox has stopped.' : 'Stops this run and requests sandbox cleanup.'}</span></div>}
      {cancelError && <p className="error-text" role="alert">{cancelError}</p>}
      {run.cleanup_required && run.status !== 'running' && <p className="agent-warning">Sandbox cleanup is still pending. Its capacity remains reserved until termination is confirmed.</p>}
      {run.state.runtime_cleanup_required && <p className="agent-warning">The agent runtime could not confirm cleanup. An operator must check the runtime before starting a replacement.</p>}
      {run.error && run.status !== 'running' && <p className={run.status === 'needs_attention' ? 'agent-warning' : 'agent-detail-copy'}>{failureText(run.error)}</p>}
      {run.result?.summary && <section className="agent-detail-section"><h3>Agent findings</h3><p className="agent-detail-copy">{run.result.summary}</p>{run.result.published === false && <p className="agent-field-note">This run finished without a submission.</p>}</section>}
      {novelty && <section className="agent-detail-section"><div className="section-heading"><h3>Researcher novelty review</h3><span className="agent-outcome">{novelty.approved ? 'Approved for experiments' : 'More research needed'}</span></div><p className="agent-detail-copy">{novelty.rationale}</p><p className="agent-field-note">Compared with {novelty.compared_paper_ids?.length ?? 0} Library papers. This pre-submission review is separate from the Library’s independent novelty decision.</p>{novelty.idea && <details className="agent-disclosure"><summary>Reviewed idea</summary><p>{novelty.idea}</p></details>}</section>}
      <section className="agent-detail-section" aria-labelledby="agent-experiments-heading"><div className="section-heading"><h3 id="agent-experiments-heading">Experiments and tests</h3><span className="agent-count">{experiments.length} recorded</span></div>
        {experiments.length === 0 && inProgress.length === 0 && <p className="muted agent-detail-copy">No sandbox executions recorded yet.</p>}
        {experiments.map((experiment, index) => <details className="agent-experiment" key={experiment.id}>
          <summary><span><strong>{index + 1}. {readable(experiment.purpose)}</strong><span className="agent-experiment-reference">{experiment.id}{experiment.step !== undefined ? ` · step ${experiment.step}` : ''}</span></span><span className={experiment.exit_code === 0 ? 'agent-experiment-success' : 'agent-experiment-failure'}>{experiment.exit_code === 0 ? 'Passed' : `Exit ${experiment.exit_code}`}{experiment.truncated ? ' · output truncated' : ''}</span></summary>
          <div className="agent-experiment-body"><h4>Command</h4><pre>{experiment.command}</pre><h4>Output</h4><pre>{experiment.output || '(No output)'}</pre><p className="agent-field-note">Exit code {experiment.exit_code} · {experiment.timeout_seconds} second command limit</p></div>
        </details>)}
        {inProgress.map((event) => <div className="agent-experiment-pending" key={event.sequence}><strong>{readable(String(event.data.purpose || 'Experiment'))} {run.status === 'running' ? 'in progress' : 'has no recorded result'}</strong><pre>{String(event.data.command || '')}</pre></div>)}
      </section>
      {run.state.publication && <PublicationView reference={run.state.publication} publication={publication} error={publicationError} />}
      <section className="agent-detail-section" aria-labelledby="agent-activity-heading"><div className="section-heading"><h3 id="agent-activity-heading">Activity</h3><span className="agent-count">{events.length} events</span></div>
        {eventsError && <p className="error-text" role="alert">{eventsError} Showing activity received so far.</p>}
        {events.length === 0 ? <p className="muted agent-detail-copy">Activity appears when the worker starts this run.</p> : <ol className="agent-timeline">{[...events].reverse().slice(0, visibleEvents).map((event) => <li key={event.sequence}><time dateTime={new Date(event.created_at * 1000).toISOString()}>{date(event.created_at)}</time><div><span>{eventLabel(event)}</span>{typeof event.data.rationale === 'string' && <details className="agent-disclosure"><summary>Review details</summary><p>{event.data.rationale}</p></details>}</div></li>)}</ol>}
        {events.length > visibleEvents && <button className="button agent-earlier-activity" onClick={() => setVisibleEvents((value) => value + 40)}>Show earlier activity · {events.length - visibleEvents} remaining</button>}
      </section>
      <details className="agent-disclosure agent-run-identifiers"><summary>Run and sandbox references</summary><dl><dt>Run</dt><dd><code>{run.id}</code></dd><dt>Compute job</dt><dd><code>{run.compute_job_id || 'Not allocated'}</code></dd><dt>Sandbox</dt><dd><code>{run.sandbox_id || 'Not allocated'}</code></dd></dl></details>
    </>}
  </section>;
}

export default function Agents({ active }: { active: boolean }) {
  const [configuration, setConfiguration] = useState<Configuration | null>(null);
  const [workers, setWorkers] = useState<WorkerStatus | null>(null);
  const [workerError, setWorkerError] = useState('');
  const [runs, setRuns] = useState<Run[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [pools, setPools] = useState<Pool[]>([]);
  const [benchmarks, setBenchmarks] = useState<BenchmarkChoice[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');
  const [poolError, setPoolError] = useState('');
  const [benchmarkError, setBenchmarkError] = useState('');
  const [poolsReady, setPoolsReady] = useState(false);
  const [harnesses, setHarnesses] = useState<HarnessCatalog | null>(null);
  const [harnessError, setHarnessError] = useState('');
  const [harnessesReady, setHarnessesReady] = useState(false);
  const [historyReady, setHistoryReady] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [launchRole, setLaunchRole] = useState<Role | null>(null);
  const [launchSeed, setLaunchSeed] = useState<RunRequest | null>(null);
  const [refreshToken, setRefreshToken] = useState(0);
  const [notice, setNotice] = useState('');
  const pendingLaunch = useRef<PendingLaunch | null>(null);

  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    const signal = controller.signal;
    let timer: number | undefined;
    setRefreshing(true);
    async function poll() {
      try {
        const config = await api<Configuration>('/agents/config', { signal });
        if (signal.aborted) return;
        setConfiguration(config);
        if (!config.configured) { setError(''); setPoolsReady(false); return; }
        const [history, capacity, choices, workerStatus, harnessCatalog] = await Promise.allSettled([
          api<{ runs: Run[]; total: number }>('/agents/runs?limit=' + PAGE_SIZE + '&offset=' + offset, { signal }),
          api<{ resources: Pool[] }>('/agents/resources', { signal }),
          api<{ benchmarks: BenchmarkChoice[] }>('/benchmarks', { signal }),
          api<WorkerStatus>('/agents/status', { signal }),
          api<HarnessCatalog>('/agents/settings/harnesses', { signal }),
        ]);
        if (signal.aborted) return;
        if (history.status === 'fulfilled') { setRuns(history.value.runs); setTotal(history.value.total); setHistoryReady(true); setError(''); }
        else setError(errorText(history.reason, 'Unable to refresh agent history.'));
        if (capacity.status === 'fulfilled') { setPools(validPools(capacity.value.resources)); setPoolsReady(true); setPoolError(''); }
        else { setPoolError(errorText(capacity.reason, 'Unable to load sandbox pools.')); setPoolsReady(false); }
        if (choices.status === 'fulfilled') { setBenchmarks(choices.value.benchmarks); setBenchmarkError(''); }
        else setBenchmarkError(errorText(choices.reason, 'Unable to load benchmarks.'));
        if (workerStatus.status === 'fulfilled') { setWorkers(workerStatus.value); setWorkerError(''); }
        else setWorkerError(errorText(workerStatus.reason, 'Unable to check worker availability.'));
        if (harnessCatalog.status === 'fulfilled') { setHarnesses(harnessCatalog.value); setHarnessesReady(true); setHarnessError(''); }
        else { setHarnessError('Unable to check installed harnesses. Refresh after checking the backend.'); setHarnessesReady(false); }
      } catch (failure) { if (!signal.aborted) { setError(errorText(failure, 'Unable to connect to the agent service.')); setPoolsReady(false); } }
      finally {
        if (!signal.aborted) { setLoading(false); setRefreshing(false); timer = window.setTimeout(() => { void poll(); }, 5000); }
      }
    }
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [active, offset, refreshToken]);

  const connected = configuration?.configured === true;
  const signInUrl = configuration?.sign_in_url?.startsWith('/signin-with-chatgpt?') ? configuration.sign_in_url : null;
  function refresh() { setRefreshing(true); setRefreshToken((value) => value + 1); }
  function newAgent(role: Role, seed: RunRequest | null = null) {
    if (seed) pendingLaunch.current = null; // An independent run must never replay its source.
    setLaunchSeed(seed); setLaunchRole(role);
  }
  return <div className="agents">
    <header className="agents-header"><div><h1>Agents</h1><p>Explore new research and turn existing ideas into benchmark attempts.</p></div><div className="agents-header-actions"><button className="button" disabled={refreshing} onClick={refresh}>{refreshing ? 'Refreshing…' : 'Refresh'}</button><button className="button" disabled={!connected} onClick={() => newAgent('attempt')}>New attempt</button><button className="button button-primary" disabled={!connected} onClick={() => newAgent('researcher')}>New agent</button></div></header>
    {notice && <p className="agent-success" role="status">{notice}</p>}
    {configuration && !connected && <div className="agent-connection-note"><h2>{configuration.requires_sign_in ? 'Sign in to manage agents' : 'Agent service is not connected'}</h2><p>{configuration.message || 'Connect the Agent Harness to launch runs and see their progress.'}</p>{signInUrl && <a className="button button-primary" href={signInUrl} target="_top">Sign in to manage agents</a>}</div>}
    {error && <p className="error-banner" role="alert">{error} {historyReady && connected && 'Showing the last available run history.'}</p>}
    {connected && poolError && <p className="error-banner" role="alert">{poolError} Launching is unavailable until sandbox capacity can be checked.</p>}
    {connected && poolsReady && !pools.length && <p className="agent-warning">No eligible sandbox pools are currently available. Existing runs remain visible; new runs need registered capacity in ResourceManager.</p>}
    {connected && harnessError && <p className="error-banner" role="alert">{harnessError}</p>}
    {connected && harnessesReady && !harnesses?.harnesses.some((item) => item.available) && <p className="agent-warning">No agent harness is available. Open Connections to check Codex sign-in, external runtimes, and model profiles.</p>}
    {connected && <div className="agent-worker-status" aria-live="polite">{workerError ? <span>{workerError}</span> : workers ? <><span className={workers.workers_online ? 'agent-worker-online' : ''}>{workers.workers_online} {workers.workers_online === 1 ? 'worker' : 'workers'} online</span><span>{workers.running_runs} running · {workers.queued_runs} queued</span>{workers.workers_online === 0 && <span>Queued agents will wait until a worker connects.</span>}</> : <span>Checking worker availability…</span>}</div>}
    {loading && <p className="agent-loading">Connecting to agents…</p>}
    {connected && <div className={selectedId ? 'agents-workspace has-selection' : 'agents-workspace'}>
      <section className="agent-history" aria-labelledby="agent-history-heading"><div className="section-heading"><h2 id="agent-history-heading">Runs <span className="agent-count">{total}</span></h2><span className="agent-field-note">Refreshes automatically</span></div>
        {historyReady && !runs.length && !error ? <div className="agent-empty"><h3>No agent runs yet</h3><p>Launch a researcher to develop an idea, or an attempt agent to work on a benchmark. Each run has its own sandbox.</p></div> : <ul className="agent-run-list">{runs.map((run) => <li key={run.id}><button className={selectedId === run.id ? 'agent-run-choice is-selected' : 'agent-run-choice'} onClick={() => { setSelectedId(run.id); setNotice(''); }} aria-pressed={selectedId === run.id}><span className="agent-run-choice-heading"><strong>{roleLabel(run.request.role)}</strong><Status status={run.status} /></span><span className="agent-run-choice-objective">{run.request.objective}</span><span className="agent-run-choice-meta"><time dateTime={new Date(run.created_at * 1000).toISOString()}>{date(run.created_at)}</time><span>{run.state.steps || 0}/{run.request.max_steps} steps</span></span></button></li>)}</ul>}
        {total > PAGE_SIZE && <div className="agent-pagination"><span>{offset + 1}–{Math.min(offset + PAGE_SIZE, total)} of {total}</span><div><button className="button" disabled={offset === 0 || refreshing} onClick={() => setOffset((value) => Math.max(0, value - PAGE_SIZE))}>Previous</button><button className="button" disabled={offset + PAGE_SIZE >= total || refreshing} onClick={() => setOffset((value) => value + PAGE_SIZE)}>Next</button></div></div>}
      </section>
      {selectedId ? <RunDetail key={selectedId} id={selectedId} active={active && connected} refreshToken={refreshToken} harnesses={harnesses} onClose={() => setSelectedId(null)} onUpdated={refresh} onSpinOut={(request) => newAgent(request.role, request)} /> : historyReady && !!runs.length && <div className="agent-selection-note"><h2>Select a run</h2><p>Inspect its progress, experiment commands and outputs, and independent verification results.</p></div>}
    </div>}
    {launchRole && <LaunchAgent role={launchRole} initialRequest={launchSeed} pools={pools} benchmarks={benchmarks} benchmarkError={benchmarkError} harnesses={harnesses || pendingHarnessCatalog} availabilityChecked={connected && poolsReady && harnessesReady} pending={pendingLaunch} onClose={() => setLaunchRole(null)} onCreated={(run) => {
      setLaunchRole(null); setSelectedId(run.id); setOffset(0); setNotice(`${roleLabel(run.request.role)} queued. It will start when a worker and sandbox capacity are available.`); refresh();
    }} />}
  </div>;
}
