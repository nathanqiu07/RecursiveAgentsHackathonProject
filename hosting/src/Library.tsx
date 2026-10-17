import { useEffect, useRef, useState } from 'react';
import type { LibraryFeed, LibraryIdea, LibraryPaper } from '../shared/library';
import './library.css';

type Filter = 'all' | 'unverified' | 'verifying' | 'verified';
const dateLabel = (value: string) => {
  const date = new Date(value);
  return Number.isFinite(date.getTime()) ? date.toLocaleString() : 'Time unavailable';
};
const safeLink = (value: string | null | undefined) => {
  try { const url = new URL(value || ''); return ['https:', 'http:'].includes(url.protocol) && !url.username && !url.password ? url.href : null; }
  catch { return null; }
};
const activePaper = (paper: LibraryPaper) => ['planning', 'verifying'].includes(paper.status);
const verifiedPaper = (paper: LibraryPaper) => paper.status === 'completed' && paper.verdict === 'verified';
function paperLabel(paper: LibraryPaper) {
  if (paper.status === 'planning') return 'Preparing verification';
  if (paper.status === 'verifying') return paper.compute_job_status === 'queued' ? 'Verification queued' : 'Verifying';
  if (verifiedPaper(paper)) return 'Verified';
  if (paper.status === 'pending_verification') return 'Awaiting verification';
  if (paper.status === 'blocked') return 'Verification blocked';
  if (paper.status === 'failed') return 'Verification failed';
  if (paper.verdict === 'not_verified') return 'Not verified';
  if (paper.verdict === 'inconclusive') return 'Inconclusive';
  return 'Unverified';
}
function ideaActivity(idea: LibraryIdea) {
  if (idea.blocker) return 'Experiments blocked';
  if (idea.run_status === 'running') return idea.experiment_count ? 'Experiments in progress' : 'Researching';
  if (idea.run_status === 'queued') return 'Research queued';
  if (idea.run_status === 'cancelled') return 'Research stopped';
  if (idea.run_status === 'failed') return 'Research interrupted';
  return idea.experiment_count ? 'Experiments recorded' : 'Awaiting experiments';
}

export default function Library({ active = true }: { active?: boolean }) {
  const [feed, setFeed] = useState<LibraryFeed | null>(null);
  const [filter, setFilter] = useState<Filter>('all');
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [refreshing, setRefreshing] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const dialog = useRef<HTMLDialogElement>(null);
  const focusedBeforeDialog = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!active) return;
    let stopped = false;
    let busy = false;
    const controller = new AbortController();
    async function load() {
      if (busy) return;
      busy = true;
      setRefreshing(true);
      try {
        const response = await fetch(`/api/library/feed?limit=100&offset=${offset}`, { cache: 'no-store', signal: controller.signal });
        const body = await response.json();
        if (!response.ok) throw new Error(body.error || 'Unable to load Library.');
        if (!stopped) { setFeed(body as LibraryFeed); setError(''); }
      } catch (failure) {
        if (!stopped) setError(failure instanceof Error ? failure.message : 'Unable to load Library.');
      } finally { busy = false; if (!stopped) { setRefreshing(false); setLoading(false); } }
    }
    setLoading(true);
    void load();
    const timer = window.setInterval(() => { void load(); }, 5000);
    return () => { stopped = true; controller.abort(); window.clearInterval(timer); };
  }, [active, offset, refresh]);

  const ideas = feed?.ideas || [];
  const papers = feed?.papers || [];
  const verifying = feed?.active_verifications || [];
  const drafts = feed?.counts.unverified || 0;
  const verified = feed?.counts.verified || 0;
  const verificationAvailable = feed?.verification_available === true;
  const entries = [
    ...ideas.map((idea) => ({ key: `idea:${idea.id}`, kind: 'idea' as const, value: idea })),
    ...papers.map((paper) => ({ key: `paper:${paper.id}`, kind: 'paper' as const, value: paper })),
  ].sort((a, b) => Date.parse(b.value.created_at) - Date.parse(a.value.created_at));
  const shown = entries.filter((entry) => filter === 'all' || (entry.kind === 'idea' ? filter === 'unverified' : filter === 'verified' ? verifiedPaper(entry.value) : filter === 'verifying' ? activePaper(entry.value) : !verifiedPaper(entry.value) && !activePaper(entry.value)));
  const detail = entries.find((entry) => entry.key === selected) || verifying.map((paper) => ({ key: `paper:${paper.id}`, kind: 'paper' as const, value: paper })).find((entry) => entry.key === selected);

  useEffect(() => {
    if (selected && detail && active) {
      if (!dialog.current?.open) {
        focusedBeforeDialog.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
        dialog.current?.showModal();
      }
    } else if (dialog.current?.open) dialog.current.close();
  }, [selected, !!detail, active]);
  function close() { dialog.current?.close(); setSelected(null); focusedBeforeDialog.current?.focus(); }

  return <div className="library-page">
    <header className="library-header"><div><p className="library-eyebrow">Research workspace</p><h1>Library</h1><p>Original ideas, experimental evidence, and independent verification.</p></div><button className="button" onClick={() => setRefresh((value) => value + 1)} disabled={refreshing}>{refreshing ? 'Refreshing…' : 'Refresh'}</button></header>
    {error && <p role="alert" className="library-warning">{error}{feed ? ' Showing the last loaded results.' : ''}</p>}
    {!!feed?.warnings.length && <div role="status" className="library-warning">{feed.warnings.map((warning) => <p key={warning}>{warning}</p>)}</div>}
    <div className="library-summary" aria-label="Library counts">
      <div><span>Unverified ideas & papers</span><strong>{feed?.drafts_available ? drafts : '—'}</strong></div>
      <div><span>In verification</span><strong>{verificationAvailable ? feed?.counts.in_verification : '—'}</strong></div>
      <div><span>Verified papers</span><strong>{verificationAvailable ? verified : '—'}</strong></div>
    </div>
    <div className="library-layout"><section className="library-collection" aria-label="Ideas and papers">
      <div className="library-filters" role="group" aria-label="Filter Library">
        {(['all', 'unverified', 'verifying', 'verified'] as Filter[]).map((item) => <button key={item} type="button" aria-pressed={filter === item} onClick={() => setFilter(item)}>{({ all: 'All research', unverified: 'Unverified', verifying: 'In verification', verified: 'Verified' })[item]}</button>)}
      </div>
      <p className="library-caption">Draft ideas remain unverified until a paper has passed independent checks.</p>
      {loading && !feed ? <p className="library-empty" role="status">Loading ideas and verification activity…</p> : shown.length ? <div className="library-cards">{shown.map((entry) => {
        const isIdea = entry.kind === 'idea';
        return <article className="library-card" key={entry.key}>
          <div className="library-card-meta"><span className={`library-badge ${isIdea ? 'library-badge-draft' : verifiedPaper(entry.value) ? 'library-badge-verified' : activePaper(entry.value) ? 'library-badge-active' : 'library-badge-draft'}`}>{isIdea ? 'Unverified draft' : paperLabel(entry.value)}</span><span>{isIdea ? 'Original idea' : 'Research paper'}</span></div>
          <button className="library-title" onClick={() => setSelected(entry.key)}>{entry.value.title}</button>
          <p className="library-card-description">{isIdea ? entry.value.hypothesis : entry.value.summary || 'Submitted for independent review. Open the paper to read its method and evidence.'}</p>
          <div className="library-card-footer"><span>{isIdea ? ideaActivity(entry.value) : entry.value.node_id ? `Node · ${entry.value.node_id}` : paperLabel(entry.value)}</span><span>{isIdea ? `Agent ${entry.value.run_id.slice(0, 8)}` : dateLabel(entry.value.created_at)}</span></div>
        </article>;
      })}</div> : <div className="library-empty"><h3>{filter === 'verified' ? 'No verified papers yet' : filter === 'verifying' ? 'No papers in verification' : filter === 'unverified' ? 'No unverified ideas on this page' : 'No research ideas yet'}</h3><p>{filter === 'verified' ? 'Papers appear here after an independent verifier confirms their evidence.' : filter === 'verifying' ? 'Submitted papers appear here when a verifier starts preparing or checking them.' : 'Ideas appear automatically when research agents record their hypotheses.'}</p></div>}
      {feed && (feed.pagination.has_more || offset > 0) && <div className="library-pagination"><button className="button" disabled={!offset || refreshing} onClick={() => { setOffset(Math.max(0, offset - 100)); setSelected(null); }}>Previous</button><span>{offset + 1}–{Math.min(offset + feed.pagination.limit, feed.pagination.total)} of {feed.pagination.total}</span><button className="button" disabled={!feed.pagination.has_more || refreshing} onClick={() => { setOffset(offset + 100); setSelected(null); }}>Next</button></div>}
    </section><aside className="library-verifier" aria-label="Current verifier activity">
      <div className="library-verifier-heading"><span className={`library-live-dot ${verifying.length ? 'library-live-dot-active' : ''}`} /><h2>Verifier now</h2></div>
      {!feed ? <p>Checking verifier activity…</p> : !verificationAvailable ? <><strong className="library-verifier-idle">Verification status unavailable</strong><p>The Library service could not be fully refreshed. Current verifier activity is unknown; check the connection or refresh to try again.</p></> : verifying.length ? <><p>{verifying.length} {verifying.length === 1 ? 'paper is' : 'papers are'} being checked.</p><div className="library-active-list">{verifying.map((paper) => <article key={paper.id}><span className="library-badge library-badge-active">{paperLabel(paper)}</span><h3>{paper.title}</h3>{paper.node_id && <p>Node · {paper.node_id}</p>}{paper.compute_job_status && <p>Compute job · {paper.compute_job_status}</p>}{paper.started_at && <p>Started {dateLabel(paper.started_at)}</p>}<button className="library-text-button" onClick={() => setSelected(`paper:${paper.id}`)}>View paper →</button></article>)}</div></> : <><strong className="library-verifier-idle">No verification running</strong><p>Draft ideas are waiting for experiments and paper submission. This panel shows the paper and node when independent verification starts.</p></>}
      <div className="library-verifier-note">An agent developing or testing an idea is shown separately from the independent verifier.</div>
      {feed && <p className="library-updated">Updated {dateLabel(feed.checked_at)}</p>}
    </aside></div>
    <dialog ref={dialog} className="library-dialog" onCancel={(event) => { event.preventDefault(); close(); }} onClose={() => setSelected(null)} aria-labelledby="library-detail-title">
      {detail && <><header><span className="library-badge">{detail.kind === 'idea' ? 'Unverified draft' : paperLabel(detail.value)}</span><button className="icon-button" onClick={close} aria-label="Close research details">×</button></header><h2 id="library-detail-title">{detail.value.title}</h2>
        {detail.kind === 'idea' ? <><p className="library-detail-lead">{detail.value.hypothesis}</p><dl className="library-detail-meta"><div><dt>Research agent</dt><dd>{detail.value.run_id}</dd></div><div><dt>Progress</dt><dd>{ideaActivity(detail.value)} · {detail.value.experiment_count} experiments recorded</dd></div></dl>
          {detail.value.blocker && <p className="library-warning">{detail.value.blocker}</p>}
          {detail.value.novelty_difference && <section><h3>Proposed contribution</h3><p>{detail.value.novelty_difference}</p></section>}
          {!!detail.value.experiment_plan.length && <section><h3>Experiment plan</h3>{detail.value.experiment_plan.map((plan, index) => <dl key={index} className="library-plan">{Object.entries(plan).map(([key, value]) => <div key={key}><dt>{({ baseline: 'Baseline', experiment: 'Experiment', metric: 'Metric', success_criterion: 'Success criterion' } as Record<string, string>)[key] || key}</dt><dd>{value}</dd></div>)}</dl>)}</section>}
          {!!detail.value.source_urls.length && <section><h3>References</h3><ul>{detail.value.source_urls.map((url) => safeLink(url) && <li key={url}><a href={safeLink(url)!} target="_blank" rel="noreferrer">{url}</a></li>)}</ul></section>}
        </> : <>{detail.value.summary && <p className="library-detail-lead">{detail.value.summary}</p>}{detail.value.blocked_reason && <p className="library-warning">{detail.value.blocked_reason}</p>}<dl className="library-detail-meta"><div><dt>Verification status</dt><dd>{paperLabel(detail.value)}</dd></div>{detail.value.node_id && <div><dt>Node</dt><dd>{detail.value.node_id}</dd></div>}{detail.value.compute_job_id && <div><dt>Compute job</dt><dd>{detail.value.compute_job_id} · {detail.value.compute_job_status || 'Awaiting status'}</dd></div>}</dl>{safeLink(detail.value.paper_url) && <a href={safeLink(detail.value.paper_url)!} target="_blank" rel="noreferrer">Open paper source ↗</a>}<section><h3>Paper</h3>{detail.value.paper_text_truncated && <p className="library-warning">Showing the first part of this paper. The full manuscript remains stored in the Library.</p>}<div className="library-paper-text">{detail.value.paper_text || 'The paper is available at its source URL.'}</div></section></>}
      </>}
    </dialog>
  </div>;
}
