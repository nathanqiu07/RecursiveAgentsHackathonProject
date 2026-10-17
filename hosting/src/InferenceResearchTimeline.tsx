import { useEffect, useState } from 'react';
import type { LibraryFeed, LibraryIdea } from '../shared/library';

// The current Nemotron / A5000 benchmark. Historical optimization runs have separate charts.
const BENCHMARK = 'b9d6b8c7-3e72-46e9-a35b-9fd17b613426';

// Display labels summarize the recorded proposals; the original text stays in each disclosure.
function shortTitle(idea: LibraryIdea) {
  const hypothesis = idea.hypothesis.toLowerCase();
  if (hypothesis.includes('grouped w4a16')) return 'Tile-based expert kernels';
  if (hypothesis.includes('event-safe')) return 'Safe workspace sharing';
  if (hypothesis.includes('phase-lifetime')) return 'One workspace across layers';
  if (hypothesis.includes('chunked prefill')) return 'Chunked prefill, shared memory';
  if (hypothesis.includes('full remaining-generation')) return 'Reserve memory before decoding';
  if (hypothesis.includes('single bounded gpu workspace')) return 'Shared prefill and expert memory';
  const words = idea.title.split(/\s+/);
  return words.slice(0, 7).join(' ') + (words.length > 7 ? '…' : '');
}

export default function InferenceResearchTimeline({ active }: { active: boolean }) {
  const [feed, setFeed] = useState<LibraryFeed | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    async function load() {
      try {
        const pages: LibraryFeed[] = [];
        let offset = 0;
        do {
          const response = await fetch(`/api/library/feed?limit=100&offset=${offset}`, { signal: controller.signal, cache: 'no-store' });
          if (!response.ok) throw new Error('Library unavailable');
          const page = await response.json() as LibraryFeed;
          pages.push(page);
          offset += 100;
          if (!page.pagination.has_more) break;
        } while (offset < 2000);
        if (!controller.signal.aborted) {
          setFeed({ ...pages[0], ideas: pages.flatMap(p => p.ideas), papers: pages.flatMap(p => p.papers),
            drafts_available: pages.every(p => p.drafts_available), verification_available: pages.every(p => p.verification_available) });
          setError(false);
        }
      } catch { if (!controller.signal.aborted) setError(true); }
    }
    void load();
    const timer = window.setInterval(() => void load(), 60000);
    return () => { controller.abort(); window.clearInterval(timer); };
  }, [active]);
  const ideas = (feed?.ideas ?? []).filter(i => i.benchmark_id === BENCHMARK);
  const runIds = new Set(ideas.map(i => i.run_id));
  const papers = (feed?.papers ?? []).filter(p => p.run_id && runIds.has(p.run_id));
  const groups = new Map<number | null, LibraryIdea[]>();
  for (const idea of ideas) {
    const generation = idea.recursive_generation ?? null;
    groups.set(generation, [...(groups.get(generation) ?? []), idea]);
  }
  return <section className="home-research" aria-labelledby="research-title">
    <div className="home-section-heading"><div><p className="home-eyebrow">LIVE AGENT RESEARCH / INFERENCE</p><h2 id="research-title">From hypothesis to paper.</h2></div><p>Nemotron Lightning · RTX A5000 · NVFP4<br />Current benchmark research, grouped by recorded recursive generation.</p></div>
    {error && <p role="status">Research could not refresh. {feed ? 'Showing the last available records.' : 'Library status is unavailable.'}</p>}
    {!feed && !error && <p>Loading agent research…</p>}
    {feed && <>
      <div className="research-summary"><strong>{feed.verification_available && feed.drafts_available ? papers.length : '—'} submitted papers</strong><span>{feed.drafts_available ? ideas.length : '—'} recorded proposals</span></div>
      {(!feed.drafts_available || !feed.verification_available) && <p role="status">Some research sources are unavailable; this timeline may be incomplete.</p>}
      {papers.length === 0 && feed.verification_available && feed.drafts_available && <p>No submitted papers are linked to the current inference proposals yet. Proposals below are research in progress, not measured improvements.</p>}
      <div className="research-chart" role="region" aria-label="Inference research by recursive iteration" tabIndex={0}>
        <div className="research-chart-axis">Recursive iteration → <span>Open a title for the full research record</span></div>
        <div className="research-chart-columns">
          {[...groups].sort(([a], [b]) => (a ?? Infinity) - (b ?? Infinity)).map(([generation, entries]) => <section className="research-iteration" key={generation ?? 'unknown'}>
            <h3><span className="research-iteration-dot" />{generation == null ? 'Iteration unrecorded' : `R${generation}`}</h3>
            <div className="research-iteration-items">{entries.map(idea => {
              const publications = papers.filter(p => p.run_id === idea.run_id);
              return <details className="research-item" key={idea.id}>
                <summary><span className="research-item-title">{shortTitle(idea)}</span><span className="research-item-status">{publications.length ? 'Paper available' : 'Proposal'} · {idea.run_id.slice(0, 8)}</span></summary>
                <div className="research-item-body"><strong>Original research</strong><p>{idea.hypothesis}</p>
                  <small>Run {idea.run_status} · {idea.experiment_count} experiment records (not verified results)</small>
                  {publications.map(paper => <details key={paper.id}><summary>Paper: {paper.title} · {paper.verdict ?? paper.status.replaceAll('_', ' ')}</summary><p>{paper.summary}</p><pre>{paper.paper_text}</pre></details>)}
                  <p>{idea.novelty_difference}</p>{idea.experiment_plan.map((plan, i) => <p key={i}><strong>{plan.metric}</strong> — {plan.experiment}<br />Baseline: {plan.baseline}<br />Success criterion: {plan.success_criterion}</p>)}
                  <small>Run ID: {idea.run_id}<br />Recorded: {idea.created_at || 'Date unavailable'}</small>
                </div>
              </details>;
            })}</div>
          </section>)}
        </div>
      </div>
      {groups.has(null) && <p className="home-source">Iteration is not recorded for these runs. They remain together until recursive lineage is available.</p>}
      <p className="home-source">Source: live agent runs and Library submissions · Checked {feed.checked_at}. External reference papers are not counted as agent publications.</p>
    </>}
  </section>;
}
