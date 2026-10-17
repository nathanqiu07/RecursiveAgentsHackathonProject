import { useEffect, useState } from 'react';
import type { LibraryFeed, LibraryIdea } from '../shared/library';

// The current Nemotron / A5000 benchmark. Historical optimization runs have separate charts.
const BENCHMARK = 'b9d6b8c7-3e72-46e9-a35b-9fd17b613426';

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
      {[...groups].sort(([a], [b]) => (a ?? Infinity) - (b ?? Infinity)).map(([generation, entries]) => <div className="research-generation" key={generation ?? 'unknown'}>
        <h3>{generation == null ? 'Generation not recorded' : `R${generation} · Recursive generation`}</h3>
        {generation == null && <p>These agents have no recorded recursive lineage. Independent runs are not successive generations.</p>}
        <div className="research-cards">{entries.map(idea => {
          const publications = papers.filter(p => p.run_id === idea.run_id);
          return <article className="research-card" key={idea.id}>
            <div className="research-card-meta"><span>PROPOSAL · UNVERIFIED</span><span>Agent {idea.run_id.slice(0, 8)}</span></div>
            <h4>{idea.title}</h4><p>{idea.hypothesis}</p>
            <small>Run {idea.run_status} · {idea.experiment_count} experiment records (not verified results)</small>
            {publications.map(paper => <details key={paper.id}><summary>Paper: {paper.title} · {paper.verdict ?? paper.status.replaceAll('_', ' ')}</summary><p>{paper.summary}</p><pre>{paper.paper_text}</pre></details>)}
            <details><summary>Research plan and provenance</summary><p>{idea.novelty_difference}</p>{idea.experiment_plan.map((plan, i) => <p key={i}><strong>{plan.metric}</strong> — {plan.experiment}<br />Baseline: {plan.baseline}<br />Success criterion: {plan.success_criterion}</p>)}<small>Run ID: {idea.run_id}<br />Recorded: {idea.created_at || 'Date unavailable'}</small></details>
          </article>;
        })}</div>
      </div>)}
      <p className="home-source">Source: live agent runs and Library submissions · Checked {feed.checked_at}. External reference papers are not counted as agent publications.</p>
    </>}
  </section>;
}
