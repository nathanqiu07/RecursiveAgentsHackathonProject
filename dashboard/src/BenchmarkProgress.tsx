import { useState } from 'react';
export type ProgressRun = {
  name: string; layer: string; benchmark?: string; status?: string;
  climb?: { baseline?: number | null; candidates?: { gen: number; accepted: boolean; cost?: number | null; parent_cost?: number | null; score?: number | null; change?: string }[]; generations?: { gen: number; start: number; end: number }[] };
};
const finite = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n);
function Chart({ runs, inference }: { runs: ProgressRun[]; inference: boolean }) {
  const options = runs.filter(r => inference ? r.layer === 'inference' && r.climb?.candidates?.length : r.benchmark === 'QAPLIB' && r.climb?.generations?.length);
  const [chosen, setChosen] = useState('');
  const preferred = inference ? 'inference-opt-2' : 'qaplib-research-1';
  const run = options.find(r => r.name === chosen) ?? options.find(r => r.name === preferred) ?? options[options.length - 1];
  const candidates = run?.climb?.candidates ?? [];
  const rounds = (run?.climb?.generations ?? []).map(g => {
    const changes = candidates.filter(c => c.gen === g.gen);
    const accepted = changes.filter(c => c.accepted);
    const start = inference ? changes.find(c => finite(c.parent_cost))?.parent_cost : g.start;
    const end = inference ? accepted[accepted.length - 1]?.cost ?? start : g.end;
    return {gen:g.gen,start,end,accepted:accepted.length,changes};
  }).filter((g): g is typeof g & {start:number;end:number} => finite(g.start) && finite(g.end));
  const values = rounds.flatMap(g => [g.start,g.end]);
  const high = Math.max(...values,0) * 1.12 || 1;
  const x = (i:number,after:boolean) => 80 + (i+.5)*440/Math.max(rounds.length,1) + (after?24:-24);
  const y = (v:number) => 235-v/high*180;
  const fmt = (n:number) => inference ? '$'+n.toFixed(5) : n.toFixed(4);
  return <article className="progress-card">
    <h3>{inference ? 'Inference benchmark' : 'QAPLIB benchmark'}</h3>
    <p>{inference ? 'Measured dev cost (USD) · lower is better' : 'Measured dev score · higher is better'}</p>
    <label>Recorded run <select value={run?.name ?? ''} onChange={e=>setChosen(e.target.value)} disabled={!options.length}>{options.map(r=><option key={r.name} value={r.name}>{r.name} · {r.status}</option>)}</select></label>
    {!rounds.length ? <p>No measured recursive generations available.</p> : <>
      <svg viewBox="0 0 600 300" role="img" aria-label={`${inference?'Inference':'QAPLIB'} measured baseline and retained results by recursive generation`}>
        {[0,1,2,3].map(i=>{const v=high*i/3;return <g key={i}><line x1="80" x2="555" y1={y(v)} y2={y(v)} stroke="#ffffff18"/><text x="72" y={y(v)+4} textAnchor="end" fill="#b1abc7" fontSize="11">{inference?'$'+v.toFixed(3):v.toFixed(2)}</text></g>})}
        {rounds.map((g,i)=><g key={g.gen}><line x1={x(i,false)} x2={x(i,true)} y1={y(g.start)} y2={y(g.end)} stroke="#65ecdc" strokeWidth="2"/><circle cx={x(i,false)} cy={y(g.start)} r="6" fill="#1b1233" stroke="#c4b9d9" strokeWidth="2"><title>R{g.gen} baseline: {fmt(g.start)}</title></circle><circle cx={x(i,true)} cy={y(g.end)} r="7" fill="#65ecdc"><title>R{g.gen} retained result: {fmt(g.end)}</title></circle><text x={x(i,false)-9} y={y(g.start)-13} textAnchor="end" fill="#c4b9d9" fontSize="12">{fmt(g.start)}</text><text x={x(i,true)+9} y={y(g.end)+21} fill="#65ecdc" fontSize="12">{fmt(g.end)}</text><text x={(x(i,false)+x(i,true))/2} y="263" textAnchor="middle" fill="#f7f4ff" fontSize="14">R{g.gen}</text></g>)}
        <text x="320" y="290" textAnchor="middle" fill="#b1abc7" fontSize="11">Recursive generation (recorded, zero-based)</text>
      </svg>
      <p className="progress-key">○ Generation baseline <span>● Retained result</span></p>
      <div className="progress-rounds">{rounds.map(g=><details key={g.gen}><summary>R{g.gen}: {g.accepted} of {g.changes.length} changes kept · {fmt(g.start)} → {fmt(g.end)}</summary><ul>{g.changes.map((c,i)=><li key={i}>{c.accepted?'Kept':'Rejected'}: {c.change}{finite(c.score) ? ` · dev score ${c.score.toFixed(4)}` : ''}</li>)}</ul></details>)}</div>
    </>}
    <p className="progress-source">{inference ? 'Crusoe inference optimization run; not the pending A5000 Nemotron baseline.' : 'Each generation uses its own recorded baseline; cross-generation changes are not all attributable to accepted edits.'}</p>
  </article>;
}
export default function BenchmarkProgress({runs}:{runs:ProgressRun[]}) {
 return <section className="home-visualizations" aria-labelledby="benchmark-progress-title"><div className="home-section-heading"><div><p className="home-eyebrow">MEASURED BENCHMARK RESULTS</p><h2 id="benchmark-progress-title">Two benchmarks. Recursive progress.</h2></div><p>R = recursive generation.<br/>Recorded results, refreshed from the benchmark export.</p></div><div className="home-viz-grid"><Chart runs={runs} inference/><Chart runs={runs} inference={false}/></div><p className="home-source">Source: datacenter.json → runs_index → climb. Expand a generation to inspect the changes. No synthetic rounds or claimed improvements are plotted.</p></section>;
}
