import { useEffect, useMemo, useRef, useState, type RefObject } from 'react';

// runs_index in datacenter.json, written by scripts/build_datacenter_view.py (index_runs, climb_series).
export type RunStatus = 'done' | 'running' | 'crashed' | 'stopped';
export type ClimbCandidate = {
  i: number; gen: number; id: string; parent: number | null; type: string | null; target: string | null;
  change: string; rationale: string; claim_gain: number | null; claim_cost_pct: number | null;
  score: number; parent_score: number | null; gain: number | null; cost: number | null; parent_cost: number | null;
  cost_pct: number | null; n: number | null; accepted: boolean; accepted_by: string | null; reason: string;
  verdict: string | null; research_query: string | null; sources: string[]; incumbent: number | null;
};
type HeldoutPoint = { kind: 'baseline' | 'cold' | 'brain' | 'final'; label: string; score: number };
export type IndexedRun = {
  name: string; layer: string; layer_name: string; benchmark: string; status: RunStatus;
  started: string | null; ended: string | null; solver: string | null; reviewer: string | null; research: boolean;
  generations: number; candidates: number; accepted: number;
  heldout: { metric: string; best: number; baseline: number; delta: number; best_label: string; baseline_label: string; lower_is_better: boolean } | null;
  cost_usd: number | null; calls: number | null;
  climb: { metric: string; baseline: number | null; baseline_label: string | null; candidates: ClimbCandidate[];
    generations: { gen: number; start: number | null; end: number | null }[]; heldout: HeldoutPoint[] };
};

const LAYERS = [
  { id: 'hall', name: 'Hall layout' }, { id: 'placement', name: 'Equipment placement' },
  { id: 'kernels', name: 'GPU kernels' }, { id: 'inference', name: 'Inference cost' },
];
const OVERLAY_COLORS = ['#18181b', '#2563eb', '#d97706', '#7c3aed', '#0891b2', '#db2777', '#65a30d', '#71717a'];
const statusText: Record<RunStatus, string> = { done: 'Done', running: 'Running', crashed: 'Crashed', stopped: 'Stopped' };
const kindText: Record<HeldoutPoint['kind'], string> = { baseline: 'base', cold: 'cold', brain: 'brain', final: 'final' };
const num = (value: number | null | undefined, digits = 3) => value == null ? 'n/a' : value.toFixed(digits);
const plus = (value: number | null | undefined, digits = 3) => value == null ? 'n/a' : `${value >= 0 ? '+' : '-'}${Math.abs(value).toFixed(digits)}`;
const pct = (value: number | null | undefined) => value == null ? 'n/a' : `${value >= 0 ? '+' : '-'}${Math.abs(value).toFixed(1)}%`;
const clock = (stamp: string | null) => stamp ? stamp.slice(11) : 'n/a';
const host = (url: string) => { try { return new URL(url).hostname.replace(/^www\./, ''); } catch { return url; } };

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(320);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(260, Math.floor(entry.contentRect.width))));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

export function StatusBadge({ status }: { status: RunStatus }) {
  return <span className={`benchmark-status dc-status-${status}`}>{statusText[status]}</span>;
}

// Incumbent before candidate i: the generation's re-measured parent score at a generation start, else the previous incumbent.
function before(cands: ClimbCandidate[], index: number) {
  const cand = cands[index];
  const prev = cands[index - 1];
  if (!prev || prev.gen !== cand.gen) return cand.parent_score ?? prev?.incumbent ?? cand.score;
  return prev.incumbent ?? cand.parent_score ?? cand.score;
}

function stepPath(cands: ClimbCandidate[], upto: number, x: (i: number) => number, y: (v: number) => number, shift = 0) {
  if (!cands.length || upto < 0) return '';
  const parts = [`M${x(-0.5)},${y(before(cands, 0) - shift)}`];
  for (let index = 0; index <= Math.min(upto, cands.length - 1); index += 1) {
    parts.push(`V${y(before(cands, index) - shift)}`, `H${x(index)}`, `V${y((cands[index].incumbent ?? before(cands, index)) - shift)}`);
  }
  parts.push(`H${x(Math.min(upto, cands.length - 1) + 0.5)}`);
  return parts.join(' ');
}

function ticks(low: number, high: number, count = 4) {
  return Array.from({ length: count + 1 }, (_, index) => low + ((high - low) * index) / count);
}

function SingleChart({ run, selected, onSelect, upto }: {
  run: IndexedRun; selected: number | null; onSelect: (index: number) => void; upto: number;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const cands = run.climb.candidates;
  const held = run.climb.heldout;
  const base = run.climb.baseline;
  const height = 280;
  const right = held.length ? Math.min(170, Math.max(120, width * 0.3)) : 12;
  const margin = { top: 26, right, bottom: 50, left: 46 };
  const values = [...cands.map((c) => c.score), ...cands.map((c) => c.parent_score ?? c.score), ...held.map((h) => h.score), ...(base != null ? [base] : [])];
  let low = Math.min(...values);
  let high = Math.max(...values);
  const pad = Math.max(0.01, (high - low) * 0.12);
  low -= pad; high += pad;
  const plotRight = width - margin.right;
  const count = Math.max(1, cands.length);
  const x = (index: number) => margin.left + (plotRight - margin.left) * ((index + 0.5) / count);
  const y = (value: number) => margin.top + (height - margin.top - margin.bottom) * (1 - (value - low) / (high - low));
  const genStarts = cands.filter((c, index) => index === 0 || cands[index - 1].gen !== c.gen);
  const heldX = (index: number) => plotRight + 14 + ((right - 28) * (index + 0.5)) / Math.max(1, held.length);
  const shown = cands.filter((c) => c.i <= upto);
  const tickDigits = high - low < 0.06 ? 3 : 2;
  return <div ref={ref} className="hc-plot">
    <svg width={width} height={height} role="img"
      aria-label={`${run.name}: ${cands.length} candidates scored on dev, ${run.accepted} accepted`}>
      {ticks(low, high).map((tick) => <g key={tick}>
        <line className="dc-grid" x1={margin.left} x2={plotRight} y1={y(tick)} y2={y(tick)} />
        <text className="dc-axis" x={margin.left - 6} y={y(tick) + 3} textAnchor="end">{tick.toFixed(tickDigits)}</text>
      </g>)}
      {genStarts.map((c) => <g key={`g${c.gen}`}>
        <line className="hc-gen" x1={x(c.i - 0.5)} x2={x(c.i - 0.5)} y1={margin.top - 8} y2={height - margin.bottom} />
        <text className="dc-axis" x={x(c.i - 0.5) + 4} y={margin.top - 12}>gen {c.gen}</text>
      </g>)}
      {base != null && <g>
        <line className="dc-reference" x1={margin.left} x2={plotRight} y1={y(base)} y2={y(base)} />
        <text className="hc-base-label" x={plotRight - 4} y={y(base) - 5} textAnchor="end">baseline {base.toFixed(3)}</text>
      </g>}
      <path className="dc-step" d={stepPath(cands, upto, x, y)} />
      {shown.map((c) => <g key={c.id} className="hc-dot" tabIndex={0} role="button"
        aria-label={`${c.id}: ${c.change}, dev ${c.score.toFixed(3)}, ${c.accepted ? 'accepted' : 'rejected'}`}
        onMouseEnter={() => onSelect(c.i)} onFocus={() => onSelect(c.i)} onClick={() => onSelect(c.i)}>
        <circle cx={x(c.i)} cy={y(c.score)} r={12} className="hc-hit" />
        <circle cx={x(c.i)} cy={y(c.score)} r={selected === c.i ? 7 : 5}
          className={`${c.accepted ? 'dc-accepted' : 'dc-rejected'}${selected === c.i ? ' hc-selected' : ''}`} />
      </g>)}
      {cands.length <= 24 && cands.map((c) => <text key={`x${c.id}`} className="dc-axis" x={x(c.i)} y={height - margin.bottom + 14} textAnchor="middle">{c.i + 1}</text>)}
      {held.length > 0 && <g>
        <line className="hc-divider" x1={plotRight + 6} x2={plotRight + 6} y1={margin.top - 8} y2={height - margin.bottom} />
        <text className="dc-axis" x={plotRight + 12} y={margin.top - 12}>held-out</text>
        {held.map((h, index) => <g key={h.kind}>
          <circle cx={heldX(index)} cy={y(h.score)} r={4.5} className={`hc-held hc-held-${h.kind}`}><title>{`${h.label}: ${h.score.toFixed(3)}`}</title></circle>
          <text className="dc-axis" x={heldX(index)} y={y(h.score) + (index % 2 ? 16 : -8)} textAnchor="middle">{h.score.toFixed(tickDigits)}</text>
          <text className="dc-axis" x={heldX(index)} y={height - margin.bottom + (index % 2 ? 26 : 14)} textAnchor="middle">{kindText[h.kind]}</text>
        </g>)}
      </g>}
      <text className="dc-axis-title" x={margin.left} y={height - 8}>x: candidate in gate order; y: {run.climb.metric}, mean of {cands[0]?.n ?? '?'} dev tasks</text>
    </svg>
  </div>;
}

function OverlayChart({ runs }: { runs: IndexedRun[] }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const height = 280;
  const margin = { top: 20, right: 12, bottom: 40, left: 50 };
  const series = runs.map((run) => ({ run, ref0: run.climb.baseline ?? run.climb.candidates[0]?.parent_score ?? 0 }));
  const values = series.flatMap(({ run, ref0 }) => run.climb.candidates.flatMap((c) => [c.score - ref0, (c.incumbent ?? c.score) - ref0, before(run.climb.candidates, c.i) - ref0]));
  let low = Math.min(0, ...values);
  let high = Math.max(0, ...values);
  const pad = Math.max(0.01, (high - low) * 0.1);
  low -= pad; high += pad;
  const count = Math.max(1, ...runs.map((run) => run.climb.candidates.length));
  const x = (index: number) => margin.left + (width - margin.left - margin.right) * ((index + 0.5) / count);
  const y = (value: number) => margin.top + (height - margin.top - margin.bottom) * (1 - (value - low) / (high - low));
  return <div ref={ref} className="hc-plot">
    <svg width={width} height={height} role="img" aria-label={`Overlay of ${runs.length} runs, dev score minus baseline`}>
      {ticks(low, high).map((tick) => <g key={tick}>
        <line className="dc-grid" x1={margin.left} x2={width - margin.right} y1={y(tick)} y2={y(tick)} />
        <text className="dc-axis" x={margin.left - 6} y={y(tick) + 3} textAnchor="end">{plus(tick, high - low < 0.06 ? 3 : 2)}</text>
      </g>)}
      <line className="dc-reference" x1={margin.left} x2={width - margin.right} y1={y(0)} y2={y(0)} />
      {series.map(({ run, ref0 }, index) => <g key={run.name} style={{ color: OVERLAY_COLORS[index % OVERLAY_COLORS.length] }}>
        <path className="hc-overlay-step" d={stepPath(run.climb.candidates, run.climb.candidates.length - 1, x, y, ref0)} />
        {run.climb.candidates.map((c) => <circle key={c.id} cx={x(c.i)} cy={y(c.score - ref0)} r={3.5}
          className={c.accepted ? 'hc-overlay-acc' : 'hc-overlay-rej'}><title>{`${run.name} ${c.id}: ${plus(c.score - ref0)} vs baseline`}</title></circle>)}
      </g>)}
      <text className="dc-axis-title" x={margin.left} y={height - 8}>x: candidate in gate order; y: dev score minus the run's baseline</text>
    </svg>
    <div className="dc-legend">{series.map(({ run }, index) => <span key={run.name} className="hc-swatch"
      style={{ color: OVERLAY_COLORS[index % OVERLAY_COLORS.length] }}><i />{run.name}</span>)}</div>
  </div>;
}

function Detail({ cand }: { cand: ClimbCandidate | undefined }) {
  if (!cand) return <div className="hc-detail hc-detail-empty">Hover or tap a dot to see the change, the claim and the gate decision.</div>;
  const parentCost = cand.parent_cost;
  return <div className="hc-detail" aria-live="polite">
    <div className="hc-detail-head">
      <strong>{cand.id}</strong>
      <span className={cand.accepted ? 'hc-pill hc-pill-acc' : 'hc-pill hc-pill-rej'}>{cand.accepted ? `Accepted${cand.accepted_by ? ` (${cand.accepted_by})` : ''}` : 'Rejected'}</span>
    </div>
    <p className="hc-change">{cand.type}{cand.target ? ` · ${cand.target}` : ''}: {cand.change}</p>
    {cand.rationale && <p className="hc-rationale">{cand.rationale}</p>}
    <dl className="hc-facts">
      <div><dt>Dev score</dt><dd>{num(cand.parent_score)} → {num(cand.score)}</dd></div>
      <div><dt>Claimed gain</dt><dd>{plus(cand.claim_gain)}</dd></div>
      <div><dt>Measured gain</dt><dd className={cand.gain != null && cand.gain > 0 ? 'dc-up' : cand.gain != null && cand.gain < 0 ? 'dc-down' : ''}>{plus(cand.gain)}</dd></div>
      <div><dt>Claimed cost</dt><dd>{pct(cand.claim_cost_pct)}</dd></div>
      <div><dt>Measured cost</dt><dd>{pct(cand.cost_pct)}</dd></div>
      <div><dt>$ per task</dt><dd>{parentCost == null ? 'n/a' : `${parentCost.toPrecision(3)} → ${cand.cost?.toPrecision(3) ?? 'n/a'}`}</dd></div>
    </dl>
    <p className="hc-reason"><span>Gate:</span> {cand.reason}</p>
    {cand.verdict && <p className="hc-reason"><span>Claim check:</span> {cand.verdict}</p>}
    {cand.sources.length > 0 && <div className="hc-sources"><span>You.com sources</span>
      <ul>{cand.sources.map((url) => <li key={url}><a href={url} target="_blank" rel="noreferrer">{host(url)}</a></li>)}</ul>
      {cand.research_query && <p className="hc-query">Query: {cand.research_query}</p>}
    </div>}
  </div>;
}

export function RunsByLayer({ runs, onOpen, current }: { runs: IndexedRun[]; onOpen: (run: IndexedRun) => void; current: string | null }) {
  return <section className="dc-layer" aria-labelledby="dc-runs-index">
    <div className="section-heading"><div>
      <h2 id="dc-runs-index">All runs by layer</h2>
      <p className="muted">{runs.length} live runs. Offline and $0 runs are left out. Click a row to open its climb.</p>
    </div></div>
    {LAYERS.map((layer) => {
      const rows = runs.filter((run) => run.layer === layer.id);
      return <div key={layer.id} className="hc-group">
        <h3>{layer.name} <span className="muted">{rows.length} run{rows.length === 1 ? '' : 's'}</span></h3>
        {!rows.length ? <p className="dc-empty">No live run.</p> : <div className="hc-table-wrap"><table className="hc-table">
          <thead><tr><th>Run</th><th>Status</th><th>Benchmark</th><th>Start</th><th>End</th><th>Solver / reviewer</th>
            <th>Research</th><th className="num">Gens</th><th className="num">Scored</th><th className="num">Accepted</th>
            <th className="num">Held-out best</th><th className="num">Baseline</th><th className="num">Delta</th><th className="num">$</th><th className="num">Calls</th></tr></thead>
          <tbody>{rows.map((run) => {
            const digits = run.layer === 'inference' ? 6 : 3;
            const good = run.heldout && (run.heldout.lower_is_better ? run.heldout.delta < 0 : run.heldout.delta > 0);
            const bad = run.heldout && (run.heldout.lower_is_better ? run.heldout.delta > 0 : run.heldout.delta < 0);
            return <tr key={run.name} className={current === run.name ? 'hc-row-current' : ''} tabIndex={0}
              onClick={() => onOpen(run)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onOpen(run); } }}>
              <td className="hc-run-name">{run.name}</td><td><StatusBadge status={run.status} /></td><td>{run.benchmark}</td>
              <td>{clock(run.started)}</td><td>{clock(run.ended)}</td>
              <td className="hc-models">{run.solver ?? 'n/a'}<br /><span className="muted">{run.reviewer ?? 'n/a'}</span></td>
              <td>{run.research ? 'on' : 'off'}</td><td className="num">{run.generations}</td><td className="num">{run.candidates}</td>
              <td className="num">{run.accepted}</td><td className="num">{num(run.heldout?.best, digits)}</td>
              <td className="num">{num(run.heldout?.baseline, digits)}</td>
              <td className={`num ${good ? 'dc-up' : bad ? 'dc-down' : ''}`}>{run.heldout ? plus(run.heldout.delta, digits) : 'n/a'}</td>
              <td className="num">{run.cost_usd == null ? 'n/a' : run.cost_usd.toFixed(4)}</td><td className="num">{run.calls ?? 'n/a'}</td>
            </tr>;
          })}</tbody>
        </table></div>}
      </div>;
    })}
  </section>;
}

export default function HillClimb({ runs, openRun, sectionRef }: {
  runs: IndexedRun[]; openRun: { name: string; nonce: number } | null; sectionRef: RefObject<HTMLElement | null>;
}) {
  const withClimb = (layer: string) => runs.filter((run) => run.layer === layer);
  const defaultRun = useMemo(() => runs.reduce<IndexedRun | null>((top, run) => (!top || run.candidates > top.candidates ? run : top), null), [runs]);
  const [layer, setLayer] = useState(defaultRun?.layer ?? 'placement');
  const [runName, setRunName] = useState<string | null>(defaultRun?.name ?? null);
  const [selected, setSelected] = useState<number | null>(null);
  const [overlay, setOverlay] = useState(false);
  const [upto, setUpto] = useState(Number.MAX_SAFE_INTEGER);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!openRun) return;
    const run = runs.find((item) => item.name === openRun.name);
    if (!run) return;
    setLayer(run.layer); setRunName(run.name); setSelected(null); setOverlay(false); setPlaying(false); setUpto(Number.MAX_SAFE_INTEGER);
  }, [openRun, runs]);

  const layerRuns = withClimb(layer);
  const run = layerRuns.find((item) => item.name === runName) ?? layerRuns.find((item) => item.candidates > 0) ?? layerRuns[0];
  const cands = run?.climb.candidates ?? [];

  useEffect(() => {
    if (!playing) return;
    if (upto >= cands.length - 1) { setPlaying(false); return; }
    const timer = window.setTimeout(() => { setUpto((value) => value + 1); setSelected((value) => (value ?? -1) + 1); }, 700);
    return () => window.clearTimeout(timer);
  }, [playing, upto, cands.length]);

  function pickLayer(id: string) {
    setLayer(id); setSelected(null); setPlaying(false); setUpto(Number.MAX_SAFE_INTEGER);
    const opening = withClimb(id).find((item) => item.candidates > 0) ?? withClimb(id)[0];
    setRunName(opening?.name ?? null);
  }
  function pickRun(name: string) { setRunName(name); setSelected(null); setPlaying(false); setUpto(Number.MAX_SAFE_INTEGER); setOverlay(false); }
  function replay() { setOverlay(false); setUpto(-1); setSelected(null); setPlaying(true); }

  const climbing = layerRuns.filter((item) => item.candidates > 0);
  const best = cands.reduce<ClimbCandidate | null>((top, c) => (c.accepted && (!top || c.score > top.score) ? c : top), null);
  const start = cands[0]?.parent_score;

  return <section className="dc-layer hc" aria-labelledby="dc-hillclimb" ref={sectionRef}>
    <div className="section-heading"><div>
      <h2 id="dc-hillclimb">Hill climb</h2>
      <p className="muted">Each dot is one candidate change scored on dev by the gate. The step line is the incumbent.</p>
    </div></div>
    <div className="hc-layers" role="tablist" aria-label="Layer">
      {LAYERS.map((item) => <button key={item.id} type="button" role="tab" aria-selected={layer === item.id}
        className={`hc-chip${layer === item.id ? ' hc-chip-on' : ''}`} onClick={() => pickLayer(item.id)}>
        {item.name} <span>{withClimb(item.id).length}</span></button>)}
    </div>
    {!layerRuns.length ? <div className="dc-not-run"><strong>No live run in this layer.</strong>
      <span>{layer === 'kernels' ? 'KernelBench needs a GPU; our Crusoe project has 0 GPU quota.' : 'Nothing on disk yet.'}</span></div> : <div className="hc-body">
      <ul className="hc-runs" aria-label="Runs">
        {layerRuns.map((item) => <li key={item.name}><button type="button" className={`hc-run${run?.name === item.name ? ' hc-run-on' : ''}`}
          onClick={() => pickRun(item.name)} aria-pressed={run?.name === item.name}>
          <span className="hc-run-top"><strong>{item.name}</strong><StatusBadge status={item.status} /></span>
          <span className="hc-run-meta">{clock(item.started)} · {item.benchmark} · {item.candidates} scored, {item.accepted} accepted{item.research ? ' · research on' : ''}</span>
        </button></li>)}
      </ul>
      <div className="hc-main">
        {run && <div className="hc-toolbar">
          <div className="hc-stats">
            <span><b>{run.candidates}</b> scored</span><span><b>{run.accepted}</b> accepted</span>
            <span>dev <b>{num(start)}</b> → <b>{num(best?.score ?? start)}</b></span>
            {run.heldout && <span>held-out <b>{num(run.heldout.best, run.layer === 'inference' ? 6 : 3)}</b> vs <b>{num(run.heldout.baseline, run.layer === 'inference' ? 6 : 3)}</b></span>}
            {run.cost_usd != null && <span><b>${run.cost_usd.toFixed(4)}</b>, {run.calls} calls</span>}
          </div>
          <div className="hc-actions">
            <label className="hc-toggle"><input type="checkbox" checked={overlay} disabled={climbing.length < 2}
              onChange={(event) => { setOverlay(event.target.checked); setPlaying(false); setUpto(Number.MAX_SAFE_INTEGER); }} />
              Overlay {climbing.length} runs</label>
            <button type="button" className="hc-button" onClick={replay} disabled={overlay || cands.length === 0}>{playing ? 'Replaying…' : 'Replay'}</button>
          </div>
        </div>}
        {run && (overlay ? <OverlayChart runs={climbing} /> : cands.length === 0 ? <div className="dc-empty">
          {run.name} has no scored candidate yet ({statusText[run.status].toLowerCase()}).</div> : <>
          <SingleChart run={run} selected={selected} onSelect={(index) => { setSelected(index); }} upto={upto} />
          <div className="dc-legend"><span className="dc-key dc-key-dev">incumbent</span>
            <span className="dc-key dc-key-accepted">accepted</span><span className="dc-key dc-key-rejected">rejected</span>
            {run.climb.baseline != null && <span className="hc-key-base">{run.climb.baseline_label}</span>}</div>
          <Detail cand={selected == null ? undefined : cands[selected]} />
        </>)}
      </div>
    </div>}
  </section>;
}
