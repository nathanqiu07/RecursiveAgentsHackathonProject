import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import './benchmarks.css';
import './datacenter.css';

// Written by scripts/build_datacenter_view.py from live run artifacts; see that script for every field.
type Status = 'done' | 'running' | 'not run';
type Bar = { label: string; kind: 'baseline' | 'cold' | 'brain' | 'final'; score: number; n: number | null };
type Candidate = { gen: number; score: number; accepted: boolean; reason: string; change: string };
type Run = {
  id: string; status: 'done' | 'running'; status_detail: string; family?: string;
  baseline: { name: string; score: number | null }; heldout: Bar[];
  dev: { gen: number; score: number }[]; candidates: Candidate[]; n_heldout: number | null;
};
type Setting = { label: string; tag: string; gain_pts: number; feasible: number; attempts: number; halls: number };
type Point = { run: string; label: string; accuracy: number; cost_per_correct_usd: number; baseline: boolean };
type Layer = {
  id: string; name: string; problem: string; verified: string; score_label: string; digits: number;
  baseline: { name: string; score: number | null };
  best: { run: string; label: string; score: number; baseline: number; delta: number } | null;
  status: Status; summary: string; runs: Run[]; settings?: Setting[]; scatter?: Point[];
};
type View = { generated_at: string; command: string; pitch: string; layers: Layer[] };

const statusLabel: Record<Status, string> = { done: 'Done', running: 'Running', 'not run': 'Not run' };
const kindLabel: Record<Bar['kind'], string> = {
  baseline: 'Baseline', cold: 'v0 cold', brain: 'v0 + shared brain', final: 'Final',
};
const fixed = (value: number | null | undefined, digits: number) => value == null ? 'n/a' : value.toFixed(digits);
const signed = (value: number, digits: number) => `${value > 0 ? '+' : value < 0 ? '-' : '±'}${Math.abs(value).toFixed(digits)}`;
const firstSentence = (text: string) => text.split(/(?<=\.)\s+(?=[A-Z])/)[0] ?? text;

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(320);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(240, Math.floor(entry.contentRect.width))));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

function ticks(min: number, max: number, count = 4) {
  const step = (max - min) / count;
  return Array.from({ length: count + 1 }, (_, index) => min + step * index);
}

function Chart({ title, note, children }: { title: string; note?: string; children: ReactNode }) {
  return <figure className="dc-chart">
    <figcaption><strong>{title}</strong>{note && <span>{note}</span>}</figcaption>
    {children}
  </figure>;
}

function HeldoutBars({ run, digits, unit }: { run: Run; digits: number; unit: string }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const height = 210;
  const margin = { top: 22, right: 8, bottom: 38, left: 40 };
  const bars = run.heldout;
  const base = run.baseline.score;
  const max = Math.max(1, ...bars.map((bar) => bar.score));
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const y = (value: number) => margin.top + plotHeight * (1 - value / max);
  const slot = plotWidth / Math.max(1, bars.length);
  const barWidth = Math.min(56, slot * 0.6);
  return <div ref={ref}>
    <svg width={width} height={height} role="img"
      aria-label={`Held-out ${unit}: ${bars.map((bar) => `${kindLabel[bar.kind]} ${fixed(bar.score, digits)}`).join(', ')}`}>
      {ticks(0, max).map((tick) => <g key={tick}>
        <line className="dc-grid" x1={margin.left} x2={width - margin.right} y1={y(tick)} y2={y(tick)} />
        <text className="dc-axis" x={margin.left - 6} y={y(tick) + 3} textAnchor="end">{tick.toFixed(2)}</text>
      </g>)}
      {bars.map((bar, index) => {
        const x = margin.left + slot * index + (slot - barWidth) / 2;
        return <g key={bar.kind}>
          <rect className={`dc-bar dc-bar-${bar.kind}`} x={x} y={y(bar.score)} width={barWidth}
            height={Math.max(0, y(0) - y(bar.score))} rx={2}><title>{bar.label}</title></rect>
          <text className="dc-value" x={x + barWidth / 2} y={y(bar.score) - 5} textAnchor="middle">{fixed(bar.score, digits)}</text>
          <text className="dc-axis" x={x + barWidth / 2} y={height - margin.bottom + 15} textAnchor="middle">
            {bar.kind === 'brain' ? 'v0 + brain' : kindLabel[bar.kind]}</text>
        </g>;
      })}
      {base != null && <g>
        <line className="dc-reference" x1={margin.left} x2={width - margin.right} y1={y(base)} y2={y(base)} />
      </g>}
      <text className="dc-axis-title" x={margin.left} y={height - 6}>{unit}, mean of {run.n_heldout ?? '?'} held-out tasks</text>
    </svg>
  </div>;
}

function DevChart({ run, digits }: { run: Run; digits: number }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const height = 210;
  const margin = { top: 16, right: 14, bottom: 38, left: 46 };
  const values = [...run.dev.map((point) => point.score), ...run.candidates.map((point) => point.score)];
  const gens = [...run.dev.map((point) => point.gen), ...run.candidates.map((point) => point.gen)];
  const lastGen = Math.max(1, ...gens);
  let low = Math.min(...values);
  let high = Math.max(...values);
  const pad = Math.max(0.005, (high - low) * 0.25);
  low -= pad; high += pad;
  const x = (gen: number) => margin.left + (width - margin.left - margin.right) * ((gen + 0.3) / (lastGen + 0.6));
  const y = (value: number) => margin.top + (height - margin.top - margin.bottom) * (1 - (value - low) / (high - low));
  const sorted = [...run.dev].sort((a, b) => a.gen - b.gen);
  const step = sorted.map((point, index) => {
    const next = sorted[index + 1];
    return `${index ? 'L' : 'M'}${x(point.gen)},${y(point.score)}${next ? ` L${x(next.gen)},${y(point.score)}` : ''}`;
  }).join(' ');
  const accepted = run.candidates.filter((candidate) => candidate.accepted).length;
  return <div ref={ref}>
    <svg width={width} height={height} role="img"
      aria-label={`Dev score by generation; ${run.candidates.length} candidates, ${accepted} accepted by the gate`}>
      {ticks(low, high).map((tick) => <g key={tick}>
        <line className="dc-grid" x1={margin.left} x2={width - margin.right} y1={y(tick)} y2={y(tick)} />
        <text className="dc-axis" x={margin.left - 6} y={y(tick) + 3} textAnchor="end">{tick.toFixed(digits)}</text>
      </g>)}
      {Array.from({ length: lastGen + 1 }, (_, gen) =>
        <text key={gen} className="dc-axis" x={x(gen)} y={height - margin.bottom + 15} textAnchor="middle">gen {gen}</text>)}
      {step && <path className="dc-step" d={step} />}
      {sorted.map((point) => <circle key={`d${point.gen}`} className="dc-dev-dot" cx={x(point.gen)} cy={y(point.score)} r={3.5} />)}
      {run.candidates.map((candidate, index) => {
        const jitter = ((index % 5) - 2) * 4;
        return <circle key={index} className={candidate.accepted ? 'dc-accepted' : 'dc-rejected'}
          cx={x(candidate.gen) + jitter} cy={y(candidate.score)} r={4}>
          <title>{`${candidate.change}: ${fixed(candidate.score, digits)} (${candidate.accepted ? 'accepted' : 'rejected'}; ${candidate.reason})`}</title>
        </circle>;
      })}
      <text className="dc-axis-title" x={margin.left} y={height - 6}>dev score, mean of the dev tasks</text>
    </svg>
  </div>;
}

function SettingsBars({ settings }: { settings: Setting[] }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const row = 36;
  const margin = { top: 6, right: 132, bottom: 30, left: 8 };
  const height = margin.top + margin.bottom + row * settings.length;
  const max = Math.max(1, ...settings.map((setting) => setting.gain_pts));
  const plotWidth = width - margin.left - margin.right;
  return <div ref={ref}>
    <svg width={width} height={height} role="img" aria-label="Points over greedy layout by model setting">
      {settings.map((setting, index) => {
        const top = margin.top + row * index;
        const barWidth = (plotWidth * Math.max(0, setting.gain_pts)) / max;
        return <g key={`${setting.label}${setting.tag}`}>
          <text className="dc-axis dc-setting-label" x={margin.left} y={top + 10}>
            {setting.label}{setting.tag === 'rep' ? ' (repeat)' : ''}{setting.halls !== 4 ? `, ${setting.halls} hall${setting.halls === 1 ? '' : 's'}` : ''}</text>
          <rect className="dc-bar-track" x={margin.left} y={top + 14} width={plotWidth} height={8} rx={2} />
          <rect className={setting.gain_pts > 0 ? 'dc-bar dc-bar-final' : 'dc-bar dc-bar-cold'} x={margin.left} y={top + 14}
            width={Math.max(2, barWidth)} height={8} rx={2} />
          <text className="dc-value" x={width - margin.right + 8} y={top + 22}>{signed(setting.gain_pts, 2)} pts
            <tspan className="dc-axis" dx={6}>{setting.feasible}/{setting.attempts} feasible</tspan></text>
        </g>;
      })}
      <text className="dc-axis-title" x={margin.left} y={height - 8}>points over greedy; feasible = layouts that passed the checker</text>
    </svg>
  </div>;
}

function CostScatter({ points }: { points: Point[] }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const height = 230;
  const margin = { top: 16, right: 16, bottom: 40, left: 46 };
  const costs = points.map((point) => point.cost_per_correct_usd);
  const accs = points.map((point) => point.accuracy);
  const xMax = Math.max(...costs) * 1.15;
  const yLow = Math.max(0, Math.min(...accs) - 0.1);
  const yHigh = Math.min(1, Math.max(...accs) + 0.05);
  const x = (value: number) => margin.left + (width - margin.left - margin.right) * (value / xMax);
  const y = (value: number) => margin.top + (height - margin.top - margin.bottom) * (1 - (value - yLow) / (yHigh - yLow || 1));
  return <div ref={ref}>
    <svg width={width} height={height} role="img" aria-label="Accuracy vs dollars per correct answer, one point per config">
      {ticks(yLow, yHigh).map((tick) => <g key={`y${tick}`}>
        <line className="dc-grid" x1={margin.left} x2={width - margin.right} y1={y(tick)} y2={y(tick)} />
        <text className="dc-axis" x={margin.left - 6} y={y(tick) + 3} textAnchor="end">{tick.toFixed(2)}</text>
      </g>)}
      {ticks(0, xMax, 3).map((tick) => <text key={`x${tick}`} className="dc-axis" x={x(tick)} y={height - margin.bottom + 15}
        textAnchor="middle">{tick === 0 ? '$0' : `$${tick.toPrecision(2)}`}</text>)}
      {points.map((point) => <g key={`${point.run}${point.label}`}>
        <circle className={point.baseline ? 'dc-point-baseline' : 'dc-point'} cx={x(point.cost_per_correct_usd)} cy={y(point.accuracy)} r={5} />
        <text className="dc-value" x={x(point.cost_per_correct_usd) + (x(point.cost_per_correct_usd) > width * 0.6 ? -8 : 8)}
          y={y(point.accuracy) - 8} textAnchor={x(point.cost_per_correct_usd) > width * 0.6 ? 'end' : 'start'}>{point.label}</text>
      </g>)}
      <text className="dc-axis-title" x={margin.left} y={height - 6}>x: $ per correct answer (lower is better); y: accuracy</text>
    </svg>
  </div>;
}

function LayerCard({ layer }: { layer: Layer }) {
  const best = layer.best;
  const isCost = layer.id === 'inference';
  return <article className="dc-card">
    <div className="dc-card-head"><h3>{layer.name}</h3>
      <span className={`benchmark-status dc-status-${layer.status.replace(' ', '-')}`}>{statusLabel[layer.status]}</span></div>
    <p className="dc-card-problem">{layer.problem}</p>
    <dl className="dc-card-numbers">
      <div><dt>Baseline</dt><dd>{isCost && layer.baseline.score != null ? `$${fixed(layer.baseline.score, 6)}` : fixed(layer.baseline.score, layer.digits)}</dd></div>
      <div><dt>Best</dt><dd>{best ? (isCost ? `$${fixed(best.score, 6)}` : fixed(best.score, layer.digits)) : 'n/a'}</dd></div>
      <div><dt>Delta</dt><dd className={best && best.delta !== 0 ? (best.delta > 0 !== isCost ? 'dc-up' : 'dc-down') : ''}>
        {best ? (isCost ? `${signed(best.delta * 100, 0)}%` : signed(best.delta, layer.digits)) : 'n/a'}</dd></div>
    </dl>
    <p className="dc-card-baseline">{layer.baseline.name} · {layer.score_label}</p>
    <p className="dc-card-summary">{firstSentence(layer.summary)}</p>
  </article>;
}

function LayerSection({ layer }: { layer: Layer }) {
  const doneRuns = layer.runs.filter((run) => run.status === 'done' && run.heldout.length);
  const devRuns = layer.runs.filter((run) => run.dev.length || run.candidates.length);
  const running = layer.runs.filter((run) => run.status === 'running');
  const unit = layer.id === 'hall' ? 'DoneTwin score / 100' : layer.id === 'inference' ? 'accuracy' : 'best-known / cost';
  const barDigits = layer.id === 'inference' ? 3 : layer.digits;
  return <section className="dc-layer" aria-labelledby={`dc-${layer.id}`}>
    <div className="section-heading"><div>
      <h2 id={`dc-${layer.id}`}>{layer.name} <span className={`benchmark-status dc-status-${layer.status.replace(' ', '-')}`}>{statusLabel[layer.status]}</span></h2>
      <p className="muted">{layer.verified}</p>
    </div></div>
    <p className="dc-summary">{layer.summary}</p>
    {layer.status === 'not run' && !layer.runs.length ? <div className="dc-not-run">
      <strong>No chart: no live run.</strong>
      <span>Baseline: {layer.baseline.name}, speedup {fixed(layer.baseline.score, layer.digits)}x by definition.</span>
    </div> : <div className="dc-charts">
      {doneRuns.map((run) => <Chart key={`h${run.id}`} title={`Held-out, ${run.id}`}
        note={`${run.family ? `${run.family}; ` : ''}${run.baseline.score != null ? `dashed line = ${run.baseline.name} ${fixed(run.baseline.score, barDigits)}` : `baseline = ${run.baseline.name}`}`}>
        <HeldoutBars run={run} digits={barDigits} unit={unit} /></Chart>)}
      {layer.settings && layer.settings.length > 0 && <Chart title="Model settings, 4 train halls"
        note="Layout role varies; scored by the DoneTwin checker">
        <SettingsBars settings={layer.settings} /></Chart>}
      {layer.id === 'inference' && (layer.scatter?.length ? <Chart title="Accuracy vs $ per correct answer"
        note="one point per config, held-out questions"><CostScatter points={layer.scatter} /></Chart>
        : <div className="dc-empty">No held-out points yet. {running.map((run) => run.id).join(', ') || 'No live run'} {running.length ? 'is running.' : ''}</div>)}
      {devRuns.map((run) => <Chart key={`d${run.id}`} title={`Dev score by generation, ${run.id}`}
        note={`${run.family ? `${run.family}; ` : ''}${run.status === 'running' ? 'running; ' : ''}${run.candidates.length} candidate${run.candidates.length === 1 ? '' : 's'}, ${run.candidates.filter((c) => c.accepted).length} accepted by the gate`}>
        <DevChart run={run} digits={layer.digits} />
        <div className="dc-legend"><span className="dc-key dc-key-dev">dev mean</span>
          <span className="dc-key dc-key-accepted">accepted</span><span className="dc-key dc-key-rejected">rejected</span></div>
      </Chart>)}
    </div>}
  </section>;
}

export default function DataCenter({ active }: { active: boolean }) {
  const [view, setView] = useState<View | null>(null);
  const [error, setError] = useState('');

  const load = useCallback(async (signal?: AbortSignal) => {
    try {
      const response = await fetch('/datacenter.json', { cache: 'no-store', signal });
      if (!response.ok) throw new Error(`datacenter.json: HTTP ${response.status}`);
      const data = await response.json() as View;
      if (signal?.aborted) return;
      setView(data);
      setError('');
    } catch (failure) {
      if (signal?.aborted) return;
      setError(failure instanceof Error ? failure.message : 'Unable to load datacenter.json');
    }
  }, []);

  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    void load(controller.signal);
    const interval = window.setInterval(() => { void load(controller.signal); }, 30000);
    return () => { controller.abort(); window.clearInterval(interval); };
  }, [active, load]);

  return <div className="benchmarks dc">
    <header className="benchmarks-header"><div>
      <h1>AI Data Center</h1>
      <p>{view?.pitch ?? 'Every layer of an AI data center as a verifiable optimization problem.'} Live Crusoe runs only.</p>
    </div>{view && <span className="node-count">Updated {view.generated_at}</span>}</header>
    {error && <p className="error-banner">{error}. Run <code>.venv/bin/python scripts/build_datacenter_view.py</code>.</p>}
    {!view && !error && <p className="benchmark-loading">Loading…</p>}
    {view && <>
      <div className="dc-cards">{view.layers.map((layer) => <LayerCard key={layer.id} layer={layer} />)}</div>
      {view.layers.map((layer) => <LayerSection key={layer.id} layer={layer} />)}
      <p className="catalog-note">Data: <code>{view.command}</code>, rebuilt from runs/ and reports/; this page reloads it every 30 s.</p>
    </>}
  </div>;
}
