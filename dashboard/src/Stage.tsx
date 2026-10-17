import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import type { ClimbCandidate, IndexedRun } from './HillClimb';
import './stage.css';

// Presenter mode for the live demo. Every animation is a pure function of one clock `t` (ms into the beat),
// so pause, restart, 1x/2x/4x and reduced motion all fall out of the same timeline.

type Layer = {
  id: string; name: string; problem?: string; score_label?: string; digits?: number; status?: string;
  summary?: string; note?: string | null;
  baseline?: { name?: string; score?: number | null } | null;
  reference?: { name?: string; score?: number | null } | null;
  best?: { run?: string; label?: string; score?: number | null; baseline?: number | null; reference?: number | null;
    reference_name?: string | null; n?: number | null; accuracy?: number | null; baseline_accuracy?: number | null } | null;
  [key: string]: unknown;
};
type View = { pitch?: string; generated_at?: string; layers?: Layer[]; runs_index?: IndexedRun[] };
type BeatKind = 'thesis' | 'twin' | 'layers' | 'replay' | 'paper' | 'close';
type Beat = { key: string; kind: BeatKind; label: string; layerId?: string; duration: number };

const TWIN_URL = 'http://127.0.0.1:8000';
const PITCH = 'Every layer of an AI data center as a verifiable optimization problem';
// Numbers supplied by the team for the SOTA and rack notes; datacenter.json does not carry them yet.
const QAPLIB_SOTA_LINE = 'QAPLIB: 3.6% from best-known solutions (SOTA)';
const QAPLIB_SOTA_V0 = 'v0 cold: 4.3%';
const RACK_NOTE = 'v0 produced an invalid placement on 1 of 6; on the other 5, +0.229';
const RACK_NOTE_RUN = 'qaplib-dc-research-1';
// The team's chosen headline run per layer; used when it is finished, else the largest held-out gain wins.
const PREFERRED_RUN: Record<string, string> = { placement: 'qaplib-dc-research-2' };
const isQaplib = (run?: IndexedRun) => /^qaplib/i.test(run?.benchmark ?? '');
const INFERENCE_CAVEATS = [
  'Accuracy 47/48 to 48/48 is one question, so no accuracy gain is claimed.',
  'Median call latency doubled (9.7 s to 19.0 s) while time per hall halved (163 s to 77 s).',
];

const isNum = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const clamp01 = (x: number) => Math.max(0, Math.min(1, x));
const ease = (x: number) => 1 - Math.pow(1 - clamp01(x), 3);
const phase = (t: number, start: number, duration: number) => ease((t - start) / duration);
const lerp = (a: number, b: number, p: number) => a + (b - a) * p;
const fixed = (value: number | null | undefined, digits = 3) => (isNum(value) ? value.toFixed(digits) : 'n/a');
const signed = (value: number | null | undefined, digits = 3) =>
  isNum(value) ? `${value >= 0 ? '+' : '-'}${Math.abs(value).toFixed(digits)}` : 'n/a';
const host = (url: string) => { try { return new URL(url).hostname.replace(/^www\./, ''); } catch { return url; } };
function sourceTitle(url: string) {
  try {
    const parts = new URL(url).pathname.split('/').filter(Boolean);
    const last = decodeURIComponent(parts[parts.length - 1] ?? '').replace(/\.(pdf|html?|aspx?)$/i, '');
    const words = last.replace(/[-_+]+/g, ' ').replace(/^\d+\s+/, '').trim();
    if (!words || /^[\d.\s]+$/.test(words)) return parts.length > 1 ? decodeURIComponent(parts[parts.length - 2]).replace(/[-_]+/g, ' ') : host(url);
    return words.length > 64 ? `${words.slice(0, 62)}...` : words;
  } catch { return url; }
}
const sourceUrl = (source: unknown) => (typeof source === 'string' ? source : (source as { url?: string })?.url ?? '');
const shortChange = (text: string, max = 22) => (text.length > max ? `${text.slice(0, max - 1)}...` : text);

function useReducedMotion() {
  const [reduced, setReduced] = useState(() => typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches);
  useEffect(() => {
    const query = window.matchMedia?.('(prefers-reduced-motion: reduce)');
    if (!query) return;
    const onChange = () => setReduced(query.matches);
    query.addEventListener('change', onChange);
    return () => query.removeEventListener('change', onChange);
  }, []);
  return Boolean(reduced);
}

const candidatesOf = (run: IndexedRun | undefined): ClimbCandidate[] => (Array.isArray(run?.climb?.candidates) ? run!.climb.candidates : []);
const lowerIsBetter = (run: IndexedRun | undefined, layer?: Layer) =>
  Boolean(run?.heldout?.lower_is_better) || /lower is better|\$/i.test(layer?.score_label ?? '');

// Held-out improvement vs v0 as a fraction, signed so that bigger is better on every metric.
function heldoutGain(run: IndexedRun) {
  const held = run.heldout;
  if (!held || !isNum(held.v0) || !isNum(held.final) || held.v0 === 0) return -Infinity;
  const relative = (held.final - held.v0) / Math.abs(held.v0);
  return held.lower_is_better ? -relative : relative;
}

function replayableRuns(runs: IndexedRun[], layerId: string) {
  return runs.filter((run) => run.layer === layerId && candidatesOf(run).length > 0);
}
function defaultRun(runs: IndexedRun[], layerId: string) {
  const done = replayableRuns(runs, layerId).filter((run) => run.status === 'done' && run.heldout);
  const preferred = done.find((run) => run.name === PREFERRED_RUN[layerId]);
  if (preferred) return preferred;
  return [...done].sort((a, b) => heldoutGain(b) - heldoutGain(a) || candidatesOf(b).length - candidatesOf(a).length)[0];
}

function replayPlan(run: IndexedRun | undefined) {
  const n = candidatesOf(run).length;
  const slot = Math.min(4000, Math.max(1900, 14000 / Math.max(1, n)));
  const candStart = 2200;
  const heldStart = candStart + n * slot + 300;
  const libStart = heldStart + 3000;
  return { n, slot, candStart, heldStart, libStart, total: libStart + 1600 };
}

function verdictOf(cand: ClimbCandidate) {
  const byCost = cand.accepted_by === 'cost' || (!isNum(cand.claim_gain) && isNum(cand.claim_cost_pct));
  return byCost
    ? { unit: 'cost', claimed: cand.claim_cost_pct, measured: cand.cost_pct, digits: 1, suffix: '%' }
    : { unit: 'gain', claimed: cand.claim_gain, measured: cand.gain, digits: 3, suffix: '' };
}

function kernelsOnMps(layer: Layer | undefined) {
  if (!layer || /not run/i.test(layer.status ?? '')) return false;
  return /\bmps\b|apple gpu|metal/i.test(JSON.stringify(layer));
}

/* ---------- beat (a): thesis ---------- */

const LOOP = [
  { title: 'Solve', sub: 'Nemotron on Crusoe', x: 40, y: 40 },
  { title: 'Shared Brain', sub: 'what worked, per layer', x: 420, y: 40 },
  { title: 'Reviewer', sub: 'reads You.com results', x: 800, y: 40 },
  { title: 'Gate', sub: 'measures claims, exact verifier', x: 800, y: 300 },
  { title: "SA's Library", sub: 'kept changes publish as papers', x: 420, y: 300 },
  { title: 'Held-out check', sub: 'unseen tasks, v0 cold vs final', x: 40, y: 300 },
];
const BOX_W = 320;
const BOX_H = 130;

function Thesis({ t, pitch }: { t: number; pitch: string }) {
  const arrows = [
    'M360 105 H420', 'M740 105 H800', 'M960 170 V300', 'M800 365 H740', 'M420 365 H360',
  ];
  return <div className="stage-slide stage-thesis">
    <p className="stage-kicker" style={{ opacity: phase(t, 0, 500) }}>Crusoe x Cerebral Valley, Recursive Agents Hackathon</p>
    <h1 className="stage-title" style={{ opacity: phase(t, 100, 700), transform: `translateY(${(1 - phase(t, 100, 700)) * 16}px)` }}>{pitch}</h1>
    <svg className="stage-loop" viewBox="0 0 1160 470" role="img" aria-label="The loop: solve, shared brain, reviewer, gate, SA's Library, held-out check">
      <defs>
        <marker id="stage-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" fill="var(--stage-ink)" />
        </marker>
      </defs>
      {LOOP.map((box, index) => {
        const p = phase(t, 900 + index * 650, 600);
        return <g key={box.title} className="stage-loop-box">
          <rect x={box.x} y={box.y} width={BOX_W} height={BOX_H} rx="10" pathLength={1}
            style={{ strokeDasharray: 1, strokeDashoffset: 1 - p, fillOpacity: phase(t, 1200 + index * 650, 400) }} />
          <text x={box.x + 24} y={box.y + 56} className="stage-loop-title" style={{ opacity: phase(t, 1150 + index * 650, 400) }}>
            <tspan className="stage-loop-num">{index + 1}</tspan> {box.title}</text>
          <text x={box.x + 24} y={box.y + 94} className="stage-loop-sub" style={{ opacity: phase(t, 1250 + index * 650, 400) }}>{box.sub}</text>
        </g>;
      })}
      {arrows.map((d, index) => {
        const p = phase(t, 1350 + index * 650, 400);
        return <path key={d} d={d} className="stage-loop-arrow" pathLength={1} markerEnd={p > 0.95 ? 'url(#stage-arrow)' : undefined}
          style={{ strokeDasharray: 1, strokeDashoffset: 1 - p }} />;
      })}
      <path d="M800 330 C 640 230, 560 230, 580 175" className="stage-loop-arrow stage-loop-return" pathLength={1}
        markerEnd={phase(t, 4900, 700) > 0.95 ? 'url(#stage-arrow)' : undefined}
        style={{ strokeDasharray: '0.02 0.02', strokeDashoffset: 0, opacity: phase(t, 4900, 700) }} />
      <text x="600" y="258" textAnchor="end" className="stage-loop-note" style={{ opacity: phase(t, 5200, 500) }}>kept change joins the brain</text>
    </svg>
  </div>;
}

/* ---------- beat (b): DoneTwin ---------- */

function Twin({ t, hall }: { t: number; hall?: Layer }) {
  return <div className="stage-slide stage-twin">
    <p className="stage-kicker">Live, verifiable</p>
    <h1 className="stage-title">DoneTwin: a data hall that checks itself</h1>
    <div className="stage-twin-card" style={{ opacity: phase(t, 200, 600), transform: `translateY(${(1 - phase(t, 200, 600)) * 20}px)` }}>
      <a className="stage-twin-button" href={TWIN_URL} target="_blank" rel="noopener noreferrer">Open live twin</a>
      <p className="stage-twin-url">{TWIN_URL.replace('http://', '')}, opens in a new tab</p>
      <p className="stage-twin-look">Look at the checker: move a rack and it re-scores the hall against hard rules (collision, 2N power, CDU capacity, hose reach, zones). Any layout that breaks one is infeasible.</p>
      {hall?.best && isNum(hall.best.score) && <p className="stage-twin-meta">Hall layout on {hall.best.n ?? 'n/a'} unseen halls: {fixed(hall.best.score, 3)} {hall.score_label ? `(${hall.score_label})` : ''}</p>}
    </div>
  </div>;
}

/* ---------- beat (c): four layers ---------- */

function layerNumbers(layer: Layer, runs: IndexedRun[]) {
  const preferred = runs.find((item) => item.name === PREFERRED_RUN[layer.id] && item.status === 'done' && item.heldout);
  const run = preferred ?? runs.find((item) => item.name === layer.best?.run);
  const lower = lowerIsBetter(run, layer);
  const v0 = preferred ? preferred.heldout!.v0 : layer.best?.baseline ?? layer.baseline?.score ?? null;
  const final = preferred ? preferred.heldout!.final : layer.best?.score ?? null;
  const digits = isNum(layer.digits) ? layer.digits : 3;
  return { run, lower, v0, final, digits };
}

function formatScore(value: number | null | undefined, digits: number, lower: boolean) {
  if (!isNum(value)) return 'n/a';
  return lower ? `$${value.toFixed(digits)}` : value.toFixed(digits);
}

function deltaText(v0: number | null, final: number | null, lower: boolean, progress = 1) {
  if (!isNum(v0) || !isNum(final)) return null;
  if (lower) {
    const cut = v0 === 0 ? 0 : (1 - final / v0) * 100 * progress;
    return `${Math.round(cut)}% less per correct answer`;
  }
  return `${signed((final - v0) * progress, 3)} vs v0`;
}

function Layers({ t, layers, runs }: { t: number; layers: Layer[]; runs: IndexedRun[] }) {
  return <div className="stage-slide stage-layers">
    <p className="stage-kicker">Four layers, held-out, v0 cold to final</p>
    <h1 className="stage-title stage-title-sm">Same loop, four exact verifiers</h1>
    <div className="stage-tiles">
      {layers.map((layer, index) => {
        const enter = phase(t, index * 280, 550);
        const count = phase(t, 700 + index * 280, 1700);
        const { run: headline, lower, v0, final, digits } = layerNumbers(layer, runs);
        const notRun = /not run/i.test(layer.status ?? '') || !isNum(final);
        const shown = isNum(v0) && isNum(final) ? lerp(v0, final, count) : final;
        const delta = deltaText(v0, final, lower, count);
        const improved = isNum(v0) && isNum(final) && (lower ? final < v0 : final > v0);
        return <article key={layer.id} className={`stage-tile${notRun ? ' is-off' : ''}`}
          style={{ opacity: enter, transform: `translateY(${(1 - enter) * 28}px)` }}>
          <header><h2>{layer.name}</h2><span className={`stage-pill stage-pill-${(layer.status ?? 'unknown').replace(/\s+/g, '-')}`}>{layer.status ?? 'unknown'}</span></header>
          <p className="stage-tile-problem">{layer.problem}</p>
          {notRun ? <p className="stage-tile-big is-muted">not run</p> : <>
            <p className="stage-tile-from">v0 cold {formatScore(v0, digits, lower)}</p>
            <p className="stage-tile-big">{formatScore(shown, digits, lower)}</p>
            <p className={`stage-tile-delta${improved ? ' is-up' : ''}`}>{delta}</p>
          </>}
          <p className="stage-tile-foot">{notRun ? 'Zero GPU quota on Crusoe.' : [isNum(layer.best?.n) ? `${layer.best!.n} unseen tasks` : (layer.id === 'inference' ? '48 unseen questions' : null), layer.score_label].filter(Boolean).join(', ')}</p>
          {layer.id === 'placement' && !notRun && <p className="stage-tile-sota">{QAPLIB_SOTA_LINE}<span>{QAPLIB_SOTA_V0}</span></p>}
          {layer.id === 'placement' && !notRun && headline?.name === RACK_NOTE_RUN && <p className="stage-tile-caveat">Rack placement: {RACK_NOTE}</p>}
          {layer.id === 'inference' && !notRun && <p className="stage-tile-caveat">{INFERENCE_CAVEATS[0]}</p>}
        </article>;
      })}
    </div>
  </div>;
}

/* ---------- beat (d): replay ---------- */

function Bars({ claimed, measured, digits, suffix, progress, kept }: { claimed: number | null; measured: number | null; digits: number; suffix: string; progress: number; kept: boolean }) {
  const scale = Math.max(Math.abs(claimed ?? 0), Math.abs(measured ?? 0), 1e-9);
  const bar = (value: number | null) => {
    const width = isNum(value) ? (Math.abs(value) / scale) * 50 * progress : 0;
    return { width: `${width}%`, left: isNum(value) && value < 0 ? `${50 - width}%` : '50%' };
  };
  return <div className="stage-bars">
    {[['claimed', claimed], ['measured', measured]].map(([name, value]) => <div key={name as string} className="stage-bar-row">
      <span className="stage-bar-name">{name as string}</span>
      <span className="stage-bar-track"><span className="stage-bar-zero" />
        <span className={`stage-bar-fill ${name === 'claimed' ? 'is-claim' : kept ? 'is-kept' : 'is-rejected'}`} style={bar(value as number | null)} /></span>
      <span className="stage-bar-value">{isNum(value) ? `${signed((value as number) * progress, digits)}${suffix}` : 'n/a'}</span>
    </div>)}
  </div>;
}

function Detail({ cand, run, onClose }: { cand: ClimbCandidate; run: IndexedRun; onClose: () => void }) {
  const v = verdictOf(cand);
  const sources = (cand.sources ?? []).map(sourceUrl).filter(Boolean);
  return <aside className="stage-detail" role="dialog" aria-label={`Candidate ${cand.id}`}>
    <header>
      <span className={`stage-stamp is-static ${cand.accepted ? 'is-kept' : 'is-rejected'}`}>{cand.accepted ? 'KEPT' : 'REJECTED'}</span>
      <button type="button" className="stage-detail-close" onClick={onClose} aria-label="Close detail">Close</button>
    </header>
    <p className="stage-detail-id">{run.name} / {cand.id} / gen {cand.gen}</p>
    <h3>{cand.change}</h3>
    {cand.rationale && <p className="stage-detail-why">{cand.rationale}</p>}
    <dl>
      <dt>Claimed vs measured</dt>
      <dd><Bars claimed={v.claimed} measured={v.measured} digits={v.digits} suffix={v.suffix} progress={1} kept={cand.accepted} />
        <span className="stage-detail-small">{v.unit === 'cost' ? 'cost change, %' : `${run.climb?.metric ?? 'dev score'} change`} on {cand.n ?? 'n/a'} dev tasks; dev score {fixed(cand.parent_score)} to {fixed(cand.score)}{isNum(cand.cost_pct) ? `, cost ${signed(cand.cost_pct, 1)}%` : ''}</span></dd>
      <dt>Gate</dt><dd>{cand.reason || 'n/a'}{cand.accepted_by ? ` (kept on ${cand.accepted_by})` : ''}</dd>
      <dt>Paper verdict</dt><dd>{cand.verdict ?? 'n/a'}</dd>
      {cand.research_query && <><dt>You.com query</dt><dd className="stage-detail-small">{cand.research_query}</dd></>}
      <dt>You.com sources</dt>
      <dd>{sources.length ? <ul className="stage-detail-sources">{sources.map((url) => <li key={url}><a href={url} target="_blank" rel="noopener noreferrer"><b>{host(url)}</b> {sourceTitle(url)}</a></li>)}</ul> : 'none for this candidate'}</dd>
    </dl>
  </aside>;
}

function Replay({ t, run, runs, layer, pickRun, speed, setSpeed, selected, onSelect }: {
  t: number; run: IndexedRun; runs: IndexedRun[]; layer?: Layer; pickRun: (name: string) => void;
  speed: number; setSpeed: (speed: number) => void; selected: number | null; onSelect: (index: number | null) => void;
}) {
  const cands = candidatesOf(run);
  const plan = replayPlan(run);
  const climb = run.climb;
  const lower = lowerIsBetter(run, layer);
  const baseline = isNum(climb?.baseline) ? climb.baseline : cands[0]?.parent_score ?? null;

  const before = (index: number) => {
    const cand = cands[index];
    const prev = cands[index - 1];
    if (!prev || prev.gen !== cand.gen) return cand.parent_score ?? prev?.incumbent ?? baseline ?? cand.score;
    return prev.incumbent ?? cand.parent_score ?? cand.score;
  };
  const after = (index: number) => cands[index].incumbent ?? before(index);

  const sota = isQaplib(run);
  const values = [baseline, sota ? 1 : null, ...cands.flatMap((c, i) => [c.score, before(i), after(i)])].filter(isNum);
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
  const pad = Math.max((hi - lo) * 0.12, 0.01);
  lo -= pad; hi += pad;
  const W = 1100; const H = 520; const L = 86; const R = 120; const T = 40; const B = 64;
  const x = (i: number) => L + ((i + 0.5) / (plan.n + 0.5)) * (W - L - R);
  const y = (v: number) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  const ticks = Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4);

  const axis = phase(t, 0, 900);
  const lines = phase(t, 900, 900);
  const slotT = (i: number) => (t - plan.candStart - i * plan.slot) / plan.slot;
  const current = Math.min(plan.n - 1, Math.floor((t - plan.candStart) / plan.slot));
  const shownIndex = selected ?? (current >= 0 ? current : null);
  const shown = shownIndex != null ? cands[shownIndex] : null;

  // Running incumbent: tweened during each candidate's rise.
  let incumbent = isNum(baseline) ? baseline : 0;
  cands.forEach((_, i) => {
    const u = slotT(i);
    if (u < 0) return;
    const start = before(i);
    if (isNum(start) && i > 0 && cands[i - 1].gen !== cands[i].gen) incumbent = start;
    else if (i === 0 && isNum(start)) incumbent = start;
    const end = after(i);
    if (isNum(end)) incumbent = lerp(incumbent, end, phase(u, 0.55, 0.3));
  });

  const genBreaks = cands.map((c, i) => (i > 0 && cands[i - 1].gen !== c.gen ? i : -1)).filter((i) => i > 0);
  const held = run.heldout;
  const heldP = phase(t, plan.heldStart, 600);
  const heldBars = lower && held
    ? [{ kind: 'cold', label: 'v0 cold', score: held.v0 }, { kind: 'final', label: 'final', score: held.final }]
    : (['cold', 'brain', 'final'] as const).map((kind) => (climb?.heldout ?? []).find((point) => point.kind === kind)).filter(Boolean)
      .map((point) => ({ kind: point!.kind, label: point!.kind === 'cold' ? 'v0 cold' : point!.kind === 'brain' ? 'v0 + brain' : 'final', score: point!.score }));
  const heldMax = Math.max(...heldBars.map((bar) => bar.score).filter(isNum), 1e-9);
  const deltaP = phase(t, plan.heldStart + 1600, 1200);
  const heldDelta = held ? deltaText(held.v0, held.final, lower, deltaP) : null;
  const papers = cands.filter((c) => c.verdict);
  const verdictCounts = papers.reduce<Record<string, number>>((acc, c) => ({ ...acc, [c.verdict!]: (acc[c.verdict!] ?? 0) + 1 }), {});
  const libP = phase(t, plan.libStart, 600);
  const options = replayableRuns(runs, run.layer);
  const sourcesNow = shown ? Array.from(new Set(cands.filter((c) => c.gen === shown.gen).flatMap((c) => (c.sources ?? []).map(sourceUrl)).filter(Boolean))).slice(0, 4) : [];
  const shownU = shownIndex != null ? (selected != null ? 1 : slotT(shownIndex)) : 0;

  return <div className="stage-slide stage-replay">
    <header className="stage-replay-head">
      <div>
        <p className="stage-kicker">Replay, {run.layer_name}</p>
        <h1 className="stage-title stage-title-sm">{run.benchmark}: {cands.length} candidate {cands.length === 1 ? 'change' : 'changes'}, {cands.filter((c) => c.accepted).length} kept</h1>
      </div>
      <div className="stage-replay-tools">
        <label className="stage-select"><span>Run</span>
          <select value={run.name} onChange={(event) => pickRun(event.target.value)}>
            {options.map((option) => <option key={option.name} value={option.name}>{option.name} ({option.status})</option>)}
          </select>
        </label>
        <div className="stage-speed" role="group" aria-label="Replay speed">
          {[1, 2, 4].map((value) => <button key={value} type="button" aria-pressed={speed === value} onClick={() => setSpeed(value)}>{value}x</button>)}
        </div>
      </div>
    </header>

    <div className="stage-replay-main">
      <div className="stage-chart-wrap">
        <svg className="stage-chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Hill climb of ${run.name}`}>
          <line x1={L} x2={L} y1={T} y2={H - B} className="stage-axis" pathLength={1} style={{ strokeDasharray: 1, strokeDashoffset: 1 - axis }} />
          <line x1={L} x2={W - R} y1={H - B} y2={H - B} className="stage-axis" pathLength={1} style={{ strokeDasharray: 1, strokeDashoffset: 1 - axis }} />
          {ticks.map((tick) => <g key={tick} style={{ opacity: axis }}>
            <line x1={L} x2={W - R} y1={y(tick)} y2={y(tick)} className="stage-grid" />
            <text x={L - 12} y={y(tick) + 6} className="stage-tick" textAnchor="end">{tick.toFixed(hi - lo < 0.1 ? 3 : 2)}</text>
          </g>)}
          <text x={L} y={T - 14} className="stage-axis-label" style={{ opacity: axis }}>{climb?.metric ?? 'dev score'}</text>
          <text x={W - R} y={H - 18} className="stage-axis-label" textAnchor="end" style={{ opacity: axis }}>candidates, in order</text>
          {genBreaks.map((i) => <g key={i} style={{ opacity: phase(slotT(i), -0.1, 0.2) }}>
            <line x1={(x(i) + x(i - 1)) / 2} x2={(x(i) + x(i - 1)) / 2} y1={T} y2={H - B} className="stage-genbreak" />
            <text x={(x(i) + x(i - 1)) / 2 + 8} y={H - B - 12} className="stage-gen-label">gen {cands[i].gen}, re-measured</text>
          </g>)}
          {isNum(baseline) && <g style={{ opacity: lines }}>
            <line x1={L} x2={L + (W - L - R) * lines} y1={y(baseline)} y2={y(baseline)} className="stage-v0-line" />
            <text x={W - R + 12} y={y(baseline) - 4} className="stage-line-label">v0</text>
            <text x={W - R + 12} y={y(baseline) + 18} className="stage-line-label is-sub">{fixed(baseline)}</text>
          </g>}
          {sota && <g style={{ opacity: phase(t, 1300, 800) }}>
            <line x1={L} x2={W - R} y1={y(1)} y2={y(1)} className="stage-sota-line" />
            <text x={L + 10} y={y(1) - 10} className="stage-line-label is-sota">1.0 = best-known (SOTA)</text>
          </g>}
          {cands.map((cand, i) => {
            const u = slotT(i);
            if (u < 0) return null;
            const startX = i === 0 ? L : x(i - 1);
            const startY = y(i === 0 ? before(0) : after(i - 1));
            const mid = (startX + x(i)) / 2;
            const flat = i > 0 && cands[i - 1].gen !== cand.gen
              ? `M${startX} ${startY} H${mid} V${y(before(i))} H${x(i)}` : `M${startX} ${startY} V${y(before(i))} H${x(i)}`;
            const rise = `M${x(i)} ${y(before(i))} V${y(after(i))}`;
            return <g key={cand.id}>
              <path d={flat} className="stage-incumbent" pathLength={1} style={{ strokeDasharray: 1, strokeDashoffset: 1 - phase(u, 0.05, 0.25) }} />
              {cand.accepted && after(i) !== before(i) && <path d={rise} className="stage-incumbent is-rise" pathLength={1}
                style={{ strokeDasharray: 1, strokeDashoffset: 1 - phase(u, 0.55, 0.3) }} />}
            </g>;
          })}
          {cands.map((cand, i) => {
            const u = slotT(i);
            if (u < 0) return null;
            const p = phase(u, 0, 0.18);
            const stamp = phase(u, 0.3, 0.15);
            const labelY = y(cand.score) + (i % 2 === 0 ? -22 : 36);
            return <g key={cand.id} className={`stage-cand${cand.accepted ? ' is-kept' : ' is-rejected'}${shownIndex === i ? ' is-current' : ''}`}
              onClick={() => onSelect(i)} role="button" aria-label={`Candidate ${cand.id}: ${cand.change}`}>
              <circle cx={x(i)} cy={y(cand.score)} r={30} className="stage-cand-hit" />
              <circle cx={x(i)} cy={y(cand.score)} r={11 * p} className="stage-cand-dot" style={{ fillOpacity: stamp > 0 ? 1 : 0.25 }} />
              <text x={i === 0 ? x(i) - 14 : i === plan.n - 1 && plan.n > 1 ? x(i) + 14 : x(i)} y={labelY}
                textAnchor={i === 0 ? 'start' : i === plan.n - 1 && plan.n > 1 ? 'end' : 'middle'} className="stage-cand-label" style={{ opacity: p }}>{shortChange(cand.change, plan.n > 5 ? 18 : 24)}</text>
            </g>;
          })}
        </svg>
      </div>

      <div className="stage-side">
        <div className="stage-counter">
          <span>incumbent, {climb?.metric ?? 'dev score'}</span>
          <strong>{fixed(incumbent, 3)}</strong>
        </div>
        <div className="stage-verdict">
          {shown ? (() => {
            const v = verdictOf(shown);
            const stamp = phase(shownU, 0.3, 0.15);
            return <>
              <p className="stage-verdict-id">{shown.id}, gen {shown.gen}{shown.target ? `, ${shown.target}` : ''}</p>
              <h3>{shown.change}</h3>
              <span className={`stage-stamp ${shown.accepted ? 'is-kept' : 'is-rejected'}`}
                style={{ opacity: stamp, transform: `rotate(-4deg) scale(${1.35 - 0.35 * stamp})` }}>{shown.accepted ? 'KEPT' : 'REJECTED'}</span>
              <Bars claimed={v.claimed} measured={v.measured} digits={v.digits} suffix={v.suffix} progress={phase(shownU, 0.3, 0.35)} kept={shown.accepted} />
              <p className="stage-verdict-reason" style={{ opacity: phase(shownU, 0.45, 0.2) }}>Gate: {shown.reason || 'n/a'}{shown.verdict ? `. Verdict: ${shown.verdict}` : ''}</p>
              <p className="stage-verdict-hint">Click any candidate for detail</p>
            </>;
          })() : <p className="stage-verdict-wait">Waiting for the first candidate</p>}
        </div>
        <div className="stage-sources">
          {sourcesNow.length > 0 && <p className="stage-sources-head">Reviewer read, You.com</p>}
          {sourcesNow.map((url, k) => {
            const p = shownIndex != null ? (selected != null ? 1 : phase(slotT(cands.findIndex((c) => c.gen === shown!.gen)), 0.1 + k * 0.08, 0.2)) : 0;
            return <div key={url} className="stage-source" style={{ opacity: p, transform: `translateX(${(1 - p) * 40}px)` }}>
              <b>{host(url)}</b><span>{sourceTitle(url)}</span>
            </div>;
          })}
        </div>
      </div>
    </div>

    <div className="stage-replay-foot" style={{ opacity: heldP }}>
      <div className="stage-held">
        <p className="stage-held-head">Held-out{held ? `, ${held.metric}` : ''}{lower ? ', lower is better' : ''}</p>
        <div className="stage-held-bars">
          {heldBars.map((bar, k) => {
            const p = phase(t, plan.heldStart + 200 + k * 260, 900);
            const height = isNum(bar.score) ? (bar.score / heldMax) * 100 * p : 0;
            return <div key={bar.kind + k} className={`stage-held-bar is-${bar.kind}`}>
              <span className="stage-held-value">{isNum(bar.score) ? (lower ? `$${(bar.score * p).toFixed(6)}` : (bar.score * p).toFixed(3)) : 'n/a'}</span>
              <span className="stage-held-col"><span style={{ height: `${height}%` }} /></span>
              <span className="stage-held-label">{bar.label}</span>
            </div>;
          })}
        </div>
        {heldDelta && <p className="stage-held-delta" style={{ opacity: phase(t, plan.heldStart + 1600, 300) }}>{lower ? heldDelta : `${heldDelta} on held-out`}</p>}
        {lower && run.layer === 'inference' && <p className="stage-held-caveat">{INFERENCE_CAVEATS[0]}</p>}
        {run.name === RACK_NOTE_RUN && <p className="stage-held-caveat">{RACK_NOTE}</p>}
      </div>
      <div className="stage-lib" style={{ opacity: libP, transform: `translateY(${(1 - libP) * 24}px)` }}>
        <p className="stage-lib-head">Published to SA's Library</p>
        {papers.length ? <>
          <p className="stage-lib-big">{papers.length} {papers.length === 1 ? 'paper' : 'papers'}</p>
          <ul>{Object.entries(verdictCounts).map(([name, count]) => <li key={name}><b>{count}</b> {name}</li>)}</ul>
        </> : <p className="stage-lib-none">No paper verdicts in this export.</p>}
      </div>
    </div>
    {selected != null && cands[selected] && <Detail cand={cands[selected]} run={run} onClose={() => onSelect(null)} />}
  </div>;
}

/* ---------- beat (5): the change as a paper ---------- */

function Paper({ t, run }: { t: number; run?: IndexedRun }) {
  const kept = candidatesOf(run).filter((c) => c.accepted);
  const cand = [...kept].sort((a, b) => (b.gain ?? -Infinity) - (a.gain ?? -Infinity))[0];
  const enter = phase(t, 0, 700);
  if (!run || !cand) return <div className="stage-slide"><p className="stage-kicker">SA's Library</p><h1 className="stage-title stage-title-sm">No kept change to show in this export.</h1></div>;
  const v = verdictOf(cand);
  const sources = (cand.sources ?? []).map(sourceUrl).filter(Boolean);
  const stamp = phase(t, 1400, 300);
  return <div className="stage-slide stage-paper-slide">
    <p className="stage-kicker">The same change, as a paper in SA's Library</p>
    <article className="stage-paper" style={{ opacity: enter, transform: `translateY(${(1 - enter) * 40}px)` }}>
      <p className="stage-paper-meta">{run.layer_name} / {run.name} / {cand.id} / solver {run.solver ?? 'n/a'}, reviewer {run.reviewer ?? 'n/a'}</p>
      <h2>{cand.change}</h2>
      {cand.rationale && <p className="stage-paper-abstract">{cand.rationale}</p>}
      <div className="stage-paper-grid">
        <div><span>Claim</span><strong>{isNum(v.claimed) ? `${signed(v.claimed, v.digits)}${v.suffix}` : 'n/a'}</strong><em>{v.unit === 'cost' ? 'cost, %' : run.climb?.metric ?? 'dev score'}</em></div>
        <div><span>Measured by the gate</span><strong className="is-kept">{isNum(v.measured) ? `${signed(v.measured, v.digits)}${v.suffix}` : 'n/a'}</strong><em>on {cand.n ?? 'n/a'} dev tasks, exact verifier</em></div>
        <div><span>Gate reason</span><strong className="stage-paper-reason">{cand.reason}</strong><em>{cand.accepted_by ? `kept on ${cand.accepted_by}` : ''}</em></div>
      </div>
      {sources.length > 0 && <div className="stage-paper-refs"><span>References, from You.com</span>
        <ol>{sources.map((url, k) => <li key={url} style={{ opacity: phase(t, 900 + k * 200, 400) }}><b>{host(url)}</b> {sourceTitle(url)}</li>)}</ol></div>}
      {cand.verdict && <span className="stage-stamp is-kept stage-paper-stamp" style={{ opacity: stamp, transform: `rotate(-6deg) scale(${1.35 - 0.35 * stamp})` }}>{cand.verdict}</span>}
    </article>
  </div>;
}

/* ---------- beat (e): close ---------- */

function Close({ t, layers }: { t: number; layers: Layer[] }) {
  const kernels = layers.find((layer) => layer.id === 'kernels');
  const lines = ['On QAPLIB and rack placement a 10 s classical search still scores higher.'];
  if (kernelsOnMps(kernels)) lines.push("GPU kernels ran on the Mac's Apple GPU because our Crusoe GPU quota was zero.");
  else if (!kernels || /not run/i.test(kernels.status ?? '')) lines.push('GPU kernels: not run, zero GPU quota');
  const hall = layers.find((layer) => layer.id === 'hall');
  if (hall?.best && isNum(hall.best.score) && isNum(hall.best.baseline) && hall.best.score === hall.best.baseline) {
    lines.push('Hall layout: no change passed the gate, so final equals v0 cold.');
  }
  return <div className="stage-slide stage-close">
    <p className="stage-kicker">What the numbers do not say</p>
    <h1 className="stage-title stage-title-sm">Caveats</h1>
    <ul className="stage-caveats">
      {lines.map((line, k) => { const p = phase(t, 300 + k * 450, 600); return <li key={line} style={{ opacity: p, transform: `translateX(${(1 - p) * 24}px)` }}>{line}</li>; })}
    </ul>
    <ul className="stage-caveats-small" style={{ opacity: phase(t, 300 + lines.length * 450, 600) }}>
      {INFERENCE_CAVEATS.map((line) => <li key={line}>Inference: {line}</li>)}
    </ul>
    <p className="stage-close-foot" style={{ opacity: phase(t, 700 + lines.length * 450, 600) }}>Every number on these slides came from an exact verifier on tasks the loop never saw.</p>
  </div>;
}

/* ---------- presenter shell ---------- */

export default function Stage({ active }: { active: boolean }) {
  const [view, setView] = useState<View | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [beatIndex, setBeatIndex] = useState(0);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [picked, setPicked] = useState<Record<string, string>>({});
  const [selected, setSelected] = useState<number | null>(null);
  const [fullscreen, setFullscreen] = useState(false);
  const reduced = useReducedMotion();
  const rootRef = useRef<HTMLDivElement>(null);
  const tRef = useRef(0);
  useEffect(() => {
    // Reduced motion: every beat opens on its final frame, including the first one.
    if (!reduced) return;
    tRef.current = Number.MAX_SAFE_INTEGER;
    setT(tRef.current);
    setPlaying(false);
  }, [reduced]);

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    const load = () => fetch('/datacenter.json', { cache: 'no-store' })
      .then((response) => { if (!response.ok) throw new Error(`/datacenter.json returned ${response.status}`); return response.json(); })
      .then((json: View) => { if (!cancelled) { setView(json); setError(null); } })
      .catch((err: unknown) => { if (!cancelled) setError(err instanceof Error ? err.message : String(err)); });
    load();
    const timer = window.setInterval(load, 60_000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [active]);

  const layers = useMemo(() => (Array.isArray(view?.layers) ? view!.layers : []), [view]);
  const runs = useMemo(() => (Array.isArray(view?.runs_index) ? view!.runs_index : []), [view]);

  const replayRuns = useMemo(() => {
    const chosen: Record<string, IndexedRun> = {};
    for (const layer of layers) {
      const options = replayableRuns(runs, layer.id);
      const pick = options.find((run) => run.name === picked[layer.id]) ?? defaultRun(runs, layer.id);
      if (pick) chosen[layer.id] = pick;
    }
    return chosen;
  }, [layers, runs, picked]);

  const bestRun = useMemo(() => {
    const defaults = layers.map((layer) => defaultRun(runs, layer.id)).filter((run): run is IndexedRun => Boolean(run));
    return [...defaults].sort((a, b) => heldoutGain(b) - heldoutGain(a))[0];
  }, [layers, runs]);

  const beats: Beat[] = useMemo(() => {
    const replayLayers = layers.filter((layer) => defaultRun(runs, layer.id))
      .sort((a, b) => heldoutGain(defaultRun(runs, b.id)!) - heldoutGain(defaultRun(runs, a.id)!));
    return [
      { key: 'thesis', kind: 'thesis', label: 'Thesis', duration: 6000 },
      { key: 'twin', kind: 'twin', label: 'DoneTwin', duration: 1000 },
      { key: 'layers', kind: 'layers', label: 'Four layers', duration: 3600 },
      ...replayLayers.map((layer): Beat => ({ key: `replay-${layer.id}`, kind: 'replay', label: `Replay: ${layer.name}`, layerId: layer.id,
        duration: replayPlan(replayRuns[layer.id]).total })),
      { key: 'paper', kind: 'paper', label: 'Paper', duration: 2000 },
      { key: 'close', kind: 'close', label: 'Close', duration: 3200 },
    ];
  }, [layers, runs, replayRuns]);

  const beat = beats[Math.min(beatIndex, beats.length - 1)];
  const duration = beat.duration;

  const startBeat = useCallback((index: number) => {
    const next = Math.max(0, Math.min(beats.length - 1, index));
    setBeatIndex(next);
    setSelected(null);
    tRef.current = reduced ? Number.MAX_SAFE_INTEGER : 0;
    setT(tRef.current);
    setPlaying(!reduced);
  }, [beats.length, reduced]);

  const restart = useCallback(() => startBeat(beatIndex), [startBeat, beatIndex]);
  const togglePlay = useCallback(() => {
    if (tRef.current >= duration) { startBeat(beatIndex); return; }
    setSelected(null);
    setPlaying((value) => !value);
  }, [duration, startBeat, beatIndex]);

  useEffect(() => {
    if (!active || !playing) return;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      tRef.current = Math.min(duration, tRef.current + (now - last) * speed);
      last = now;
      setT(tRef.current);
      if (tRef.current >= duration) { setPlaying(false); return; }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [active, playing, speed, duration]);

  const toggleFullscreen = useCallback(() => {
    if (document.fullscreenElement) void document.exitFullscreen();
    else void rootRef.current?.requestFullscreen?.().catch(() => undefined);
  }, []);
  useEffect(() => {
    const onChange = () => setFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener('fullscreenchange', onChange);
    return () => document.removeEventListener('fullscreenchange', onChange);
  }, []);

  const keys = useRef({ next: () => {}, prev: () => {}, togglePlay, restart, toggleFullscreen, close: () => false as boolean });
  keys.current = {
    next: () => startBeat(beatIndex + 1), prev: () => startBeat(beatIndex - 1), togglePlay, restart, toggleFullscreen,
    close: () => { if (selected == null) return false; setSelected(null); return true; },
  };
  useEffect(() => {
    if (!active) return;
    // Capture phase, so the dashboard's own tab-strip arrow keys do not switch tabs while presenting.
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && /^(INPUT|SELECT|TEXTAREA)$/.test(target.tagName)) return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const k = keys.current;
      switch (event.key) {
        case 'ArrowRight': case 'PageDown': k.next(); break;
        case 'ArrowLeft': case 'PageUp': k.prev(); break;
        case ' ': k.togglePlay(); break;
        case 'r': case 'R': k.restart(); break;
        case 'f': case 'F': k.toggleFullscreen(); break;
        case 'Escape': if (!k.close()) return; break;
        default: return;
      }
      event.preventDefault();
      event.stopPropagation();
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [active]);

  const onSelect = (index: number | null) => { setSelected(index); if (index != null) setPlaying(false); };
  const pickRun = (layerId: string, name: string) => { setPicked((prev) => ({ ...prev, [layerId]: name })); startBeat(beatIndex); };
  const clamped = Math.min(t, duration);
  const progress = duration > 0 ? clamped / duration : 1;

  let body: React.ReactNode = null;
  if (!view) body = <div className="stage-slide"><p className="stage-kicker">Stage</p><h1 className="stage-title stage-title-sm">{error ? `Could not load data: ${error}` : 'Loading /datacenter.json'}</h1></div>;
  else if (beat.kind === 'thesis') body = <Thesis t={clamped} pitch={(view.pitch ?? PITCH).replace(/\.$/, '')} />;
  else if (beat.kind === 'twin') body = <Twin t={clamped} hall={layers.find((layer) => layer.id === 'hall')} />;
  else if (beat.kind === 'layers') body = <Layers t={clamped} layers={layers} runs={runs} />;
  else if (beat.kind === 'replay' && beat.layerId && replayRuns[beat.layerId]) body = <Replay key={replayRuns[beat.layerId].name} t={clamped}
    run={replayRuns[beat.layerId]} runs={runs} layer={layers.find((layer) => layer.id === beat.layerId)}
    pickRun={(name) => pickRun(beat.layerId!, name)} speed={speed} setSpeed={setSpeed} selected={selected} onSelect={onSelect} />;
  else if (beat.kind === 'paper') body = <Paper t={clamped} run={bestRun} />;
  else if (beat.kind === 'close') body = <Close t={clamped} layers={layers} />;

  return <div ref={rootRef} className={`stage${fullscreen ? ' is-fullscreen' : ''}`} style={{ '--stage-progress': progress } as CSSProperties}>
    <div className="stage-body">{body}</div>
    <footer className="stage-controls">
      <nav className="stage-beats" aria-label="Beats">
        {beats.map((item, index) => <button key={item.key} type="button" className={`stage-beat${index === beatIndex ? ' is-on' : ''}${index < beatIndex ? ' is-done' : ''}`}
          onClick={() => startBeat(index)} title={item.label} aria-current={index === beatIndex ? 'step' : undefined}>
          <span className="stage-beat-fill" style={{ transform: `scaleX(${index < beatIndex ? 1 : index === beatIndex ? progress : 0})` }} />
          <span className="stage-beat-label">{item.label}</span>
        </button>)}
      </nav>
      <div className="stage-buttons">
        <button type="button" onClick={() => startBeat(beatIndex - 1)} aria-label="Previous beat">Back</button>
        <button type="button" onClick={togglePlay} aria-label={playing ? 'Pause' : 'Play'}>{playing ? 'Pause' : 'Play'}</button>
        <button type="button" onClick={restart} aria-label="Restart beat">Restart</button>
        <button type="button" onClick={() => startBeat(beatIndex + 1)} aria-label="Next beat">Next</button>
        <button type="button" onClick={toggleFullscreen} aria-label="Full screen">{fullscreen ? 'Exit' : 'Full screen'}</button>
        <span className="stage-keys">Arrows, Space, R, F</span>
      </div>
    </footer>
  </div>;
}
