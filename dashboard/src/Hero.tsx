import { useCallback, useEffect, useState } from 'react';
import './benchmarks.css';
import './hero.css';

// The Overview tab. Numbers come from /datacenter.json (scripts/build_datacenter_view.py), the same file the
// AI Data Center tab reads. Its shape is still changing, so every field is optional and missing ones render "n/a".
type Score = { name?: string | null; score?: number | null } | null;
type Best = {
  run?: string; label?: string; score?: number | null; baseline?: number | null; delta?: number | null;
  n?: number | null;
  accuracy?: number | null; baseline_accuracy?: number | null; [key: string]: unknown;
} | null;
type Point = { baseline?: boolean; questions?: number | null; correct?: number | null };
type Layer = {
  id?: string; name?: string; status?: string; score_label?: string; digits?: number; note?: string | null;
  baseline?: Score; best?: Best; summary?: string; scatter?: Point[]; [key: string]: unknown;
};
type View = { generated_at?: string; layers?: Layer[] };

// The name slot. The docs carry no product name, so it holds the PITCH.md tagline.



const REFRESH_MS = 60000;
// Explains the flat hall result; used when datacenter.json carries no note of its own.
const HALL_NOTE = 'v0 starts from the greedy layout, so v0 cold equals greedy here.';
// QAPLIB is the one layer with a published SOTA (best-known solutions). Figures from the team's QAPLIB held-out run;
// datacenter.json does not carry them.
const PLACEMENT_SOTA = 'QAPLIB: 3.6% from best-known solutions (SOTA), 6 unseen instances';

const LAYERS = [
  { id: 'hall', name: 'Hall layout', unit: 'halls', verifier: 'DoneTwin scorer, 0 to 100' },
  { id: 'placement', name: 'Equipment placement', unit: 'rack placements', verifier: 'Recomputed QAP cost vs best known' },
  { id: 'kernels', name: 'GPU kernels', unit: 'problems', verifier: 'Timed kernels vs PyTorch, hack guards' },
  { id: 'inference', name: 'Inference cost', unit: 'questions', verifier: 'Code-computed answers, billed Crusoe cost' },
] as const;
type LayerId = (typeof LAYERS)[number]['id'];

type Tile = {
  id: LayerId; index: number; name: string; unit: string; verifier: string;
  status: 'done' | 'running' | 'not run'; v0: string; final: string; delta: string;
  tone: 'up' | 'flat' | 'down' | 'none'; n: number | null; extra: string; device: string;
  scoreLabel: string; deltaSuffix: string; sota: string;
};

const isNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const leadSentence = (text: string) => text.split(/(?<=\.)\s+(?=[A-Z])/)[0] ?? text;

function readStatus(value: unknown, hasResult: boolean): Tile['status'] {
  if (value === 'done' || value === 'running' || value === 'not run') return value;
  // A missing status while the view is being rebuilt reads as in progress, unless a result is already there.
  return hasResult ? 'done' : 'running';
}

function money(value: number) {
  return `$${value < 0.01 ? value.toFixed(6) : value.toFixed(4)}`;
}

// The JSON may say where kernels ran (for example Apple MPS on the Mac); show whatever it says, verbatim.
function deviceNote(layer: Layer | undefined) {
  if (!layer) return '';
  const keys = ['device', 'hardware', 'platform', 'backend', 'gpu', 'accelerator', 'runs_on'];
  for (const source of [layer, layer.best ?? {}]) {
    for (const key of keys) {
      const value = (source as Record<string, unknown>)[key];
      if (typeof value === 'string' && value.trim()) return value.trim();
    }
  }
  return '';
}

function buildTile(meta: (typeof LAYERS)[number], index: number, layer: Layer | undefined): Tile {
  const best = layer?.best ?? null;
  const lowerIsBetter = meta.id === 'inference' || /lower is better/i.test(layer?.score_label ?? '');
  const digits = isNumber(layer?.digits) ? Math.min(layer.digits, 6) : 3;
  const v0 = isNumber(best?.baseline) ? best.baseline : isNumber(layer?.baseline?.score) ? layer.baseline.score : null;
  const final = isNumber(best?.score) ? best.score : null;
  const format = (value: number | null) => value == null ? 'n/a' : lowerIsBetter ? money(value) : value.toFixed(digits);

  let delta = 'n/a';
  let tone: Tile['tone'] = 'none';
  if (v0 != null && final != null) {
    if (lowerIsBetter && v0 > 0) {
      // Lower is better, so a drop reads as a cut: "31% less", never "at 31% of the cost".
      const pct = Math.round((final / v0 - 1) * 100);
      delta = pct === 0 ? '±0%' : pct < 0 ? `${Math.abs(pct)}% less` : `${pct}% more`;
      tone = pct < 0 ? 'up' : pct > 0 ? 'down' : 'flat';
    } else {
      const gain = final - v0;
      const shown = Math.abs(gain).toFixed(digits);
      const isFlat = Number(shown) === 0;
      delta = isFlat ? `±${shown}` : `${gain > 0 ? '+' : '−'}${shown}`;
      tone = isFlat ? 'flat' : gain > 0 ? 'up' : 'down';
    }
  }


  let n = isNumber(best?.n) ? best.n : null;
  if (n == null && Array.isArray(layer?.scatter)) {
    const finalPoint = layer.scatter.find((point) => point && point.baseline === false);
    if (finalPoint && isNumber(finalPoint.questions)) n = finalPoint.questions;
  }

  let extra = '';
  const v0Point = Array.isArray(layer?.scatter) ? layer.scatter.find((point) => point?.baseline === true) : undefined;
  const finalPoint = Array.isArray(layer?.scatter) ? layer.scatter.find((point) => point?.baseline === false) : undefined;
  if (v0Point && finalPoint && isNumber(v0Point.correct) && isNumber(v0Point.questions)
    && isNumber(finalPoint.correct) && isNumber(finalPoint.questions)) {
    const moved = Math.abs(finalPoint.correct - v0Point.correct);
    const scope = moved === 0 ? 'no change' : moved === 1 ? 'one question, no gain claimed' : `${moved} questions`;
    extra = `Accuracy ${v0Point.correct}/${v0Point.questions} to ${finalPoint.correct}/${finalPoint.questions}, ${scope}.`;
  } else if (isNumber(best?.baseline_accuracy) && isNumber(best?.accuracy)) {
    extra = `Accuracy ${best.baseline_accuracy.toFixed(3)} → ${best.accuracy.toFixed(3)}`;
  } else if (final == null && layer?.summary) {
    extra = leadSentence(layer.summary);
  } else if (typeof layer?.note === 'string' && layer.note) {
    extra = layer.note;
  } else if (meta.id === 'hall') {
    extra = HALL_NOTE;
  }

  return {
    id: meta.id, index: index + 1, name: meta.name, unit: meta.unit, verifier: meta.verifier,
    status: readStatus(layer?.status, final != null), v0: format(v0), final: format(final), delta, tone, n,
    extra, device: deviceNote(layer),
    scoreLabel: lowerIsBetter ? '$ per correct answer' : layer?.score_label ?? '',
    deltaSuffix: final == null ? '' : lowerIsBetter ? 'per correct answer' : meta.id === 'placement' ? 'rack placement, vs v0' : '',
    sota: meta.id === 'placement' ? PLACEMENT_SOTA : '',
  };
}

function useDatacenter(active: boolean) {
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
      // Keep the last good numbers on screen; only the stamp line reports the failure.
      setError(failure instanceof Error ? failure.message : 'Unable to load datacenter.json');
    }
  }, []);
  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    void load(controller.signal);
    const interval = window.setInterval(() => { void load(controller.signal); }, REFRESH_MS);
    return () => { controller.abort(); window.clearInterval(interval); };
  }, [active, load]);
  const layers = Array.isArray(view?.layers) ? view.layers : [];
  const tiles = LAYERS.map((meta, index) => buildTile(meta, index, layers.find((layer) => layer?.id === meta.id)));
  return { tiles, loaded: view != null, generatedAt: view?.generated_at ?? '', error };
}

const statusText = { done: 'Measured', running: 'In progress', 'not run': 'Not measured' };

const LOOP = [
  ['01', 'Research', 'Develop an original method, grounded in prior work.'],
  ['02', 'Experiment', 'Test the idea against a reproducible baseline.'],
  ['03', 'Verify', 'Measure the result independently. Keep what holds up.'],
  ['04', 'Evolve', 'Publish the evidence. Give the next team a better starting point.'],
];

function EvolutionGraphic() {
  return <div className="evolution-graphic" aria-label="Illustration of research flowing through experiments and verification into the next version">
    <div className="evolution-grid" />
    <div className="evolution-orbit orbit-one" /><div className="evolution-orbit orbit-two" />
    <div className="evolution-core"><span>RECURSIVE AGENTS</span><strong>N<span>+1</span></strong><small>Learn. Test. Improve.</small></div>
    <div className="orbit-card orbit-research"><span className="orbit-icon">↗</span><div><small>ORIGINAL RESEARCH</small><strong>A new hypothesis</strong></div></div>
    <div className="orbit-card orbit-experiment"><span className="orbit-icon">⌁</span><div><small>REAL EXPERIMENTS</small><strong>Evidence over intuition</strong></div></div>
    <div className="orbit-card orbit-verify"><span className="orbit-icon">✓</span><div><small>INDEPENDENT VERIFIER</small><strong>Keep or reject</strong></div></div>
    <div className="evolution-caption"><span /> The improvement loop · concept</div>
  </div>;
}

export default function Hero({ active, onOpenLayers, onOpenBenchmarks }: { active: boolean; onOpenLayers: () => void; onOpenBenchmarks: () => void }) {
  const { tiles, loaded, generatedAt, error } = useDatacenter(active);
  return <div className="hero-page">
    <header className="home-intro">
      <div className="home-copy">
        <p className="home-eyebrow"><span /> RECURSIVE AGENTS / VERSION N+1</p>
        <h1>Bring a verifier.<br />Get a team that <em>improves.</em></h1>
        <p className="home-lede">Agents develop original research, run experiments, and rewrite how they work. Every improvement has to prove itself.</p>
        <div className="home-actions"><button className="home-primary" onClick={onOpenBenchmarks}>Explore benchmarks <span aria-hidden="true">↗</span></button><a className="home-secondary" href="#home-loop">How it works <span aria-hidden="true">↓</span></a></div>
        <p className="home-footnote">Built with NVIDIA Nemotron · Crusoe · You.com</p>
      </div>
      <EvolutionGraphic />
    </header>
    <section className="home-visualizations" aria-labelledby="home-visualizations-title">
      <div className="home-section-heading"><div><p className="home-eyebrow">FROM THE DECK</p><h2 id="home-visualizations-title">The ideas. The evidence.</h2></div><p>Recorded runs from the presentation.<br />These charts are a snapshot, not a live feed.</p></div>
      <div className="home-viz-grid">
        <figure><figcaption><h3>Benchmarks: the hill-climb</h3><p>What the agents predicted, what the verifier measured, and which changes were blocked.</p></figcaption><a href="/deck/chart.svg" target="_blank" rel="noreferrer" aria-label="Open benchmark improvement chart at full size"><img src="/deck/chart.svg" alt="Eleven proposed changes across Data hall, QAPLIB and ALE. QAPLIB claimed a 0.070 gain but measured a 0.137 drop. No change clears the keep line." /></a><span>Measured gain · claim · rejected before scoring</span></figure>
        <figure><figcaption><h3>Research papers, round by round</h3><p>Each card is an agent-authored proposal, placed in the round it was written.</p></figcaption><a href="/deck/timeline.svg" target="_blank" rel="noreferrer" aria-label="Open research paper timeline at full size"><img src="/deck/timeline.svg" alt="Eleven research proposals over rounds zero to two. Data hall continues across all rounds; QAPLIB and ALE stop after round zero due to rate limits." /></a><span>One card per proposal · measured development gain</span></figure>
      </div>
    </section>
    <section className="home-loop" id="home-loop" aria-labelledby="home-loop-title">
      <div className="home-section-heading"><div><p className="home-eyebrow">THE METHOD</p><h2 id="home-loop-title">Every version earns its place.</h2></div><p>A shared memory of findings.<br />An independent check on every claim.</p></div>
      <div className="home-steps">{LOOP.map(([number, title, description]) => <article key={number}><span className="home-step-number">{number}</span><h3>{title}<span aria-hidden="true">↗</span></h3><p>{description}</p></article>)}</div>
    </section>
    <section className="home-results" aria-labelledby="home-results-title">
      <div className="home-section-heading"><div><p className="home-eyebrow">ONE LOOP. MANY PROBLEMS.</p><h2 id="home-results-title">From the floor plan<br />to the inference engine.</h2></div><button className="home-text-button" onClick={onOpenLayers}>Explore the four layers <span aria-hidden="true">↗</span></button></div>
      <div className="home-result-grid">{tiles.map(tile => <article key={tile.id} className="home-result-card"><div className="home-result-top"><span>0{tile.index} / {tile.name}</span><span className="home-result-status">{loaded ? statusText[tile.status] : 'Loading'}</span></div><strong className={'home-result-value home-result-' + tile.tone}>{loaded ? tile.delta === 'n/a' ? 'Pending' : tile.delta : '…'}</strong><p className="home-result-path">{tile.v0} <span aria-hidden="true">→</span> {tile.final}{tile.scoreLabel && <span className="home-result-unit">{tile.scoreLabel}</span>}</p><p className="home-result-description">{tile.verifier}</p>{tile.extra && <p className="home-result-note">{tile.extra}</p>}{tile.device && <p className="home-result-note">{tile.device}</p>}</article>)}</div>
      <p className="home-source">Reported baseline → final results from the shared run data. {generatedAt && <>Snapshot: {generatedAt}.</>} {error && <span role="status">Results could not refresh; showing the last available snapshot.</span>}</p>
    </section>
    <footer className="home-outro"><div className="home-monogram">N+1</div><div><h2>The next idea starts here.</h2><p>Bring a measurable problem. Let the team get to work.</p></div><button className="home-primary" onClick={onOpenBenchmarks}>Find a benchmark <span aria-hidden="true">↗</span></button></footer>
  </div>;
}
