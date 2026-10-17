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
const PROJECT_NAME = 'Bring a verifier. Get a team that improves.';
const THESIS = 'Every layer of an AI data center as a verifiable optimization problem';
const TWIN_URL = 'http://127.0.0.1:8000';
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

const statusText: Record<Tile['status'], string> = { done: 'Done', running: 'Running', 'not run': 'Not run' };

function StatusPill({ status }: { status: Tile['status'] }) {
  return <span className={`benchmark-status hero-status-${status.replace(' ', '-')}`}>{statusText[status]}</span>;
}

function Actions({ onOpenLayers, align }: { onOpenLayers: () => void; align?: 'center' }) {
  return <div className={`hero-actions${align === 'center' ? ' hero-actions-center' : ''}`}>
    <button type="button" className="hero-button hero-button-primary" onClick={onOpenLayers}>See the four layers</button>
    <a className="hero-button hero-button-secondary" href={TWIN_URL} target="_blank" rel="noreferrer">Open the live twin</a>
  </div>;
}

const HOW = 'A Nemotron agent team on Crusoe improves itself through a Shared Brain and Version N+1 rewrites, and keeps a change only when a verifier measures a gain.';

function Eyebrow() {
  return <p className="hero-eyebrow">Recursive Agents Hackathon <span aria-hidden="true">·</span> Shared Brain + Version N+1</p>;
}

function HeroCentered({ onOpenLayers }: { onOpenLayers: () => void }) {
  return <header className="hero hero-a">
    <Eyebrow />
    <p className="hero-name">{PROJECT_NAME}</p>
    <h1 className="hero-title">{THESIS}</h1>
    <p className="hero-lede">{HOW}</p>
    <Actions onOpenLayers={onOpenLayers} align="center" />
  </header>;
}

function HeroSplit({ onOpenLayers, tiles, loaded }: { onOpenLayers: () => void; tiles: Tile[]; loaded: boolean }) {
  return <header className="hero hero-b">
    <div className="hero-b-text">
      <Eyebrow />
      <p className="hero-name">{PROJECT_NAME}</p>
      <h1 className="hero-title">{THESIS}</h1>
      <p className="hero-lede">{HOW}</p>
      <Actions onOpenLayers={onOpenLayers} />
    </div>
    <aside className="hero-b-panel" aria-label="Live results, v0 cold to final on held-out tasks">
      <p className="hero-panel-title">Live: v0 cold → final, held-out</p>
      <ol className="hero-b-list">
        {tiles.map((tile) => <li key={tile.id} className="hero-b-row">
          <div className="hero-b-row-head">
            <span className="hero-b-row-name"><span className="hero-index">{tile.index}</span>{tile.name}</span>
            <StatusPill status={tile.status} />
          </div>
          <div className="hero-b-row-numbers">
            <span className={`hero-b-delta hero-tone-${tile.tone}`}>{loaded ? tile.delta : '…'}</span>
            {tile.deltaSuffix && <span className="hero-b-path">{tile.deltaSuffix}</span>}
            <span className="hero-b-path">{tile.v0} → {tile.final}</span>
          </div>
          {tile.sota && <p className="hero-b-sota">{tile.sota}</p>}
        </li>)}
      </ol>
    </aside>
  </header>;
}

function HeroNumbers({ onOpenLayers, tiles, loaded }: { onOpenLayers: () => void; tiles: Tile[]; loaded: boolean }) {
  return <header className="hero hero-c">
    <div className="hero-c-band" aria-label="Change from v0 cold to final on held-out tasks, per layer">
      {tiles.map((tile) => <div key={tile.id} className="hero-c-stat">
        <span className={`hero-c-delta hero-tone-${tile.tone}`}>{loaded ? tile.delta : '…'}</span>
        {tile.deltaSuffix && <span className="hero-c-suffix">{tile.deltaSuffix}</span>}
        <span className="hero-c-label">{tile.name}</span>
        <span className="hero-c-path">
          {tile.status === 'done' ? `${tile.v0} → ${tile.final}` : statusText[tile.status]}
          {tile.status === 'done' && tile.n != null ? ` on ${tile.n} held-out ${tile.unit}` : ''}
        </span>
        {tile.sota && <span className="hero-c-sota">{tile.sota}</span>}
      </div>)}
    </div>
    <div className="hero-c-text">
      <Eyebrow />
      <p className="hero-name">{PROJECT_NAME}</p>
      <h1 className="hero-title">{THESIS}</h1>
      <p className="hero-lede">{HOW}</p>
      <Actions onOpenLayers={onOpenLayers} />
    </div>
  </header>;
}

// --- The loop diagram: one data list, drawn wide (desktop) or tall (phone). ---
const STEPS = [
  { title: 'Solve', caption: 'Nemotron roles on Crusoe work the task and leave Shared Brain notes.' },
  { title: 'Review', caption: 'A Reviewer reads You.com results and proposes changes with numeric claims.' },
  { title: 'Gate', caption: 'An exact verifier measures each claim on dev tasks: keep or reject.' },
  { title: 'Publish', caption: "Kept changes publish as papers to SA's Library." },
  { title: 'Held-out check', caption: 'The final version runs on tasks the team never saw.' },
];

function wrap(text: string, maxChars: number) {
  const lines: string[] = [];
  let line = '';
  for (const word of text.split(' ')) {
    if (line && (line + ' ' + word).length > maxChars) { lines.push(line); line = word; }
    else line = line ? `${line} ${word}` : word;
  }
  if (line) lines.push(line);
  return lines;
}

function StepBox({ x, y, w, h, index, maxChars }: { x: number; y: number; w: number; h: number; index: number; maxChars: number }) {
  const step = STEPS[index];
  const isExit = index === STEPS.length - 1;
  return <g className={isExit ? 'hero-loop-exit' : 'hero-loop-step'}>
    <rect x={x} y={y} width={w} height={h} rx={10} />
    <text className="hero-loop-num" x={x + 18} y={y + 30}>{index + 1}</text>
    <text className="hero-loop-title" x={x + 40} y={y + 30}>{step.title}</text>
    {wrap(step.caption, maxChars).map((line, lineIndex) =>
      <text key={lineIndex} className="hero-loop-caption" x={x + 18} y={y + 58 + lineIndex * 21}>{line}</text>)}
  </g>;
}

function LoopWide() {
  const w = 206; const h = 160; const gap = 42; const top = 20;
  const xs = STEPS.map((_, index) => index * (w + gap));
  const width = xs[xs.length - 1] + w;
  const bottom = top + h;
  const loopY = bottom + 46;
  return <svg className="hero-loop-wide" viewBox={`-2 0 ${width + 4} ${loopY + 34}`} role="img"
    aria-label="Loop: solve, review, gate, publish, then back to solve as the next version; the final version runs on held-out tasks.">
    <defs>
      <marker id="hero-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" className="hero-loop-head" />
      </marker>
    </defs>
    {STEPS.map((_, index) => index < STEPS.length - 1 && <line key={index} className={index === 3 ? 'hero-loop-edge hero-loop-edge-exit' : 'hero-loop-edge'}
      x1={xs[index] + w + 4} y1={top + h / 2} x2={xs[index + 1] - 6} y2={top + h / 2} markerEnd="url(#hero-arrow)" />)}
    <path className="hero-loop-edge" d={`M${xs[3] + w / 2},${bottom + 4} V${loopY} H${xs[0] + w / 2} V${bottom + 8}`} markerEnd="url(#hero-arrow)" />
    <text className="hero-loop-edge-label" x={(xs[0] + xs[3] + w) / 2} y={loopY + 24} textAnchor="middle">kept changes become version N+1, and the loop repeats</text>
    <text className="hero-loop-edge-label" x={xs[4] - gap / 2} y={top + h / 2 - 12} textAnchor="middle">end</text>
    {STEPS.map((_, index) => <StepBox key={index} x={xs[index]} y={top} w={w} h={h} index={index} maxChars={20} />)}
  </svg>;
}

function LoopTall() {
  const w = 280; const h = 122; const gap = 34; const left = 4;
  const ys = STEPS.map((_, index) => 4 + index * (h + gap));
  const height = ys[ys.length - 1] + h + 6;
  const loopX = left + w + 22;
  return <svg className="hero-loop-tall" viewBox={`0 0 ${loopX + 16} ${height}`} role="img"
    aria-label="Loop: solve, review, gate, publish, then back to solve as the next version; the final version runs on held-out tasks.">
    <defs>
      <marker id="hero-arrow-tall" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" className="hero-loop-head" />
      </marker>
    </defs>
    {STEPS.map((_, index) => index < STEPS.length - 1 && <line key={index} className={index === 3 ? 'hero-loop-edge hero-loop-edge-exit' : 'hero-loop-edge'}
      x1={left + w / 2} y1={ys[index] + h + 3} x2={left + w / 2} y2={ys[index + 1] - 5} markerEnd="url(#hero-arrow-tall)" />)}
    <path className="hero-loop-edge" d={`M${left + w + 3},${ys[3] + h / 2} H${loopX} V${ys[0] + h / 2} H${left + w + 7}`} markerEnd="url(#hero-arrow-tall)" />
    <text className="hero-loop-edge-label" x={loopX + 10} y={(ys[0] + ys[3]) / 2 + h / 2} textAnchor="middle"
      transform={`rotate(90 ${loopX + 10} ${(ys[0] + ys[3]) / 2 + h / 2})`}>version N+1, repeat</text>
    {STEPS.map((_, index) => <StepBox key={index} x={left} y={ys[index]} w={w} h={h} index={index} maxChars={28} />)}
  </svg>;
}

function Loop() {
  return <section className="hero-section" aria-labelledby="hero-loop-heading">
    <div className="hero-section-head">
      <h2 id="hero-loop-heading">The loop</h2>
      <p>One harness, reused unchanged in every layer. A change is kept for a dev gain of 0.05 or more with no lost passes, or a cost cut of 15% or more at equal score.</p>
    </div>
    <div className="hero-loop">
      <LoopWide />
      <LoopTall />
    </div>
  </section>;
}

function LayerTiles({ tiles, loaded, generatedAt, error }: { tiles: Tile[]; loaded: boolean; generatedAt: string; error: string }) {
  return <section className="hero-section" aria-labelledby="hero-layers-heading">
    <div className="hero-section-head">
      <h2 id="hero-layers-heading">Four layers, live</h2>
      <p>Every gain is v0 cold to final, measured on held-out tasks.</p>
    </div>
    <div className="hero-tiles">
      {tiles.map((tile) => <article key={tile.id} className="hero-tile">
        <div className="hero-tile-head">
          <h3><span className="hero-index">{tile.index}</span>{tile.name}</h3>
          <StatusPill status={tile.status} />
        </div>
        <p className="hero-tile-verifier">{tile.verifier}</p>
        <div className="hero-tile-main">
          <span className={`hero-tile-delta hero-tone-${tile.tone}`}>{loaded ? tile.delta : '…'}</span>
          {tile.deltaSuffix && <span className="hero-tile-suffix">{tile.deltaSuffix}</span>}
          <span className="hero-tile-path">
            <span>v0 cold {tile.v0}</span>
            <span>final {tile.final}</span>
          </span>
        </div>
        {tile.sota && <p className="hero-tile-sota">{tile.sota}</p>}
        <p className="hero-tile-meta">
          {tile.deltaSuffix ? '' : tile.scoreLabel}{tile.n != null && tile.status !== 'not run' ? `${tile.deltaSuffix ? '' : ', '}${tile.n} held-out ${tile.unit}` : ''}
        </p>
        {tile.extra && <p className="hero-tile-extra">{tile.extra}</p>}
        {tile.device && <p className="hero-tile-extra">Runs on: {tile.device}</p>}
      </article>)}
    </div>
    <p className="hero-stamp">
      {error ? `Last refresh failed (${error}); showing the last numbers loaded. ` : ''}
      {loaded ? `Data from datacenter.json, updated ${generatedAt || 'n/a'}. Reloads every 60 s.` : !error ? 'Loading datacenter.json…' : ''}
    </p>
  </section>;
}

const PROOF = [
  { title: 'Exact verifiers, no LLM judge', body: "DoneTwin's scorer for halls, recomputed QAP cost for placement, timed kernels with reward-hack guards, and code-computed answers for inference." },
  { title: 'Held-out tasks the team never saw', body: 'Train, dev and held-out splits are disjoint. Touching a held-out task before the final report raises an error.' },
  { title: 'Every claim beside its measurement', body: "Each change carries a numeric claim. The measured result is logged next to it and published to SA's Library." },
];

function Proof() {
  return <section className="hero-section" aria-labelledby="hero-proof-heading">
    <div className="hero-section-head"><h2 id="hero-proof-heading">How we know it's real</h2></div>
    <ol className="hero-proof">
      {PROOF.map((point, index) => <li key={point.title}>
        <span className="hero-proof-num">{index + 1}</span>
        <div><h3>{point.title}</h3><p>{point.body}</p></div>
      </li>)}
    </ol>
  </section>;
}

const STACK = [
  ['Crusoe Managed Inference', 'every model call, with cost per role'],
  ['NVIDIA Nemotron', 'Nano, Super, Lightning'],
  ['You.com search', 'sources for the Reviewer'],
  ['DoneTwin', 'data-hall twin and scorer, by Rikin'],
  ['Dashboard and Library', 'by SA'],
] as const;

function BuiltWith() {
  return <section className="hero-section hero-built" aria-labelledby="hero-built-heading">
    <div className="hero-section-head"><h2 id="hero-built-heading">Built with</h2></div>
    <ul className="hero-stack">
      {STACK.map(([name, detail]) => <li key={name}><strong>{name}</strong><span>{detail}</span></li>)}
    </ul>
    <p className="hero-team">Team: Nathan, SA, Rikin. Crusoe x Cerebral Valley, Recursive Agents Hackathon.</p>
  </section>;
}

function readVariant(): 'a' | 'b' | 'c' {
  if (typeof window === 'undefined') return 'b';
  const value = new URLSearchParams(window.location.search).get('hero')?.toLowerCase();
  return value === 'a' || value === 'c' ? value : 'b';
}

export default function Hero({ active, onOpenLayers }: { active: boolean; onOpenLayers: () => void }) {
  const [variant] = useState(readVariant);
  const { tiles, loaded, generatedAt, error } = useDatacenter(active);
  return <div className={`hero-page hero-page-${variant}`}>
    {variant === 'a' && <HeroCentered onOpenLayers={onOpenLayers} />}
    {variant === 'b' && <HeroSplit onOpenLayers={onOpenLayers} tiles={tiles} loaded={loaded} />}
    {variant === 'c' && <HeroNumbers onOpenLayers={onOpenLayers} tiles={tiles} loaded={loaded} />}
    <Loop />
    <LayerTiles tiles={tiles} loaded={loaded} generatedAt={generatedAt} error={error} />
    <Proof />
    <BuiltWith />
  </div>;
}
