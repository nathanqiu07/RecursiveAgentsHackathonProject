/* Live demo: one keyboard-driven page over /datacenter.json (rebuilt every 60 s). */
(function () {
  'use strict';

  const REDUCED = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  const DEFAULT_RUN = 'qaplib-dc-research-2';
  const ACCENTS = { hall: 'var(--cyan)', placement: 'var(--lime)', kernels: 'var(--violet)', inference: 'var(--flame)' };
  const $ = (sel) => document.querySelector(sel);
  const isNum = (v) => typeof v === 'number' && isFinite(v);
  const fixed = (v, d = 3) => (isNum(v) ? v.toFixed(d) : 'n/a');
  const signed = (v, d = 3) => (isNum(v) ? (v >= 0 ? '+' : '-') + Math.abs(v).toFixed(d) : 'n/a');
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const money = (v) => (isNum(v) ? '$' + (v < 0.01 ? v.toFixed(6) : v.toFixed(2)) : 'n/a');

  let data = null;

  // ---------- stage + navigation ----------
  const stage = $('#stage');
  function fit() {
    const s = Math.min(innerWidth / 1600, (innerHeight - 44) / 900);
    stage.style.transform = `translate(-50%,-50%) translateY(-18px) scale(${s})`;
  }
  addEventListener('resize', fit); fit();

  const beats = [...document.querySelectorAll('.beat')];
  let cur = 0;
  window.camTarget = 0;
  function go(n) {
    const next = Math.max(0, Math.min(beats.length - 1, n));
    if (next === cur && beats[cur].classList.contains('on')) return;
    cur = next;
    beats.forEach((b, i) => b.classList.toggle('on', i === cur));
    $('#count').textContent = `${cur + 1} / ${beats.length}`;
    window.camTarget = +beats[cur].dataset.cam;
    if (beats[cur].dataset.name === 'replay') replay.start();
    else replay.stop();
    try { history.replaceState(null, '', '#b' + (cur + 1)); } catch (e) { /* file:// or sandbox: hash is a convenience only */ }
  }
  $('#next').onclick = () => go(cur + 1);
  $('#prev').onclick = () => go(cur - 1);
  addEventListener('keydown', (e) => {
    if (e.target && /^(SELECT|INPUT|TEXTAREA)$/.test(e.target.tagName)) return;
    if (['ArrowRight', 'PageDown', ' '].includes(e.key)) { e.preventDefault(); go(cur + 1); }
    else if (['ArrowLeft', 'PageUp'].includes(e.key)) { e.preventDefault(); go(cur - 1); }
    else if (e.key === 'r' || e.key === 'R') { if (beats[cur].dataset.name === 'replay') replay.start(); else replayBeatsAnim(); }
    else if (e.key === 'f' || e.key === 'F') {
      if (!document.fullscreenElement) document.documentElement.requestFullscreen?.().catch(() => {});
      else document.exitFullscreen?.();
    }
  });
  // Restart the CSS entrance animations of the current beat.
  function replayBeatsAnim() {
    const b = beats[cur]; b.classList.remove('on'); void b.offsetWidth; b.classList.add('on');
  }

  // ---------- data helpers ----------
  const runsAll = () => (data && Array.isArray(data.runs_index) ? data.runs_index : []);
  const candsOf = (run) => (run && run.climb && Array.isArray(run.climb.candidates) ? run.climb.candidates : []);
  const lowerBetter = (run) => Boolean(run && run.heldout && run.heldout.lower_is_better);

  function host(url) { try { return new URL(url).hostname.replace(/^www\./, ''); } catch (e) { return String(url); } }
  const sourceUrl = (s) => (typeof s === 'string' ? s : (s && s.url) || '');
  // Title from the URL path; empty when it looks like an id, so only the domain shows.
  function sourceTitle(s) {
    if (s && typeof s === 'object' && s.title && !/^[\w.\-/]*\d[\w.\-/]*$/.test(s.title)) return s.title;
    try {
      const url = sourceUrl(s);
      const parts = new URL(url).pathname.split('/').filter(Boolean);
      const last = decodeURIComponent(parts[parts.length - 1] || '').replace(/\.(pdf|html?|aspx?)$/i, '');
      const words = last.replace(/[-_+%]+/g, ' ').replace(/^\d+\s+/, '').trim();
      if (!words || !/\s/.test(words) || /\d+\.\d+/.test(words) || /^[\d.\sv]+$/.test(words)) return '';
      return words.length > 60 ? words.slice(0, 58) + '...' : words;
    } catch (e) { return ''; }
  }
  const short = (t, n) => (t.length > n ? t.slice(0, n - 1) + '...' : t);
  // A mutation in plain words, from its knob and value (ported from Stage.tsx).
  function plainLabel(c) {
    const change = c.change || '';
    const target = (c.target || '').replace(/^tools?\./, '').replace(/\.enabled$/, '');
    const raw = change.includes('=') ? change.split('=').slice(1).join('=').trim() : '';
    const truthy = /true/i.test(raw) ? true : /false/i.test(raw) ? false : null;
    const words = (t) => t.replace(/[_.]+/g, ' ').trim();
    const step = change.match(/^(\w+)\s+(step|prompt)\s*:\s*(.*)$/i);
    if (step) {
      const role = step[1].toLowerCase(), kind = step[2].toLowerCase();
      if (/runtime.?error/i.test(step[3]) && /abort/i.test(step[3])) return `${role} step: abort on runtime error`;
      const tool = step[3].match(/\b([a-z]+_[a-z_]+)\b/i);
      return tool ? `${role} ${kind}: use ${words(tool[1])}` : `${role} ${kind} edit`;
    }
    if (c.type === 'prompt_edit') return `${words(target || 'role')} prompt edit`;
    if (/thinking|reasoning/i.test(target)) return truthy === false ? 'reasoning off' : 'reasoning on';
    if (/^attempts?$/i.test(target) && /^\d+$/.test(raw)) return `${raw} attempt${raw === '1' ? '' : 's'}`;
    if (c.type === 'tool_set' || /^tools?\b/i.test(change)) return `${words(target)} tool ${truthy === false ? 'off' : 'on'}`;
    if (raw && target) return short(`${words(target)} ${truthy === null ? raw.replace(/^"|"$/g, '') : truthy ? 'on' : 'off'}`, 30);
    return short(change || 'change', 26);
  }
  function verdictOf(c) {
    const v = (c.verdict || '').toLowerCase();
    if (v.includes('smaller')) return { text: 'effect smaller than claimed', cls: 'small' };
    if (v.includes('not reproduced')) return { text: 'not reproduced', cls: 'bad' };
    if (v.includes('reproduced')) return { text: 'accepted', cls: 'ok' };
    return c.accepted ? { text: 'accepted', cls: 'ok' } : { text: 'rejected', cls: 'bad' };
  }

  // ---------- beat 1: hero tiles ----------
  function layerDelta(L) {
    const b = L.best || {};
    const lower = /lower is better|\$/i.test(L.score_label || '');
    if (!isNum(b.delta)) return { text: 'n/a', cls: 'na', better: false, lower };
    if (lower) {
      const pct = Math.round(Math.abs(b.delta) * 100);
      return b.delta < 0 ? { text: `${pct}% less per correct answer`, cls: 'up', better: true, lower }
        : { text: `${pct}% more per correct answer`, cls: 'flat', better: false, lower };
    }
    return { text: `${signed(b.delta, L.digits || 3)} vs v0 cold`, cls: b.delta > 0 ? 'up' : 'flat', better: b.delta > 0, lower };
  }
  function layerNotes(L) {
    const notes = [];
    const b = L.best || {};
    if (L.id === 'hall') notes.push(L.note ? 'v0 starts from the greedy layout.' : 'v0 starts from the greedy layout.');
    if (L.id === 'placement' && L.sota && isNum(L.sota.final_gap) && isNum(L.sota.v0_gap)) {
      notes.push(`QAPLIB: ${L.sota.final_gap.toFixed(1)}% from best-known solutions (SOTA), vs ${L.sota.v0_gap.toFixed(1)}% for v0 cold.`);
    }
    if (L.id === 'kernels') {
      (L.caveats || []).filter((c) => !/eager|cuda/i.test(c)).slice(0, 1).forEach((c) => notes.push(c + '.'));
      notes.push(`${L.device || 'Apple GPU (MPS)'}, our Crusoe GPU quota was zero; no kernel beat PyTorch eager yet.`);
    }
    if (L.id === 'inference') {
      const m = /on (\d+) unseen questions/.exec(L.summary || '');
      const n = m ? +m[1] : null;
      if (n && isNum(b.baseline_accuracy) && isNum(b.accuracy)) {
        notes.push(`Accuracy ${Math.round(b.baseline_accuracy * n)}/${n} → ${Math.round(b.accuracy * n)}/${n} is one question, no gain claimed.`);
      } else notes.push('Accuracy change is one question, no gain claimed.');
    }
    return notes;
  }
  function tileValue(L, v) {
    if (!isNum(v)) return 'n/a';
    if (/\$/.test(L.score_label || '')) return '$' + v.toFixed(6);
    return v.toFixed(L.digits || 3);
  }
  function renderHero() {
    const tiles = $('#tiles');
    const layers = data && Array.isArray(data.layers) ? data.layers : [];
    const order = ['hall', 'placement', 'kernels', 'inference'];
    const fallbackNames = { hall: 'Hall layout', placement: 'Equipment placement', kernels: 'GPU kernels', inference: 'Inference cost' };
    tiles.innerHTML = order.map((id, i) => {
      const L = layers.find((x) => x.id === id);
      if (!L) {
        return `<div class="tile" style="--accent:${ACCENTS[id]};--d:${0.1 + i * 0.12}s"><div class="lname">${fallbackNames[id]}</div><span class="chip run">running</span><div class="pair"><span class="fin">running</span></div></div>`;
      }
      const b = L.best || {};
      const v0 = L.baseline && isNum(L.baseline.score) ? L.baseline.score : b.baseline;
      const d = layerDelta(L);
      const live = runsAll().filter((r) => r.layer === id && r.status === 'running');
      const chip = !isNum(b.score) ? '<span class="chip run">running</span>'
        : d.better ? '<span class="chip">beats v0 cold</span>'
        : '<span class="chip flat">no gain over v0 cold yet</span>';
      const runline = live.length ? `<div class="runline"><span class="chip run">live</span> ${esc(live.map((r) => r.name).join(', '))} running</div>` : '';
      const unit = /\$/.test(L.score_label || '') ? '$ per correct answer' : (L.score_label || 'score');
      return `<div class="tile" style="--accent:${ACCENTS[id]};--d:${0.1 + i * 0.12}s">
        <div class="lname">${esc(L.name || fallbackNames[id])}</div>
        <div class="lprob">${esc(L.problem || '')}</div>
        ${chip}
        <div class="metric">${esc(unit.replace(/\s*\(lower is better\)/i, ''))}${d.lower ? ', lower is better' : ''}</div>
        <div class="pair${/\$/.test(L.score_label || '') ? ' money' : ''}"><span class="v0">${tileValue(L, v0)}</span><span class="arrow">→</span><span class="fin">${tileValue(L, b.score)}</span></div>
        <div class="pl"><span>v0 cold</span><span>final, held-out${isNum(b.n) ? ', n=' + b.n : ''}</span></div>
        <div class="delta ${d.cls}">${esc(d.text)}</div>
        ${runline}
        <ul class="notes">${layerNotes(L).map((n) => `<li>${esc(n)}</li>`).join('')}</ul>
      </div>`;
    }).join('');
  }

  // ---------- beat 4: replay ----------
  const SVGNS = 'http://www.w3.org/2000/svg';
  const W = 980, H = 520, PAD = { l: 70, r: 30, t: 30, b: 70 };
  const replay = (function () {
    let runName = DEFAULT_RUN, speed = 1, raf = 0, t = 0, last = 0, plan = null, playing = false;

    function replayableRuns() {
      return runsAll().filter((r) => candsOf(r).length > 0);
    }
    function pickRun() {
      const list = replayableRuns();
      if (!list.length) return null;
      return list.find((r) => r.name === runName) || list.find((r) => r.name === DEFAULT_RUN) || list[0];
    }
    function fillPicker() {
      const sel = $('#rp-run');
      const list = replayableRuns();
      const want = (pickRun() || {}).name;
      const html = list.map((r) => `<option value="${esc(r.name)}"${r.name === want ? ' selected' : ''}>${esc(r.name)} · ${esc(r.layer_name || r.layer)}${r.status === 'running' ? ' (running)' : ''}</option>`).join('');
      if (sel.innerHTML !== html) sel.innerHTML = html;
    }

    function build() {
      const run = pickRun();
      const svg = $('#rp-chart');
      svg.innerHTML = '';
      $('#rp-cand').innerHTML = '<div class="meta">replay</div><h4>running</h4>';
      $('#rp-srcs').innerHTML = '<div class="hd">Reviewer read, You.com</div>';
      $('#rp-held').innerHTML = '';
      if (!run) { plan = null; return; }
      runName = run.name;
      const climb = run.climb || {};
      const cands = candsOf(run);
      const lower = lowerBetter(run) && /\$|cost/i.test(climb.metric || '');
      const base = isNum(climb.baseline) ? climb.baseline : (cands[0] && cands[0].parent_score);
      // incumbent after each decision
      let inc = base;
      const incs = cands.map((c) => { if (isNum(c.incumbent)) inc = c.incumbent; else if (c.accepted && isNum(c.score)) inc = c.score; return inc; });
      const vals = [base, ...cands.map((c) => c.score), ...incs].filter(isNum);
      let lo = Math.min(...vals), hi = Math.max(...vals);
      if (!vals.length) { lo = 0; hi = 1; }
      const span = hi - lo || Math.abs(hi) * 0.1 || 0.1;
      lo -= span * 0.12; hi += span * 0.12;
      const x = (i) => PAD.l + ((i + 1) / (cands.length + 1)) * (W - PAD.l - PAD.r);
      const y = (v) => {
        const f = (v - lo) / (hi - lo);
        return PAD.t + (lower ? f : 1 - f) * (H - PAD.t - PAD.b);
      };
      const el = (tag, attrs, parent = svg) => { const e = document.createElementNS(SVGNS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; };
      // axes + grid
      for (let k = 0; k <= 4; k++) {
        const v = lo + (k / 4) * (hi - lo);
        el('line', { x1: PAD.l, x2: W - PAD.r, y1: y(v), y2: y(v), stroke: 'rgba(165,156,201,.14)' });
        el('text', { x: PAD.l - 10, y: y(v) + 4, 'text-anchor': 'end', fill: '#a59cc9', 'font-size': 12 }).textContent = v.toFixed(3);
      }
      // generation bands
      const gens = [...new Set(cands.map((c) => c.gen))];
      gens.forEach((g) => {
        const idx = cands.map((c, i) => (c.gen === g ? i : -1)).filter((i) => i >= 0);
        const x0 = x(idx[0]) - 26, x1 = x(idx[idx.length - 1]) + 26;
        el('rect', { x: x0, y: PAD.t, width: x1 - x0, height: H - PAD.t - PAD.b, fill: 'rgba(139,92,246,.07)', rx: 10 });
        el('text', { x: (x0 + x1) / 2, y: H - PAD.b + 26, 'text-anchor': 'middle', fill: '#a59cc9', 'font-size': 12, 'letter-spacing': '.1em' }).textContent = `GEN ${g}`;
      });
      // v0 baseline
      if (isNum(base)) {
        el('line', { x1: PAD.l, x2: W - PAD.r, y1: y(base), y2: y(base), stroke: '#ffc93c', 'stroke-width': 2, 'stroke-dasharray': '7 7', opacity: 0.9 });
        el('text', { x: W - PAD.r, y: y(base) + (lower ? -8 : 18), 'text-anchor': 'end', fill: '#ffc93c', 'font-size': 12 }).textContent = `v0 cold ${base.toFixed(3)}`;
      }
      // incumbent step path
      let d = `M ${PAD.l} ${y(base)}`;
      incs.forEach((v, i) => { d += ` H ${x(i)} V ${y(v)}`; });
      d += ` H ${W - PAD.r}`;
      const incPath = el('path', { d, fill: 'none', stroke: '#22d3ee', 'stroke-width': 3, 'stroke-linejoin': 'round' });
      const totalLen = incPath.getTotalLength ? incPath.getTotalLength() : 2000;
      incPath.style.strokeDasharray = totalLen;
      incPath.style.strokeDashoffset = totalLen;
      // lengths to reach each candidate x along the path
      const reach = cands.map((c, i) => {
        let lenAt = 0; let px = PAD.l, py = y(base);
        for (let j = 0; j <= i; j++) { lenAt += Math.abs(x(j) - px) + Math.abs(y(incs[j]) - (j === 0 ? py : y(incs[j - 1]))); px = x(j); py = y(incs[j]); }
        return lenAt;
      });
      // dots + labels + claim ghost
      const dots = cands.map((c, i) => {
        const g = el('g', { class: 'dot', opacity: 0 });
        const cy = isNum(c.score) ? y(c.score) : y(base);
        if (isNum(c.claim_gain) && isNum(c.parent_score) && !lower) {
          const cl = c.parent_score + c.claim_gain;
          el('line', { x1: x(i), x2: x(i), y1: y(c.parent_score), y2: y(cl), stroke: 'rgba(245,241,255,.35)', 'stroke-width': 2, 'stroke-dasharray': '3 4' }, g);
          el('circle', { cx: x(i), cy: y(cl), r: 6, fill: 'none', stroke: 'rgba(245,241,255,.6)', 'stroke-width': 2 }, g);
        }
        el('circle', { cx: x(i), cy, r: 11, fill: c.accepted ? '#b6f24a' : '#ff5a6e', stroke: '#0b0820', 'stroke-width': 3 }, g);
        const above = i % 2 === 0;
        const ly = Math.max(PAD.t + 14, Math.min(H - PAD.b - 8, cy + (above ? -22 : 30)));
        el('text', { x: x(i), y: ly, 'text-anchor': 'middle', fill: '#f5f1ff', 'font-size': 13, 'font-weight': 600 }).textContent = '';
        const lab = el('text', { x: x(i), y: ly, 'text-anchor': 'middle', fill: '#f5f1ff', 'font-size': 13, 'font-weight': 600 }, g);
        lab.textContent = short(plainLabel(c), 24);
        return g;
      });
      el('text', { x: PAD.l, y: H - 8, fill: '#a59cc9', 'font-size': 12 }).textContent = 'proposed changes, in order →';
      el('text', { x: W - PAD.r, y: H - 8, 'text-anchor': 'end', fill: '#a59cc9', 'font-size': 12 }).textContent = 'hollow ring = the paper\'s claimed score';

      $('#rp-kicker').textContent = `Replay · ${run.layer_name || run.layer} · ${run.name}${run.status === 'running' ? ' (still running)' : ''}`;
      $('#rp-title').textContent = `${cands.length} proposed changes, ${cands.filter((c) => c.accepted).length} kept`;
      $('#rp-metric').textContent = `${climb.metric || 'dev score'}${lower ? ' (lower is better, axis flipped)' : ''}`;

      // held-out + library panel skeleton
      const held = Array.isArray(climb.heldout) ? climb.heldout : [];
      const hk = { cold: 'v0 cold', brain: 'v0 + brain', final: 'final' };
      const hcol = { cold: '#ffc93c', brain: '#8b5cf6', final: '#b6f24a' };
      const hv = held.map((h) => h.score).filter(isNum);
      const hmax = hv.length ? Math.max(...hv) : 1;
      const hmin = hv.length ? Math.min(...hv) : 0;
      $('#rp-held').innerHTML = `<div class="hd">Held-out tasks it never saw${lowerBetter(run) ? ' (lower is better)' : ''}</div>
        <div class="hbars">${held.length ? held.map((h) => `<div class="hbar"><div class="num">${fixed(h.score, 3)}</div><div class="col" data-h="${isNum(h.score) ? Math.max(8, (lowerBetter(run) ? hmin / h.score : h.score / hmax) * 100) : 4}" style="background:${hcol[h.kind] || '#22d3ee'}"></div><div class="lab">${esc(hk[h.kind] || h.label || h.kind)}</div></div>`).join('') : '<div class="more">held-out: running</div>'}</div>
        <div class="lib">${cands.filter((c) => c.accepted).length} kept change${cands.filter((c) => c.accepted).length === 1 ? '' : 's'} published to SA's Library as papers</div>`;

      // timeline (ms at 1x): ~2.3 s per candidate, then held-out and library
      const per = Math.min(2600, Math.max(1500, 13500 / Math.max(1, cands.length)));
      const events = [];
      cands.forEach((c, i) => {
        const t0 = 600 + i * per;
        events.push({ at: t0, fn: () => showCand(run, c, i, dots, incPath, totalLen, reach) });
        events.push({ at: t0 + per * 0.55, fn: () => stamp(c) });
      });
      const tHeld = 600 + cands.length * per + 400;
      events.push({ at: tHeld, fn: () => { incPath.style.strokeDashoffset = 0; showHeld(); } });
      events.push({ at: tHeld + 1800, fn: () => $('#rp-held .lib') && $('#rp-held .lib').classList.add('show') });
      plan = { events, end: tHeld + 2200, idx: 0 };
    }

    function showCand(run, c, i, dots, incPath, totalLen, reach) {
      dots[i].setAttribute('opacity', 1);
      incPath.style.transition = REDUCED ? 'none' : `stroke-dashoffset ${0.9 / speed}s ease-out`;
      incPath.style.strokeDashoffset = Math.max(0, totalLen - reach[i]);
      const claim = c.claim_gain, meas = c.gain;
      const costMode = c.accepted_by === 'cost' || (!isNum(claim) || claim === 0) && isNum(c.claim_cost_pct) && isNum(c.cost_pct) && Math.abs(c.claim_cost_pct) > 1;
      let rows;
      if (costMode) {
        const m = Math.max(Math.abs(c.claim_cost_pct || 0), Math.abs(c.cost_pct || 0), 1);
        rows = [['claimed cost', c.claim_cost_pct, m, '#a59cc9', '%'], ['measured cost', c.cost_pct, m, c.accepted ? '#b6f24a' : '#ff5a6e', '%']];
      } else {
        const m = Math.max(Math.abs(claim || 0), Math.abs(meas || 0), 1e-6);
        rows = [['claimed', claim, m, '#a59cc9', ''], ['measured', meas, m, c.accepted ? '#b6f24a' : '#ff5a6e', '']];
      }
      const fmtRow = (v, unit) => (isNum(v) ? (unit === '%' ? `${v >= 0 ? '+' : ''}${v.toFixed(0)}%` : signed(v, 3)) : 'n/a');
      $('#rp-cand').innerHTML = `<div class="meta">gen ${esc(c.gen)} · change ${i + 1} of ${candsOf(run).length} · ${esc(c.type || 'change')}</div>
        <h4>${esc(plainLabel(c))}</h4>
        <p class="why">${esc(c.rationale || c.change || '')}</p>
        <div class="bars">${rows.map((r) => `<span>${r[0]}</span><div class="track"><div class="fill" data-w="${isNum(r[1]) ? Math.abs(r[1]) / r[2] * 100 : 0}" style="background:${r[3]}"></div></div><span class="val">${fmtRow(r[1], r[4])}</span>`).join('')}</div>
        <div class="stampbox ${c.accepted ? 'st-kept' : 'st-rej'}">${c.accepted ? 'KEPT' : 'REJECTED'}</div>`;
      requestAnimationFrame(() => document.querySelectorAll('#rp-cand .fill').forEach((f) => { f.style.width = f.dataset.w + '%'; }));
      // sources the Reviewer read for this generation
      const gen = candsOf(run).filter((x) => x.gen === c.gen);
      const seen = new Set();
      const srcs = [];
      gen.forEach((x) => (x.sources || []).forEach((s) => { const u = sourceUrl(s); if (u && !seen.has(u)) { seen.add(u); srcs.push(s); } }));
      const box = $('#rp-srcs');
      const key = 'g' + c.gen;
      if (box.dataset.gen !== key) {
        box.dataset.gen = key;
        box.innerHTML = '<div class="hd">Reviewer read, You.com</div>' + (srcs.length
          ? srcs.slice(0, 4).map((s) => { const title = sourceTitle(s); return `<div class="src"><b>${esc(host(sourceUrl(s)))}</b>${title ? `<span>${esc(title)}</span>` : ''}</div>`; }).join('')
          : '<div class="more">no web search for this generation</div>');
        [...box.querySelectorAll('.src')].forEach((n, k) => setTimeout(() => n.classList.add('show'), REDUCED ? 0 : (150 + k * 180) / speed));
      }
    }
    function stamp() { const s = $('#rp-cand .stampbox'); if (s) s.classList.add('show'); }
    function showHeld() {
      const h = $('#rp-held'); h.classList.add('show');
      h.querySelectorAll('.col').forEach((c, k) => setTimeout(() => { c.style.height = c.dataset.h + '%'; }, REDUCED ? 0 : (k * 380) / speed));
    }

    function tick(now) {
      if (!playing || !plan) return;
      t += (now - last) * speed; last = now;
      while (plan.idx < plan.events.length && plan.events[plan.idx].at <= t) { plan.events[plan.idx].fn(); plan.idx++; }
      if (t < plan.end) raf = requestAnimationFrame(tick); else playing = false;
    }
    function start() {
      stop();
      fillPicker();
      build();
      if (!plan) return;
      t = 0; plan.idx = 0;
      if (REDUCED) { plan.events.forEach((e) => e.fn()); plan.idx = plan.events.length; return; }
      playing = true; last = performance.now(); raf = requestAnimationFrame(tick);
    }
    function stop() { playing = false; cancelAnimationFrame(raf); }
    // Jump to any time, used by tests and reduced motion.
    function seek(ms) { if (!plan) return; while (plan.idx < plan.events.length && plan.events[plan.idx].at <= ms) { plan.events[plan.idx].fn(); plan.idx++; } t = ms; }

    $('#rp-run').addEventListener('change', (e) => { runName = e.target.value; start(); e.target.blur(); });
    document.querySelectorAll('.speed button').forEach((b) => b.addEventListener('click', () => {
      speed = +b.dataset.speed;
      document.querySelectorAll('.speed button').forEach((x) => x.classList.toggle('on', x === b));
      b.blur();
    }));
    $('#rp-replay').addEventListener('click', (e) => { start(); e.currentTarget.blur(); });
    return { start, stop, seek, fillPicker, isPlaying: () => playing, current: () => runName };
  })();
  window.__demoReplay = replay;

  // ---------- beat 5: papers ----------
  function renderPapers() {
    const all = runsAll();
    const counts = { accepted: 0, 'effect smaller than claimed': 0, 'not reproduced': 0, rejected: 0 };
    let papers = 0;
    all.forEach((r) => candsOf(r).forEach((c) => { if (!c.verdict && !c.research_query) return; papers++; counts[verdictOf(c).text]++; }));
    const cls = { accepted: 'ok', 'effect smaller than claimed': 'small', 'not reproduced': 'bad', rejected: 'bad' };
    $('#pp-totals').innerHTML = `<span class="vt"><b>${papers}</b>papers across all runs</span>` +
      Object.keys(counts).filter((k) => counts[k] > 0).map((k) => `<span class="vt"><b class="${cls[k] === 'ok' ? 'up' : cls[k] === 'small' ? 'flat' : ''}" style="${cls[k] === 'bad' ? 'color:var(--red)' : ''}">${counts[k]}</b>${k}</span>`).join('');
    const run = all.find((r) => r.name === replay.current() && candsOf(r).length) || all.find((r) => r.name === DEFAULT_RUN) || all.find((r) => candsOf(r).length);
    const tl = $('#pp-timeline');
    if (!run) { tl.innerHTML = '<div class="more">papers: running</div>'; return; }
    $('#pp-sub').textContent = `Papers per generation in ${run.name} (${run.layer_name || run.layer}). The gate reruns each claim on the dev set and stamps the verdict.`;
    const cands = candsOf(run);
    const gens = [...new Set(cands.map((c) => c.gen))].sort((a, b) => a - b);
    let k = 0;
    tl.innerHTML = gens.map((g) => {
      const list = cands.filter((c) => c.gen === g);
      const shown = list.slice(0, 3);
      return `<div class="gen"><div class="gh"><span>Generation ${esc(g)}</span><span>${list.filter((c) => c.accepted).length} kept / ${list.length}</span></div>
        ${shown.map((c) => {
          const v = verdictOf(c);
          const costMode = c.accepted_by === 'cost';
          const nums = costMode
            ? `<span>claimed <b>${isNum(c.claim_cost_pct) ? c.claim_cost_pct.toFixed(0) + '%' : 'n/a'}</b> cost</span><span>measured <b>${isNum(c.cost_pct) ? c.cost_pct.toFixed(0) + '%' : 'n/a'}</b></span>`
            : `<span>claimed <b>${signed(c.claim_gain)}</b></span><span>measured <b>${signed(c.gain)}</b></span>`;
          return `<div class="paper" style="--d:${0.1 + (k++) * 0.12}s"><div class="pm">${esc(c.id || 'paper')} · ${esc(c.type || '')}</div><h4>${esc(plainLabel(c))}</h4><div class="nums">${nums}</div><span class="pv ${v.cls}">${esc(v.text.toUpperCase())}</span></div>`;
        }).join('')}
        ${list.length > shown.length ? `<div class="more">+${list.length - shown.length} more</div>` : ''}</div>`;
    }).join('');
  }

  // ---------- beat 6: receipts ----------
  function renderReceipts() {
    const all = runsAll();
    const proposed = all.reduce((s, r) => s + (isNum(r.candidates) ? r.candidates : candsOf(r).length), 0);
    const kept = all.reduce((s, r) => s + (isNum(r.accepted) ? r.accepted : candsOf(r).filter((c) => c.accepted).length), 0);
    const calls = all.reduce((s, r) => s + (isNum(r.calls) ? r.calls : 0), 0);
    const cost = all.reduce((s, r) => s + (isNum(r.cost_usd) ? r.cost_usd : 0), 0);
    const layers = data && Array.isArray(data.layers) ? data.layers : [];
    const beat = layers.filter((L) => layerDelta(L).better).length;
    const running = all.filter((r) => r.status === 'running').length;
    const stats = data ? [
      [proposed, 'changes proposed'],
      [kept, 'kept by the gate'],
      [proposed - kept, 'rejected'],
      [calls ? calls.toLocaleString('en-US') : 'n/a', 'Nemotron calls, finished runs'],
      [cost ? '$' + cost.toFixed(2) : 'n/a', 'inference cost, finished runs'],
      [`${beat}/${layers.length}`, 'layers beat v0 cold on held-out'],
    ] : [['running', ''], ['running', ''], ['running', ''], ['running', ''], ['running', ''], ['running', '']];
    $('#rc-stats').innerHTML = stats.map((s, i) => `<div class="stat" style="--d:${0.08 * i}s"><div class="v">${esc(s[0])}</div><div class="l">${esc(s[1])}</div></div>`).join('');
    const rows = layers.map((L) => {
      const b = L.best || {}; const d = layerDelta(L);
      const v0 = L.baseline && isNum(L.baseline.score) ? L.baseline.score : b.baseline;
      const nr = all.filter((r) => r.layer === L.id).length;
      return `<tr><td>${esc(L.name)}</td><td>${esc(b.run || 'running')}</td><td class="num">${nr}</td><td class="num">${tileValue(L, v0)}</td><td class="num">${tileValue(L, b.score)}</td><td class="num ${d.cls}">${esc(d.text.replace(' vs v0 cold', ''))}</td></tr>`;
    }).join('');
    $('#rc-table').innerHTML = `<thead><tr><th>Layer</th><th>Best run</th><th class="num">Runs</th><th class="num">v0 cold</th><th class="num">Final, held-out</th><th class="num">Δ vs v0 cold</th></tr></thead><tbody>${rows || '<tr><td colspan="6">running</td></tr>'}</tbody>
      <tfoot><tr><td colspan="6" style="color:var(--dim);font-family:var(--mono);font-size:13px;border:0;padding-top:12px">${all.length} runs in the index, ${all.filter((r) => r.status === 'done').length} done, ${running} running. Every number on this page is computed from datacenter.json${data && data.generated_at ? ', built ' + esc(data.generated_at) : ''}.</td></tr></tfoot>`;
  }

  // ---------- data loop ----------
  function renderAll() {
    renderHero();
    renderPapers();
    renderReceipts();
    replay.fillPicker();
    if (beats[cur].dataset.name === 'replay' && !replay.isPlaying()) replay.start();
  }
  async function load() {
    try {
      const res = await fetch('/datacenter.json?t=' + Date.now(), { cache: 'no-store' });
      if (!res.ok) throw new Error('datacenter.json HTTP ' + res.status);
      data = await res.json();
      $('#asof').textContent = `live data · built ${data.generated_at || 'n/a'}`;
    } catch (err) {
      $('#asof').textContent = 'data: unavailable, retrying in 60 s';
      console.warn('demo: could not load datacenter.json', err);
    }
    renderAll();
  }
  renderAll();
  load();
  setInterval(load, 60000);
  { const m = location.hash.match(/^#b(\d)$/); if (m) go(+m[1] - 1); }

  // ---------- 3D hill background (from the deck) ----------
  (function () {
    if (!window.THREE) return;
    const c = document.getElementById('bg');
    let r;
    try { r = new THREE.WebGLRenderer({ canvas: c, antialias: true, alpha: true }); } catch (e) { return; }
    r.setClearColor(0x000000, 0); r.setPixelRatio(Math.min(devicePixelRatio, 2));
    const sc = new THREE.Scene(); sc.fog = new THREE.FogExp2(0x0b0820, 0.028);
    const cam = new THREE.PerspectiveCamera(50, 1, 0.1, 200);
    const N = 90, S = 60;
    const g = new THREE.PlaneGeometry(S, S, N, N); g.rotateX(-Math.PI / 2);
    const hgt = (x, z) => 9 * Math.exp(-((x - 4) ** 2 + (z + 2) ** 2) / 90) + 3 * Math.exp(-((x + 12) ** 2 + (z - 10) ** 2) / 40) + 0.6 * Math.sin(x * 0.5) * Math.cos(z * 0.4);
    const pos = g.attributes.position, col = [];
    const A = new THREE.Color(0x22d3ee), B = new THREE.Color(0x8b5cf6), C = new THREE.Color(0xff6a2b), tmp = new THREE.Color();
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), z = pos.getZ(i), y = hgt(x, z); pos.setY(i, y);
      const t = Math.min(1, Math.max(0, y / 9)); t < 0.5 ? tmp.copy(A).lerp(B, t * 2) : tmp.copy(B).lerp(C, (t - 0.5) * 2); col.push(tmp.r, tmp.g, tmp.b);
    }
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    sc.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({ vertexColors: true, wireframe: true, transparent: true, opacity: 0.45 })));
    sc.add(new THREE.Points(g, new THREE.PointsMaterial({ size: 0.22, vertexColors: true })));
    const path = []; for (let i = 0; i <= 200; i++) { const a = i / 200; const x = -22 + 26 * a + 3 * Math.sin(a * 9), z = 18 - 20 * a + 2 * Math.cos(a * 7); path.push(new THREE.Vector3(x, hgt(x, z) + 0.5, z)); }
    const curve = new THREE.CatmullRomCurve3(path);
    sc.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 300, 0.08, 8), new THREE.MeshBasicMaterial({ color: 0xb6f24a })));
    const orb = new THREE.Mesh(new THREE.SphereGeometry(0.45, 24, 24), new THREE.MeshBasicMaterial({ color: 0xffffff })); sc.add(orb);
    const halo = new THREE.Mesh(new THREE.SphereGeometry(1.1, 24, 24), new THREE.MeshBasicMaterial({ color: 0xb6f24a, transparent: true, opacity: 0.25 })); sc.add(halo);
    const sheets = []; for (let i = 0; i < 14; i++) {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(1.4, 1.8), new THREE.MeshBasicMaterial({ color: [0xffc93c, 0x22d3ee, 0xff6a2b, 0x8b5cf6][i % 4], transparent: true, opacity: 0.45, side: THREE.DoubleSide }));
      m.position.set((Math.sin(i * 12.9898) * 0.5) * 40, 8 + ((i * 7) % 10), (Math.cos(i * 78.233) * 0.5) * 40); m.userData.s = 0.5 + (i % 5) * 0.4; sc.add(m); sheets.push(m);
    }
    const cams = [[0, 16, 34], [-30, 12, 10], [12, 7, 30], [24, 20, 22], [-22, 22, -18], [30, 28, -6], [0, 40, 0.1]];
    const cp = new THREE.Vector3(...cams[window.camTarget || 0]);
    function size() { r.setSize(innerWidth, innerHeight, false); cam.aspect = innerWidth / innerHeight; cam.updateProjectionMatrix(); }
    addEventListener('resize', size); size();
    const t0 = performance.now();
    function frame(now) {
      // rAF timestamps can predate t0; a negative t would index the curve out of range.
      const t = Math.max(0, (now - t0) / 1000);
      const tg = cams[window.camTarget] || cams[0];
      const target = new THREE.Vector3(tg[0] + Math.sin(t * 0.1) * 3, tg[1], tg[2] + Math.cos(t * 0.1) * 3);
      if (REDUCED) cp.copy(new THREE.Vector3(...tg)); else cp.lerp(target, 0.03);
      cam.position.copy(cp); cam.lookAt(2, 3, 0);
      const a = REDUCED ? 0.6 : (t * 0.06) % 1, p = curve.getPointAt(a); orb.position.copy(p); halo.position.copy(p);
      if (!REDUCED) {
        halo.scale.setScalar(1 + 0.25 * Math.sin(t * 4));
        sheets.forEach((m, i) => { m.rotation.x = t * 0.3 * m.userData.s; m.rotation.y = t * 0.2 * m.userData.s; m.position.y += Math.sin(t + i) * 0.01; });
      }
      r.render(sc, cam);
      // Reduced motion: render only when the beat changes.
      if (!REDUCED) requestAnimationFrame(frame);
    }
    frame(performance.now());
    if (REDUCED) { let lastCam = window.camTarget; setInterval(() => { if (window.camTarget !== lastCam) { lastCam = window.camTarget; frame(performance.now()); } }, 200); }
  })();
})();
