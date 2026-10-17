import type { LibraryFeed, LibraryIdea, LibraryPaper } from '../shared/library.ts';

const UUID = /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i;
const MAX_SCAN = 1000;
const headers = { 'Cache-Control': 'no-store' };
const record = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value);
const text = (value: unknown, max = 5000) => typeof value === 'string' ? value.replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g, '').slice(0, max).trim() : '';
const nullable = (value: unknown, max = 5000) => text(value, max) || null;
const identifier = (value: unknown) => typeof value === 'string' && UUID.test(value) ? value.toLowerCase() : null;
function date(value: unknown): string {
  if (typeof value !== 'number' && typeof value !== 'string') return '';
  const parsed = new Date(typeof value === 'number' ? value * 1000 : value);
  return Number.isFinite(parsed.getTime()) ? parsed.toISOString() : '';
}
function safeUrl(value: unknown) {
  if (typeof value !== 'string' || value.length > 2048 || /[\s\\\u0000-\u001f\u007f]/.test(value)) return null;
  try {
    const url = new URL(value);
    if (url.protocol !== 'https:' || url.username || url.password || url.hash) return null;
    return url.href;
  } catch { return null; }
}
function connection(env: Record<string, unknown>, kind: 'harness' | 'library') {
  const raw = kind === 'harness' ? env.AGENT_HARNESS_URL : env.LIBRARY_API_URL ?? env.HARNESS_LIBRARY_URL;
  const key = kind === 'harness' ? env.AGENT_HARNESS_API_KEY : env.LIBRARY_API_KEY ?? env.HARNESS_LIBRARY_API_KEY;
  if (typeof raw !== 'string' || !raw || (key !== undefined && key !== '' && (typeof key !== 'string' || /[^\x20-\x7e]/.test(key)))) return null;
  try {
    const url = new URL(raw);
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.pathname !== '/' || url.search || url.hash
      || /[\s\\\u0000-\u001f\u007f]/.test(raw)) return null;
    return { url: url.origin, key: typeof key === 'string' && key ? key : null };
  } catch { return null; }
}
async function body(response: Response) {
  if (!response.ok || (response.headers.get('Content-Encoding') && response.headers.get('Content-Encoding') !== 'identity')) throw new Error('Unavailable');
  const reader = response.body?.getReader();
  if (!reader) throw new Error('Empty');
  let size = 0;
  const chunks: Uint8Array[] = [];
  try {
    while (true) {
      const value = await reader.read();
      if (value.done) break;
      size += value.value.byteLength;
      if (size > 16 * 1024 * 1024) throw new Error('Response too large');
      chunks.push(value.value);
    }
  } finally { await reader.cancel().catch(() => {}); }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  const value: unknown = JSON.parse(new TextDecoder().decode(bytes));
  if (!record(value)) throw new Error('Invalid response');
  return value;
}
function paper(value: unknown): LibraryPaper | null {
  if (!record(value) || !identifier(value.id) || !record(value.submission)) return null;
  const statuses = ['pending_verification', 'planning', 'verifying', 'blocked', 'completed', 'failed'];
  if (!statuses.includes(String(value.status))) return null;
  const submission = value.submission;
  const report = record(value.report) ? value.report : {};
  const submittedBy = nullable(submission.submitted_by, 200);
  const runId = submittedBy?.startsWith('researcher:') ? identifier(submittedBy.slice('researcher:'.length)) : null;
  return {
    id: identifier(value.id)!, title: text(submission.title, 500) || 'Untitled paper',
    paper_text: text(submission.paper_text, 100000), paper_text_truncated: typeof submission.paper_text === 'string' && submission.paper_text.length > 100000, paper_url: safeUrl(submission.paper_url),
    status: value.status as LibraryPaper['status'],
    // A novelty review is never scientific verification; only the Library's final report counts.
    verdict: value.status === 'completed' && ['verified', 'not_verified', 'inconclusive'].includes(String(report.verdict))
      ? report.verdict as LibraryPaper['verdict'] : null,
    summary: nullable(report.summary, 10000), blocked_reason: nullable(value.blocked_reason, 2000),
    node_id: nullable(value.node_id, 200), compute_job_id: identifier(value.compute_job_id),
    compute_job_status: ['queued', 'provisioning', 'running', 'completed', 'failed', 'cancelled'].includes(String(value.compute_job_status)) ? String(value.compute_job_status) : null,
    submitted_by: submittedBy, run_id: runId, created_at: date(value.created_at), started_at: date(value.started_at) || null,
  };
}
function ideas(run: Record<string, unknown>, publications: LibraryPaper[]): LibraryIdea[] {
  const runId = identifier(run.id);
  const request = record(run.request) ? run.request : {};
  if (!runId || request.role !== 'researcher' || !record(run.state)) return [];
  const state = run.state;
  const proposals = Array.isArray(state.research_ideas) ? [...state.research_ideas] : [];
  if (record(state.research_proposal)) proposals.push(state.research_proposal);
  if (!proposals.length && record(state.research_summary) && text(state.research_summary.hypothesis)) proposals.push(state.research_summary);
  const seen = new Set<string>();
  return proposals.slice(0, 200).flatMap((proposal: unknown) => {
    if (!record(proposal)) return [];
    const hypothesis = text(proposal.hypothesis, 10000);
    if (!hypothesis) return [];
    const key = JSON.stringify([hypothesis, proposal.created_at ?? null]);
    if (seen.has(key)) return [];
    seen.add(key);
    const plans = Array.isArray(proposal.experiment_plan) ? proposal.experiment_plan : record(proposal.experiment_plan) ? [proposal.experiment_plan] : [];
    const sources = Array.isArray(proposal.source_urls) ? proposal.source_urls : [];
    const statuses = ['queued', 'running', 'needs_attention', 'completed', 'failed', 'cancelled'];
    const runStatus = statuses.includes(String(run.status)) ? String(run.status) : 'unknown';
    return [{
      id: `draft:${runId}:${seen.size - 1}`, run_id: runId, title: text(proposal.title, 160) || hypothesis.slice(0, 110),
      hypothesis, novelty_difference: text(proposal.novelty_difference, 10000),
      source_urls: [...new Set(sources.map(safeUrl).filter((value): value is string => value !== null))].slice(0, 20),
      experiment_plan: plans.slice(0, 10).flatMap((plan) => record(plan) ? [{ baseline: text(plan.baseline), experiment: text(plan.experiment), metric: text(plan.metric), success_criterion: text(plan.success_criterion) }] : []),
      run_status: runStatus, status: 'unverified' as const, verification_status: 'unverified' as const,
      experiment_count: Array.isArray(state.experiments) ? state.experiments.length : Number.isSafeInteger(state.experiment_count) && Number(state.experiment_count) >= 0 ? Number(state.experiment_count) : 0,
      created_at: date(proposal.created_at) || date(run.created_at), updated_at: date(run.updated_at),
      // Correlation records provenance only; it does not mark any draft as verified.
      paper_id: publications.find((item) => item.run_id === runId)?.id ?? null,
      blocker: runStatus === 'needs_attention' ? 'Research experiments need attention. Open the agent run for details.' : null,
    }];
  });
}

/** Read-only aggregation. Drafts never enter the independent verification queue. */
export async function libraryFeedApi(request: Request, path: string[], env: Record<string, unknown>, requester: typeof fetch = fetch): Promise<Response | null> {
  if (path[0] !== 'library') return null;
  const json = (value: unknown, status = 200, extra = {}) => Response.json(value, { status, headers: { ...headers, ...extra } });
  if (path.length !== 2 || path[1] !== 'feed') return json({ error: 'Library route not found.' }, 404);
  if (request.method !== 'GET') return json({ error: 'Use GET for the Library feed.' }, 405, { Allow: 'GET' });
  const query = new URL(request.url).searchParams;
  if ([...query].some(([key, value]) => !['limit', 'offset'].includes(key) || query.getAll(key).length !== 1 || !/^\d+$/.test(value))) return json({ error: 'Invalid Library pagination.' }, 400);
  const limit = Number(query.get('limit') ?? 100), offset = Number(query.get('offset') ?? 0);
  if (!Number.isSafeInteger(limit) || limit < 1 || limit > 100 || !Number.isSafeInteger(offset) || offset < 0 || offset > 2 ** 31) return json({ error: 'Invalid Library pagination.' }, 400);
  const warnings: string[] = [];
  const signal = AbortSignal.any([request.signal, AbortSignal.timeout(15000)]);
  const read = async (kind: 'harness' | 'library', destination: string) => {
    const selected = connection(env, kind);
    if (!selected) throw new Error('Not configured');
    const outgoing: Record<string, string> = { Accept: 'application/json', 'Accept-Encoding': 'identity' };
    if (selected.key) outgoing.Authorization = `Bearer ${selected.key}`;
    return body(await requester(selected.url + destination, { headers: outgoing, redirect: 'error', signal }));
  };
  const list = async (kind: 'harness' | 'library', key: 'runs' | 'papers') => {
    const entries: unknown[] = [];
    // Library list entries can include 1 MiB manuscripts; keep each page under the body cap.
    const pageSize = key === 'papers' ? 10 : 100;
    try {
      while (entries.length < MAX_SCAN) {
        const response = await read(kind, `/${key}?limit=${pageSize}&offset=${entries.length}`);
        if (!Array.isArray(response[key]) || !Number.isSafeInteger(response.total) || Number(response.total) < 0) throw new Error('Invalid list');
        const page = response[key] as unknown[];
        entries.push(...page.slice(0, pageSize));
        if (entries.length >= Number(response.total)) return { entries, complete: true };
        if (!page.length) throw new Error('Incomplete list');
      }
      warnings.push(`Showing the newest ${MAX_SCAN} ${key}; older records are outside this feed snapshot.`);
    } catch { warnings.push(kind === 'harness' ? 'Research drafts are temporarily unavailable or incomplete.' : 'Library verification status is temporarily unavailable or incomplete.'); }
    return { entries, complete: false };
  };
  const [runList, paperList] = await Promise.all([list('harness', 'runs'), list('library', 'papers')]);
  const papers = paperList.entries.map(paper).filter((value): value is LibraryPaper => value !== null);
  const researchers = runList.entries.filter((run): run is Record<string, unknown> => record(run) && identifier(run.id) !== null && record(run.request) && run.request.role === 'researcher');
  const drafts: LibraryIdea[] = [];
  let cursor = 0, unavailable = false;
  await Promise.all(Array.from({ length: Math.min(6, researchers.length) }, async () => {
    while (cursor < researchers.length) {
      const summary = researchers[cursor++];
      try {
        const detail = await read('harness', `/runs/${identifier(summary.id)}`);
        if (detail.id !== summary.id) throw new Error('Run mismatch');
        drafts.push(...ideas(detail, papers));
      } catch { unavailable = true; drafts.push(...ideas(summary, papers)); }
    }
  }));
  if (unavailable) warnings.push('Some draft details could not be refreshed; available summaries are shown.');
  const combined = [...drafts.map((item) => ({ kind: 'idea', item })), ...papers.map((item) => ({ kind: 'paper', item }))]
    .sort((a, b) => b.item.created_at.localeCompare(a.item.created_at) || a.item.id.localeCompare(b.item.id));
  const page = combined.slice(offset, offset + limit);
  const output: LibraryFeed = {
    ideas: page.filter((entry) => entry.kind === 'idea').map((entry) => entry.item as LibraryIdea),
    papers: page.filter((entry) => entry.kind === 'paper').map((entry) => entry.item as LibraryPaper),
    active_verifications: papers.filter((item) => ['planning', 'verifying'].includes(item.status)),
    counts: {
      unverified: drafts.length + papers.filter((item) => !['planning', 'verifying'].includes(item.status) && item.verdict !== 'verified').length,
      in_verification: papers.filter((item) => ['planning', 'verifying'].includes(item.status)).length,
      verified: papers.filter((item) => item.verdict === 'verified').length,
    },
    verification_available: paperList.complete && papers.length === paperList.entries.length,
    drafts_available: runList.complete && !unavailable,
    pagination: { limit, offset, total: combined.length, has_more: offset + limit < combined.length },
    warnings, checked_at: new Date().toISOString(),
  };
  return json(output);
}
