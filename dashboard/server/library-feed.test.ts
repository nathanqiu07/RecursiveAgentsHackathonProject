import assert from 'node:assert/strict';
import test from 'node:test';
import { libraryFeedApi } from './library-feed.ts';

const RUN = '11111111-1111-4111-8111-111111111111';
const PAPER = '22222222-2222-4222-8222-222222222222';
const env = { AGENT_HARNESS_URL: 'https://harness.example', AGENT_HARNESS_API_KEY: 'private-harness-key',
  LIBRARY_API_URL: 'https://library.example', LIBRARY_API_KEY: 'private-library-key' };
const idea = { hypothesis: 'Reduce GPU register pressure with selective fusion.', novelty_difference: 'A held-out controlled comparison.',
  source_urls: ['https://arxiv.org/abs/1234.5678', 'javascript:alert(1)', 'https://name:secret@example.org/paper'],
  experiment_plan: [{ baseline: 'Unfused graph', experiment: 'Selective fusion', metric: 'Latency', success_criterion: 'Preserve output and improve latency' }], created_at: 1700000000 };
const run = { id: RUN, request: { role: 'researcher' }, status: 'needs_attention', created_at: 1699999999, updated_at: 1700000010,
  state: { research_ideas: [idea], research_proposal: idea, novelty: { approved: true }, experiments: [{ id: 'evidence' }], messages: ['PRIVATE TRANSCRIPT'] } };
function publication(status = 'verifying') {
  return { id: PAPER, created_at: '2023-11-14T22:14:00Z', started_at: '2023-11-14T22:14:01Z', status,
    submission: { title: 'Selective fusion', paper_text: 'Original paper', submitted_by: `researcher:${RUN}` },
    novelty: { verdict: 'novel' }, report: { verdict: 'verified', summary: 'Independent report' },
    compute_job_status: 'queued', node_id: 'api3-a4000' };
}
type Handler = (url: URL, init: RequestInit) => Response;
function fetcher(handler: Handler): typeof fetch {
  return ((url: string | URL | Request, init: RequestInit = {}) => Promise.resolve(handler(new URL(String(url)), init))) as typeof fetch;
}
async function call(handler: Handler, query = '', settings: Record<string, unknown> = env, method = 'GET') {
  const result = await libraryFeedApi(new Request(`http://localhost/api/library/feed${query}`, { method }), ['library', 'feed'], settings, fetcher(handler));
  assert.ok(result);
  return { response: result, body: await result.json() };
}
function reply(url: URL, papers: unknown[] = [publication()]) {
  if (url.pathname === '/papers') return Response.json({ papers, total: papers.length });
  if (url.pathname === '/runs') return Response.json({ runs: [run], total: 1 });
  if (url.pathname === `/runs/${RUN}`) return Response.json(run);
  throw new Error('Unexpected route');
}

test('all historical drafts remain unverified; latest proposal is deduplicated and active verifier work is separate', async () => {
  const { body } = await call((url, init) => {
    assert.equal(init.method, undefined);
    assert.equal(init.redirect, 'error');
    assert.equal(new Headers(init.headers).get('Authorization'), url.host === 'harness.example' ? 'Bearer private-harness-key' : 'Bearer private-library-key');
    return reply(url);
  });
  assert.equal(body.ideas.length, 1);
  assert.equal(body.ideas[0].status, 'unverified');
  assert.equal(body.ideas[0].verification_status, 'unverified');
  assert.equal(body.ideas[0].paper_id, PAPER);
  assert.equal(body.ideas[0].experiment_count, 1);
  assert.equal(body.ideas[0].created_at, '2023-11-14T22:13:20.000Z');
  assert.deepEqual(body.ideas[0].source_urls, ['https://arxiv.org/abs/1234.5678']);
  assert.equal(body.active_verifications[0].id, PAPER);
  assert.equal(body.active_verifications[0].compute_job_status, 'queued');
  assert.equal(body.active_verifications[0].verdict, null);
  assert.deepEqual(body.counts, { unverified: 1, in_verification: 1, verified: 0 });
  assert.equal(body.verification_available, true);
  assert.equal(body.drafts_available, true);
  const serialized = JSON.stringify(body);
  assert.ok(!serialized.includes('private-'));
  assert.ok(!serialized.includes('PRIVATE TRANSCRIPT'));
  assert.ok(!serialized.includes('"novelty":'));
});

test('only completed Library-owned scientific report establishes verified status', async () => {
  const { body } = await call((url) => reply(url, [publication('completed')]));
  assert.equal(body.papers[0].verdict, 'verified');
  assert.deepEqual(body.counts, { unverified: 1, in_verification: 0, verified: 1 });
  assert.equal(body.active_verifications.length, 0);
  const pending = await call((url) => reply(url, [publication('pending_verification')]));
  assert.equal(pending.body.papers[0].verdict, null);
  assert.equal(pending.body.active_verifications.length, 0);
  const planning = await call((url) => reply(url, [publication('planning')]));
  assert.equal(planning.body.active_verifications[0].status, 'planning');
});

test('pagination spans drafts and papers while active verification remains visible on every page', async () => {
  const { body } = await call((url) => reply(url), '?limit=1&offset=1');
  assert.equal(body.ideas.length, 1);
  assert.equal(body.papers.length, 0);
  assert.equal(body.active_verifications.length, 1);
  assert.deepEqual(body.counts, { unverified: 1, in_verification: 1, verified: 0 });
  assert.deepEqual(body.pagination, { limit: 1, offset: 1, total: 2, has_more: false });
});

test('one upstream failure retains independent data and does not expose errors or credentials', async () => {
  const { body } = await call((url) => url.host === 'library.example' ? Response.json({ error: 'SECRET diagnostic' }, { status: 500 }) : reply(url));
  assert.equal(body.ideas.length, 1);
  assert.equal(body.papers.length, 0);
  assert.equal(body.warnings.length, 1);
  assert.equal(body.verification_available, false);
  assert.equal(body.drafts_available, true);
  assert.ok(!JSON.stringify(body).includes('SECRET'));
});

test('bad query and mutation are rejected before upstream reads', async () => {
  for (const query of ['?limit=0', '?limit=101', '?offset=-1', '?limit=1&limit=2', '?url=https://attacker.example']) {
    const { response } = await call(() => { throw new Error('Should not call'); }, query);
    assert.equal(response.status, 400);
  }
  const { response } = await call(() => { throw new Error('Should not call'); }, '', env, 'POST');
  assert.equal(response.status, 405);
});

test('list paging includes older runs and all historical hypotheses', async () => {
  const summaries = Array.from({ length: 100 }, () => ({ id: RUN, request: { role: 'attempt' } }));
  const { body } = await call((url) => {
    if (url.pathname === '/runs') return Response.json({ runs: url.searchParams.get('offset') === '0' ? summaries : [run], total: 101 });
    if (url.pathname === `/runs/${RUN}`) return Response.json({ ...run, state: { ...run.state, research_ideas: [{ ...idea, hypothesis: 'Earlier distinct proposal.' }, idea] } });
    return Response.json({ papers: [], total: 0 });
  });
  assert.equal(body.ideas.length, 2);
  assert.equal(body.pagination.total, 2);
});

test('empty healthy sources are available while missing configuration and draft detail failure are explicit', async () => {
  const empty = await call((url) => Response.json(url.pathname === '/runs' ? { runs: [], total: 0 } : { papers: [], total: 0 }));
  assert.equal(empty.body.verification_available, true);
  assert.equal(empty.body.drafts_available, true);
  assert.deepEqual(empty.body.counts, { unverified: 0, in_verification: 0, verified: 0 });
  const missing = await call((url) => reply(url), '', { AGENT_HARNESS_URL: env.AGENT_HARNESS_URL });
  assert.equal(missing.body.verification_available, false);
  const failed = await call((url) => url.pathname === `/runs/${RUN}` ? Response.json({}, { status: 500 }) : reply(url));
  assert.equal(failed.body.drafts_available, false);
  assert.equal(failed.body.verification_available, true);
});

test('paper excerpts declare truncation without changing full-snapshot counts', async () => {
  const source = publication('completed');
  source.submission.paper_text = 'x'.repeat(100001);
  const { body } = await call((url) => reply(url, [source]), '?limit=1');
  assert.equal(body.papers[0].paper_text.length, 100000);
  assert.equal(body.papers[0].paper_text_truncated, true);
  assert.deepEqual(body.counts, { unverified: 1, in_verification: 0, verified: 1 });
});
