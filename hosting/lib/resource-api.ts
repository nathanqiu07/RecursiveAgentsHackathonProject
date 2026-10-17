import { z } from 'zod';
import { finishSchema, jobSchema, leaseSchema, nodeSchema } from '../shared/models';
import type { Capacity, CloudCapacity, Job, Offering, ResourceNode } from '../shared/models';
import { crusoeCatalog } from './resource-catalog';

const LEASE_MS = 120_000;
const CAPACITY_TTL_MS = 60_000;
type ResourceEnv = { DB: D1Database; CRUSOE_ACCESS_KEY_ID?: string; CRUSOE_SECRET_KEY?: string };
type NodeRow = { id: string; definition: string; last_seen: number; created_at: number; updated_at: number };
type JobRow = {
  id: string; definition: string; node_id: string | null; status: Job['status'];
  allocation_held: number; claim_token: string | null; lease_until: number | null;
  error: string | null; created_at: number; updated_at: number;
};
class ResourceError extends Error {
  status: number;
  constructor(status: number, message: string) { super(message); this.status = status; }
}
const zero = (): Capacity => ({ cpus: 0, memory_mb: 0, gpus: 0 });
const json = (value: unknown, status = 200) => Response.json(value, { status, headers: { 'Cache-Control': 'no-store' } });
const querySchema = z.object({
  node_id: z.string().max(100).optional(),
  status: z.enum(['queued', 'running', 'completed', 'failed', 'cancelled']).optional(),
  limit: z.coerce.number().int().min(1).max(500).default(100),
  offset: z.coerce.number().int().min(0).max(2 ** 31).default(0),
}).strict();

async function body(request: Request): Promise<unknown> {
  const text = await request.text();
  if (new TextEncoder().encode(text).byteLength > 256 * 1024) throw new ResourceError(413, 'Request exceeds 256 KiB');
  try { return JSON.parse(text); } catch { throw new ResourceError(400, 'Request body must be valid JSON'); }
}

function jobView(row: JobRow): Job {
  return {
    ...jobSchema.parse(JSON.parse(row.definition)), id: row.id, node_id: row.node_id, status: row.status,
    allocation_held: Boolean(row.allocation_held), lease_until: row.lease_until, error: row.error,
    created_at: row.created_at, updated_at: row.updated_at,
  };
}

async function expire(db: D1Database, now: number) {
  // Lost contact cannot establish that execution stopped; the reservation stays held.
  await db.prepare(`UPDATE resource_jobs SET status='failed',error='Worker heartbeat expired',updated_at=?
    WHERE status='running' AND lease_until<=?`).bind(now, now).run();
}

async function getJob(db: D1Database, id: string): Promise<Job> {
  const row = await db.prepare('SELECT * FROM resource_jobs WHERE id=?').bind(id).first<JobRow>();
  if (!row) throw new ResourceError(404, 'Job not found');
  return jobView(row);
}

async function nodeView(db: D1Database, row: NodeRow, now: number): Promise<ResourceNode> {
  const definition = nodeSchema.parse(JSON.parse(row.definition));
  const held = await db.prepare(`SELECT * FROM resource_jobs WHERE node_id=? AND allocation_held=1 ORDER BY created_at,id`)
    .bind(row.id).all<JobRow>();
  const jobs = held.results.map(jobView);
  const reserved = jobs.reduce((sum, job) => ({
    cpus: (Math.round(sum.cpus * 1000) + Math.round(job.cpus * 1000)) / 1000,
    memory_mb: sum.memory_mb + job.memory_mb, gpus: sum.gpus + job.gpus,
  }), zero());
  const remaining = {
    cpus: Math.max(0, (Math.round(definition.cpus * 1000) - Math.round(reserved.cpus * 1000)) / 1000),
    memory_mb: Math.max(0, definition.memory_mb - reserved.memory_mb),
    gpus: Math.max(0, definition.gpus - reserved.gpus),
  };
  const slots = Math.max(0, definition.max_concurrent_jobs - jobs.length);
  const status: ResourceNode['status'] = row.last_seen <= now - LEASE_MS ? 'offline'
    : jobs.some((job) => job.status !== 'running') ? 'cleanup_required'
      : slots === 0 || remaining.cpus <= 0 || remaining.memory_mb <= 0 ? 'busy' : 'available';
  return { ...definition, id: row.id, status, reserved, available: status === 'available' ? remaining : zero(),
    available_slots: status === 'available' ? slots : 0, jobs, last_seen: row.last_seen,
    created_at: row.created_at, updated_at: row.updated_at };
}

async function getNode(db: D1Database, id: string, now: number) {
  const row = await db.prepare('SELECT * FROM resource_nodes WHERE id=?').bind(id).first<NodeRow>();
  if (!row) throw new ResourceError(404, 'Node not found');
  return nodeView(db, row, now);
}

async function listNodes(db: D1Database, now: number) {
  const rows = await db.prepare('SELECT * FROM resource_nodes ORDER BY id').all<NodeRow>();
  return Promise.all(rows.results.map((row) => nodeView(db, row, now)));
}

async function listJobs(db: D1Database, query: z.infer<typeof querySchema>) {
  const bindings = [query.node_id ?? null, query.node_id ?? null, query.status ?? null, query.status ?? null];
  const rows = await db.prepare(`SELECT * FROM resource_jobs WHERE (? IS NULL OR node_id=?)
    AND (? IS NULL OR status=?) ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?`)
    .bind(...bindings, query.limit, query.offset).all<JobRow>();
  const count = await db.prepare(`SELECT count(*) AS total FROM resource_jobs WHERE (? IS NULL OR node_id=?)
    AND (? IS NULL OR status=?)`).bind(...bindings).first<{ total: number }>();
  return { jobs: rows.results.map(jobView), total: count?.total ?? 0, limit: query.limit, offset: query.offset };
}

async function startJob(db: D1Database, id: string, nodeId: string, now: number) {
  const job = await getJob(db, id);
  if (job.status !== 'queued') throw new ResourceError(409, 'Only queued jobs can start');
  await getNode(db, nodeId, now);
  const token = crypto.randomUUID();
  // This one conditional write checks all held allocations and reserves the job atomically in D1.
  const result = await db.prepare(`UPDATE resource_jobs SET node_id=?,status='running',allocation_held=1,
    claim_token=?,lease_until=?,updated_at=? WHERE id=? AND status='queued' AND EXISTS (
      SELECT 1 FROM resource_nodes n WHERE n.id=? AND n.last_seen>?
      AND NOT EXISTS(SELECT 1 FROM resource_jobs h WHERE h.node_id=n.id AND h.allocation_held=1 AND h.status!='running')
      AND (SELECT count(*) FROM resource_jobs h WHERE h.node_id=n.id AND h.allocation_held=1)<json_extract(n.definition,'$.max_concurrent_jobs')
      AND COALESCE((SELECT sum(CAST(round(json_extract(h.definition,'$.cpus')*1000) AS INTEGER)) FROM resource_jobs h
        WHERE h.node_id=n.id AND h.allocation_held=1),0)+?<=CAST(round(json_extract(n.definition,'$.cpus')*1000) AS INTEGER)
      AND COALESCE((SELECT sum(json_extract(h.definition,'$.memory_mb')) FROM resource_jobs h
        WHERE h.node_id=n.id AND h.allocation_held=1),0)+?<=json_extract(n.definition,'$.memory_mb')
      AND COALESCE((SELECT sum(json_extract(h.definition,'$.gpus')) FROM resource_jobs h
        WHERE h.node_id=n.id AND h.allocation_held=1),0)+?<=json_extract(n.definition,'$.gpus')
    )`).bind(nodeId, token, now + LEASE_MS, now, id, nodeId, now - LEASE_MS,
      Math.round(job.cpus * 1000), job.memory_mb, job.gpus).run();
  if (!result.meta.changes) throw new ResourceError(409, 'Node does not have enough available capacity or job is already assigned');
  return { job: await getJob(db, id), lease_token: token };
}

async function cloudInventory(env: ResourceEnv, now: number) {
  const sync = await env.DB.prepare(`SELECT last_success_at,error FROM resource_provider_sync WHERE provider='crusoe'`)
    .first<{ last_success_at: number | null; error: string | null }>();
  const capacities = await env.DB.prepare('SELECT * FROM resource_cloud_capacities ORDER BY product_name,location')
    .all<CloudCapacity>();
  const configured = Boolean(env.CRUSOE_ACCESS_KEY_ID && env.CRUSOE_SECRET_KEY);
  return {
    configured, status: !configured ? 'not_configured' : sync?.error ? 'error'
      : sync?.last_success_at != null ? 'synced' : 'never_synced',
    last_success_at: sync?.last_success_at ?? null, error: sync?.error ?? null,
    stale: !configured || Boolean(sync?.error) || sync?.last_success_at == null || now - sync.last_success_at >= CAPACITY_TTL_MS,
    offerings: crusoeCatalog as Offering[], capacities: capacities.results,
  };
}

async function syncCrusoe(env: ResourceEnv) {
  const keyId = env.CRUSOE_ACCESS_KEY_ID;
  const secret = env.CRUSOE_SECRET_KEY;
  if (!keyId || !secret) throw new ResourceError(503, 'Connect Crusoe Cloud API credentials to check live capacity');
  try {
    if (!/^[A-Za-z0-9_-]+$/.test(keyId) || !/^[A-Za-z0-9_-]+={0,2}$/.test(secret)) {
      throw new ResourceError(502, 'Crusoe credentials are not in the expected format');
    }
    const path = '/v1/capacities';
    const timestamp = new Date().toISOString().replace(/\.\d{3}Z$/, '+00:00');
    const decoded = atob(secret.replaceAll('-', '+').replaceAll('_', '/'));
    const key = await crypto.subtle.importKey('raw', Uint8Array.from(decoded, (char) => char.charCodeAt(0)),
      { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
    const signature = new Uint8Array(await crypto.subtle.sign('HMAC', key,
      new TextEncoder().encode(`${path}\n\nGET\n${timestamp}\n`)));
    const encoded = btoa(String.fromCharCode(...signature)).replaceAll('+', '-').replaceAll('/', '_').replace(/=+$/, '');
    const response = await fetch(`https://api.cloud.crusoe.ai${path}`, {
      headers: { Authorization: `Bearer 1.0:${keyId}:${encoded}`, 'X-Crusoe-Timestamp': timestamp, Accept: 'application/json' },
      redirect: 'error', signal: AbortSignal.timeout(15000),
    });
    if (!response.ok) throw new ResourceError(502, response.status === 401 || response.status === 403
      ? 'Crusoe credentials were rejected or do not have capacity access' : 'Crusoe capacity service is currently unavailable');
    if (!response.body) throw new ResourceError(502, 'Crusoe returned an empty response');
    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let length = 0;
    try {
      while (true) {
        const part = await reader.read();
        if (part.done) break;
        length += part.value.byteLength;
        if (length > 2 * 1024 * 1024) throw new ResourceError(502, 'Crusoe capacity response is too large');
        chunks.push(part.value);
      }
    } finally { await reader.cancel().catch(() => {}); }
    const joined = new Uint8Array(length);
    let position = 0;
    for (const chunk of chunks) { joined.set(chunk, position); position += chunk.byteLength; }
    const payload = z.object({ items: z.array(z.object({
      type: z.string().min(1).max(200), location: z.string().min(1).max(200),
      quantity: z.number().int().nonnegative(), num_slices: z.number().int().nonnegative().optional(),
    })).max(10000) }).safeParse(JSON.parse(new TextDecoder().decode(joined)));
    if (!payload.success) throw new ResourceError(502, 'Crusoe returned an unexpected capacity format');
    const seen = new Set<string>();
    const now = Date.now();
    const statements = [env.DB.prepare('DELETE FROM resource_cloud_capacities')];
    for (const item of payload.data.items) {
      const pair = JSON.stringify([item.type, item.location]);
      if (seen.has(pair)) throw new ResourceError(502, 'Crusoe returned duplicate capacity records');
      seen.add(pair);
      statements.push(env.DB.prepare(`INSERT INTO resource_cloud_capacities
        (product_name,location,quantity,num_slices,checked_at) VALUES(?,?,?,?,?)`)
        .bind(item.type, item.location, item.quantity, item.num_slices ?? null, now));
    }
    statements.push(env.DB.prepare(`INSERT INTO resource_provider_sync(provider,last_success_at,error) VALUES('crusoe',?,NULL)
      ON CONFLICT(provider) DO UPDATE SET last_success_at=excluded.last_success_at,error=NULL`).bind(now));
    await env.DB.batch(statements);
  } catch (error) {
    const detail = error instanceof ResourceError ? error.message : 'Unable to read live Crusoe capacity; try again shortly';
    await env.DB.prepare(`INSERT INTO resource_provider_sync(provider,error) VALUES('crusoe',?)
      ON CONFLICT(provider) DO UPDATE SET error=excluded.error`).bind(detail).run();
    throw new ResourceError(502, detail);
  }
}

/** Preserve the existing Resources API on Sites, backed by durable D1 records. */
export async function resourceApi(request: Request, path: string[], env: ResourceEnv): Promise<Response | null> {
  if (!['nodes', 'jobs', 'resources', 'providers'].includes(path[0])) return null;
  if (!env.DB) return json({ error: 'Resource storage is unavailable' }, 503);
  const db = env.DB;
  const method = request.method;
  const now = Date.now();
  const url = new URL(request.url);
  try {
    await expire(db, now);
    if (path[0] === 'nodes') {
      if (path.length === 1 && method === 'GET') return json({ nodes: await listNodes(db, now) });
      if (path.length === 2 && path[1] === 'register' && method === 'POST') {
        const node = nodeSchema.parse(await body(request));
        const result = await db.prepare(`INSERT INTO resource_nodes(id,definition,last_seen,created_at,updated_at)
          VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET definition=excluded.definition,last_seen=excluded.last_seen,
          updated_at=excluded.updated_at WHERE NOT EXISTS (
            SELECT 1 FROM resource_jobs WHERE node_id=excluded.id AND allocation_held=1)`)
          .bind(node.id, JSON.stringify(node), now, now, now).run();
        if (!result.meta.changes) throw new ResourceError(409, 'Finish and confirm cleanup of assigned jobs before changing node specifications');
        return json(await getNode(db, node.id, now), 201);
      }
      if (path.length === 2 && method === 'GET') return json(await getNode(db, path[1], now));
      if (path.length === 3 && path[2] === 'heartbeat' && method === 'POST') {
        const result = await db.prepare('UPDATE resource_nodes SET last_seen=? WHERE id=?').bind(now, path[1]).run();
        if (!result.meta.changes) throw new ResourceError(404, 'Node not found');
        return json(await getNode(db, path[1], now));
      }
      if (path.length === 3 && path[2] === 'jobs' && method === 'GET') {
        await getNode(db, path[1], now);
        const query = querySchema.omit({ node_id: true }).parse(Object.fromEntries(url.searchParams));
        return json(await listJobs(db, { ...query, node_id: path[1] }));
      }
    }
    if (path[0] === 'resources' && path[1] === 'available' && path.length === 2 && method === 'GET') {
      const query = z.object({
        cpus: z.coerce.number().finite().nonnegative().default(0),
        memory_mb: z.coerce.number().int().nonnegative().default(0),
        gpus: z.coerce.number().int().nonnegative().default(0),
        gpu_memory_mb: z.coerce.number().int().nonnegative().default(0),
        gpu_model: z.string().max(200).optional(), provider: z.string().max(100).optional(),
        region: z.string().max(100).optional(), allocation_mode: z.enum(['always_on', 'on_demand']).optional(),
      }).strict().parse(Object.fromEntries(url.searchParams));
      const nodes = (await listNodes(db, now)).filter((node) => node.status === 'available'
        && node.available.cpus >= query.cpus && node.available.memory_mb >= query.memory_mb
        && node.available.gpus >= query.gpus && node.gpu_memory_mb >= query.gpu_memory_mb
        && (!query.gpu_model || node.gpu_model.toLowerCase() === query.gpu_model.toLowerCase())
        && (!query.provider || node.provider === query.provider) && (!query.region || node.region === query.region)
        && (!query.allocation_mode || node.allocation_mode === query.allocation_mode));
      return json({ nodes });
    }
    if (path[0] === 'jobs') {
      if (path.length === 1 && method === 'GET') return json(await listJobs(db, querySchema.parse(Object.fromEntries(url.searchParams))));
      if (path.length === 1 && method === 'POST') {
        const definition = jobSchema.parse(await body(request));
        const id = crypto.randomUUID();
        await db.prepare(`INSERT INTO resource_jobs(id,definition,status,created_at,updated_at) VALUES(?,?,'queued',?,?)`)
          .bind(id, JSON.stringify(definition), now, now).run();
        return json(await getJob(db, id), 201);
      }
      if (path.length === 2 && method === 'GET') return json(await getJob(db, path[1]));
      if (path.length === 3 && path[2] === 'start' && method === 'POST') {
        const input = z.object({ node_id: nodeSchema.shape.id }).strict().parse(await body(request));
        return json(await startJob(db, path[1], input.node_id, now));
      }
      if (path.length === 3 && path[2] === 'heartbeat' && method === 'POST') {
        const input = leaseSchema.parse(await body(request));
        const job = await db.prepare(`SELECT node_id FROM resource_jobs WHERE id=? AND claim_token=? AND status='running'`)
          .bind(path[1], input.lease_token).first<{ node_id: string }>();
        if (!job) throw new ResourceError(409, 'Job lease is no longer active');
        // Each guarded statement is safe if cancellation races with this heartbeat.
        const results = await db.batch([
          db.prepare(`UPDATE resource_jobs SET lease_until=?,updated_at=? WHERE id=? AND claim_token=? AND status='running'`)
            .bind(now + LEASE_MS, now, path[1], input.lease_token),
          db.prepare(`UPDATE resource_nodes SET last_seen=? WHERE id=? AND EXISTS (
            SELECT 1 FROM resource_jobs WHERE id=? AND claim_token=? AND status='running')`)
            .bind(now, job.node_id, path[1], input.lease_token),
        ]);
        if (!results[0].meta.changes) throw new ResourceError(409, 'Job lease is no longer active');
        return json(await getJob(db, path[1]));
      }
      if (path.length === 3 && path[2] === 'finish' && method === 'POST') {
        const input = finishSchema.parse(await body(request));
        const result = await db.prepare(`UPDATE resource_jobs SET
          error=CASE WHEN status='running' THEN ? ELSE error END,
          status=CASE WHEN status='running' THEN ? ELSE status END,allocation_held=0,claim_token=NULL,
          lease_until=NULL,updated_at=? WHERE id=? AND claim_token=? AND allocation_held=1`)
          .bind(input.error ?? null, input.status, now, path[1], input.lease_token).run();
        if (!result.meta.changes) throw new ResourceError(409, 'Job allocation or cleanup token is no longer active');
        return json(await getJob(db, path[1]));
      }
      if (path.length === 3 && path[2] === 'cancel' && method === 'POST') {
        await getJob(db, path[1]);
        const result = await db.prepare(`UPDATE resource_jobs SET status='cancelled',lease_until=NULL,updated_at=?
          WHERE id=? AND status IN ('queued','running')`).bind(now, path[1]).run();
        if (!result.meta.changes) throw new ResourceError(409, 'Job is already finished');
        return json(await getJob(db, path[1]));
      }
    }
    if (path[0] === 'providers' && path[1] === 'crusoe') {
      if (path.length === 2 && method === 'GET') return json(await cloudInventory(env, now));
      if (path.length === 3 && path[2] === 'sync' && method === 'POST') {
        await syncCrusoe(env);
        return json(await cloudInventory(env, Date.now()));
      }
    }
    return null;
  } catch (error) {
    if (error instanceof z.ZodError) return json({ error: 'Invalid request', details: error.issues.map(({ path, message }) => ({ path, message })) }, 400);
    if (error instanceof ResourceError) return json({ error: error.message }, error.status);
    throw error;
  }
}
