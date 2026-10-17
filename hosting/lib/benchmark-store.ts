import { benchmarkSchema, submissionSchema } from '../shared/benchmarks';
import type { Benchmark, BenchmarkInput, Submission, SubmissionInput, SubmissionStatus } from '../shared/benchmarks';
import { ApiError } from './api-error';
import { validateSubmissionPayload, validateSubmissionSchema } from './submission-schema';
import { validateRequestTemplate } from './verifier-adapter';

type BenchmarkRow = { id: string; definition: string; version: number; created_at: number };
type SubmissionRow = {
  id: string; benchmark_id: string; definition: string; status: SubmissionStatus; score: number | null;
  feedback: string | null; error: string | null; external_job_id: string | null;
  phase: 'submit' | 'dispatching' | 'poll' | 'done'; next_check_at: number; deadline_at: number;
  lease_token: string | null; lease_until: number | null; created_at: number; updated_at: number; completed_at: number | null;
};
export type VerificationClaim = {
  benchmark: Benchmark; submission: Submission; token: string; phase: 'submit' | 'poll'; deadline_at: number;
};

export class BenchmarkStore {
  db: D1Database;
  now: () => number;
  constructor(db: D1Database, now = Date.now) { this.db = db; this.now = now; }

  async benchmarkView(row: BenchmarkRow): Promise<Benchmark> {
    const definition = JSON.parse(row.definition) as BenchmarkInput;
    const counts = await this.db.prepare(`SELECT count(*) AS submission_count,
      count(CASE WHEN status='verified' THEN 1 END) AS verified_count,
      ${definition.score_direction === 'lower' ? 'min' : 'max'}(CASE WHEN status='verified' THEN score END) AS best_score
      FROM benchmark_submissions WHERE benchmark_id=?`).bind(row.id).first<{
        submission_count: number; verified_count: number; best_score: number | null;
      }>();
    return { ...definition, id: row.id, version: row.version, created_at: row.created_at,
      submission_count: counts?.submission_count ?? 0, verified_count: counts?.verified_count ?? 0, best_score: counts?.best_score ?? null };
  }

  async getBenchmark(id: string): Promise<Benchmark> {
    const row = await this.db.prepare('SELECT * FROM benchmarks WHERE id=?').bind(id).first<BenchmarkRow>();
    if (!row) throw new ApiError(404, 'Benchmark not found');
    return this.benchmarkView(row);
  }

  async listBenchmarks(): Promise<Benchmark[]> {
    const rows = await this.db.prepare(`SELECT b.*,count(s.id) AS submission_count,
      count(CASE WHEN s.status='verified' THEN 1 END) AS verified_count,
      CASE json_extract(b.definition,'$.score_direction') WHEN 'lower'
        THEN min(CASE WHEN s.status='verified' THEN s.score END)
        ELSE max(CASE WHEN s.status='verified' THEN s.score END) END AS best_score
      FROM benchmarks b LEFT JOIN benchmark_submissions s ON s.benchmark_id=b.id
      GROUP BY b.id ORDER BY b.created_at DESC,b.id DESC LIMIT 500`).all<BenchmarkRow & {
        submission_count: number; verified_count: number; best_score: number | null;
      }>();
    return rows.results.map(({ definition, ...row }) => ({ ...JSON.parse(definition), ...row }));
  }

  async createBenchmark(input: unknown): Promise<Benchmark> {
    const definition = benchmarkSchema.parse(input);
    validateSubmissionSchema(definition.submission_schema);
    validateRequestTemplate(definition.verifier.request_template);
    const id = crypto.randomUUID();
    await this.db.prepare('INSERT INTO benchmarks(id,definition,created_at) VALUES(?,?,?)')
      .bind(id, JSON.stringify(definition), this.now()).run();
    return this.getBenchmark(id);
  }

  submissionView(row: SubmissionRow): Submission {
    return { ...JSON.parse(row.definition), id: row.id, benchmark_id: row.benchmark_id, status: row.status,
      score: row.score, feedback: row.feedback, error: row.error, external_job_id: row.external_job_id,
      created_at: row.created_at, updated_at: row.updated_at, completed_at: row.completed_at };
  }

  async getSubmission(id: string): Promise<Submission> {
    const row = await this.db.prepare('SELECT * FROM benchmark_submissions WHERE id=?').bind(id).first<SubmissionRow>();
    if (!row) throw new ApiError(404, 'Submission not found');
    return this.submissionView(row);
  }

  async createSubmission(benchmarkId: string, input: unknown): Promise<Submission> {
    const benchmark = await this.getBenchmark(benchmarkId);
    const definition = submissionSchema.parse(input);
    validateSubmissionPayload(benchmark.submission_schema, definition.payload);
    const id = crypto.randomUUID();
    const now = this.now();
    await this.db.prepare(`INSERT INTO benchmark_submissions
      (id,benchmark_id,definition,status,phase,next_check_at,deadline_at,created_at,updated_at)
      VALUES(?,?,?,'queued','submit',?,?,?,?)`).bind(id, benchmarkId, JSON.stringify(definition),
      now, now + benchmark.verifier.timeout_seconds * 1000, now, now).run();
    return this.getSubmission(id);
  }

  async listSubmissions(benchmarkId: string, limit = 100, offset = 0) {
    await this.getBenchmark(benchmarkId);
    const rows = await this.db.prepare(`SELECT * FROM benchmark_submissions WHERE benchmark_id=?
      ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?`).bind(benchmarkId, limit, offset).all<SubmissionRow>();
    const count = await this.db.prepare('SELECT count(*) AS total FROM benchmark_submissions WHERE benchmark_id=?')
      .bind(benchmarkId).first<{ total: number }>();
    return { submissions: rows.results.map((row) => this.submissionView(row)), total: count?.total ?? 0, limit, offset };
  }

  async leaderboard(benchmarkId: string) {
    const benchmark = await this.getBenchmark(benchmarkId);
    const rows = await this.db.prepare(`SELECT * FROM benchmark_submissions WHERE benchmark_id=? AND status='verified'
      ORDER BY score ${benchmark.score_direction === 'lower' ? 'ASC' : 'DESC'},created_at ASC,id ASC LIMIT 1000`)
      .bind(benchmarkId).all<SubmissionRow>();
    let rank = 0;
    const entries = rows.results.map((row, index) => {
      if (index === 0 || row.score !== rows.results[index - 1].score) rank = index + 1;
      return { ...this.submissionView(row), rank };
    });
    const count = await this.db.prepare(`SELECT count(*) AS total FROM benchmark_submissions WHERE benchmark_id=? AND status='verified'`)
      .bind(benchmarkId).first<{ total: number }>();
    return { entries, total: count?.total ?? 0 };
  }

  async claimDue(): Promise<VerificationClaim | null> {
    const now = this.now();
    await this.db.batch([
      this.db.prepare(`UPDATE benchmark_submissions SET status='failed',phase='done',
        error='Verification dispatch outcome is unknown. Check the verifier before submitting again.',
        lease_token=NULL,lease_until=NULL,completed_at=?,updated_at=? WHERE phase='dispatching' AND lease_until<=?`)
        .bind(now, now, now),
      this.db.prepare(`UPDATE benchmark_submissions SET status='failed',phase='done',error='Verification time limit exceeded',
        lease_token=NULL,lease_until=NULL,completed_at=?,updated_at=? WHERE phase!='done' AND deadline_at<=?`)
        .bind(now, now, now),
    ]);
    const token = crypto.randomUUID();
    // One conditional UPDATE owns both selection and claim; competing isolates cannot claim the same run.
    const row = await this.db.prepare(`UPDATE benchmark_submissions SET lease_token=?,lease_until=?,
      phase=CASE WHEN phase='submit' THEN 'dispatching' ELSE 'poll' END,status='running',updated_at=?
      WHERE id=(SELECT id FROM benchmark_submissions WHERE phase IN ('submit','poll') AND next_check_at<=?
        AND deadline_at>? AND (lease_until IS NULL OR lease_until<=?) ORDER BY next_check_at,created_at,id LIMIT 1)
      AND phase IN ('submit','poll') AND (lease_until IS NULL OR lease_until<=?) RETURNING *`)
      .bind(token, now + 60000, now, now, now, now, now).first<SubmissionRow>();
    if (!row) return null;
    return { benchmark: await this.getBenchmark(row.benchmark_id), submission: this.submissionView(row), token,
      phase: row.phase === 'dispatching' ? 'submit' : 'poll', deadline_at: row.deadline_at };
  }

  async nextDue(): Promise<number | null> {
    const row = await this.db.prepare(`SELECT min(next_check_at) AS due FROM benchmark_submissions
      WHERE phase IN ('submit','poll') AND (lease_until IS NULL OR lease_until<=?)`).bind(this.now()).first<{ due: number | null }>();
    return row?.due ?? null;
  }

  async pending(claim: VerificationClaim, jobId: string, status: 'queued' | 'running', error: string | null = null) {
    const now = this.now();
    await this.db.prepare(`UPDATE benchmark_submissions SET phase='poll',status=?,external_job_id=?,error=?,
      next_check_at=?,lease_token=NULL,lease_until=NULL,updated_at=? WHERE id=? AND lease_token=? AND phase!='done'`)
      .bind(status, jobId, error, now + claim.benchmark.verifier.poll_interval_seconds * 1000, now, claim.submission.id, claim.token).run();
  }

  async finish(claim: VerificationClaim, status: 'verified' | 'failed' | 'rejected', score: number | null = null,
    feedback: string | null = null, error: string | null = null) {
    const now = this.now();
    if (status === 'verified' && (score === null || !Number.isFinite(score))) throw new ApiError(400, 'Verified score must be finite');
    if (now >= claim.deadline_at) { status = 'failed'; score = null; error = 'Verification time limit exceeded'; }
    await this.db.prepare(`UPDATE benchmark_submissions SET status=?,score=?,feedback=?,error=?,phase='done',
      lease_token=NULL,lease_until=NULL,completed_at=?,updated_at=? WHERE id=? AND lease_token=? AND phase!='done'`)
      .bind(status, status === 'verified' ? score : null, feedback, error, now, now, claim.submission.id, claim.token).run();
  }
}
