import { integer, real, text, sqliteTable, index, primaryKey, check } from 'drizzle-orm/sqlite-core';
import { sql } from 'drizzle-orm';

export const benchmarks = sqliteTable('benchmarks', {
  id: text('id').primaryKey(), definition: text('definition').notNull(),
  version: integer('version').notNull().default(1), createdAt: integer('created_at').notNull(),
});

export const submissions = sqliteTable('benchmark_submissions', {
  id: text('id').primaryKey(), benchmarkId: text('benchmark_id').notNull().references(() => benchmarks.id),
  definition: text('definition').notNull(), status: text('status').notNull(), score: real('score'),
  feedback: text('feedback'), error: text('error'), externalJobId: text('external_job_id'),
  phase: text('phase').notNull(), nextCheckAt: integer('next_check_at').notNull(),
  deadlineAt: integer('deadline_at').notNull(), leaseToken: text('lease_token'), leaseUntil: integer('lease_until'),
  createdAt: integer('created_at').notNull(), updatedAt: integer('updated_at').notNull(), completedAt: integer('completed_at'),
}, (table) => [
  index('benchmark_submission_history').on(table.benchmarkId, table.createdAt),
  index('benchmark_submission_queue').on(table.phase, table.nextCheckAt),
  check('benchmark_submission_status', sql`${table.status} IN ('queued','running','verified','failed','rejected')`),
  check('benchmark_submission_phase', sql`${table.phase} IN ('submit','dispatching','poll','done')`),
  check('benchmark_verified_score', sql`(${table.status}='verified' AND ${table.score} IS NOT NULL) OR (${table.status}!='verified' AND ${table.score} IS NULL)`),
]);

export const resourceNodes = sqliteTable('resource_nodes', {
  id: text('id').primaryKey(), definition: text('definition').notNull(), lastSeen: integer('last_seen').notNull(),
  createdAt: integer('created_at').notNull(), updatedAt: integer('updated_at').notNull(),
});
export const resourceJobs = sqliteTable('resource_jobs', {
  id: text('id').primaryKey(), definition: text('definition').notNull(), nodeId: text('node_id').references(() => resourceNodes.id),
  status: text('status').notNull(), allocationHeld: integer('allocation_held').notNull().default(0),
  claimToken: text('claim_token'), leaseUntil: integer('lease_until'), error: text('error'),
  createdAt: integer('created_at').notNull(), updatedAt: integer('updated_at').notNull(),
}, (table) => [
  index('resource_jobs_node_allocation').on(table.nodeId, table.allocationHeld),
  index('resource_jobs_status_created').on(table.status, table.createdAt),
]);
export const resourceCloudCapacities = sqliteTable('resource_cloud_capacities', {
  productName: text('product_name').notNull(), location: text('location').notNull(), quantity: integer('quantity').notNull(),
  numSlices: integer('num_slices'), checkedAt: integer('checked_at').notNull(),
}, (table) => [primaryKey({ columns: [table.productName, table.location] })]);
export const resourceProviderSync = sqliteTable('resource_provider_sync', {
  provider: text('provider').primaryKey(), lastSuccessAt: integer('last_success_at'), error: text('error'),
});
