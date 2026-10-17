CREATE TABLE `benchmarks` (
	`id` text PRIMARY KEY NOT NULL,
	`definition` text NOT NULL,
	`version` integer DEFAULT 1 NOT NULL,
	`created_at` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `resource_cloud_capacities` (
	`product_name` text NOT NULL,
	`location` text NOT NULL,
	`quantity` integer NOT NULL,
	`num_slices` integer,
	`checked_at` integer NOT NULL,
	PRIMARY KEY(`product_name`, `location`)
);
--> statement-breakpoint
CREATE TABLE `resource_jobs` (
	`id` text PRIMARY KEY NOT NULL,
	`definition` text NOT NULL,
	`node_id` text,
	`status` text NOT NULL,
	`allocation_held` integer DEFAULT 0 NOT NULL,
	`claim_token` text,
	`lease_until` integer,
	`error` text,
	`created_at` integer NOT NULL,
	`updated_at` integer NOT NULL,
	FOREIGN KEY (`node_id`) REFERENCES `resource_nodes`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE INDEX `resource_jobs_node_allocation` ON `resource_jobs` (`node_id`,`allocation_held`);--> statement-breakpoint
CREATE INDEX `resource_jobs_status_created` ON `resource_jobs` (`status`,`created_at`);--> statement-breakpoint
CREATE TABLE `resource_nodes` (
	`id` text PRIMARY KEY NOT NULL,
	`definition` text NOT NULL,
	`last_seen` integer NOT NULL,
	`created_at` integer NOT NULL,
	`updated_at` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `resource_provider_sync` (
	`provider` text PRIMARY KEY NOT NULL,
	`last_success_at` integer,
	`error` text
);
--> statement-breakpoint
CREATE TABLE `benchmark_submissions` (
	`id` text PRIMARY KEY NOT NULL,
	`benchmark_id` text NOT NULL,
	`definition` text NOT NULL,
	`status` text NOT NULL,
	`score` real,
	`feedback` text,
	`error` text,
	`external_job_id` text,
	`phase` text NOT NULL,
	`next_check_at` integer NOT NULL,
	`deadline_at` integer NOT NULL,
	`lease_token` text,
	`lease_until` integer,
	`created_at` integer NOT NULL,
	`updated_at` integer NOT NULL,
	`completed_at` integer,
	FOREIGN KEY (`benchmark_id`) REFERENCES `benchmarks`(`id`) ON UPDATE no action ON DELETE no action,
	CONSTRAINT "benchmark_submission_status" CHECK("benchmark_submissions"."status" IN ('queued','running','verified','failed','rejected')),
	CONSTRAINT "benchmark_submission_phase" CHECK("benchmark_submissions"."phase" IN ('submit','dispatching','poll','done')),
	CONSTRAINT "benchmark_verified_score" CHECK(("benchmark_submissions"."status"='verified' AND "benchmark_submissions"."score" IS NOT NULL) OR ("benchmark_submissions"."status"!='verified' AND "benchmark_submissions"."score" IS NULL))
);
--> statement-breakpoint
CREATE INDEX `benchmark_submission_history` ON `benchmark_submissions` (`benchmark_id`,`created_at`);--> statement-breakpoint
CREATE INDEX `benchmark_submission_queue` ON `benchmark_submissions` (`phase`,`next_check_at`);