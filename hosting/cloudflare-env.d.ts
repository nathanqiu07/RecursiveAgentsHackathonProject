declare namespace Cloudflare {
  interface Env {
    DB?: D1Database;
    BUCKET?: R2Bucket;
    AGENT_HARNESS_URL?: string;
    AGENT_HARNESS_API_KEY?: string;
    AGENT_HARNESS_ALLOWED_EMAILS?: string;
    AGENT_HARNESS_ALLOWED_USER_IDS?: string;
  }
}
