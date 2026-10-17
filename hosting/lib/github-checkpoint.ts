export class PublicCheckpointError extends Error {
  rejected: boolean;
  constructor(message: string, rejected = false) { super(message); this.rejected = rejected; }
}

export function checkpointReference(value: unknown): { apiUrl: string; sha: string } {
  if (typeof value !== 'string') throw new PublicCheckpointError('A public GitHub base checkpoint pinned to a full commit SHA is required.', true);
  let url: URL;
  try { url = new URL(value); }
  catch { throw new PublicCheckpointError('Base checkpoint must be a public GitHub commit or tree URL pinned to a full commit SHA.', true); }
  const match = url.pathname.match(/^\/([A-Za-z0-9-]+)\/([A-Za-z0-9_.-]+)\/(?:commit|tree)\/([a-fA-F0-9]{40}|[a-fA-F0-9]{64})\/?$/);
  if (url.protocol !== 'https:' || url.hostname !== 'github.com' || url.port || url.username || url.password
    || url.search || url.hash || !match) {
    throw new PublicCheckpointError('Base checkpoint must be a public https://github.com/OWNER/REPO/commit/FULL_SHA or /tree/FULL_SHA URL.', true);
  }
  return { apiUrl: `https://api.github.com/repos/${encodeURIComponent(match[1])}/${encodeURIComponent(match[2])}/git/commits/${match[3]}`, sha: match[3] };
}

/** Public access is confirmed with an anonymous GitHub API read before any verifier dispatch. */
export async function verifyPublicCheckpoint(value: unknown, request: typeof fetch = fetch, signal?: AbortSignal): Promise<void> {
  const { apiUrl, sha } = checkpointReference(value);
  try {
    const response = await request(apiUrl, {
      method: 'GET', redirect: 'error', credentials: 'omit',
      headers: { Accept: 'application/vnd.github+json', 'User-Agent': 'Crusoe-Benchmark-Verifier' },
      signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(10000)]) : AbortSignal.timeout(10000),
    });
    if (response.status !== 200) {
      await response.body?.cancel();
      if (response.status === 404) throw new PublicCheckpointError('Base checkpoint is not publicly accessible on GitHub: the repository or commit was not found.', true);
      throw new PublicCheckpointError(`Public GitHub checkpoint could not be confirmed (HTTP ${response.status}).`);
    }
    if (!response.body) throw new PublicCheckpointError('GitHub returned an empty checkpoint response.');
    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let length = 0;
    try {
      while (true) {
        const next = await reader.read();
        if (next.done) break;
        length += next.value.byteLength;
        if (length > 1024 * 1024) throw new PublicCheckpointError('GitHub checkpoint response exceeds 1 MiB.');
        chunks.push(next.value);
      }
    } finally { await reader.cancel().catch(() => {}); }
    const bytes = new Uint8Array(length);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    let result: unknown;
    try { result = JSON.parse(new TextDecoder().decode(bytes)); }
    catch { throw new PublicCheckpointError('GitHub returned an invalid checkpoint response.'); }
    const resolved = result && typeof result === 'object' && Object.hasOwn(result, 'sha')
      ? (result as Record<string, unknown>).sha : undefined;
    if (typeof resolved !== 'string' || resolved.toLowerCase() !== sha.toLowerCase()) {
      throw new PublicCheckpointError('Public GitHub checkpoint did not resolve to the submitted commit SHA.', true);
    }
  } catch (error) {
    if (error instanceof PublicCheckpointError) throw error;
    throw new PublicCheckpointError('Could not confirm public access to the GitHub base checkpoint. The verifier was not called.');
  }
}
