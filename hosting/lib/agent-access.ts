/** Sites supplies trusted visitor identity; a public Site does not grant compute access. */
export function agentAccess(request: Request, path: string[], env: Record<string, unknown>): Response | null {
  if (path[0] !== 'agents') return null;
  const config = path.length === 2 && path[1] === 'config' && request.method === 'GET';
  let message = '';
  let signIn = false;
  if (typeof env.AGENT_HARNESS_URL !== 'string' || !env.AGENT_HARNESS_URL) return null;
  try {
    if (new URL(env.AGENT_HARNESS_URL).protocol !== 'https:') {
      message = 'Configure a secure HTTPS harness endpoint to enable hosted agent controls.';
    }
  } catch { message = 'Check the server agent harness connection settings.'; }
  const emails = typeof env.AGENT_HARNESS_ALLOWED_EMAILS === 'string'
    ? env.AGENT_HARNESS_ALLOWED_EMAILS.split(',').map((email) => email.trim().toLowerCase()).filter(Boolean) : [];
  const ids = typeof env.AGENT_HARNESS_ALLOWED_USER_IDS === 'string'
    ? env.AGENT_HARNESS_ALLOWED_USER_IDS.split(',').map((id) => id.trim()).filter(Boolean) : [];
  let status = 503;
  if (!message && !emails.length && !ids.length) message = 'Configure the agent operator allowlist in the dashboard server settings.';
  const userId = request.headers.get('oai-authenticated-user-id');
  const email = request.headers.get('oai-authenticated-user-email');
  if (!message && (!userId || !email)) {
    message = 'Sign in to view and launch agents.';
    signIn = true;
    status = 401;
  }
  if (!message && !ids.includes(userId!) && !emails.includes(email!.trim().toLowerCase())) {
    message = 'Your account does not have permission to manage agents.';
    status = 403;
  }
  if (!message) return null;
  return Response.json(config ? {
    configured: false, message,
    ...(signIn ? { requires_sign_in: true, sign_in_url: '/signin-with-chatgpt?return_to=%2F' } : {}),
  } : { error: message }, { status: config ? 200 : status, headers: { 'Cache-Control': 'no-store' } });
}
