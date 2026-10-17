import assert from 'node:assert/strict';
import test from 'node:test';
import { agentAccess } from '../../hosting/lib/agent-access.ts';

const URL = 'https://private-harness.example';
const SECRET = 'private-harness-server-key';
const OPERATOR = 'operator@example.test';
const USER_ID = 'site-operator-123';
const env = {
  AGENT_HARNESS_URL: URL,
  AGENT_HARNESS_API_KEY: SECRET,
  AGENT_HARNESS_ALLOWED_EMAILS: OPERATOR,
};

function request(path = 'runs', headers: Record<string, string> = {}) {
  return new Request(`https://dashboard.example/api/agents/${path}`, { headers });
}

function identity(email = OPERATOR, userId = USER_ID) {
  return { 'oai-authenticated-user-id': userId, 'oai-authenticated-user-email': email };
}

async function denied(path: string, settings: Record<string, unknown> = env,
  headers: Record<string, string> = {}) {
  const response = agentAccess(request(path, headers), ['agents', ...path.split('/')], settings);
  assert.ok(response);
  assert.equal(response.headers.get('Cache-Control'), 'no-store');
  const body = await response.json() as Record<string, unknown>;
  const serialized = JSON.stringify(body);
  for (const sensitive of [SECRET, URL, USER_ID, OPERATOR]) {
    assert.ok(!serialized.includes(sensitive), `Response leaked ${sensitive}`);
  }
  return { response, body };
}

test('unconfigured hosted harness delegates status to the connection helper without probing', () => {
  for (const settings of [{}, { AGENT_HARNESS_URL: '' }]) {
    assert.equal(agentAccess(request('config'), ['agents', 'config'], settings), null);
    assert.equal(agentAccess(request(), ['agents', 'runs'], settings), null);
  }
  assert.equal(agentAccess(request(), ['benchmarks'], env), null);
});

test('hosted agent controls require a configured HTTPS endpoint before user access', async () => {
  for (const endpoint of ['http://private-harness.example', 'file:///private/harness',
    'ftp://private-harness.example', 'not-an-endpoint']) {
    const settings = { ...env, AGENT_HARNESS_URL: endpoint };
    const config = await denied('config', settings, identity());
    assert.equal(config.response.status, 200);
    assert.equal(config.body.configured, false);
    assert.equal(config.body.requires_sign_in, undefined);
    assert.ok(!JSON.stringify(config.body).includes(endpoint));
    const runs = await denied('runs', settings, identity());
    assert.equal(runs.response.status, 503);
  }
});

test('unset or blank operator allowlists deny compute control even to a signed-in visitor', async () => {
  for (const settings of [
    { AGENT_HARNESS_URL: URL, AGENT_HARNESS_API_KEY: SECRET },
    { ...env, AGENT_HARNESS_ALLOWED_EMAILS: ' ,  ', AGENT_HARNESS_ALLOWED_USER_IDS: ' , ' },
  ]) {
    const config = await denied('config', settings, identity());
    assert.equal(config.response.status, 200);
    assert.equal(config.body.configured, false);
    assert.ok(String(config.body.message).includes('allowlist'));
    assert.equal((await denied('runs', settings, identity())).response.status, 503);
  }
});

test('anonymous visitors receive a sign-in action from config and 401 on agent routes', async () => {
  const config = await denied('config');
  assert.equal(config.response.status, 200);
  assert.equal(config.body.configured, false);
  assert.equal(config.body.requires_sign_in, true);
  assert.equal(config.body.sign_in_url, '/signin-with-chatgpt?return_to=%2F');
  assert.ok(String(config.body.sign_in_url).startsWith('/'));
  for (const path of ['runs', 'resources', 'status', `runs/${USER_ID}/cancel`]) {
    const actual = await denied(path);
    assert.equal(actual.response.status, 401);
    assert.deepEqual(Object.keys(actual.body), ['error']);
  }
});

test('signed-in visitors outside both operator allowlists receive 403', async () => {
  const headers = identity('visitor@example.test', 'other-site-user');
  const config = await denied('config', env, headers);
  assert.equal(config.response.status, 200);
  assert.equal(config.body.configured, false);
  assert.equal(config.body.requires_sign_in, undefined);
  assert.equal(config.body.sign_in_url, undefined);
  assert.equal((await denied('runs', env, headers)).response.status, 403);
});

test('an allowed operator email is case-insensitive and may be comma-separated', () => {
  const settings = { ...env, AGENT_HARNESS_ALLOWED_EMAILS: 'first@example.test, OPERATOR@EXAMPLE.TEST, ' };
  assert.equal(agentAccess(request('config', identity('Operator@Example.Test')),
    ['agents', 'config'], settings), null);
  assert.equal(agentAccess(request('runs', identity('  operator@example.test  ')),
    ['agents', 'runs'], settings), null);
});

test('an allowed trusted site user ID grants access independently of the email allowlist', () => {
  const settings = { ...env, AGENT_HARNESS_ALLOWED_USER_IDS: `other-id, ${USER_ID}, ` };
  assert.equal(agentAccess(request('runs', identity('different@example.test')),
    ['agents', 'runs'], settings), null);
  const idOnly = { AGENT_HARNESS_URL: URL, AGENT_HARNESS_ALLOWED_USER_IDS: USER_ID };
  assert.equal(agentAccess(request('config', identity()), ['agents', 'config'], idOnly), null);
});

test('access requires both trusted user ID and email headers even if one matches an allowlist', async () => {
  const settings = { ...env, AGENT_HARNESS_ALLOWED_USER_IDS: USER_ID };
  const cases: Array<Record<string, string>> = [
    { 'oai-authenticated-user-email': OPERATOR },
    { 'oai-authenticated-user-id': USER_ID },
    identity(OPERATOR, ''),
    identity('', USER_ID),
  ];
  for (const headers of cases) {
    const actual = await denied('runs', settings, headers);
    assert.equal(actual.response.status, 401);
    const config = await denied('config', settings, headers);
    assert.equal(config.body.requires_sign_in, true);
  }
});
