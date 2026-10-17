import { useEffect, useRef, useState, type FormEvent } from 'react';
import './connections.css';
import HarnessProfiles from './HarnessProfiles';

type ProviderName = 'crusoe' | 'you';
type ProviderCheck = { ok: boolean; message: string; checked_at: number };
type Provider = { configured: boolean; api_key_set: boolean; model?: string | null; source: string; enabled: boolean; check: ProviderCheck | null };
type Settings = { providers: Record<ProviderName, Provider>; revision: number };
type Readiness = { ready: boolean; checks: Array<{ id: string; label: string; ok: boolean; required: boolean; message: string }> };
type Model = { id: string; supported_parameters?: string[] };
type Access = { configured: boolean; message?: string; requires_sign_in?: boolean; sign_in_url?: string };

class ConnectionError extends Error {
  constructor(status: number) {
    super(status === 401 ? 'Sign in to manage agent connections.'
      : status === 403 ? 'Your account does not have access to manage agent connections.'
        : status === 400 || status === 422 ? 'Check the connection details and try again.'
          : status === 409 ? 'The settings changed during this request. Refresh and try again.'
            : status === 429 ? 'The provider is busy or its rate limit was reached. Try again shortly.'
              : 'Unable to complete this request. Check that the harness backend is running.');
  }
}

// Credentials travel only in same-origin POST bodies. Error responses are never
// rendered verbatim because an upstream service could reflect a submitted key.
async function api<T>(path: string, signal: AbortSignal, body?: Record<string, string>): Promise<T> {
  const response = await fetch(`/api/agents${path}`, {
    signal, cache: 'no-store', credentials: 'same-origin',
    ...(body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {}),
  });
  if (!response.ok) throw new ConnectionError(response.status);
  try { return await response.json() as T; }
  catch { throw new Error('The backend returned an unreadable response. Refresh and try again.'); }
}

function checkDate(seconds: number) {
  return new Date(seconds * 1000).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
}

function ProviderStatus({ provider }: { provider: Provider }) {
  const text = !provider.enabled || !provider.api_key_set ? 'Not connected'
    : !provider.configured ? 'Model needed'
      : !provider.check ? 'Saved · not checked'
        : provider.check.ok ? 'Connection checked' : 'Check failed';
  return <span className={`connection-status ${provider.configured && provider.check?.ok ? 'connection-status-good' : ''}`}>{text}</span>;
}

export default function Connections({ active = true }: { active?: boolean }) {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [access, setAccess] = useState<Access | null>(null);
  const [readiness, setReadiness] = useState<Readiness | null>(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [readinessError, setReadinessError] = useState('');
  const [keys, setKeys] = useState<Record<ProviderName, string>>({ crusoe: '', you: '' });
  const [model, setModel] = useState('');
  const [models, setModels] = useState<Model[]>([]);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const currentController = useRef<AbortController | null>(null);
  const generation = useRef(0);
  const guard = useRef(false);

  async function refreshReadiness(signal: AbortSignal) {
    try {
      const result = await api<Readiness>('/settings/readiness', signal);
      if (!signal.aborted) { setReadiness(result); setReadinessError(''); }
    } catch {
      if (!signal.aborted) { setReadiness(null); setReadinessError('Readiness could not be checked. Refresh to try again.'); }
    }
  }

  async function load() {
    if (guard.current) return;
    currentController.current?.abort();
    const controller = new AbortController();
    currentController.current = controller;
    const requestGeneration = ++generation.current;
    setLoading(true); setError(''); setNotice('');
    try {
      const configuration = await api<Access>('/config', controller.signal);
      if (controller.signal.aborted || generation.current !== requestGeneration) return;
      setAccess(configuration);
      if (!configuration.configured) { setSettings(null); setReadiness(null); return; }
      const [configurationResult] = await Promise.all([
        api<Settings>('/settings/providers', controller.signal), refreshReadiness(controller.signal),
      ]);
      if (controller.signal.aborted || generation.current !== requestGeneration) return;
      setSettings(configurationResult); setModel(configurationResult.providers.crusoe.model || '');
    } catch (failure) {
      if (!controller.signal.aborted && generation.current === requestGeneration) {
        setError(failure instanceof Error ? failure.message : 'Unable to load connections.');
      }
    } finally {
      if (!controller.signal.aborted && generation.current === requestGeneration) setLoading(false);
    }
  }

  useEffect(() => {
    if (active) void load();
    return () => {
      currentController.current?.abort(); generation.current += 1; guard.current = false;
      setBusy(''); setKeys({ crusoe: '', you: '' });
    };
  }, [active]); // Refresh settings on activation; typed keys never persist between tabs.

  const changes = (name: ProviderName) => Boolean(keys[name].trim()) || (name === 'crusoe' && model.trim() !== (settings?.providers.crusoe.model || ''));

  async function perform(name: ProviderName, action: 'save' | 'test' | 'models' | 'remove') {
    if (!settings || guard.current) return;
    guard.current = true;
    const controller = new AbortController();
    currentController.current?.abort(); currentController.current = controller;
    const requestGeneration = ++generation.current;
    setBusy(`${name}-${action}`); setError(''); setNotice('');
    const valid = () => !controller.signal.aborted && generation.current === requestGeneration;
    try {
      let updated = settings;
      if (action === 'save' || (action !== 'remove' && changes(name))) {
        const body: Record<string, string> = {};
        if (keys[name].trim()) body.api_key = keys[name].trim();
        if (name === 'crusoe') body.model = model.trim();
        updated = await api<Settings>(`/settings/providers/${name}`, controller.signal, body);
        if (!valid()) return;
        setSettings(updated); setKeys((previous) => ({ ...previous, [name]: '' }));
        if (name === 'crusoe') { setModel(updated.providers.crusoe.model || ''); setModels([]); setModelsLoaded(false); }
      }
      if (action === 'test' || action === 'remove') {
        updated = await api<Settings>(`/settings/providers/${name}/${action}`, controller.signal, {});
        if (!valid()) return;
        setSettings(updated);
        if (action === 'remove') {
          setKeys((previous) => ({ ...previous, [name]: '' }));
          if (name === 'crusoe') { setModel(updated.providers.crusoe.model || ''); setModels([]); setModelsLoaded(false); }
        }
      }
      if (action === 'models') {
        const result = await api<{ models: Model[] }>('/settings/providers/crusoe/models', controller.signal);
        if (!valid()) return;
        setModels(result.models); setModelsLoaded(true);
      }
      if (!valid()) return;
      setNotice(action === 'remove' ? `${name === 'crusoe' ? 'Crusoe' : 'You.com'} disconnected for new runs.`
        : action === 'save' ? 'Connection settings saved. Run a connection check before launching.'
          : action === 'models' ? `${modelsLoaded ? 'Refreshed' : 'Loaded'} available Crusoe models.`
            : updated.providers[name].check?.ok ? 'Connection check passed.' : 'Connection check did not pass. See the provider details below.');
      await refreshReadiness(controller.signal);
    } catch (failure) {
      if (valid()) setError(failure instanceof Error ? failure.message : 'Unable to update this connection.');
    } finally {
      if (valid()) { guard.current = false; setBusy(''); }
    }
  }

  const disabled = loading || Boolean(busy);
  const crusoe = settings?.providers.crusoe;
  const you = settings?.providers.you;

  function submit(event: FormEvent<HTMLFormElement>, name: ProviderName) {
    event.preventDefault(); void perform(name, 'save');
  }

  return <div className="connections">
    <div className="connections-heading">
      <div><h1>Connections</h1><p>Connect the model and research tools used by your agents.</p></div>
      <button type="button" className="button button-secondary" disabled={disabled} onClick={() => void load()}>{loading ? 'Refreshing…' : 'Refresh status'}</button>
    </div>
    {error && <p className="connection-error" role="alert">{error}</p>}
    {notice && <p className="connection-notice" role="status">{notice}</p>}
    {loading && !settings && <p className="connection-loading">Loading connections…</p>}
    {access && !access.configured && <div className="connection-unavailable"><h2>Connect the harness backend</h2><p>{access.message || 'The dashboard needs a running agent harness to save provider settings.'}</p>
      {access.requires_sign_in && access.sign_in_url && <a className="button button-secondary" href={access.sign_in_url}>Sign in</a>}
    </div>}
    {settings && crusoe && you && <>
      <section className="connection-readiness" aria-labelledby="connection-readiness-title">
        <div className="connection-section-heading"><h2 id="connection-readiness-title">Agent readiness</h2><span className={`connection-status ${readiness?.ready ? 'connection-status-good' : ''}`}>{readiness?.ready ? 'Ready to launch' : readiness ? 'Setup needed' : 'Check unavailable'}</span></div>
        {readiness && <ul className="connection-checks">{readiness.checks.map((check) => <li key={check.id}>
          <span className={`connection-check-icon ${check.ok ? 'connection-check-good' : check.required ? 'connection-check-needed' : ''}`} aria-hidden="true">{check.ok ? '✓' : check.required ? '!' : '–'}</span>
          <div><strong>{check.label}{!check.required && <span className="connection-optional">Optional</span>}</strong><p>{check.message}</p></div>
          <span className="connection-check-label">{check.ok ? 'Ready' : check.required ? 'Needs setup' : 'Optional'}</span>
        </li>)}</ul>}
        {readinessError && <p className="connection-note">{readinessError}</p>}
        <p className="connection-note">Provider settings apply to new agent runs. Active runs keep their current connections.</p>
      </section>
      <HarnessProfiles active={active} />
      <div className="connection-grid">
        <section className="connection-card" aria-labelledby="crusoe-connection-title">
          <div className="connection-section-heading"><h2 id="crusoe-connection-title">Crusoe inference</h2><ProviderStatus provider={crusoe} /></div>
          <p className="connection-description">Optional model access for the built-in research harness. Codex uses its existing sign-in.</p>
          <form onSubmit={(event) => submit(event, 'crusoe')}>
            <fieldset disabled={disabled} className="connection-fields"><legend className="connection-visually-hidden">Crusoe inference credentials</legend>
              <label htmlFor="crusoe-api-key">Foundry API key<span className="connection-key-state">{crusoe.api_key_set ? 'A key is stored. Enter a new key to replace it.' : 'No API key stored.'}</span>
                <input id="crusoe-api-key" type="password" autoComplete="new-password" spellCheck={false} maxLength={4096} value={keys.crusoe} placeholder={crusoe.api_key_set ? 'Leave blank to keep the saved key' : 'Paste your Crusoe inference key'} onChange={(event) => setKeys((previous) => ({ ...previous, crusoe: event.target.value }))} />
              </label>
              <p className="connection-note">Use your <a href="https://console.crusoecloud.com/foundry/models" target="_blank" rel="noreferrer">Crusoe Foundry</a> inference key. Cloud capacity access keys are managed separately.</p>
              <div className="connection-model-loader"><label htmlFor="crusoe-model-choice">Available models</label><button type="button" className="connection-text-button" onClick={() => void perform('crusoe', 'models')} disabled={disabled || (!crusoe.api_key_set && !keys.crusoe.trim())}>{busy === 'crusoe-models' ? 'Loading…' : changes('crusoe') ? 'Save + load models' : 'Load models'}</button></div>
              <select id="crusoe-model-choice" value={models.some((item) => item.id === model) ? model : ''} disabled={!models.length || disabled} onChange={(event) => { if (event.target.value) setModel(event.target.value); }}>
                <option value="">{modelsLoaded ? models.length ? 'Choose a model or enter its ID below' : 'No models returned; enter an ID below' : 'Load models using your saved key'}</option>
                {models.map((item) => <option key={item.id} value={item.id}>{item.id}</option>)}
              </select>
              <label htmlFor="crusoe-model-id">Model ID<input id="crusoe-model-id" type="text" autoComplete="off" spellCheck={false} maxLength={256} value={model} placeholder="Choose a model above or enter its ID" onChange={(event) => setModel(event.target.value)} /></label>
              <p className="connection-note">Choose a model with tool calling and structured JSON support. Connection checks verify the selected model.</p>
              <div className="connection-actions"><button className="button" type="submit" disabled={disabled || !changes('crusoe')}>{busy === 'crusoe-save' ? 'Saving…' : 'Save connection'}</button>
                <button className="button button-secondary" type="button" disabled={disabled || !model.trim() || (!crusoe.api_key_set && !keys.crusoe.trim())} onClick={() => void perform('crusoe', 'test')}>{busy === 'crusoe-test' ? 'Checking…' : changes('crusoe') ? 'Save + test' : 'Test connection'}</button>
                <button className="connection-text-button connection-disconnect" type="button" disabled={disabled || (!crusoe.enabled && !crusoe.api_key_set)} onClick={() => void perform('crusoe', 'remove')}>{busy === 'crusoe-remove' ? 'Disconnecting…' : 'Disconnect'}</button></div>
            </fieldset>
          </form>
          {crusoe.source === 'environment' && <p className="connection-note">Using settings from the backend environment.</p>}
          {crusoe.check && <div className={`connection-test-result ${crusoe.check.ok ? 'connection-test-good' : 'connection-test-failed'}`}><p>{crusoe.check.message}</p><time dateTime={new Date(crusoe.check.checked_at * 1000).toISOString()}>Checked {checkDate(crusoe.check.checked_at)}</time></div>}
          <p className="connection-endpoint">Endpoint <code>api.inference.crusoecloud.com/v1</code></p>
        </section>
        <section className="connection-card" aria-labelledby="you-connection-title">
          <div className="connection-section-heading"><h2 id="you-connection-title">You.com search</h2><ProviderStatus provider={you} /></div>
          <p className="connection-description">Web research to find papers, compare approaches, and explore new ideas.</p>
          <form onSubmit={(event) => submit(event, 'you')}>
            <fieldset disabled={disabled} className="connection-fields"><legend className="connection-visually-hidden">You.com search credentials</legend>
              <label htmlFor="you-api-key">Search API key<span className="connection-key-state">{you.api_key_set ? 'A key is stored. Enter a new key to replace it.' : 'No API key stored.'}</span>
                <input id="you-api-key" type="password" autoComplete="new-password" spellCheck={false} maxLength={4096} value={keys.you} placeholder={you.api_key_set ? 'Leave blank to keep the saved key' : 'Paste your You.com API key'} onChange={(event) => setKeys((previous) => ({ ...previous, you: event.target.value }))} />
              </label>
              <p className="connection-note">Create a Search API key in the <a href="https://you.com/platform" target="_blank" rel="noreferrer">You.com platform</a>. Search is optional; agents can still read the Library without it.</p>
              <div className="connection-actions"><button className="button" type="submit" disabled={disabled || !changes('you')}>{busy === 'you-save' ? 'Saving…' : 'Save connection'}</button>
                <button className="button button-secondary" type="button" disabled={disabled || (!you.api_key_set && !keys.you.trim())} onClick={() => void perform('you', 'test')}>{busy === 'you-test' ? 'Checking…' : changes('you') ? 'Save + test' : 'Test connection'}</button>
                <button className="connection-text-button connection-disconnect" type="button" disabled={disabled || (!you.enabled && !you.api_key_set)} onClick={() => void perform('you', 'remove')}>{busy === 'you-remove' ? 'Disconnecting…' : 'Disconnect'}</button></div>
            </fieldset>
          </form>
          {you.source === 'environment' && <p className="connection-note">Using settings from the backend environment.</p>}
          {you.check && <div className={`connection-test-result ${you.check.ok ? 'connection-test-good' : 'connection-test-failed'}`}><p>{you.check.message}</p><time dateTime={new Date(you.check.checked_at * 1000).toISOString()}>Checked {checkDate(you.check.checked_at)}</time></div>}
          <p className="connection-endpoint">Endpoint <code>ydc-index.io/v1/search</code></p>
        </section>
      </div>
      <p className="connection-storage-note">Keys are encrypted on the harness backend and are never returned to the browser. Connection tests make small provider requests and may incur usage charges.</p>
    </>}
  </div>;
}
