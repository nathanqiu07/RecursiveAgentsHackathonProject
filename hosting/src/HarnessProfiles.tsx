import { useEffect, useRef, useState, type FormEvent } from 'react';
import { harnessLabel, type HarnessCatalog, type HarnessId, type ModelProfile } from './harnesses';

type AuthMode = ModelProfile['auth_mode'];
type InferenceProvider = 'crusoe' | 'openai' | 'nvidia' | 'custom';
type Draft = { id?: string; name: string; harness: HarnessId; model: string; base_url: string; auth_mode: AuthMode; api_key: string; provider?: InferenceProvider };
const inferenceProviders = [
  { id: 'crusoe', label: 'Crusoe', baseUrl: 'https://api.inference.crusoecloud.com/v1', keyLabel: 'Crusoe Foundry API key' },
  { id: 'openai', label: 'OpenAI', baseUrl: 'https://api.openai.com/v1', keyLabel: 'OpenAI API key' },
  { id: 'nvidia', label: 'NVIDIA API Endpoints', baseUrl: 'https://integrate.api.nvidia.com/v1', keyLabel: 'NVIDIA API key' },
  { id: 'custom', label: 'Custom compatible endpoint', baseUrl: '', keyLabel: 'Provider API key' },
] as const;
const crusoeEndpoint = inferenceProviders[0].baseUrl;
const nvidiaEndpoint = inferenceProviders[2].baseUrl;
const normalizedEndpoint = (value: string | null | undefined) => (value || '').trim().replace(/\/+$/, '');
const inferProvider = (value: string | null | undefined): InferenceProvider => inferenceProviders.find((item) => item.baseUrl && normalizedEndpoint(item.baseUrl) === normalizedEndpoint(value))?.id || 'custom';
const effectiveEndpoint = (harness: HarnessId, value: string | null | undefined) => normalizedEndpoint(value) || (harness === 'nemoclaw' ? nvidiaEndpoint : harness === 'claude-code' ? 'https://api.anthropic.com' : '');
const emptyDraft = (): Draft => ({ name: '', harness: 'codex', model: '', base_url: '', auth_mode: 'codex-oauth', api_key: '' });

async function request<T>(path: string, signal: AbortSignal, body?: Record<string, string>) {
  const response = await fetch(`/api/agents/settings/harnesses${path}`, { signal, cache: 'no-store', credentials: 'same-origin',
    ...(body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {}),
  });
  if (!response.ok) throw new Error(response.status === 400 || response.status === 422 ? 'Check the profile details and try again.' : 'Unable to update harness profiles. Check that the backend is running and you have access.');
  return await response.json() as T;
}

export default function HarnessProfiles({ active }: { active: boolean }) {
  const [catalog, setCatalog] = useState<HarnessCatalog | null>(null);
  const [draft, setDraft] = useState<Draft>(emptyDraft);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [refresh, setRefresh] = useState(0);
  const controller = useRef<AbortController | null>(null);
  const guard = useRef(false);

  useEffect(() => {
    if (!active) return;
    const pending = new AbortController(); controller.current = pending;
    setLoading(true); setError('');
    void request<HarnessCatalog>('', pending.signal).then((value) => {
      if (!pending.signal.aborted) setCatalog(value);
    }).catch(() => {
      if (!pending.signal.aborted) setError('Unable to load harnesses and model profiles. Refresh to try again.');
    }).finally(() => { if (!pending.signal.aborted) setLoading(false); });
    return () => { pending.abort(); controller.current?.abort(); guard.current = false; setBusy(''); setDraft((value) => ({ ...value, api_key: '' })); };
  }, [active, refresh]);

  const selected = catalog?.harnesses.find((item) => item.id === draft.harness);
  const editing = catalog?.profiles.find((item) => item.id === draft.id);
  const disabled = loading || Boolean(busy);
  const provider = draft.provider || inferProvider(draft.base_url || (draft.harness === 'nemoclaw' ? nvidiaEndpoint : ''));
  const providerPreset = inferenceProviders.find((item) => item.id === provider)!;
  const endpointChanged = Boolean(editing?.api_key_set && normalizedEndpoint(draft.base_url) !== normalizedEndpoint(editing.base_url));
  const needsKey = draft.auth_mode === 'api-key' && (!editing?.api_key_set || endpointChanged);

  function chooseHarness(harness: HarnessId) {
    setDraft({ name: '', harness, model: '', base_url: harness === 'nemoclaw' ? crusoeEndpoint : '', api_key: '', auth_mode: harness === 'codex' ? 'codex-oauth' : harness === 'claude-code' || harness === 'nemoclaw' ? 'api-key' : 'environment', ...(harness === 'nemoclaw' ? { provider: 'crusoe' } : {}) });
    setNotice('');
  }

  function chooseProvider(value: InferenceProvider) {
    const preset = inferenceProviders.find((item) => item.id === value)!;
    setDraft((previous) => ({ ...previous, provider: value, base_url: preset.baseUrl, auth_mode: 'api-key',
      api_key: (previous.provider || inferProvider(previous.base_url || nvidiaEndpoint)) === value && effectiveEndpoint(previous.harness, previous.base_url) === effectiveEndpoint(previous.harness, preset.baseUrl) ? previous.api_key : '' }));
  }

  function changeEndpoint(value: string) {
    setDraft((previous) => ({ ...previous, base_url: value, ...(previous.harness === 'nemoclaw' ? { provider: inferProvider(value) } : {}),
      api_key: effectiveEndpoint(previous.harness, previous.base_url) === effectiveEndpoint(previous.harness, value) ? previous.api_key : '' }));
  }

  async function mutate(action: 'save' | 'remove', id?: string) {
    if (guard.current) return;
    guard.current = true; setBusy(action === 'remove' ? id || 'remove' : 'save'); setError(''); setNotice('');
    const pending = new AbortController(); controller.current?.abort(); controller.current = pending;
    try {
      const body: Record<string, string> = action === 'remove' ? {} : {
        ...(draft.id ? { id: draft.id } : {}), name: draft.name.trim(), harness: draft.harness,
        model: draft.model.trim(), auth_mode: draft.auth_mode,
        ...(draft.auth_mode === 'api-key' && draft.base_url.trim() ? { base_url: draft.base_url.trim() } : {}),
        ...(draft.auth_mode === 'api-key' && draft.api_key.trim() ? { api_key: draft.api_key.trim() } : {}),
      };
      const value = await request<HarnessCatalog>(action === 'remove' ? `/profiles/${id}/remove` : '/profiles', pending.signal, body);
      if (pending.signal.aborted) return;
      setCatalog(value); setDraft(action === 'save' ? { ...emptyDraft(), harness: draft.harness, auth_mode: draft.auth_mode, base_url: draft.base_url, provider: draft.provider } : emptyDraft()); setNotice(action === 'save' ? 'Model profile saved. Select it when launching an agent.' : 'Model profile removed.');
    } catch {
      if (!pending.signal.aborted) setError('Unable to save this change. Check the profile details and backend connection.');
    } finally { if (!pending.signal.aborted) { guard.current = false; setBusy(''); } }
  }

  function submit(event: FormEvent<HTMLFormElement>) { event.preventDefault(); void mutate('save'); }

  return <section className="connection-harnesses" aria-labelledby="harness-profiles-title">
    <div className="connection-section-heading"><h2 id="harness-profiles-title">Agent harnesses &amp; model profiles</h2><button type="button" className="connection-text-button" disabled={disabled} onClick={() => setRefresh((value) => value + 1)}>{loading ? 'Refreshing…' : 'Refresh harnesses'}</button></div>
    <p className="connection-description">Codex uses the existing sign-in by default. Save multiple model profiles for each harness, then choose one for each run.</p>
    {error && <p className="connection-error" role="alert">{error}</p>}
    {notice && <p className="connection-notice" role="status">{notice}</p>}
    {loading && !catalog && <p className="connection-loading">Checking installed harnesses…</p>}
    {catalog && <>
      <ul className="connection-harness-list">{catalog.harnesses.map((item) => <li key={item.id}><div><strong>{item.label}{item.id === catalog.default_harness && <span className="connection-optional">Default</span>}</strong><span className={`connection-status ${item.available ? 'connection-status-good' : ''}`}>{item.available ? 'Available' : 'Needs setup'}</span></div><p>{item.message}</p></li>)}</ul>
      <div className="connection-profile-workspace">
        <div className="connection-saved-profiles"><h3>Saved model profiles</h3>{!catalog.profiles.filter((item) => item.enabled).length ? <p className="connection-note">No saved profiles yet. Codex uses the worker’s existing sign-in and default model when available.</p>
          : <ul>{catalog.profiles.filter((item) => item.enabled).map((profile) => <li key={profile.id}><div><strong>{profile.name}</strong><p>{harnessLabel(profile.harness)} · {profile.model || 'Runtime default'}</p><span>{profile.auth_mode === 'codex-oauth' ? 'Codex sign-in' : profile.auth_mode === 'environment' ? 'Backend environment' : profile.api_key_set ? 'API key stored' : 'API key needed'}</span></div><div className="connection-profile-actions"><button type="button" className="connection-text-button" disabled={disabled} onClick={() => { setDraft({ ...profile, base_url: profile.base_url || '', api_key: '', ...(profile.harness === 'nemoclaw' ? { provider: inferProvider(profile.base_url || nvidiaEndpoint) } : {}) }); setError(''); setNotice(''); }}>Edit</button><button type="button" className="connection-text-button" disabled={disabled} onClick={() => void mutate('remove', profile.id)}>{busy === profile.id ? 'Removing…' : 'Remove'}</button></div></li>)}</ul>}
        </div>
        <form className="connection-profile-form" onSubmit={submit}><h3>{draft.id ? 'Edit model profile' : 'Add a model profile'}</h3><fieldset disabled={disabled} className="connection-fields"><legend className="connection-visually-hidden">Model profile details</legend>
          <label htmlFor="profile-harness">Harness<select id="profile-harness" value={draft.harness} onChange={(event) => chooseHarness(event.target.value as HarnessId)}>{catalog.harnesses.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
          {selected && <p className="connection-note">{selected.message}</p>}
          <label htmlFor="profile-name">Profile name<input id="profile-name" required maxLength={100} value={draft.name} placeholder="e.g. Codex research" onChange={(event) => setDraft((value) => ({ ...value, name: event.target.value }))} /></label>
          <label htmlFor="profile-model">Model ID<input id="profile-model" list="profile-model-options" required maxLength={256} autoComplete="off" spellCheck={false} value={draft.model} placeholder={draft.harness === 'nemoclaw' ? 'Exact model ID from the inference provider' : 'Choose or enter a model ID'} onChange={(event) => setDraft((value) => ({ ...value, model: event.target.value }))} /><datalist id="profile-model-options">{(selected?.models || []).map((item) => <option key={item.id} value={item.id}>{item.label || item.id}</option>)}</datalist></label>
          {draft.harness === 'nemoclaw' && <p className="connection-note">Model IDs depend on the provider. For Nemotron through Crusoe, copy the exact model ID from your Crusoe catalog.</p>}
          <label htmlFor="profile-auth">Authentication<select id="profile-auth" value={draft.auth_mode} onChange={(event) => setDraft((value) => ({ ...value, auth_mode: event.target.value as AuthMode, api_key: '', base_url: event.target.value === 'api-key' && value.harness === 'nemoclaw' ? crusoeEndpoint : '', ...(value.harness === 'nemoclaw' ? { provider: 'crusoe' } : {}) }))}>
            {draft.harness === 'codex' ? <option value="codex-oauth">Existing Codex sign-in</option> : <><option value="environment">Backend environment</option><option value="api-key">API key</option></>}
          </select></label>
          {draft.auth_mode === 'codex-oauth' && <p className="connection-note">Uses the Codex account already signed in on the worker. OAuth credentials stay on the worker.</p>}
          {draft.auth_mode === 'environment' && <p className="connection-note">{draft.harness === 'builtin' ? 'Uses the Crusoe connection below or the backend model environment.' : 'Uses this harness’s credentials and endpoint from the worker environment.'}</p>}
          {draft.auth_mode === 'api-key' && <>
            {draft.harness === 'nemoclaw' && <><label htmlFor="profile-inference-provider">Inference provider<select id="profile-inference-provider" value={provider} onChange={(event) => chooseProvider(event.target.value as InferenceProvider)}>{inferenceProviders.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
              <p className="connection-note">{provider === 'openai' ? 'Use an OpenAI API key for this connection. The existing Codex sign-in applies to Codex profiles.' : provider === 'crusoe' ? 'Use a Crusoe Foundry inference key and an exact model ID available in your Crusoe catalog, including Nemotron when offered.' : provider === 'nvidia' ? 'Use an NVIDIA API key and the exact model ID offered by this endpoint.' : 'Use the API key and exact model ID for your OpenAI-compatible endpoint.'} Confirm the model ID when changing providers.</p></>}
            <label htmlFor="profile-base-url">API base URL<input id="profile-base-url" type="url" required={draft.harness === 'nemoclaw'} autoComplete="off" spellCheck={false} maxLength={2048} value={draft.base_url} placeholder={draft.harness === 'claude-code' ? 'https://api.anthropic.com' : draft.harness === 'nemoclaw' ? providerPreset.baseUrl || 'https://your-provider.example/v1' : 'https://your-provider.example/v1'} onChange={(event) => changeEndpoint(event.target.value)} /></label>
            {draft.harness === 'claude-code' && <p className="connection-note">Claude Code requires an Anthropic Messages compatible endpoint.</p>}
            <label htmlFor="profile-api-key">{draft.harness === 'nemoclaw' ? providerPreset.keyLabel : 'API key'}<span className="connection-key-state">{endpointChanged ? 'The endpoint changed. Enter a key for the new provider.' : editing?.api_key_set ? 'Leave blank to keep the stored key.' : 'The key is encrypted on the backend.'}</span><input id="profile-api-key" type="password" required={needsKey} autoComplete="new-password" spellCheck={false} maxLength={4096} value={draft.api_key} onChange={(event) => setDraft((value) => ({ ...value, api_key: event.target.value }))} /></label>
          </>}
          <div className="connection-actions"><button className="button" type="submit" disabled={disabled}>{busy === 'save' ? 'Saving…' : draft.id ? 'Save profile' : 'Add profile'}</button>{draft.id && <button className="button button-secondary" type="button" onClick={() => setDraft(emptyDraft())}>Cancel edit</button>}</div>
        </fieldset></form>
      </div>
    </>}
  </section>;
}
